"""RAG pipeline orchestration.

The end-to-end flow is:

rewrite query → retrieve (dense or hybrid) → rerank → generate answer → return metrics.
"""

from __future__ import annotations

import logging
import re
import time
import hashlib
from .cache_manager import CacheManager
from typing import Any

from .config import NUM_FUSION_QUERIES, RETRIEVAL_K, SEMANTIC_CACHE_THRESHOLD
from .llm import get_llm as _get_llm
from .qdrant_conn import close_qdrant_client, is_qdrant_client_closed_error
from .reranker import rerank
from .retriever import build_retriever, retrieve_documents, retrieve_documents_async
import json
from .utils import format_docs, ThinkingParser
from .metrics import log_query
from .embeddings import get_embeddings


logger = logging.getLogger(__name__)


QUERY_REWRITE_PROMPT = """
Rewrite the question into a standalone question, then generate {num_fusion} alternate
phrasings of that standalone question to help retrieve relevant documents from a vector
database by covering different perspectives on the same question.

Chat History:
{history}

Question:
{question}

Respond with exactly {num_fusion_plus_one} lines, one question per line, nothing else:
- Line 1: the standalone question.
- Lines 2-{num_fusion_plus_one}: the alternate phrasings.
"""


RAG_PROMPT = """
Answer the question using the provided context. Follow the instructions strictly.

CORE RULES:
1. **Absolute Accuracy**: Accuracy is the #1 priority. If the information is not in the context, say you don't know. DO NOT hallucinate or guess.
2. **Strict Semantic & Concept Alignment**: Prevent term confusion and "semantic drift" across all domains:
   - For **Data/Financials**: Carefully distinguish similar but distinct terms (e.g., "Income before taxes", "Net income", "Comprehensive income") and trace table rows/columns precisely.
   - For **Policies/Legal**: Match exact clauses and conditional terms. Do not generalize (e.g., distinguish "Termination with cause" from "General termination").
   - For **Technical/Manuals**: Keep precise track of version numbers, device models, and specific parameter values (e.g., distinguish "Soft reboot" vs "Hard reset").
3. **Dynamic Unit Detection**: Identify the currency and scale (e.g., Millions, Thousands, Billions) directly from the document context, table headers, or footnotes. 
   - Look for symbols like "$m", "$k", "in millions", etc. 
   - If no unit is specified, report the raw number and mention that the unit was not found in the source.
4. **Year/Version Validation**: ALWAYS double check column headers, document dates, or version labels. Do not mix up data from different periods, years, or versions.
5. **Table & List Integrity**: Maintain the relationship between headers and values. For truncated tables, trace row labels to columns carefully.
6. **Adaptive Context**: 
   - For **Data/Financials**: Provide precise figures with identified units and a brief explanation.
   - For **Policies/Technical/Manuals**: Provide comprehensive, step-by-step explanations or conditions.
   - For **Conceptual/Scientific (Books/Research)**: Explain definitions, methods, or key concepts in a structured way. 
   - **Medical Caution**: For medical data, be extremely literal and precise. Never suggest actions; only report what is written.

EXAMPLE OF ANALYSIS:
Context: "User Manual: Press RED for 5s. Table: | Action | Duration | \n | Reset | 10s |"
User: "Bagaimana cara reset?"
<thinking>
Step 1 (Deconstruction): User asks for reset procedure.
Step 2 (Retrieval): Manual says RED button for 5s. Table says 10s for Reset.
Step 3 (Cross-Validation/Synthesis): There is a conflict between text (5s) and table (10s). The table specifically labels the action as 'Reset'.
Step 4 (Strategy): Report both if ambiguous, but prioritize the specific label.
</thinking>
Berdasarkan tinjauan pada dokumen panduan pengguna, terdapat dua informasi terkait prosedur reset yang perlu diperhatikan. Pada bagian tabel spesifikasi teknis, durasi reset ditetapkan selama 10 detik. Namun, pada instruksi naratif di bagian teks, disebutkan penekanan tombol MERAH selama 5 detik. Untuk hasil yang lebih akurat, disarankan untuk mengikuti ketentuan pada tabel spesifikasi (10 detik) karena labelnya lebih spesifik untuk tindakan 'Reset'.

Context:
{context}

Question:
{question}

Instructions:
1. **START with <thinking>** and perform a deep, step-by-step analysis.
2. **Thinking Format**: 
   - Write each step on a NEW LINE as a clear bullet point (e.g., `- **Step 1 (Deconstruction)**: ...`).
   - NEVER use single backticks (`) or triple backticks (```) anywhere inside the thinking block.
3. **Thinking Steps**: 
   - **Deconstruction**: Breakdown the user intent and identify key entities, years, exact requested terms, and concepts.
   - **Retrieval & Evidence**: Extract ALL relevant snippets. Don't stop at the first match.
   - **Contextual Reasoning**: Analyze the relationships between the snippets.
   - **Cross-Validation**: Verify units, dates, versions, and labels. Explicitly check if a retrieved term/concept matches the user's requested term exactly, or if it is a similar-sounding but different concept.
   - **Synthesis**: Formulate the final logic that leads to the answer.
4. **CLOSE with </thinking>**.
5. **Final Response Formatting**: 
   - Provide a **concise, direct, and professional narrative** (around 2 to 4 sentences or a short paragraph).
   - **NO WORDINESS**: Do not be overly verbose or beat around the bush, but do not give a 1-sentence answer either. Get straight to the point while providing necessary context.
   - **NO BACKTICKS**: NEVER use single backticks (`) or triple backticks (```) to highlight numbers, text, or anything else. If you need to emphasize something, use bold text (**) instead.
   - **NO INTRO OR FILLER**: Do NOT use filler phrases like "Berikut adalah...", "Informasi ini ditemukan pada bagian...", or "Based on the documents...". Start directly with the factual answer.
   - Use a polite and professional tone in the SAME LANGUAGE as the question.
6. DO NOT cite sources manually.

Sources:
{sources}
"""


_cache_manager: CacheManager | None = None

def get_cache_manager() -> CacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def _get_exact_cache_key(standalone_query: str) -> str:
    """L1 exact-match cache key, keyed on the normalized standalone query."""
    normalized = standalone_query.strip().lower()
    return f"rag_exact:{hashlib.md5(normalized.encode()).hexdigest()}"


def format_history(history: Any, max_items: int = 5) -> str:
    """Format chat history into a prompt-friendly string, limited to last N items."""
    if not history:
        return ""

    if isinstance(history, str):
        return history

    # If it's a list, take the last max_items
    items = list(history)[-max_items:]
    return "\n".join(str(item) for item in items)


def _parse_rewrite_response(raw: str, original_question: str) -> tuple[str, list[str]]:
    """Split the combined rewrite+fusion LLM response into
    (standalone_query, fusion_queries), falling back to the original
    question if the response is empty or malformed."""
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return original_question, []
    standalone = lines[0] if len(lines[0]) >= 2 else original_question
    return standalone, lines[1 : 1 + NUM_FUSION_QUERIES]


def rewrite_query(question: str, history: Any) -> tuple[str, list[str]]:
    """Rewrite a question into a standalone query plus fusion variations, with local caching."""
    cache = get_cache_manager()
    history_text = format_history(history)
    # Key must hash the same history text used to generate the rewrite, or two
    # conversations with the same latest message but different earlier turns
    # collide and serve each other's cached standalone query.
    rewrite_cache_key = f"rewrite_cache:{hashlib.md5(f'{question}:{history_text}'.encode()).hexdigest()}"

    cached_rewrite = cache.get(rewrite_cache_key)
    if cached_rewrite:
        return cached_rewrite["standalone"], cached_rewrite["fusion"]

    prompt = QUERY_REWRITE_PROMPT.format(
        history=history_text,
        question=question,
        num_fusion=NUM_FUSION_QUERIES,
        num_fusion_plus_one=NUM_FUSION_QUERIES + 1,
    )
    raw = _get_llm().invoke(prompt).content
    standalone, fusion = _parse_rewrite_response(raw, question)

    cache.set(rewrite_cache_key, {"standalone": standalone, "fusion": fusion}, ttl=3600)
    return standalone, fusion


async def rewrite_query_async(query: str, history: list[str]) -> tuple[str, list[str]]:
    """Async counterpart of `rewrite_query` - same combined rewrite+fusion prompt and caching."""
    cache = get_cache_manager()
    history_text = format_history(history)
    rewrite_cache_key = f"rewrite_cache:{hashlib.md5(f'{query}:{history_text}'.encode()).hexdigest()}"

    cached_rewrite = cache.get(rewrite_cache_key)
    if cached_rewrite:
        return cached_rewrite["standalone"], cached_rewrite["fusion"]

    prompt = QUERY_REWRITE_PROMPT.format(
        history=history_text,
        question=query,
        num_fusion=NUM_FUSION_QUERIES,
        num_fusion_plus_one=NUM_FUSION_QUERIES + 1,
    )
    try:
        llm = _get_llm()
        response = await llm.ainvoke(prompt)
        content = response.content
        if not isinstance(content, str):
            return query, []
        standalone, fusion = _parse_rewrite_response(content, query)
        cache.set(rewrite_cache_key, {"standalone": standalone, "fusion": fusion}, ttl=3600)
        return standalone, fusion
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        return query, []


def _lookup_cache(
    cache: CacheManager, standalone_query: str, exact_key: str
) -> tuple[Any, Any]:
    """L1 exact then L2 semantic cache lookup.

    Returns `(cached_result, query_embedding)`. `query_embedding` is the
    embedding computed for the L2 check (None on an L1 hit, since it's
    never needed) - callers that go on to run the pipeline on a full miss
    reuse it instead of re-embedding.
    """
    cached = cache.get(exact_key)
    if cached:
        return cached, None

    embedder = get_embeddings()
    query_embedding = embedder.embed_query(standalone_query)

    semantic_key = cache.get_semantic(
        query_embedding, threshold=SEMANTIC_CACHE_THRESHOLD, query_text=standalone_query
    )
    if semantic_key:
        cached_sem = cache.get(semantic_key)
        if cached_sem:
            cache.set(exact_key, cached_sem, ttl=3600)
            return cached_sem, query_embedding

    return None, query_embedding


def rag_pipeline(question: str, history: Any) -> dict[str, Any]:
    """Run the RAG pipeline with a 2-layer lean cache (L1 exact, L2 semantic)."""
    total_start = time.monotonic()
    cache = get_cache_manager()

    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            standalone_query, fusion_queries = rewrite_query(question, history)
            logger.info(f"Standalone Query: {standalone_query}")
            exact_key = _get_exact_cache_key(standalone_query)
            cached, query_embedding = _lookup_cache(cache, standalone_query, exact_key)
            if cached is not None:
                is_l1 = query_embedding is None
                logger.info(
                    f"RAG {'L1 Exact' if is_l1 else 'L2 Semantic'} Cache HIT for: {standalone_query[:50]}"
                )
                try:
                    log_query(
                        question=question,
                        answer=cached["answer"],
                        response_time=round(time.monotonic() - total_start, 2),
                        retrieval_time=0,
                        generation_time=0,
                        docs_retrieved=cached.get("metrics", {}).get("docs_retrieved", 0),
                        chunks_processed=cached.get("metrics", {}).get("chunks_processed", 0),
                        retrieval_scores=cached.get("metrics", {}).get("retrieval_scores", []),
                        thoughts=cached.get("thoughts", ""),
                        standalone_query=standalone_query
                    )
                except Exception:
                    logger.warning("Failed to log cached query metrics", exc_info=True)
                return cached

            if query_embedding is None:
                query_embedding = get_embeddings().embed_query(standalone_query)

            result = _run_rag_pipeline_with_query(standalone_query, question, history, fusion_queries)
            # Don't cache "nothing relevant found" - retrieval may succeed once
            # more documents are indexed later (mirrors the streaming path,
            # which also skips caching on empty retrieval).
            if result.get("metrics", {}).get("docs_retrieved", 0) > 0:
                cache.set(exact_key, result, ttl=3600)
                cache.add_semantic(query_embedding, exact_key, query_text=standalone_query)
            return result
        except Exception as exc:
            if attempt < (max_attempts - 1) and is_qdrant_client_closed_error(exc):
                logger.warning("Qdrant client was closed during query; recreating and retrying once")
                close_qdrant_client()
                continue
            raise
    raise RuntimeError("Unexpected RAG retry state")


def _run_rag_pipeline_with_query(
    standalone_query: str, original_question: str, history: Any, fusion_queries: list[str] = ()
) -> dict[str, Any]:
    """Internal helper for core RAG steps."""
    total_start = time.monotonic()
    retriever, vectorstore, client = build_retriever()

    retrieval_start = time.monotonic()
    retrieved_docs, retrieval_meta = retrieve_documents(
        standalone_query,
        dense_retriever=retriever,
        client=client,
        fusion_queries=fusion_queries,
    )

    if not retrieved_docs:
        retrieval_time = time.monotonic() - retrieval_start
        total_time = time.monotonic() - total_start
        res = {
            "answer": "Maaf, tidak ditemukan informasi relevan.",
            "thoughts": "",
            "sources": [],
            "metrics": {
                "total_time": round(total_time, 2),
                "retrieval_time": round(retrieval_time, 2),
                "generation_time": 0,
                "docs_retrieved": 0,
                "chunks_processed": 0,
                "retrieval_scores": [],
            },
        }
        try:
            log_query(
                question=original_question,
                response_time=total_time,
                retrieval_time=retrieval_time,
                generation_time=0,
                docs_retrieved=0,
                chunks_processed=0,
                standalone_query=standalone_query,
            )
        except Exception:
            logger.warning("Failed to log empty-retrieval query metrics", exc_info=True)
        return res

    reranked_docs, retrieval_scores, initial_ranks = rerank(standalone_query, retrieved_docs)
    retrieval_time = time.monotonic() - retrieval_start

    context, sources_str, sources_json = format_docs(reranked_docs, retrieval_scores)
    prompt = RAG_PROMPT.format(context=context, question=original_question, sources=sources_str)

    generation_start = time.monotonic()

    # Non-streaming mode gets the full response in one shot, so a plain
    # regex extraction is enough - no need for the incremental ThinkingParser
    # used by the streaming path.
    raw_answer = _get_llm().invoke(prompt).content
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw_answer, re.DOTALL)
    full_thoughts = thinking_match.group(1).strip() if thinking_match else ""
    full_answer = re.sub(r"<thinking>.*?</thinking>", "", raw_answer, flags=re.DOTALL).strip()

    generation_time = time.monotonic() - generation_start
    total_time = time.monotonic() - total_start

    res = {
        "answer": full_answer,
        "thoughts": full_thoughts,
        "sources": sources_json,
        "metrics": {
            "total_time": round(total_time, 2),
            "retrieval_time": round(retrieval_time, 2),
            "generation_time": round(generation_time, 2),
            "docs_retrieved": len(retrieved_docs),
            "chunks_processed": len(reranked_docs),
            "retrieval_scores": retrieval_scores,
        },
    }

    try:
        log_query(
            question=original_question,
            response_time=total_time,
            answer_length=len(full_answer),
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            docs_retrieved=len(retrieved_docs),
            chunks_processed=len(reranked_docs),
            retrieval_scores=retrieval_scores,
            initial_ranks=initial_ranks,
            thoughts=full_thoughts,
            standalone_query=standalone_query,
        )
    except Exception:
        logger.warning("Failed to log query metrics", exc_info=True)

    return res


async def rag_pipeline_stream_async(query: str, chat_history: list[str]):
    total_start = time.monotonic()
    cache = get_cache_manager()

    try:
        yield json.dumps({"step": "rewriting"}) + "\n"
        standalone_query, fusion_queries = await rewrite_query_async(query, chat_history)
        logger.info(f"Stream Standalone Query: {standalone_query}")
        exact_key = _get_exact_cache_key(standalone_query)

        cached, query_embedding = _lookup_cache(cache, standalone_query, exact_key)
        if cached is not None:
            is_l1 = query_embedding is None
            logger.info(f"Stream: {'L1 Exact' if is_l1 else 'L2 Semantic'} Cache HIT")
            yield json.dumps({"step": "cached", "answer": cached["answer"]}) + "\n"
            yield json.dumps({
                "step": "done",
                "answer": cached["answer"],
                "thoughts": cached.get("thoughts", ""),
                "sources": cached["sources"],
                "metrics": cached.get("metrics", {})
            }) + "\n"

            try:
                if is_l1:
                    log_query(
                        question=query,
                        response_time=round(time.monotonic() - total_start, 2),
                        answer_length=len(cached.get("answer", "")),
                        retrieval_time=0,
                        generation_time=0,
                        docs_retrieved=cached.get("metrics", {}).get("docs_retrieved", 0),
                        chunks_processed=cached.get("metrics", {}).get("chunks_processed", 0),
                        retrieval_scores=cached.get("metrics", {}).get("retrieval_scores", []),
                        thoughts=cached.get("thoughts", ""),
                        standalone_query=standalone_query,
                    )
                else:
                    log_query(
                        question=query,
                        response_time=round(time.monotonic() - total_start, 2),
                        answer_length=len(cached["answer"]),
                        retrieval_time=0,
                        generation_time=0,
                        thoughts=cached.get("thoughts", ""),
                        standalone_query=standalone_query,
                    )
            except Exception:
                pass
            return

        if query_embedding is None:
            query_embedding = get_embeddings().embed_query(standalone_query)

        yield json.dumps({"step": "retrieving", "query": standalone_query}) + "\n"
        retriever, vectorstore, client = build_retriever()
        
        retrieval_start = time.monotonic()
        try:
            docs, metadata = await retrieve_documents_async(
                standalone_query, dense_retriever=retriever, client=client, fusion_queries=fusion_queries
            )
        except Exception as e:
            if is_qdrant_client_closed_error(e):
                close_qdrant_client()
                retriever, vectorstore, client = build_retriever()
                docs, metadata = await retrieve_documents_async(
                    standalone_query, dense_retriever=retriever, client=client, fusion_queries=fusion_queries
                )
            else:
                raise
        retrieval_time = time.monotonic() - retrieval_start

        if not docs:
            yield json.dumps({
                "step": "done",
                "answer": "Maaf, tidak ditemukan informasi relevan.",
                "sources": [],
                "metrics": metadata
            }) + "\n"
            try:
                log_query(
                    question=query,
                    response_time=round(time.monotonic() - total_start, 2),
                    retrieval_time=retrieval_time,
                    generation_time=0,
                    docs_retrieved=0,
                    chunks_processed=0,
                    standalone_query=standalone_query,
                )
            except Exception:
                logger.warning("Failed to log empty-retrieval query metrics (stream)", exc_info=True)
            return

        yield json.dumps({"step": "reranking"}) + "\n"
        top_docs, retrieval_scores, initial_ranks = rerank(standalone_query, docs)
        metadata["retrieval_scores"] = retrieval_scores
        metadata["reranked_docs"] = len(top_docs)

        yield json.dumps({"step": "generating"}) + "\n"
        context_text, sources_str, sources_json = format_docs(top_docs, retrieval_scores)
        # Use the user's original wording (not the rewritten standalone query) so
        # the answer matches the language/tone the user actually typed - the sync
        # path already does this via `original_question`.
        prompt = RAG_PROMPT.format(context=context_text, question=query, sources=sources_str)
        
        llm = _get_llm()
        generation_start = time.monotonic()
        parser = ThinkingParser()
        full_answer = ""
        full_thoughts = ""
        
        async for chunk in llm.astream(prompt):
            if chunk.content:
                for msg_type, content in parser.feed(chunk.content):
                    if msg_type == "thinking":
                        full_thoughts += content
                        yield json.dumps({"step": "thinking", "content": content}) + "\n"
                    else:
                        full_answer += content
                        yield json.dumps({"step": "token", "content": content}) + "\n"
        
        # Flush remaining buffer
        for msg_type, content in parser.flush():
            if msg_type == "thinking":
                full_thoughts += content
                yield json.dumps({"step": "thinking", "content": content}) + "\n"
            else:
                full_answer += content
                yield json.dumps({"step": "token", "content": content}) + "\n"
        
        generation_time = time.monotonic() - generation_start
        total_time = time.monotonic() - total_start

        final_result = {
            "step": "done",
            "answer": full_answer,
            "thoughts": full_thoughts,
            "sources": sources_json,
            "metrics": {
                **metadata,
                "total_time": round(total_time, 2),
                "retrieval_time": round(retrieval_time, 2),
                "generation_time": round(generation_time, 2),
            }
        }
        
        cache_data = {
            "answer": full_answer,
            "thoughts": full_thoughts,
            "sources": sources_json,
            "metrics": final_result["metrics"]
        }
        cache.set(exact_key, cache_data, ttl=3600)
        cache.add_semantic(query_embedding, exact_key, query_text=standalone_query)
        
        yield json.dumps(final_result) + "\n"

        try:
            log_query(
                question=query,
                response_time=total_time,
                answer_length=len(full_answer),
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                docs_retrieved=len(docs),
                chunks_processed=len(top_docs),
                retrieval_scores=metadata.get("retrieval_scores", []),
                initial_ranks=initial_ranks,
                thoughts=full_thoughts,
                standalone_query=standalone_query,
            )
        except Exception:
            pass

    except Exception as e:
        logger.exception("Pipeline error")
        yield json.dumps({"step": "error", "message": str(e)}) + "\n"
