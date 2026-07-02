# RAG Codebase Audit Findings

Audit result of the RAG pipeline (embeddings, chunking, retrieval, reranking, caching, generation).
Ordered by severity, most severe first.

## 1. Embedding model mismatch is never checked

**File:** `backend/app/index_data.py:257-268`

The Qdrant collection's dimension/model is only set once, at first-ever index.
There is no check that the current `EMBEDDING_MODEL` config still matches the model that created the existing collection.

**Failure scenario:** Change `INVENIOAI_EMBEDDING_MODEL` and upload a new PDF without clearing the collection.
If dimensions differ, the upsert crashes.
If dimensions coincidentally match but the model differs, old and new vectors silently live in incompatible embedding spaces in the same collection, degrading and poisoning retrieval quality with no error at all.

## 2. Table-aware chunking still splits tables mid-row

**File:** `backend/app/index_data.py:208-221`

The "table-aware" splitter falls back to `"\n| "` as a separator once `"\n\n\n"`/`"\n\n"` are not found.
For any markdown table larger than `CHUNK_SIZE` (1500 chars, common for financial/spec tables), `RecursiveCharacterTextSplitter` splits between table rows.
Overlap only carries the tail of the previous chunk, so later chunks contain data rows with no header row, exactly the "split mid-table, lose context" failure the fix was supposed to prevent.

**Failure scenario:** A 40-row financial table exceeding 1500 chars. Rows past the first chunk lose the column headers, so the LLM cannot map values to the correct metric/year.

## 3. Final prompt uses different query text on stream vs non-stream paths

**File:** `backend/app/rag_pipeline.py:323` vs `backend/app/rag_pipeline.py:509`

The non-streaming path builds the final answer prompt with `question=original_question`, but the streaming path (`rag_pipeline_stream_async`) uses `question=standalone_query` (the rewritten query) instead.
`QUERY_REWRITE_PROMPT` has no language-preservation instruction, so the LLM can rewrite the question into a different language or reword it, while `RAG_PROMPT` explicitly instructs to use the same language as the question.

**Failure scenario:** In `/query/stream`, an Indonesian question gets rewritten to English, and the answer comes back in the wrong language or loses conversational framing. Does not happen on `/query`.

## 4. Rewrite cache key uses a different history window than the rewrite itself

**File:** `backend/app/rag_pipeline.py:153-154` and `179-180`

`rewrite_query`/`rewrite_query_async` compute the rewrite-cache key using `format_history(history, max_items=1)` (only the latest turn), but generate the actual rewritten query using `format_history(history)` (last 5 turns, default).

**Failure scenario:** Two different conversations whose latest message happens to be identical but whose earlier turns differ hash to the same `rewrite_cache_key`. The second conversation is silently served the first conversation's cached standalone query, an unrelated retrieval context.

## 5. Sync `/query` path does not guard against empty retrieval

**File:** `backend/app/rag_pipeline.py`, `_run_rag_pipeline_with_query` (contrast with the guard at line 493 in the streaming path)

The synchronous `/query` path never checks whether `retrieved_docs`/`reranked_docs` is empty before formatting the prompt and calling the LLM. `format_docs([])` yields an empty context string, and the prompt is still sent to the LLM, which may hallucinate an answer instead of returning the "not found" message the streaming endpoint explicitly returns for the same condition.

**Failure scenario:** The same question via `/query` vs `/query/stream` behaves inconsistently when nothing relevant is retrieved.

## 6. Race condition in semantic cache read-modify-write

**File:** `backend/app/cache_manager.py:73-142` (`get_semantic`/`add_semantic`)

Read-modify-write on the `semantic_registry` list (`disk_cache.get` -> mutate -> `disk_cache.set`) has no lock.

**Failure scenario:** Two concurrent requests both read the same registry snapshot, each append their own entry, and the second `set` silently overwrites the first's addition, losing a semantic-cache entry under concurrent load.

## 7. Number/year guard is a blunt exact-match check

**File:** `backend/app/cache_manager.py:108-110`

The number/year guard requires the incoming and cached query's numeric-token sets to match exactly, rather than checking whether the differing number is actually relevant to the cached answer.

**Failure scenario:** An incidental number difference not tied to the actual answer (e.g. an unrelated figure the user mentions) forces a cache miss even when the queries are truly equivalent. Minor compared to the above; not a correctness/safety bug, but shows the fix is a blunt set-equality check.
