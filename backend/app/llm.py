"""Shared Groq LLM singleton.

Both the RAG generation step (`rag_pipeline.py`) and the MultiQueryRetriever
(`retriever.py`) need a Groq chat model. A single cached instance avoids
recreating a client (and its connection pool) on every request.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq

from .config import GROQ_API_KEY, LLM_MODEL


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY belum di-set. Isi di .env (lihat .env.example) sebelum menjalankan query."
        )

    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0,
    )
