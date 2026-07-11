# Backend Improvement Plan - InvenioAI

Rencana eksekusi berdasarkan `docs/backend-improvement-spec.md`.
Setiap increment adalah vertical slice yang shippable dan diverifikasi sebelum lanjut.
Tanggal: 2026-07-11.

## Urutan Increment

Prioritas: cepat + aman dulu, lalu kredibilitas, lalu performa, lalu efisiensi.

---

## Increment A - Konsistensi retrieval + config drift

Memenuhi: R1, R6.
Target file: `backend/app/retriever.py`, `backend/app/config.py`, `backend/app/rag_pipeline.py`, `CLAUDE.md`.

Langkah:

1. Putuskan perilaku hybrid: pertahankan similarity + RRF dense+sparse (lebih sederhana) dan perbarui docstring `retriever.py:116` serta `CLAUDE.md` agar tidak lagi mengklaim MMR di hybrid.
   Catatan: RRF dense+sparse (Qdrant native) sudah benar dan boleh diklaim; yang perlu dikoreksi adalah label "RAG Fusion" level-query yang sebenarnya hanya multi-query union. Putuskan implementasi RRF level-query nyata atau relabel menjadi "multi-query retrieval".
2. Selaraskan `RERANKER_MODEL` di `config.py:130` dengan docstring dan `CLAUDE.md` (satu nama definitif).
3. Hapus import mati `RETRIEVAL_K` di `rag_pipeline.py:16` dan return value `vectorstore` yang tidak dipakai.
4. Ganti fallback reranker dari skor `0.0` ke nilai netral yang tidak dihitung sebagai relevant/irrelevant di metrics, atau tandai entri sebagai "no-score".

Verifikasi: `pytest` hijau; `npm run lint` (tsc backend jika ada) bersih; grep memastikan tidak ada klaim MMR-hybrid tersisa.

Estimasi: kecil.

---

## Increment B - Metrik IR kredibel

Memenuhi: R2.
Target file: `backend/app/metrics.py`, `backend/eval/` (baru), `frontend*/` label dashboard.

Langkah:

1. Relabel metrik berbasis skor reranker di response `/metrics` dan dashboard menjadi "reranker-score proxy".
2. Buat `backend/eval/qrels.jsonl` (~30-50 query dengan chunk-id relevan) dan `backend/eval/run_eval.py`.
3. Implementasi perhitungan Precision/Recall/nDCG/MRR terhadap qrels nyata (bukan skor reranker).
4. Tambah target dokumentasi cara menjalankan eval di `CLAUDE.md`.

Verifikasi: jalankan `python backend/eval/run_eval.py`, hasilkan angka; snapshot sebagai baseline untuk increment berikutnya.

Estimasi: sedang (butuh kurasi qrels manual).

---

## Increment C - Concurrency aman

Memenuhi: R3.
Target file: `backend/app/rag_pipeline.py`, mungkin `embeddings.py`, `reranker.py`.

Langkah:

1. Bungkus `embed_query`, `rerank`, dan cache IO di jalur `rag_pipeline_stream_async` dengan `anyio.to_thread.run_sync` atau `loop.run_in_executor`.
2. Pertimbangkan `AsyncQdrantClient` untuk retrieval async sejati.
3. Uji beban ringan (mis. beberapa request paralel) untuk konfirmasi tidak ada blocking.

Verifikasi: uji konkuren menunjukkan latensi request paralel tidak terserialisasi; `pytest` hijau.

Estimasi: sedang.

---

## Increment D - Caching efektif + efisiensi LLM

Memenuhi: R4, R5.
Target file: `backend/app/rag_pipeline.py`, `backend/app/cache_manager.py`, `backend/app/config.py`.

Langkah:

1. Turunkan threshold semantic cache ke env-configurable (default 0.93) dengan number-guard tetap.
2. Migrasi semantic registry dari blob JSON ke vector search (Qdrant collection khusus), fallback ke diskcache bila tidak tersedia.
3. Jadikan query-rewrite kondisional: skip bila history kosong.
4. Kalibrasi `NUM_FUSION_QUERIES` terhadap eval set dari Increment B.

Verifikasi: cache hit ratio naik terhadap trace uji; eval set tidak turun kualitas; jumlah pemanggilan LLM per query berkurang.

Estimasi: sedang-besar.

---

## Backlog Optimasi (opsional, setelah A-D)

Diambil dari section Improvisasi di `docs/backend-audit.md`:

- RRF nyata + weighting dense/sparse via `client.query_points` + `FusionQuery`.
- Faithfulness / grounding check (RAGAS atau LLM-judge ringan).
- Contextual / sentence-window chunking.
- Adaptive retrieval (query router untuk sapaan / pertanyaan sederhana).
- Timeout + retry backoff + circuit breaker pada Groq dan Qdrant.
- Observability: structured logging JSON, trace id, ekspor metrik Prometheus.
- Validasi input Pydantic (batas panjang question/history), rate limiting.
- Pydantic response models menggantikan `Dict[str, Any]`.
- `pydantic-settings` menggantikan helper `_env_*`.
- Dockerfile multi-stage dengan pre-download bobot model.

## Cara Kerja

- Sebelum tiap increment: tampilkan plan detail, tunggu persetujuan.
- Setelah tiap increment: jalankan verifikasi, laporkan hasil apa adanya, lalu buka increment berikutnya.
- Commit per increment mengikuti Conventional Commits.
