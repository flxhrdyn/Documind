# Backend Audit - InvenioAI

Audit kematangan backend untuk portfolio AI Engineer.
Fokus: bug nyata, kesalahan konsep RAG, isu infra, dan skalabilitas.
Tanggal: 2026-07-11.

## Ringkasan

Backend sudah punya fitur canggih (hybrid retrieval, reranking, CoT, dual-layer cache), tetapi ada beberapa titik di mana implementasi tidak sesuai klaim dokumentasi, metrik yang menyesatkan, dan risiko concurrency.
Temuan diurutkan berdasarkan prioritas.

## Bug nyata

### 1. MMR mati diam-diam di mode hybrid

- File: `backend/app/retriever.py:116`
- Kode memaksa `search_type = "similarity"` ketika hybrid aktif.
  Hybrid adalah default (`USE_HYBRID_SEARCH=1`), jadi di jalur produksi MMR tidak pernah dipakai.
- Dampak: klaim "dense MMR + sparse BM42" di `CLAUDE.md` dan komentar kode tidak benar untuk konfigurasi default.
  Diversity retrieval hilang.
- Rekomendasi: aktifkan MMR yang kompatibel dengan hybrid, atau perbarui dokumentasi agar jujur bahwa hybrid memakai similarity + RRF fusion tanpa MMR.

### 2. IR metrics sirkular (kesalahan konsep RAG)

- File: `backend/app/metrics.py:253-269`
- nDCG, HitRate, MRR, Precision, Recall dihitung dari skor reranker itu sendiri sebagai ground-truth relevance.
  Ini sirkular: reranker mengurutkan berdasar skornya, lalu skor yang sama dipakai sebagai label "relevan".
- `recall_at_k` bahkan mengakui "total relevant unknown" dan memakai jumlah retrieved sebagai perkiraan, sehingga recall cenderung selalu mendekati 1.
- Dampak: metrik terlihat impresif tetapi tidak valid secara IR (tidak ada qrels / label relevance).
  Ini red-flag umum saat review portfolio.
- Rekomendasi: relabel metrik ini sebagai "reranker-score proxy", atau tambahkan eval set kecil dengan qrels manual untuk IR metric yang sah.

### 3. "RAG Fusion" salah nama / salah konsep

Penting: ada dua "fusion" berbeda di sistem ini, hanya satu yang benar.

- BENAR - RRF fusion dense + sparse (level vektor).
  Mode hybrid (`retriever.py:88-98`, `RetrievalMode.HYBRID`) memakai RRF native server-side Qdrant untuk menggabungkan hasil dense dan sparse/BM42.
  Ini implementasi RRF yang sah dan boleh diklaim.
- SALAH - "RAG Fusion" level query.
  `NUM_FUSION_QUERIES` (dilabeli "RAG Fusion" di `config.py:156`) sebenarnya masuk `MultiQueryRetriever` (`retriever.py:126`) yang hanya melakukan `unique_union` = union + dedup berdasarkan konten.
  RAG Fusion sesungguhnya wajib menggabungkan ranked-list tiap varian query via Reciprocal Rank Fusion; tidak ada kode RRF level-query di mana pun (`grep` untuk RRF/fusion hanya menemukan RRF dense+sparse dan label komentar).
- Dampak: klaim "RAG Fusion multi-query" di `CLAUDE.md` tidak akurat; yang ada adalah multi-query union.
- Rekomendasi: pilih salah satu -
  (a) implementasi RRF nyata di level query (skor `1/(k+rank)` per varian, jumlahkan lintas varian) sehingga istilah "RAG Fusion" jadi benar, atau
  (b) ganti istilah menjadi "multi-query retrieval" dan sisakan klaim RRF hanya untuk fusion dense+sparse.

### 4. Event loop diblokir di jalur async streaming

- File: `backend/app/rag_pipeline.py:439`, `rag_pipeline.py:528`
- `embed_query`, `rerank`, dan disk cache IO adalah operasi sync CPU-bound, dipanggil langsung di async generator tanpa `run_in_executor`.
- Dampak: satu request berat memblokir seluruh event loop, jadi semua request lain ikut tertahan di bawah concurrency.
- Rekomendasi: bungkus operasi blocking dengan `asyncio.get_event_loop().run_in_executor(...)` atau `anyio.to_thread.run_sync`.

## Konsep dan skalabilitas

### 5. Semantic cache threshold 0.995 mematikan fitur

- File: `backend/app/rag_pipeline.py:232`, `rag_pipeline.py:442`
- Default `get_semantic(threshold=0.90)` di-override menjadi 0.995 = nyaris exact-match, ditambah number-guard regex.
- Dampak: semantic cache hampir tidak pernah hit, jadi fitur efektif mati.
  Dokumentasi menyebut ambang > 0.90.
- Rekomendasi: turunkan ke sekitar 0.92-0.95 dan andalkan number-guard untuk mencegah false positive.

### 6. Semantic registry O(n) blob JSON

- File: `backend/app/cache_manager.py:64-76`
- Setiap `add_semantic` membaca dan menulis ulang seluruh list (hingga 1000 vektor x 384 dim) di satu key Redis.
  Setiap query melakukan scan linear penuh.
- Dampak: tidak scalable, beban memori dan IO tumbuh linear.
- Rekomendasi: gunakan vector search (Qdrant collection terpisah atau Redis vector index) untuk semantic cache.

### 7. Redundansi query expansion

- File: `backend/app/rag_pipeline.py:224`, `retriever.py:126`
- Setiap query melakukan 1 LLM rewrite, lalu MultiQuery membuat N varian query lagi.
  Ekspansi ganda.
- Biaya per query: 1 rewrite + N generasi LLM + N x 20 retrieval + rerank + 1 generasi jawaban.
- Dampak: latency dan token boros, fungsi rewrite dan multi-query tumpang tindih.
- Rekomendasi: pilih salah satu strategi, atau jadikan rewrite kondisional (hanya saat ada history).

### 8. Config drift

- File: `backend/app/config.py:130`, `config.py:125`
- `RERANKER_MODEL` default `ms-marco-MultiBERT-L-12`, tetapi `CLAUDE.md` dan docstring menyebut `ms-marco-MiniLM-L-12-v2`.
- `IR_RELEVANCE_THRESHOLD=0.7` mengasumsikan skor ternormalisasi [0,1]; distribusi skor MultiBERT belum tentu sesuai.
- Rekomendasi: samakan nama model di config, docstring, dan CLAUDE.md; verifikasi distribusi skor reranker aktif dan kalibrasi ulang threshold.

## Minor

- `RETRIEVAL_K` di-import tetapi tidak dipakai di `backend/app/rag_pipeline.py:16`.
- `vectorstore` dikembalikan `build_retriever` tetapi tidak dipakai pemanggil.
- Fallback reranker memberi skor `0.0` sehingga semua dokumen dinilai "irrelevant" di IR metrics secara diam-diam, menskew data.
- Tidak ada timeout pada pemanggilan Groq LLM; request yang menggantung memblokir tanpa batas.

## Urutan increment yang disarankan

- A: Fix #1 (MMR hybrid) + #8 (config drift). Cepat, aman, menutup celah dokumentasi yang tidak akurat.
- B: Fix #2 (IR metrics). Paling penting untuk kredibilitas portfolio.
- C: Fix #4 (event-loop blocking) via `run_in_executor`.
- D: Fix #5 (threshold) + #7 (redundansi expansion).

## Improvisasi dan Optimasi

Peningkatan opsional di luar perbaikan bug, untuk mematangkan sistem sebagai portfolio.

### Kualitas retrieval dan RAG

- **RRF fusion nyata plus weighting**: implementasi Reciprocal Rank Fusion di level query dan sediakan bobot dense vs sparse.
  Kalau `langchain-qdrant` tidak mengekspos knob fusion, panggil `client.query_points` langsung dengan `prefetch` + `FusionQuery(RRF/DBSF)`.
- **Eval set + qrels**: buat direktori `backend/eval/` berisi ~30-50 pasang query dan chunk-id relevan (qrels manual), lalu jalankan Precision/Recall/nDCG/MRR sesungguhnya di CI.
  Ini mengubah metrik dari kosmetik menjadi bukti kuantitatif.
- **Grounding / faithfulness check**: tambahkan verifikasi bahwa setiap klaim di jawaban punya sitasi valid, dan skor faithfulness (mis. via RAGAS atau LLM-judge ringan).
  Kurangi halusinasi terukur.
- **Contextual / late chunking**: pertimbangkan menambahkan ringkasan konteks per chunk (contextual retrieval ala Anthropic) atau sentence-window retrieval untuk meningkatkan presisi.
- **Adaptive retrieval**: skip retrieval atau kurangi `k` untuk pertanyaan sederhana / sapaan (query router), hemat token dan latency.
- **Rerank dua tahap**: retrieval lebar (k besar) lalu rerank ke top-N kecil sudah ada; tambahkan filter skor minimum agar dokumen sampah tidak masuk konteks LLM.
- **Kalibrasi `RETRIEVAL_K` dan `RERANK_TOP_K`**: jalankan sweep terhadap eval set untuk menemukan titik optimal, jangan pakai angka statis tanpa dasar.

### Performa dan infra

- **Vector-native semantic cache**: pindahkan semantic cache ke collection Qdrant khusus atau Redis vector index (HNSW), hilangkan scan linear O(n).
- **Async penuh**: pakai `AsyncQdrantClient` dan bungkus embedding/rerank di thread executor agar event loop tidak terblokir (terkait Fix #4).
- **Batasi biaya LLM**: jadikan query-rewrite kondisional (hanya saat ada history), dan turunkan `NUM_FUSION_QUERIES` bila eval menunjukkan gain marginal.
- **Timeout dan retry**: set timeout eksplisit + retry backoff pada Groq dan Qdrant, dengan circuit breaker agar kegagalan hulu tidak menggantungkan request.
- **Streaming SSE yang benar**: pastikan heartbeat dan penanganan disconnect klien pada `/query/stream`, plus header anti-buffering (`X-Accel-Buffering: no`).
- **Warm-up terukur**: preload model sudah ada; tambahkan endpoint `/health` dan `/ready` terpisah agar orchestrator tahu kapan siap melayani.
- **Observability**: tambahkan structured logging (JSON) + trace id per request, dan ekspor metrik Prometheus (latency histogram, cache hit ratio, token usage) alih-alih hanya `metrics.json`.

### Kualitas kode dan keamanan

- **Validasi input**: batasi panjang `question` dan ukuran `history` di model Pydantic untuk cegah prompt abuse dan biaya membengkak.
- **Rate limiting**: tambahkan limiter (mis. slowapi) pada endpoint query dan upload.
- **Pydantic response models**: ganti `Dict[str, Any]` dengan skema response terstruktur agar kontrak API jelas dan terdokumentasi otomatis di OpenAPI.
- **Test coverage**: tambahkan unit test untuk cache-key collision, number-guard semantic cache, dan ThinkingParser; plus test integrasi retrieval end-to-end dengan Qdrant in-memory.
- **Konfigurasi tersentral**: pertimbangkan `pydantic-settings` (BaseSettings) menggantikan helper `_env_*` manual untuk validasi tipe dan dokumentasi config yang konsisten.
- **Dockerfile multi-stage**: pisahkan build dan runtime, pin versi model, dan pre-download bobot model saat build agar cold-start deterministik.
