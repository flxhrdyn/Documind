# Backend Improvement Spec - InvenioAI

Spesifikasi perbaikan dan optimasi backend berdasarkan `docs/backend-audit.md`.
Tanggal: 2026-07-11.

## Tujuan

Mematangkan backend RAG agar layak sebagai portfolio AI Engineer: menghapus bug, menyelaraskan implementasi dengan klaim dokumentasi, dan membuat metrik yang kredibel.

## Prinsip

- Increment tipis yang shippable end-to-end, bukan refactor besar sekaligus.
- Setiap perubahan disertai bukti (test lulus atau eval numerik), bukan asumsi.
- Tidak menambah abstraksi di luar kebutuhan tiap increment.

## Ruang Lingkup

Dalam lingkup: retrieval, reranking, caching, metrics, endpoint API, konfigurasi.
Di luar lingkup: perubahan frontend, migrasi Streamlit ke React, perubahan model deployment.

## Requirement per Area

### R1 - Konsistensi retrieval (Fix #1, #3)

- Mode hybrid harus jujur soal strategi yang dipakai.
  Baik aktifkan MMR yang kompatibel hybrid, atau perbarui docstring dan `CLAUDE.md` agar menyatakan hybrid = similarity + RRF fusion tanpa MMR.
- Bedakan dua fusion secara eksplisit: RRF dense+sparse (Qdrant native, sudah benar) vs "RAG Fusion" level query (saat ini hanya multi-query union, belum RRF).
- Istilah "RAG Fusion" hanya dipakai bila RRF level-query nyata diimplementasikan (skor `1/(k+rank)` per varian, dijumlahkan lintas varian); jika tidak, ganti menjadi "multi-query retrieval" dan sisakan klaim RRF hanya untuk dense+sparse.
- Kriteria selesai: tidak ada klaim di dokumentasi yang bertentangan dengan kode; `retriever.py` punya satu perilaku yang terdefinisi jelas.

### R2 - Metrik IR yang kredibel (Fix #2)

- Metrik berbasis skor reranker harus dilabeli sebagai proxy, bukan IR metric absolut.
- Sediakan jalur eval sah: direktori `backend/eval/` berisi query dan qrels manual, plus skrip yang menghitung Precision/Recall/nDCG/MRR terhadap qrels.
- Kriteria selesai: dashboard membedakan "reranker-score proxy" dari "IR metric (qrels)"; skrip eval dapat dijalankan dan menghasilkan angka.

### R3 - Concurrency aman (Fix #4)

- Operasi sync CPU-bound (embedding, rerank, disk cache IO) di jalur async tidak boleh memblokir event loop.
- Kriteria selesai: operasi blocking dibungkus executor; uji beban ringan menunjukkan request paralel tidak saling menahan.

### R4 - Caching efektif dan scalable (Fix #5, #6)

- Threshold semantic cache diturunkan ke rentang yang benar-benar menghasilkan hit (target 0.92-0.95) dengan number-guard tetap aktif.
- Semantic cache berpindah dari blob JSON O(n) ke vector search (Qdrant collection atau Redis vector index).
- Kriteria selesai: cache hit ratio terukur meningkat; tidak ada read-modify-write seluruh registry per query.

### R5 - Efisiensi biaya LLM (Fix #7)

- Query-rewrite dijadikan kondisional (hanya saat ada history).
- `NUM_FUSION_QUERIES` dikalibrasi terhadap eval, bukan angka statis.
- Kriteria selesai: jumlah pemanggilan LLM per query turun tanpa penurunan kualitas pada eval set.

### R6 - Konfigurasi dan kebersihan (Fix #8, minor)

- Nama `RERANKER_MODEL` selaras di config, docstring, dan `CLAUDE.md`.
- `IR_RELEVANCE_THRESHOLD` dikalibrasi ke distribusi skor reranker aktif.
- Hapus import dan return value mati (`RETRIEVAL_K`, `vectorstore`).
- Fallback reranker tidak lagi mengirim skor `0.0` yang menskew metrik secara diam-diam.
- Kriteria selesai: `npm run lint` backend bersih; tidak ada dead code terkait.

## Requirement Non-Fungsional

- Setiap increment tidak menurunkan kualitas jawaban pada eval set.
- Tidak ada regresi pada test suite `pytest`.
- Perubahan config backward-compatible via env var (default aman).

## Risiko

- Mengubah retrieval dapat menggeser kualitas jawaban; mitigasi dengan eval set sebelum dan sesudah.
- Migrasi semantic cache ke vector store menambah dependensi operasional; mitigasi dengan fallback ke diskcache bila store tidak tersedia.
