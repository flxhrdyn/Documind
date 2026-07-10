# Audit Infra, Deploy & Sisa Modul - InvenioAI

Tanggal: 2026-07-10.
Scope: `Dockerfile`, `.dockerignore`/`.gitignore`, `uv.lock`, `backend/app/utils.py`.
Lanjutan dari `docs/audit-frontend-ux.md` dan `docs/audit-backend-rag-core.md`.

## Tinggi

### 1. `uv.lock` tidak tercatat di git meski proyek diklaim "lock-managed dengan uv"

File: `.gitignore:35` (`uv.lock` masuk daftar ignore di bawah section "Virtual Environment"), dikonfirmasi
via `git ls-files` - `uv.lock` tidak ada di index git, hanya `requirements.txt` (root, backend, frontend)
yang tercatat.

`pyproject.toml` mendeklarasikan dependency dengan version range longgar (mis. `"fastapi>=0.110.0"`,
`"langchain>=0.3.7"`). Tanpa `uv.lock` ter-commit, `uv sync`/`uv pip install` oleh kontributor lain atau CI
akan resolve versi terbaru yang memenuhi range tersebut kapan pun dijalankan - bukan versi persis yang
sedang dipakai. Ini bertentangan dengan tujuan lockfile (build reproducible) dan berisiko "works on my
machine" karena drift versi antar environment/waktu.

Perbaikan: hapus `uv.lock` dari `.gitignore`, commit lockfile-nya. Kalau memang sengaja tidak dikunci,
perbaiki dokumentasi (README/CLAUDE.md) agar tidak mengklaim proyek "lock-managed".

## Sedang

### 2. ~~Klaim batas ukuran upload cuma berlaku di jalur Streamlit UI, bukan di API backend~~ (FIXED)

File: `Dockerfile:49` (`STREAMLIT_SERVER_MAX_UPLOAD_SIZE=100`) vs `backend/app/index_api.py`
(`_save_uploaded_pdf`).

Status: diperbaiki bersamaan dengan `docs/audit-backend-rag-core.md` temuan #7 - backend sekarang punya
guard ukuran sendiri (`INVENIOAI_MAX_UPLOAD_SIZE_MB`, default 100MB), jadi tidak lagi bergantung semata
pada `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` yang cuma berlaku di widget UI.

Perbaikan: tetap tambahkan validasi ukuran eksplisit di `_save_uploaded_pdf` / route handler backend, jangan
mengandalkan konfigurasi Streamlit sebagai satu-satunya lapisan proteksi.

### 3. `sources_json["score"]` di `format_docs` selalu bernilai default 0.0 - metadata skor rerank tidak pernah diteruskan ke sumber jawaban

File: `backend/app/utils.py:38` - `"score": metadata.get("score", 0.0)`.

`metadata` di sini adalah `doc.metadata` (metadata chunk dari Qdrant/LangChain `Document`), bukan skor hasil
reranking. Skor FlashRank dihitung terpisah di `reranker.py` sebagai list `retrieval_scores` yang sejajar
index dengan `reranked_docs`, tapi tidak pernah disuntikkan balik ke `doc.metadata["score"]` sebelum dikirim
ke `format_docs()` (lihat `rag_pipeline.py:319-322` dan `:503-508`). Akibatnya field `score` yang dikirim ke
frontend di setiap item `sources` selalu `0.0`, membuat UI/README yang mengklaim menampilkan relevansi
sumber tidak punya data nyata untuk itu.

Perbaikan: setelah `rerank()`, tempelkan skor ke masing-masing `Document.metadata["score"]` sebelum memanggil
`format_docs()`, atau ubah signature `format_docs()` untuk menerima `retrieval_scores` sebagai parameter
terpisah dan pasangkan by index.

## Rendah

### 4. `ThinkingParser` tidak menangani tag `<thinking>` yang muncul lebih dari sekali

File: `backend/app/utils.py:45-149`.

State machine parser cuma punya dua fase: sebelum `thinking_done` dan sesudahnya (baris 134-138, "After
thinking is done, everything is a token"). Kalau LLM (karena variasi prompt/model) menghasilkan lebih dari
satu blok `<thinking>...</thinking>` dalam satu respons, blok kedua dan seterusnya akan diperlakukan sebagai
teks jawaban biasa (bocor ke `full_answer`) alih-alih `full_thoughts`. Risiko rendah karena prompt eksplisit
minta satu blok thinking di awal, tapi tidak ada guard kalau model menyimpang.

Perbaikan: opsional, dokumentasikan asumsi "satu blok thinking per respons" secara eksplisit di
`RAG_PROMPT`/`ThinkingParser`, atau tangani multi-blok kalau ternyata terjadi di observasi produksi.

## Verifikasi positif (bukan bug, dicatat agar tidak diulang-cek)

- `.dockerignore` sudah benar mengecualikan `.env`, `.git`, `venv`, `uploaded_docs`, `qdrant_storage`,
  `metrics.json` - tidak ada kebocoran secret/data ke image.
- Multi-stage `Dockerfile` (builder + final) sudah memisahkan build tools dari image akhir, model ML
  di-download saat build (bukan runtime) - startup lebih cepat & tidak bergantung akses internet saat boot.

## Belum diaudit

- CI/CD (`.github/workflows/*.yml`) belum dicek isinya secara detail.
- `backend/tests/` - belum dicek seberapa besar celah di tiga laporan audit ini sudah/belum tercakup test
  yang ada.
- `docker-compose.yml` Redis tanpa password terekspos ke host (`ports: 6379:6379`) - sudah disinggung
  sepintas di `docs/audit-backend-rag-core.md` temuan #1 tapi belum diverifikasi dampak keamanannya secara
  menyeluruh (mis. apakah port itu dibuka ke jaringan publik di deployment nyata atau cuma lokal).
