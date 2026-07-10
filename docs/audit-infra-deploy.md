# Audit Infra, Deploy & Sisa Modul - InvenioAI

Tanggal: 2026-07-10. Diperbaiki: 2026-07-10.
Scope: `Dockerfile`, `docker-compose.yml`, `.dockerignore`/`.gitignore`, `uv.lock`, `backend/app/utils.py`,
`.github/workflows/*.yml`.
Lanjutan dari `docs/audit-frontend-ux.md` dan `docs/audit-backend-rag-core.md`.

## Tinggi

### 1. ~~`uv.lock` tidak tercatat di git meski proyek diklaim "lock-managed dengan uv"~~ (FIXED)

File: `.gitignore` (baris `uv.lock` di section "Virtual Environment" dihapus), `uv.lock` (di-regenerate).

Status: diperbaiki. `uv lock` dijalankan ulang untuk memastikan lockfile sinkron dengan `pyproject.toml`
saat ini (resolve 127 packages, tanpa error), lalu `uv.lock` dibuang dari `.gitignore` sehingga sekarang
muncul sebagai untracked file yang siap di-`git add`/commit. **Belum di-commit** di sesi ini - itu langkah
terpisah yang perlu persetujuan eksplisit (bukan bagian dari audit/fix otomatis).

### 2. Redis di `docker-compose.yml` expose port ke host tanpa password (FIXED)

File: `docker-compose.yml` (service `redis`).

Status: diperbaiki. `ports: ["6379:6379"]` di service `redis` dihapus. `backend` sudah terhubung ke Redis
lewat `REDIS_URL=redis://redis:6379/0` yang resolve lewat DNS internal Docker Compose - port mapping ke
host itu tidak diperlukan sama sekali untuk fungsi apa pun, dan sebelumnya membuka instance Redis tanpa
`requirepass` ke jaringan host. Kalau firewall host permisif (umum di VM cloud tanpa security group ketat),
ini risiko klasik Redis unauthenticated RCE/data exfiltration.

## Sedang

### 3. ~~Klaim batas ukuran upload cuma berlaku di jalur Streamlit UI, bukan di API backend~~ (FIXED)

File: `Dockerfile:49` (`STREAMLIT_SERVER_MAX_UPLOAD_SIZE=100`) vs `backend/app/index_api.py`
(`_save_uploaded_pdf`).

Status: diperbaiki bersamaan dengan `docs/audit-backend-rag-core.md` temuan #7 - backend sekarang punya
guard ukuran sendiri (`INVENIOAI_MAX_UPLOAD_SIZE_MB`, default 100MB), jadi tidak lagi bergantung semata
pada `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` yang cuma berlaku di widget UI. Frontend juga sudah menambahkan
guard client-side yang sama (`docs/audit-frontend-ux.md` #2).

### 4. ~~`sources_json["score"]` di `format_docs` selalu bernilai default 0.0~~ (FIXED)

File: `backend/app/utils.py` (`format_docs`), `backend/app/rag_pipeline.py` (kedua pemanggil).

Status: diperbaiki. `format_docs()` sekarang menerima parameter opsional `scores: Optional[List[float]]`
yang dipasangkan by-index dengan `docs` - dipanggil dengan `retrieval_scores` hasil `rerank()` di kedua
jalur (`_run_rag_pipeline_with_query` untuk `/query`, dan `rag_pipeline_stream_async` untuk
`/query/stream`). Field `score` yang dikirim ke frontend sekarang berisi skor rerank sungguhan, bukan 0.0
konstan.

## Rendah

### 5. ~~`ThinkingParser` tidak menangani tag `<thinking>` yang muncul lebih dari sekali~~ (FIXED)

File: `backend/app/utils.py` (`ThinkingParser`).

Status: diperbaiki. State `thinking_done` yang membuat blok `<thinking>` kedua dan seterusnya diperlakukan
sebagai teks jawaban biasa (bocor ke `full_answer`) dihapus - parser sekarang selalu mendeteksi tag
`<thinking>` baru setiap kali tidak sedang di dalam blok thinking, jadi bisa menangani jumlah blok berapa
pun. Diverifikasi manual:
`<thinking>Block A</thinking>Answer1<thinking>Block B</thinking>Answer2` menghasilkan
`[("thinking","Block A"), ("token","Answer1"), ("thinking","Block B"), ("token","Answer2")]` - sebelumnya
"Block B" akan bocor sebagai token/jawaban.

## Baru ditemukan saat audit CI/CD dan frontend theme

### 6. ~~`backend/requirements.txt` dan root `pyproject.toml` sudah divergen isi paketnya~~ (FIXED)

File: `pyproject.toml`, `backend/requirements.txt`, `uv.lock`.

Diverifikasi konkret lewat grep import di `backend/app/`: `llama_parse` benar-benar dipakai
(`index_data.py:147`, `from llama_parse import LlamaParse`) tapi **hilang dari `pyproject.toml`** - artinya
`uv sync` di environment dev akan menghasilkan environment yang crash `ImportError` begitu proses indexing
PDF sungguhan dijalankan. `llama_index` dan `tf-keras` juga dipakai transitif oleh stack yang sama tapi
sama-sama hilang dari `pyproject.toml`. Sebaliknya, `rank-bm25` ada di `backend/requirements.txt` tapi
**tidak pernah di-import di mana pun** - sisa dependency mati dari implementasi hybrid BM25 lokal lama
(sebelum native Qdrant BM42 dipakai, konsisten dengan temuan config mati `HYBRID_DENSE_WEIGHT`/
`HYBRID_SPARSE_WEIGHT` di `docs/audit-backend-rag-core.md` #10).

Status: diperbaiki. `tf-keras`, `llama-parse`, `llama-index` ditambahkan ke `dependencies` di
`pyproject.toml` (env dev `uv sync` sekarang punya semua paket yang backend benar-benar butuhkan).
`rank-bm25` dihapus dari `backend/requirements.txt` (dead dependency). `uv lock` dijalankan ulang untuk
resolve lockfile baru - berhasil tanpa konflik versi. Test suite penuh dijalankan setelahnya: **84 passed,
0 failed, 2 skipped** (termasuk `test_sparse_embedding_output` yang sebelumnya gagal karena isu cache model
lokal - sekarang lolos juga).

### 7. `--invenio-bg-secondary` dipakai di CSS tapi tidak pernah didefinisikan di `theme.py`

File: `frontend/theme.py` (`CSS_VARS`), dipakai di `frontend/streamlit_app.py` (background chat message,
header tabel).

`streamlit_app.py` memakai `var(--invenio-bg-secondary)` di dua tempat (background `[data-testid="stChatMessage"]`
dan `th` header tabel), tapi `CSS_VARS` di `theme.py` cuma mendefinisikan `--invenio-accent`,
`--invenio-bg-card`, `--invenio-border` - `--invenio-bg-secondary` tidak pernah dideklarasikan. CSS custom
property yang undefined tanpa nilai fallback membuat declaration itu invalid di computed-value time; untuk
`background-color` (properti non-inherited), browser jatuh ke initial value `transparent`. Chat bubble dan
header tabel jadi tidak dapat background themed yang dimaksud desainnya.

Status: diperbaiki. `--invenio-bg-secondary: var(--secondary-bg-color);` ditambahkan ke `CSS_VARS`,
konsisten dengan `COLORS["bg_secondary"]` yang sudah ada nilai yang sama di dict Python-nya tapi tidak
pernah dicerminkan ke CSS var yang sebenarnya dipakai.

### 8. `streamlit_app.py` import `COLORS` dari `theme.py` tapi tidak pernah memakainya

File: `frontend/streamlit_app.py`.

`from theme import COLORS, CSS_VARS` - dicek lewat grep, `COLORS[...]` tidak pernah dipakai di file ini
(cuma `CSS_VARS` yang dipakai). `COLORS` cuma dipakai di `frontend/pages/dashboard.py`.

Status: diperbaiki. Import dipersempit jadi `from theme import CSS_VARS` saja.

### 9. CI/CD sudah menjalankan test dan cukup aman - tidak ada temuan blocking

File: `.github/workflows/ci.yml`, `.github/workflows/sync-to-hf-space.yml`.

Diperiksa: `ci.yml` menjalankan `pytest backend/` di setiap push/PR ke branch utama (bukan cuma lint atau
build check kosong). `sync-to-hf-space.yml` deploy ke HF Space hanya setelah CI sukses, pakai GitHub
Secrets untuk kredensial (`HF_TOKEN`, `HF_SPACE_ID`) - tidak ada secret hardcoded. Snapshot sync ke Space
memakai `git push --force` sebagai fallback kalau push normal gagal - ini terhadap mirror Space milik
sendiri (bukan repo utama), jadi force-push di sini adalah trade-off yang wajar untuk deployment
snapshot-based, bukan bug. Tidak ada langkah lint/type-check terpisah, tapi ini di luar cakupan "blocking
production readiness" - dicatat sebagai potensi peningkatan, bukan temuan.

## Verifikasi positif (bukan bug, dicatat agar tidak diulang-cek)

- `.dockerignore` sudah benar mengecualikan `.env`, `.git`, `venv`, `uploaded_docs`, `qdrant_storage`,
  `metrics.json` - tidak ada kebocoran secret/data ke image.
- Multi-stage `Dockerfile` (builder + final) sudah memisahkan build tools dari image akhir, model ML
  di-download saat build (bukan runtime) - startup lebih cepat & tidak bergantung akses internet saat boot.
- `docker-compose.yml` YAML tetap valid setelah fix #2 (diverifikasi `yaml.safe_load`).

## Verifikasi

- `uv lock` sukses resolve 127 packages tanpa error setelah edit `.gitignore`, lalu resolve ulang sukses
  (menambahkan `tensorflow`, `tf-keras`, `llama-index`, `llama-parse`, dan dependency transitifnya) setelah
  fix #6.
- `docker-compose.yml` tervalidasi sebagai YAML yang valid setelah fix Redis.
- `python -m pytest backend/tests` - **84 passed, 0 failed, 2 skipped** (naik dari 83 passed/1 failed
  sebelum fix #6 - `test_sparse_embedding_output` yang tadinya gagal karena cache model lokal korup
  sekarang lolos juga).
- `python -m py_compile` lolos untuk `streamlit_app.py`, `theme.py`, `dashboard.py` setelah fix #7/#8.
- Uji end-to-end nyata (upload PDF sungguhan lewat backend yang benar-benar jalan) dicoba di sesi ini:
  backend start bersih dan terhubung ke Qdrant Cloud, tapi upload gagal karena model sparse BM42
  (`model.onnx`, berkas besar) macet di-download dari Hugging Face Hub di lingkungan sandbox ini
  (`.incomplete` stuck di 147KB, tidak nambah lagi setelah dicoba ulang dengan cache dibersihkan). Ini
  keterbatasan jaringan sandbox, bukan bug kode - perlu diverifikasi ulang di mesin dengan akses jaringan
  penuh ke HF Hub.

## Test coverage - celah yang ditambal (2026-07-10)

Audit coverage menemukan bahwa fix-fix berikut dari sesi audit ini belum punya test khusus (cuma
diverifikasi manual). Test baru ditambahkan untuk semuanya:

- `backend/tests/test_index_api.py`: cache RAG di-clear + retriever cache di-invalidate saat hapus satu
  dokumen (`test_delete_document_clears_cache_and_invalidates_retriever`); `_find_duplicate_document` (3
  test); `_index_uploaded_pdf` menolak upload duplikat dengan HTTP 409 dan membersihkan file yang baru
  disimpan.
- `backend/tests/test_index_data.py`: guard ketidakcocokan model embedding, `TestEmbeddingModelGuard` (3
  test - rekam marker pertama kali, lolos kalau cocok, tolak kalau beda); rollback batch indexing saat
  gagal separuh jalan (`TestIndexDocumentsRollback`); table header ke-carry ke chunk lanjutan pakai
  splitter asli (bukan mock) dengan tabel 120 baris.
- `backend/tests/test_auth.py` (baru): `require_api_key` nonaktif default, tolak key hilang/salah, terima
  key benar, endpoint `/metrics` benar-benar 401/200 sesuai state, `/` tetap terbuka meski auth aktif.
- `backend/tests/test_retriever.py` (baru): `build_retriever()` reuse stack untuk client Qdrant yang sama,
  `invalidate_retriever_cache()` memaksa rebuild, cache otomatis rebuild kalau instance client Qdrant
  berganti, error yang benar tanpa `GROQ_API_KEY`/koleksi belum ada.
- `backend/tests/test_thinking_parser.py`: blok `<thinking>` ganda tidak lagi bocor ke jawaban.
- `backend/tests/test_utils.py`: `format_docs()` memakai skor rerank yang diteruskan, bukan selalu 0.0.

Total: **+26 test baru**. Suite penuh: **110 passed, 0 failed, 2 skipped** (naik dari 84 passed sebelum
penambahan test ini).

## Belum diaudit

- Upload/index PDF end-to-end sungguhan (lewat UI atau API langsung) belum berhasil diverifikasi di sesi
  manapun karena keterbatasan jaringan sandbox untuk download model BM42 - perlu dicoba di environment lain.
