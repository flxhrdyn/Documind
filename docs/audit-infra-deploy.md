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

## Baru ditemukan saat audit CI/CD

### 6. `backend/requirements.txt` dan root `pyproject.toml` sudah divergen isi paketnya

File: `backend/requirements.txt` vs `pyproject.toml`.

`backend/requirements.txt` (dipakai Docker build dan `.github/workflows/ci.yml`) berisi
`rank-bm25`, `tf-keras`, `llama-parse`, `llama-index` yang **tidak ada** di daftar `dependencies` root
`pyproject.toml` (dipakai `uv sync`/`uv lock` untuk environment dev lokal). Ini konsisten dengan desain yang
didokumentasikan di `CLAUDE.md` ("Backend and frontend have separate requirements.txt ... for independent
container builds"), tapi berarti environment dev yang di-setup lewat `uv sync` bisa saja tidak punya paket
yang sebenarnya dibutuhkan backend runtime (mis. `llama-parse` yang dipakai `index_data.py`).

Status: **tidak diperbaiki di sesi ini** - memutuskan mana yang jadi sumber kebenaran (root `pyproject.toml`
vs `backend/requirements.txt`) dan menyatukannya adalah keputusan arsitektur yang butuh testing environment
dev yang tidak tersedia di sesi ini (risiko merusak setup `uv sync` tanpa cara memverifikasi). Dicatat untuk
tindak lanjut: audit `pyproject.toml` dependencies vs `backend/requirements.txt` line-by-line, putuskan satu
sumber kebenaran atau dokumentasikan eksplisit bahwa keduanya sengaja independen.

### 7. CI/CD sudah menjalankan test dan cukup aman - tidak ada temuan blocking

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

- `uv lock` sukses resolve 127 packages tanpa error setelah edit `.gitignore`.
- `docker-compose.yml` tervalidasi sebagai YAML yang valid setelah fix Redis.
- `python -m pytest backend/tests` - 83 passed, 2 skipped (1 gagal pra-eksisting tidak terkait, soal
  download model di lingkungan lokal).

## Belum diaudit

- `backend/tests/` - belum dicek seberapa besar celah di tiga laporan audit ini sudah/belum tercakup test
  yang ada (butuh sesi terpisah untuk audit coverage test).
- Environment dev lokal (`uv sync`) belum diuji end-to-end untuk mengonfirmasi temuan #6 di atas benar-benar
  menyebabkan kegagalan runtime nyata.
