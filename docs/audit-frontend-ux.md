# Audit Frontend UX & Integration - InvenioAI

Tanggal: 2026-07-09. Diperbaiki: 2026-07-10.
Scope: `frontend/streamlit_app.py`, `frontend/pages/dashboard.py`, `frontend/theme.py`.
Tujuan: kesiapan produksi dan UX.

## Kritis

### 1. ~~Chat history global, bukan per-user (data leak lintas user)~~ (FIXED)

File: `frontend/streamlit_app.py` (`st.session_state.messages`).

Status: diperbaiki, digabung dengan fix #10. `CHAT_HISTORY_CACHE_KEY`, `load_persistent_history()`,
`save_persistent_history()`, `clear_persistent_history()`, dan seluruh dependency ke `CacheManager` backend
dihapus total. Chat history sekarang hidup murni di `st.session_state.messages` - state yang sudah
di-scope per browser session oleh Streamlit sendiri, jadi tidak mungkin bocor/tertimpa antar user lagi.

Trade-off yang disengaja: history tidak lagi bertahan lewat full page reload/reconnect (sebelumnya
"bertahan" tapi lewat mekanisme yang justru menyebabkan bug kritis ini). Untuk demo RAG tanpa akun user,
ini trade-off yang benar - lebih aman daripada nyaman.

## Tinggi

### 2. ~~Tidak ada batas ukuran file upload PDF (client maupun server)~~ (FIXED)

File: `frontend/streamlit_app.py` (`MAX_UPLOAD_SIZE_MB`, guard di `uploaded_file.size`).

Status: diperbaiki di kedua sisi. Backend sudah dapat guard ukuran di sesi audit sebelumnya
(`docs/audit-backend-rag-core.md` #7, `INVENIOAI_MAX_UPLOAD_SIZE_MB`). Frontend sekarang membaca env var
yang sama dan menolak file di client (`st.error`) sebelum sempat memanggil `create_upload_job` kalau
`uploaded_file.size` melebihi batas - tidak menunggu roundtrip ke server dulu untuk kasus umum ini.

### 3. ~~Timeout query 60 detik, exception mentah ditampilkan ke user~~ (FIXED)

File: `frontend/streamlit_app.py` (`QUERY_READ_TIMEOUT_SECONDS`, `format_error_message`).

Status: diperbaiki. Read timeout naik jadi 120 detik default (bisa diatur via
`INVENIOAI_QUERY_TIMEOUT_SECONDS`) alih-alih 60 detik hardcoded. Semua exception jaringan sekarang lewat
`format_error_message()` yang sudah dirombak untuk menerima `requests.Response` ATAU
`requests.exceptions.RequestException` dan selalu menghasilkan pesan ramah pengguna, tidak pernah
`str(exception)` mentah.

## Sedang

### 4. ~~Polling upload job blocking, interval tetap 1 detik~~ (FIXED)

File: `frontend/streamlit_app.py` (`_next_poll_interval`, `wait_for_upload_job`).

Status: diperbaiki. Interval polling sekarang backoff 1s -> 2s -> 4s -> capped di 5s, bukan tetap 1
detik sepanjang durasi tunggu (bisa sampai 3600s). Tidak menghilangkan blocking sinkron sepenuhnya (di
luar scope tanpa merombak model threading Streamlit), tapi mengurangi jumlah request polling secara
signifikan untuk upload yang lama.

### 5. ~~Query kosong/whitespace dan submit ganda tidak dicegah~~ (FIXED)

File: `frontend/streamlit_app.py` (blok `# Input`).

Status: diperbaiki. `prompt = raw_prompt.strip() if raw_prompt else ""` menyaring whitespace-only input.
Submit yang identik dengan pesan user terakhir di-deteksi dan ditolak dengan `st.toast()` alih-alih
dikirim ulang ke backend (mencegah biaya pipeline RAG ganda untuk klik/enter berulang).

### 6. ~~Dashboard bisa crash kalau schema metrics.json tidak lengkap~~ (FIXED)

File: `frontend/pages/dashboard.py` (`df_display`).

Status: diperbaiki (hardening defensif). Investigasi lebih lanjut menunjukkan `per_query_ir_metrics()` di
backend (`backend/app/metrics.py`) sudah menormalkan setiap row lewat `.get(..., default)` untuk semua
kolom yang dipakai `display_cols`, jadi skenario `KeyError` yang dijelaskan di temuan asli sebenarnya sudah
tidak reproduce dengan kode backend saat ini. Tetap dipasang `df_table.reindex(columns=..., fill_value=None)`
sebagai pengaman murah terhadap schema drift di masa depan, sesuai rekomendasi awal.

### 7. ~~Error saat hapus dokumen bocorin response/stack trace backend~~ (FIXED)

File: `frontend/streamlit_app.py` (semua pemanggil `format_error_message`).

Status: diperbaiki. `format_error_message()` sekarang dipakai konsisten di semua path request (query,
upload, delete satu dokumen, delete semua dokumen, fetch metrics) - tidak ada lagi jalur yang menampilkan
`e.response.text`/`str(e)` mentah secara terpisah-pisah seperti sebelumnya.

### 8. ~~Kegagalan network di-swallow jadi "No documents yet"~~ (FIXED)

File: `frontend/streamlit_app.py` (`_fetch_indexed_documents`, `get_indexed_files`).

Status: diperbaiki. Kedua fungsi sekarang mengembalikan `(docs, backend_reachable)` alih-alih cuma
`list[str]`. Sidebar menampilkan banner error koneksi terpisah dari "No documents yet." kalau
`backend_reachable=False`, dan chat input guard membedakan "belum ada dokumen" vs "backend tidak
terjangkau" alih-alih selalu menyuruh user upload PDF.

### 9. ~~Delete-all documents tanpa konfirmasi~~ (FIXED)

File: `frontend/streamlit_app.py` (`st.session_state.confirm_delete_all`).

Status: diperbaiki. Klik pertama "Delete All Documents" cuma memunculkan warning + tombol konfirmasi
("Yes, delete all" / "Cancel"), eksekusi DELETE sungguhan cuma terjadi setelah klik kedua yang eksplisit.

## Arsitektural (akar masalah #1)

### 10. ~~Frontend import langsung modul internal backend via `sys.path` hack~~ (FIXED, untuk chat history)

File: `frontend/streamlit_app.py`.

Status: diperbaiki untuk jalur yang menyebabkan temuan #1. `sys.path.append(...)` dan
`from app.cache_manager import CacheManager` dihapus total dari `streamlit_app.py` - frontend sekarang
hanya bicara ke backend lewat REST API (`API_BASE_URL` + `API_HEADERS`), sesuai arsitektur yang
didokumentasikan.

**Catatan cakupan**: `frontend/pages/dashboard.py` masih melakukan import langsung serupa
(`from backend.app.metrics import ...`, `from backend.app.config import RETRIEVAL_K`) untuk baca
`metrics.json` lokal. Ini di luar scope temuan #1 (bukan data user yang bisa bocor lintas sesi - cuma
agregat metrics read-only), jadi sengaja tidak diubah di sesi ini. Kalau frontend/backend perlu dipisah
jadi container yang benar-benar independen (co-deploy tidak lagi diasumsikan), ini juga perlu diganti ke
endpoint API metrics backend yang sudah ada (`GET /metrics`).

## Wiring API key (tindak lanjut dari `docs/audit-backend-rag-core.md` #4)

File: `frontend/streamlit_app.py` (`API_HEADERS`).

Backend menambahkan auth opsional berbasis `X-API-Key` di sesi audit sebelumnya, tapi saat itu frontend
belum ikut mengirim header ini. Sekarang `API_HEADERS` dibaca dari `INVENIOAI_API_KEY` (kosong = tidak
mengirim header apa pun, sama seperti sebelumnya) dan dilampirkan ke **semua** request ke backend
(`/query/stream`, `/upload/jobs*`, `/documents*`, `/metrics*`). Auth end-to-end sekarang berfungsi penuh
kalau `INVENIOAI_API_KEY` diaktifkan di kedua sisi.

## Verifikasi

- `python -m py_compile` lolos untuk `streamlit_app.py` dan `dashboard.py`.
- Smoke-run `streamlit run frontend/streamlit_app.py --server.headless true` - server start bersih, HTTP
  200, tidak ada exception di log startup (tanpa backend jalan, untuk menguji jalur "backend unreachable"
  dari temuan #8 tidak meng-crash aplikasi).
- Tidak ada sisa referensi ke `CacheManager`, `sys.path`, atau fungsi `*_persistent_history` (dicek via
  grep di seluruh `streamlit_app.py`).

## Belum diaudit

- `frontend/theme.py` - hanya definisi CSS/warna, belum direview mendalam untuk masalah UX visual.
- Uji end-to-end sungguhan di browser (klik-klik nyata dengan backend hidup) belum dilakukan - verifikasi
  di atas terbatas pada smoke test headless dan pembacaan kode.
