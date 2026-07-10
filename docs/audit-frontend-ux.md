# Audit Frontend UX & Integration - InvenioAI

Tanggal: 2026-07-09.
Scope: `frontend/streamlit_app.py`, `frontend/pages/dashboard.py`, `frontend/theme.py`.
Tujuan: kesiapan produksi dan UX.

## Kritis

### 1. Chat history global, bukan per-user (data leak lintas user)

File: `frontend/streamlit_app.py:25-37, 791-792, 810`.

`CHAT_HISTORY_CACHE_KEY = "invenio_persistent_chat_history"` adalah string konstan, tidak di-scope per sesi.
`save_persistent_history()` menulis seluruh `st.session_state.messages` ke key ini setiap giliran chat.
`load_persistent_history()` membaca key yang sama untuk sesi browser baru mana pun.

Skenario: User A tanya soal PDF rahasia, User B buka app di tab/browser lain dan langsung melihat (atau
menimpa) seluruh riwayat chat User A.

Perbaikan: key harus per-sesi (mis. UUID session yang disimpan di cookie/`st.query_params`), bukan string
tetap.

## Tinggi

### 2. Tidak ada batas ukuran file upload PDF (client maupun server)

File: `frontend/streamlit_app.py:544` (uploader hanya cek ekstensi), `frontend/streamlit_app.py:418`
(`uploaded_file.getvalue()` buffer penuh ke memori), `backend/app/index_api.py` (tidak ada guard ukuran).

Skenario: user upload file 2GB (atau biner yang di-rename jadi .pdf), browser buffer penuh ke memori,
upload diam-diam berjalan lama hanya dengan spinner generik, berpotensi OOM container.

Perbaikan: cap ukuran byte di client sebelum `create_upload_job`, plus guard di server.

### 3. Timeout query 60 detik, exception mentah ditampilkan ke user

File: `frontend/streamlit_app.py:711-716, 780`.

`timeout=(5, 60)` pada request streaming. README bilang rata-rata query ~15 detik, tapi kalau lebih lambat
(model cold-start, generasi panjang), muncul `ReadTimeout` yang ditangkap `except Exception as e` lalu
ditampilkan sebagai `❌ **Connection Error:** {e}` - bocorin representasi exception internal (socket/urllib3)
ke end user.

Perbaikan: naikkan/parameterisasi read timeout, format pesan timeout jadi ramah user, jangan `str(e)` mentah.

## Sedang

### 4. Polling upload job blocking, interval tetap 1 detik

File: `frontend/streamlit_app.py:497-527`.

Tidak ada exponential backoff. Seluruh thread sesi Streamlit blok sinkron selama proses indexing (default
600s, bisa sampai 3600s via `INVENIOAI_UPLOAD_TIMEOUT_SECONDS`). Kalau koneksi putus di tengah polling, tidak
ada resume - user harus upload ulang dari awal.

Perbaikan: backoff interval polling (1s -> 3s -> 5s), jangan blok run script.

### 5. Query kosong/whitespace dan submit ganda tidak dicegah

File: `frontend/streamlit_app.py:786`.

`st.chat_input` hanya guard terhadap `None`, bukan string berisi whitespace saja, dan tidak ada pengecekan
duplikasi terhadap pesan sebelumnya. Query semacam ini tetap dikirim ke backend dan kena biaya penuh
pipeline RAG (embeddings + LLM).

Perbaikan: `if prompt and prompt.strip():` plus pengecekan kesamaan dengan pesan terakhir.

### 6. Dashboard bisa crash kalau schema metrics.json tidak lengkap

File: `frontend/pages/dashboard.py:148-151, 250`.

`avg_resp` sudah di-guard terhadap pembagian nol dengan benar, tapi
`df_display = df_table[list(display_cols.keys())]` akan `KeyError` dan meng-crash seluruh halaman Dashboard
jika ada entri lama di `metrics.json` yang kehilangan salah satu kolom
(question/response_time/retrieval_time/docs_retrieved/ndcg/hit_rate) - misalnya setelah perubahan schema
atau entri yang setengah tertulis akibat query yang crash.

Perbaikan: `.reindex(columns=..., fill_value=None)` sebelum memilih kolom tampilan.

### 7. Error saat hapus dokumen bocorin response/stack trace backend

File: `frontend/streamlit_app.py:592-600, 617-618`.

Saat `requests.delete(...).raise_for_status()` gagal, except block menampilkan
`e.response.json().get("detail", e.response.text)` atau `e.response.text`/`str(e)` mentah lewat `st.error`.
Kalau backend 500 dengan traceback Python di body (default FastAPI mode debug), traceback tersebut tampil
utuh ke end user.

Perbaikan: pakai helper `format_error_message()` (sudah ada di baris 399) secara konsisten juga untuk path
delete/metrics, bukan cuma query/upload.

### 8. Kegagalan network di-swallow jadi "No documents yet"

File: `frontend/streamlit_app.py:367-384, 387-396, 787`.

`_fetch_indexed_documents` / `get_indexed_files` menangkap semua exception lewat `except Exception: return
[]`. Kegagalan network (backend down, DNS error, timeout) membuat sidebar cuma nampilin "No documents yet."
tanpa indikasi backend tidak terjangkau, dan chat input malah menyuruh user upload PDF padahal dokumen
sebenarnya sudah ter-index.

Perbaikan: bedakan state "belum ada dokumen" vs "backend tidak terjangkau", tampilkan banner error koneksi.

### 9. Delete-all documents tanpa konfirmasi

File: `frontend/streamlit_app.py:607-618`.

Tombol "Delete All Documents" langsung eksekusi tanpa dialog konfirmasi, dan saat sukses juga menghapus
`st.session_state.messages` serta chat history persisten. Misclick menghapus seluruh knowledge base dan
riwayat chat secara ireversibel, tanpa undo.

Perbaikan: tambahkan checkbox/klik kedua sebagai konfirmasi sebelum eksekusi DELETE.

## Arsitektural (akar masalah #1)

### 10. Frontend import langsung modul internal backend via `sys.path` hack

File: `frontend/streamlit_app.py:16-18`.

`sys.path.append(... / "backend")` lalu `from app.cache_manager import CacheManager` membuat frontend harus
co-deploy di filesystem/container yang sama dengan backend dan berbagi konfigurasi Redis/diskcache
(`backend/app/config.py`), alih-alih berkomunikasi murni lewat HTTP API yang sudah didokumentasikan
(`INVENIOAI_API_BASE_URL`).

Ini adalah akar penyebab temuan #1 (cache key global), dan menghalangi frontend/backend untuk pernah di-scale
atau di-deploy terpisah - padahal arsitektur env-driven `API_BASE_URL` menyiratkan itu seharusnya mungkin.

Perbaikan: frontend hanya boleh bicara ke backend lewat REST API; state chat history harus per-sesi di sisi
frontend atau lewat endpoint backend yang menerima session/user identifier.

## Belum diaudit

Area berikut belum dicek pada sesi ini - lanjutkan kalau mau audit menyeluruh:

- `backend/app/rag_pipeline.py`, `retriever.py`, `reranker.py`, `cache_manager.py`, `embeddings.py`,
  `qdrant_conn.py`, `metrics.py` (logika inti RAG).
- `backend/app/main.py`, `index_api.py`, `index_data.py`, `config.py` (API layer, keamanan, konkurensi).
- Dockerfile, docker-compose.yml, start.sh, CI/CD, dependency pinning (kesiapan deployment/infra).
