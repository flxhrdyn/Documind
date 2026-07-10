# Audit Backend RAG Core & API - InvenioAI

Tanggal: 2026-07-10.
Scope: `backend/app/rag_pipeline.py`, `retriever.py`, `cache_manager.py`, `index_api.py`, `main.py`.
Metode: baca manual (tanpa subagent), fokus pada bug logika, konsep RAG, dan kesiapan produksi.
Lanjutan dari `docs/audit-frontend-ux.md`.

## Kritis

### 1. ~~Semantic caching diam-diam mati total saat `CACHE_TYPE=redis`~~ (FIXED)

File: `backend/app/cache_manager.py`.

Status: diperbaiki. `get_semantic()`/`add_semantic()` sekarang lewat `_load_semantic_registry()` /
`_save_semantic_registry()` yang mendukung kedua backend - untuk Redis, registry disimpan sebagai satu
key (`semantic_registry`) berisi JSON list, mirip pola diskcache yang sudah ada (linear scan, cap 1000
entri). Read-modify-write di `add_semantic()` juga sekarang dilindungi `threading.Lock()`, sekalian
memperbaiki temuan #17 (race condition) untuk kasus intra-process.

Catatan/batasan: lock ini cuma melindungi dalam satu proses. Kalau backend dijalankan multi-worker/replica
(beberapa proses Python terpisah mengakses Redis yang sama), race condition read-modify-write masih mungkin
terjadi lintas proses karena Redis sendiri tidak dikunci di level operasi ini. Untuk skala itu, perlu
Redis-native atomic op (mis. Lua script atau `WATCH`/`MULTI`) - dicatat sebagai potensi tindak lanjut kalau
deployment sudah butuh horizontal scaling.

README masih perlu diverifikasi ulang apakah klaim "Dual-Layer Semantic Caching (DiskCache / Redis)"
sekarang akurat untuk kedua mode - sudah benar secara fungsional, tinggal pastikan tidak ada klaim performa
lain yang perlu disesuaikan.

### 2. ~~Hapus 1 dokumen tidak membersihkan cache - jawaban basi tetap disajikan~~ (FIXED)

Status: diperbaiki. `delete_document()` sekarang memanggil `CacheManager().clear()` setelah delete sukses
(sama seperti `clear_documents()`), jadi cache basi ikut kehapus setiap kali knowledge base berubah.
Trade-off: ini clear seluruh cache (bukan cuma entri terkait dokumen yang dihapus) - cache lain yang masih
valid juga ikut hilang, tapi lebih aman daripada menyajikan jawaban salah. Kalau granularitas jadi masalah
performa nyata, bisa dioptimasi nanti dengan invalidasi per-dokumen.

File: `backend/app/index_api.py:252-328` (`delete_document`) dibandingkan `331-374` (`clear_documents`).

`clear_documents()` (hapus semua dokumen) secara eksplisit clear cache di baris 366-372. Tapi
`delete_document()` (hapus satu file) tidak pernah memanggil `cache.clear()` atau invalidasi cache spesifik
dokumen tersebut.

Skenario: User upload dokumen A, tanya sesuatu (ter-cache: exact/semantic/deep). User hapus dokumen A,
upload dokumen B dengan topik mirip. Pertanyaan yang serupa akan hit cache lama dan menyajikan jawaban dari
dokumen A yang sudah dihapus - jawaban salah, terlihat sebagai halusinasi/data lama padahal sebenarnya
bug caching.

Perbaikan: panggil `CacheManager().clear()` (atau invalidasi lebih presisi per-dokumen jika cache key
menyimpan referensi source) di `delete_document()` juga.

## Tinggi

### 3. ~~CORS `allow_origins=["*"]` dikombinasikan dengan `allow_credentials=True`~~ (FIXED)

File: `backend/app/main.py:128-135`, `backend/app/config.py` (`ALLOWED_ORIGINS`).

Status: diperbaiki. `ALLOWED_ORIGINS` sekarang env-driven (`INVENIOAI_ALLOWED_ORIGINS`, default `*`) dan
`allow_credentials` otomatis `False` selama masih wildcard, otomatis `True` begitu origin eksplisit
di-set (mis. untuk frontend React yang direncanakan). Dikonfirmasi bahwa komunikasi Streamlit->backend
saat ini murni server-to-server (Python `requests`), jadi tidak terpengaruh CORS sama sekali.

### 4. ~~Tidak ada autentikasi/otorisasi di endpoint mana pun~~ (FIXED, backend; frontend menyusul)

File: `backend/app/auth.py` (baru), diterapkan sebagai dependency di `backend/app/main.py`
(`/query`, `/query/stream`, `/metrics`, `/metrics/sync`) dan seluruh router `index_api.py` (`/upload`,
`/upload/jobs*`, `/documents*`).

Status: sisi backend selesai. `INVENIOAI_API_KEY` opsional (default kosong = auth nonaktif, supaya
demo/dev lokal tidak berubah perilakunya). Kalau di-set, semua endpoint di atas mewajibkan header
`X-API-Key` yang cocok, selain `GET /` yang tetap terbuka untuk health check.

**Belum lengkap end-to-end**: Streamlit frontend belum dikirimi header ini di request-nya - kalau
`INVENIOAI_API_KEY` diaktifkan di production sekarang, UI Streamlit sendiri akan kena 401. Perlu tindak
lanjut di batch audit frontend: baca env yang sama di sisi Streamlit dan lampirkan `X-API-Key` di semua
panggilan `requests.*`.

## Sedang

### 5. ~~Prompt streaming pakai `standalone_query` bukan pertanyaan asli user~~ (FIXED)

File: `backend/app/rag_pipeline.py` (`rag_pipeline_stream_async`).

Status: diperbaiki. Prompt final di jalur streaming sekarang pakai `question=query` (pertanyaan asli user),
konsisten dengan jalur sync yang sudah pakai `original_question`. `standalone_query` tetap dipakai untuk
retrieval saja.

### 6. ~~`build_retriever()` bikin ulang `ChatGroq` + `MultiQueryRetriever` di setiap query, tanpa cache~~ (FIXED)

File: `backend/app/llm.py` (baru), `backend/app/retriever.py` (`build_retriever`,
`invalidate_retriever_cache`), `backend/app/index_api.py` (`_on_knowledge_base_changed`).

Status: diperbaiki. Dua bagian:

1. **LLM singleton dipakai bersama** - `_get_llm()` yang tadinya duplikat (satu di `rag_pipeline.py`, satu
   `ChatGroq` baru di `retriever.py` khusus untuk `MultiQueryRetriever`) sekarang jadi satu
   `get_llm()` di `backend/app/llm.py` (`@lru_cache(maxsize=1)`), dipakai oleh `rag_pipeline.py` (lewat
   alias `_get_llm`, supaya mock test lama tetap jalan tanpa perubahan) dan `retriever.py`.
2. **Retriever stack di-cache process-wide** - `build_retriever()` sekarang menyimpan
   `(retriever, vectorstore, client)` di cache module-level, dilindungi lock. Cache hanya rebuild kalau
   instance Qdrant client berubah (deteksi otomatis lewat identity check, murah - tidak ada network call)
   atau setelah `invalidate_retriever_cache()` dipanggil eksplisit. `index_api.py` memanggil ini lewat
   helper `_on_knowledge_base_changed()` setiap kali upload/hapus dokumen sukses, supaya query berikutnya
   tetap re-validasi koleksi Qdrant terbaru.

Efeknya: `client.get_collections()` (network round-trip) dan konstruksi `MultiQueryRetriever`/`ChatGroq`
baru tidak lagi terjadi di setiap query - hanya saat cache invalid (pertama kali, atau setelah dokumen
berubah).

### 7. ~~`_save_uploaded_pdf` tidak membatasi ukuran file (server-side)~~ (FIXED)

File: `backend/app/index_api.py` (`_save_uploaded_pdf`), `backend/app/config.py` (`MAX_UPLOAD_SIZE_MB`).

Status: diperbaiki. Upload sekarang ditulis ke disk per-chunk (1MB per baca, bukan `file.file.read()`
sekaligus) dan dihentikan begitu total melebihi `INVENIOAI_MAX_UPLOAD_SIZE_MB` (default 100MB) - file
parsial otomatis dihapus dan client menerima HTTP 413. Ini juga menutup celah yang disinggung di
`docs/audit-infra-deploy.md` #2 (batas ukuran sebelumnya cuma ada di sisi Streamlit UI, bukan API).

### 8. ~~Upload ulang file dengan nama sama menghasilkan duplikat chunk di Qdrant~~ (FIXED)

File: `backend/app/index_api.py` (`_save_uploaded_pdf`, `_find_duplicate_document`,
`_index_uploaded_pdf`), `backend/app/index_data.py` (`index_documents`).

Status: diperbaiki. `_save_uploaded_pdf` sekarang menghitung sha256 dari isi file sambil streaming ke disk
(tanpa baca dua kali), dan mengembalikan `(file_path, content_hash)`. Sebelum indexing dijalankan,
`_find_duplicate_document()` mengecek apakah sudah ada chunk di Qdrant dengan `metadata.content_hash` yang
sama persis - kalau ada, upload ditolak dengan HTTP 409 dan pesan menyebutkan nama file yang sudah
ter-index, file yang baru disimpan dihapus lagi, tanpa membuang waktu parsing LlamaParse. Kalau lolos,
`content_hash` ikut disimpan di metadata setiap chunk lewat `index_documents(..., content_hash=...)` supaya
pengecekan berikutnya bisa jalan. Ini tetap membiarkan file dengan nama sama tapi isi beda diterima (dengan
suffix UUID seperti sebelumnya) - yang dicegah cuma duplikasi isi identik.

### 9. ~~Batch indexing tidak punya penanganan kegagalan parsial~~ (FIXED)

File: `backend/app/index_data.py` (`index_documents`, blok `try/except` di sekitar loop batch).

Status: diperbaiki. Loop batch sekarang dibungkus `try/except`: kalau batch ke-N gagal (mis. Qdrant timeout,
koneksi putus di tengah upload besar), batch 1..N-1 yang sudah ke-upsert untuk dokumen tersebut dihapus
lewat filter `metadata.source_file`/`metadata.source` (pola yang sama dengan `delete_document()`) sebelum
exception dilempar ulang ke `_index_uploaded_pdf`/`_run_upload_job`. Kalau rollback delete-nya sendiri gagal
(skenario langka), itu di-log jelas sebagai kondisi yang butuh pembersihan manual, bukan disembunyikan.

### 10. ~~`HYBRID_DENSE_WEIGHT` / `HYBRID_SPARSE_WEIGHT` adalah config mati~~ (FIXED - dihapus)

Status: diselidiki, bukan diimplementasikan. Dicek langsung ke `langchain_qdrant.QdrantVectorStore` yang
dipakai (`inspect.signature`) - versi library yang ter-install **tidak** expose parameter fusion weight
sama sekali; hanya native RRF fusion Qdrant tanpa knob bobot. Meneruskan bobot ini butuh bypass wrapper
`QdrantVectorStore` dan panggil `client.query_points` manual dengan konfigurasi Fusion custom - perubahan
besar dengan risiko regresi tidak sepadan untuk saat ini.

Keputusan: `HYBRID_DENSE_WEIGHT`/`HYBRID_SPARSE_WEIGHT` dihapus dari `config.py` dan `retriever.py` (sesuai
opsi kedua di rekomendasi awal - "hapus variabel ini kalau memang tidak dipakai"). Sekalian dibersihkan:
`.env.example` sebelumnya juga mendokumentasikan sederet env var lain
(`INVENIOAI_HYBRID_LEXICAL_K`, `INVENIOAI_HYBRID_FUSION_LIMIT`, `INVENIOAI_HYBRID_MAX_LEXICAL_DOCS`,
`INVENIOAI_HYBRID_RRF_K`, `INVENIOAI_HYBRID_LEXICAL_WEIGHT`) yang ternyata **tidak pernah dibaca di kode
mana pun** (dikonfirmasi grep) - sisa dokumentasi basi dari implementasi hybrid berbasis BM25 lokal yang
sudah digantikan native BM42 Qdrant. Semua ikut dihapus dari `.env.example`.

### 11. ~~`total_documents_indexed` tidak update real-time~~ (FIXED)

File: `backend/app/index_api.py` (`_sync_document_count`, dipanggil dari `_index_uploaded_pdf`,
`delete_document`, `clear_documents`).

Status: diperbaiki. Alih-alih memakai `log_document_indexed()` (increment naif yang tidak match jumlah
dokumen unik sebenarnya), fix ini menambahkan helper `_sync_document_count()` yang reuse logika
`list_documents()` (scan dokumen unik dari Qdrant) dan memanggil `sync_indexed_docs_count()` - dipanggil
setelah upload sukses dan setelah delete (satu dokumen atau semua). `clear_documents()` langsung
`sync_indexed_docs_count(0)` karena collection-nya sudah dihapus seluruhnya. Dashboard sekarang akurat
tanpa perlu restart server.

### 12. ~~Threshold IR metrics (0.7) diasumsikan skala skor reranker 0-1, tidak divalidasi~~ (SEBAGIAN FIXED)

File: `backend/app/config.py` (`IR_RELEVANCE_THRESHOLD`), `backend/app/metrics.py`.

Status: sebagian diperbaiki. Angka `0.7` yang tadinya hardcoded di 6 tempat berbeda
(`precision_at_k`, `recall_at_k`, `mrr`, `hit_rate_at_k`, `compute_ir_metrics`, `per_query_ir_metrics`)
sekarang satu sumber kebenaran lewat `INVENIOAI_IR_RELEVANCE_THRESHOLD` (default tetap 0.7, didokumentasikan
di `.env.example` beserta asumsinya). Operator sekarang bisa kalibrasi ulang tanpa ubah kode.

**Belum lengkap**: verifikasi rentang skor aktual `RERANKER_MODEL` (`ms-marco-MiniLM-L-12-v2` via FlashRank)
belum dilakukan di sesi ini - butuh menjalankan model sungguhan dengan query/dokumen nyata dan sampling
distribusi skornya, yang di luar jangkauan perbaikan kode statis. Threshold 0.7 tetap asumsi berdasar nama
"masuk akal", bukan angka yang sudah dikonfirmasi dari observasi produksi.

### 13. ~~Ketidakcocokan model embedding tidak pernah dicek saat re-index ke koleksi yang sudah ada~~ (FIXED)

File: `backend/app/index_data.py` (`_check_or_record_embedding_model`, dipanggil dari `index_documents`).

Status: diperbaiki. Saat koleksi Qdrant pertama kali dibuat, disisipkan satu "marker point" tersembunyi
(ID tetap, payload `{"__meta__": True, "embedding_model": ...}`) yang mencatat `EMBEDDING_MODEL` aktif.
Setiap kali `index_documents()` dipanggil berikutnya, marker ini di-retrieve dan dibandingkan dengan
`EMBEDDING_MODEL` yang sedang aktif - kalau beda, index ditolak dengan pesan jelas **sebelum** proses parsing
LlamaParse dijalankan (gagal cepat, tidak buang kuota parsing). Marker point otomatis tidak ikut terhitung
sebagai dokumen oleh `list_documents()` karena tidak punya field `metadata.source_file`/`source`.

### 14. ~~Chunking "table-aware" tetap bisa memotong tabel di tengah baris~~ (FIXED)

File: `backend/app/index_data.py` (`_find_table_header`, dipakai di `process_pdf_documents`).

Status: diperbaiki. Saat header-splitting per section, section yang mengandung tabel markdown sekarang
mendeteksi baris header + separator tabel pertamanya (`_find_table_header`) dan menyimpannya sementara di
`chunk.metadata["_table_header"]`. Setelah `RecursiveCharacterTextSplitter` menghasilkan `final_chunks`,
setiap chunk yang mulai di tengah tabel (`page_content` diawali `|` tapi belum mengandung header itu di
awal) langsung ditempeli ulang baris header sebelum disimpan; key metadata sementara itu dibuang lagi
supaya tidak ikut ke payload Qdrant.

Diverifikasi manual dengan tabel 120 baris (~5300 karakter, exceeds `CHUNK_SIZE=1500`): hasilnya 4 chunk,
dan keempatnya sekarang diawali baris header `| Metric | 2022 | 2023 |` - sebelumnya cuma chunk pertama
yang punya header, 3 chunk sisanya berisi baris data telanjang tanpa label kolom.

### 15. ~~Cache key hasil rewrite pakai jendela riwayat chat yang beda dari riwayat yang benar-benar dipakai~~ (FIXED)

File: `backend/app/rag_pipeline.py` (`rewrite_query`, `rewrite_query_async`).

Status: diperbaiki. Kedua fungsi sekarang menghitung `history_text` sekali (`format_history(history)`,
history penuh) dan memakainya baik untuk hash `rewrite_cache_key` maupun untuk prompt rewrite yang
sebenarnya - key sekarang konsisten dengan input yang benar-benar memengaruhi hasil, tidak ada lagi
kolisi antar percakapan berbeda.

### 16. ~~Jalur sync `/query` tidak menjaga kasus hasil retrieval kosong~~ (FIXED)

File: `backend/app/rag_pipeline.py` (`_run_rag_pipeline_with_query`, `rag_pipeline`).

Status: diperbaiki. `_run_rag_pipeline_with_query` sekarang mengembalikan pesan "Maaf, tidak ditemukan
informasi relevan." (setara jalur streaming) begitu `retrieved_docs` kosong, tanpa memanggil LLM. Sebagai
efek samping yang benar, `rag_pipeline()` juga diubah supaya **tidak menyimpan** hasil "tidak ditemukan" ke
cache (dicek lewat `metrics.docs_retrieved == 0`) - meniru perilaku jalur streaming yang juga tidak
meng-cache kondisi ini, karena retrieval bisa berhasil nanti setelah dokumen baru diindex.

Test terkait (`test_caching.py`, `test_semantic_caching_integration.py`) diperbarui supaya mock hasil RAG
menyertakan `metrics.docs_retrieved` yang realistis, sesuai perilaku baru ini.

### 17. ~~Race condition pada read-modify-write registry semantic cache~~ (FIXED, intra-process)

File: `backend/app/cache_manager.py` (`get_semantic`/`add_semantic`).

Status: diperbaiki sekaligus saat mengerjakan fix #1 - `add_semantic()` sekarang membungkus seluruh
read-modify-write dengan `self._semantic_lock` (`threading.Lock`). Lihat catatan batasan multi-proses di
temuan #1: lock ini melindungi dalam satu proses Python, belum melindungi race lintas proses/replica yang
berbagi Redis yang sama.

### 18. Guard angka/tahun di semantic cache terlalu kaku (exact-match set) - SENGAJA TIDAK DIKERJAKAN

File: `backend/app/cache_manager.py` (`get_semantic`, guard angka/tahun).

Status: dilewati secara sadar. Rekomendasi awal sendiri menandai ini "opsional" dan "dampak minor" - efeknya
cuma false-negative cache miss (query dianggap beda padahal ekuivalen), bukan jawaban salah. Tidak ada
laporan/observasi produksi yang menunjukkan ini signifikan, jadi tidak sepadan menambah kompleksitas heuristik
sekarang. Tetap tercatat di sini kalau nanti terbukti perlu.

## Belum diaudit

Area berikut belum dicek pada sesi ini:

- `backend/app/utils.py` (`format_docs`, `ThinkingParser`) - sudah dicek terpisah di
  `docs/audit-infra-deploy.md`.
- Dockerfile, docker-compose.yml (selain rujukan `CACHE_TYPE=redis` di atas) - sudah dicek terpisah di
  `docs/audit-infra-deploy.md`. CI/CD dan dependency pinning masih belum.
- `backend/tests/` - belum dicek seberapa besar celah di atas sudah/belum tercakup oleh test yang ada.
