"""InvenioAI Streamlit chat UI.

This Streamlit app is a thin client for the FastAPI backend. Configure the
backend URL with `INVENIOAI_API_BASE_URL` (defaults to `http://localhost:8000`).
"""

import json
import os
import threading
import time

import requests
import streamlit as st

st.set_page_config(
    page_title="InvenioAI | Intelligent RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

import importlib
import theme
importlib.reload(theme)
from theme import CSS_VARS

API_BASE_URL = os.getenv("INVENIOAI_API_BASE_URL", "http://localhost:8000").rstrip("/")

# Sent with every backend request. Empty unless INVENIOAI_API_KEY is set,
# matching the backend's opt-in auth (backend/app/auth.py).
_api_key = (os.getenv("INVENIOAI_API_KEY") or "").strip()
API_HEADERS = {"X-API-Key": _api_key} if _api_key else {}


def _get_max_upload_size_mb() -> int:
    raw = (os.getenv("INVENIOAI_MAX_UPLOAD_SIZE_MB") or "100").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 100


MAX_UPLOAD_SIZE_MB = _get_max_upload_size_mb()


def _is_hf_spaces_runtime() -> bool:
    return bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST"))


def _get_query_read_timeout_seconds() -> int:
    # README states ~15s average, but cold model starts / long generations can
    # run well past a fixed 60s cap and would surface as a raw ReadTimeout.
    default_timeout = 120
    raw = (os.getenv("INVENIOAI_QUERY_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return default_timeout
    try:
        value = int(raw)
    except ValueError:
        return default_timeout
    return max(30, min(value, 600))


QUERY_READ_TIMEOUT_SECONDS = _get_query_read_timeout_seconds()


def _get_upload_timeout_seconds() -> int:
    # Upload + indexing may take longer on cold starts, especially in HF Spaces.
    default_timeout = 600
    raw = (os.getenv("INVENIOAI_UPLOAD_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return default_timeout

    try:
        value = int(raw)
    except ValueError:
        return default_timeout

    return max(30, min(value, 3600))


UPLOAD_TIMEOUT_SECONDS = _get_upload_timeout_seconds()
UPLOAD_JOB_POLL_INTERVAL_SECONDS = 1.0
UPLOAD_JOB_WAIT_SECONDS = UPLOAD_TIMEOUT_SECONDS
UPLOAD_DURATION_HISTORY_KEY = "upload_duration_history"
MAX_UPLOAD_DURATION_SAMPLES = 10


def _get_assistant_typing_enabled() -> bool:
    raw = (os.getenv("INVENIOAI_ASSISTANT_TYPING_EFFECT") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _get_assistant_word_delay_seconds() -> float:
    raw = (os.getenv("INVENIOAI_ASSISTANT_TYPING_WORD_DELAY_SECONDS") or "0.016").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 0.016
    return max(0.0, min(value, 0.08))


def _get_assistant_typing_max_words() -> int:
    raw = (os.getenv("INVENIOAI_ASSISTANT_TYPING_MAX_WORDS") or "140").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 140
    return max(20, min(value, 500))


ASSISTANT_TYPING_ENABLED = _get_assistant_typing_enabled()
ASSISTANT_TYPING_WORD_DELAY_SECONDS = _get_assistant_word_delay_seconds()
ASSISTANT_TYPING_MAX_WORDS = _get_assistant_typing_max_words()


def _is_chat_active() -> bool:
    return True


def _get_upload_duration_history() -> list[float]:
    history = st.session_state.get(UPLOAD_DURATION_HISTORY_KEY)
    if isinstance(history, list):
        clean: list[float] = []
        for value in history:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric > 0:
                clean.append(numeric)
        st.session_state[UPLOAD_DURATION_HISTORY_KEY] = clean[-MAX_UPLOAD_DURATION_SAMPLES:]
        return st.session_state[UPLOAD_DURATION_HISTORY_KEY]

    st.session_state[UPLOAD_DURATION_HISTORY_KEY] = []
    return st.session_state[UPLOAD_DURATION_HISTORY_KEY]


def _record_upload_duration(seconds: float) -> None:
    if seconds <= 0:
        return
    history = _get_upload_duration_history()
    history.append(seconds)
    st.session_state[UPLOAD_DURATION_HISTORY_KEY] = history[-MAX_UPLOAD_DURATION_SAMPLES:]


def fetch_metrics() -> tuple[dict | None, str | None]:
    """Fetch aggregate metrics from backend."""
    try:
        resp = requests.get(f"{API_BASE_URL}/metrics", headers=API_HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # If docs is 0, try to force a sync once on frontend startup/refresh
            if data.get("total_documents_indexed", 0) == 0:
                try:
                    requests.post(f"{API_BASE_URL}/metrics/sync", headers=API_HEADERS, timeout=5)
                    # Re-fetch after sync
                    resp = requests.get(f"{API_BASE_URL}/metrics", headers=API_HEADERS, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                except requests.exceptions.RequestException:
                    pass
            return data, None
        return None, format_error_message(resp)
    except requests.exceptions.RequestException as e:
        return None, format_error_message(e)


def _estimate_upload_eta_seconds(elapsed_seconds: float) -> int | None:
    history = _get_upload_duration_history()
    if not history:
        return None

    avg_seconds = sum(history) / len(history)
    remaining = int(round(avg_seconds - elapsed_seconds))
    if remaining <= 0:
        return 0
    return remaining


def _render_assistant_message(reply: str) -> None:
    """Render assistant reply with a safe typing effect.

    We animate short previews for UX, then always render the final full markdown
    atomically to avoid broken layout after page switches/reruns.
    """
    if not ASSISTANT_TYPING_ENABLED:
        st.markdown(reply)
        return

    words = reply.split()
    if not words:
        st.markdown(reply)
        return

    preview_words = min(len(words), ASSISTANT_TYPING_MAX_WORDS)
    placeholder = st.empty()
    current: list[str] = []

    for word in words[:preview_words]:
        if not _is_chat_active():
            break
        current.append(word)
        placeholder.markdown(" ".join(current) + " ▌")
        time.sleep(ASSISTANT_TYPING_WORD_DELAY_SECONDS)

    placeholder.markdown(reply)

# Active page tracking is now handled by Streamlit's multi-page system


# ── Design System ─────────────────────────────────────────────────────────────

# Inject Google Fonts
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
{CSS_VARS}

/* ── Global Typography ── */
/* Apply font only to text-heavy elements to avoid breaking icons */
h1, h2, h3, h4, h5, h6, p, li, button, input, textarea, [data-testid="stChatMessage"] {{
    font-family: 'Outfit', sans-serif !important;
}}

/* ── Branding ── */
.branding {{
    font-weight: 700 !important;
}}

/* ── Welcome Screen Centering ── */
.welcome-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    min-height: 65vh; /* Adjusted to center in viewport */
}}

.welcome-title {{
    font-weight: 700 !important;
    font-size: 3.5rem !important;
    margin-bottom: 0.5rem !important;
}}

.welcome-subtitle {{
    font-size: 1.2rem !important;
    opacity: 0.8;
}}

/* ── Sidebar Spacing ── */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0.75rem !important;
}}

/* Tighten spacing between elements in the sidebar to match Dashboard */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0.75rem !important;
}}

/* ── Container Layout ── */
.block-container {{
    max-width: 1100px !important;
    padding-top: 2rem !important;
}}

/* ── UI Cleanup ── */
footer {{ visibility: hidden; }}
/* Keep header visible for the menu, but make it clean */
header {{ 
    background-color: transparent !important;
}}

/* ── Chat Input Focus (override default red focus ring with brand accent) ── */
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"]:focus-within {{
    border-color: var(--invenio-accent) !important;
    box-shadow: 0 0 0 1px var(--invenio-accent) !important;
}}

/* ── Chat Message Styling ── */
[data-testid="stChatMessage"] {{
    background-color: var(--invenio-bg-secondary) !important;
    border: 1px solid var(--invenio-border) !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
    margin-bottom: 1rem !important;
}}

/* ── Expander Styling ── */
.stExpander {{
    border: 1px solid var(--invenio-border) !important;
    border-radius: 8px !important;
    background-color: var(--invenio-bg-card) !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
    margin-bottom: 0.5rem !important;
}}

/* Ultra-aggressive fix for status/thinking empty boxes */
div[data-testid="stStatusWidget"], 
div[data-testid="stStatusWidget"] > details,
div[data-testid="stStatusWidget"] > details > summary {{
    border: none !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
}}

.stCaption {{
    word-break: break-all !important;
}}

/* ── Status Bar Alignment ── */
.stStatusWidget {{
    border: 1px solid var(--invenio-border) !important;
    border-radius: 8px !important;
}}

/* ── Avatar Alignment ── */
div[data-testid="stChatMessageAvatar"] {{
    margin-top: 8px !important;
}}

/* ── Button Transitions ── */
button {{
    transition: all 0.2s ease !important;
}}
/* ── Table Styling ── */
table {{
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 1rem 0 !important;
    font-size: 0.9rem !important;
}}

th {{
    background-color: var(--invenio-bg-secondary) !important;
    color: var(--invenio-accent) !important;
    text-align: left !important;
    padding: 10px !important;
    border-bottom: 2px solid var(--invenio-border) !important;
}}

td {{
    padding: 8px 10px !important;
    border-bottom: 1px solid var(--invenio-border) !important;
}}

tr:hover {{
    background-color: rgba(128, 128, 128, 0.05) !important;
}}

/* Ensure horizontal scroll for wide tables */
.stMarkdown div[style*="overflow-x: auto"] {{
    scrollbar-width: thin;
    scrollbar-color: var(--invenio-border) transparent;
}}
</style>
""", unsafe_allow_html=True)


def _fetch_indexed_documents(api_base_url: str) -> tuple[list[str], bool]:
    """Fetch filenames from backend. Returns `(docs, backend_reachable)` so
    callers can tell "no documents yet" apart from "couldn't reach backend" -
    they need very different messaging.

    We handle caching manually to avoid caching empty states during startup.
    """
    now = time.time()

    # Check manual cache in session_state
    if "docs_cache" in st.session_state:
        cache_data, cache_time = st.session_state.docs_cache
        if now - cache_time < 30: # 30s TTL
            return cache_data, True

    try:
        resp = requests.get(f"{api_base_url}/documents", headers=API_HEADERS, timeout=5)
    except requests.exceptions.RequestException:
        return [], False

    if resp.status_code != 200:
        return [], True

    payload = resp.json() or {}
    if isinstance(payload, list):
        docs = [str(d) for d in payload if d]
    else:
        docs = [str(d) for d in (payload.get("documents") or []) if d]

    # ONLY cache if we found documents.
    # If empty, don't cache so we keep polling on next rerun.
    if docs:
        st.session_state.docs_cache = (docs, now)
    return docs, True


def get_indexed_files() -> tuple[list[str], bool]:
    """Returns `(docs, backend_reachable)` - see `_fetch_indexed_documents`."""
    try:
        return _fetch_indexed_documents(API_BASE_URL)
    except Exception:
        return [], False


def format_error_message(exc) -> str:
    """Turn a request failure into a user-facing message without leaking
    internal exception/traceback details.

    Accepts a `requests.Response` (non-2xx status, no exception raised), a
    `requests.exceptions.RequestException` (network-level - no response
    object), or any exception carrying an HTTP `response`
    (e.g. from `Response.raise_for_status()`).
    """
    response = exc if isinstance(exc, requests.Response) else getattr(exc, "response", None)
    if response is not None:
        text = response.text.lower()
        if "quota" in text or "rate_limit" in text or "429" in text:
            return (
                "⚠️ **Groq API Rate Limit exceeded.** "
                "You have reached the request limit for your Groq plan. "
                "Wait a moment or check your Groq dashboard."
            )
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        return f"❌ **Error {response.status_code}:** {detail}"

    if isinstance(exc, requests.exceptions.ConnectionError):
        return f"❌ **Connection Error:** Could not connect to the backend at {API_BASE_URL}."
    if isinstance(exc, requests.exceptions.Timeout):
        return "⏱️ **Timeout:** The request took too long. Please try again."
    return f"❌ **Error:** {type(exc).__name__}. Please try again."


def create_upload_job(uploaded_file) -> tuple[str | None, str | None]:
    try:
        resp = requests.post(
            f"{API_BASE_URL}/upload/jobs",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            headers=API_HEADERS,
            timeout=120,
        )
    except requests.exceptions.RequestException as e:
        return None, format_error_message(e)

    if resp.status_code != 200:
        return None, format_error_message(resp)

    try:
        payload = resp.json()
    except Exception:
        return None, f"❌ **Error {resp.status_code}:** Unexpected response from backend."

    job_id = payload.get("job_id")
    if not job_id:
        return None, "❌ **Error:** Missing upload job id from backend response."
    return str(job_id), None


def fetch_upload_job(job_id: str) -> tuple[dict | None, str | None]:
    try:
        resp = requests.get(f"{API_BASE_URL}/upload/jobs/{job_id}", headers=API_HEADERS, timeout=15)
    except requests.exceptions.RequestException as e:
        return None, format_error_message(e)

    if resp.status_code != 200:
        return None, format_error_message(resp)

    try:
        return resp.json(), None
    except Exception:
        return None, f"❌ **Error {resp.status_code}:** Unexpected response from backend."


def _render_upload_job_status(
    status_slot,
    status: str,
    filename: str,
    elapsed_seconds: float,
    eta_seconds: int | None,
) -> None:
    if status_slot is None:
        return

    label = f"**{filename}**" if filename else "**Document**"
    eta_suffix = ""
    if eta_seconds is not None:
        eta_suffix = " (ETA now)" if eta_seconds == 0 else f" (ETA ~{eta_seconds}s)"

    if status == "pending":
        status_slot.info(f"{label} queued for indexing... ({elapsed_seconds:.0f}s){eta_suffix}")
        return
    if status == "running":
        status_slot.info(f"{label} is starting up... ({elapsed_seconds:.0f}s){eta_suffix}")
        return
    if status == "parsing":
        status_slot.info(f"{label} is being parsed by AI Cloud... ({elapsed_seconds:.0f}s){eta_suffix}")
        return
    if status == "indexing":
        status_slot.info(f"{label} is being saved to vector store... ({elapsed_seconds:.0f}s){eta_suffix}")
        return
    if status in ("succeeded", "failed"):
        # Terminal states: leave the final st.success/st.error (rendered by the
        # caller with the full detail message) as the only notice, instead of
        # duplicating a generic one here first.
        status_slot.empty()
        return

    status_slot.info(f"{label} status: {status} ({elapsed_seconds:.0f}s)")





def _next_poll_interval(current: float) -> float:
    """Back off polling frequency: 1s -> 3s -> 5s (capped), instead of a
    fixed 1s interval for the whole (up to 3600s) indexing wait."""
    return min(current * 2, 5.0)


def wait_for_upload_job(job_id: str, *, status_slot=None, filename: str = "") -> tuple[bool, str]:
    started = time.monotonic()
    poll_interval = UPLOAD_JOB_POLL_INTERVAL_SECONDS
    while True:
        elapsed = time.monotonic() - started
        eta_seconds = _estimate_upload_eta_seconds(elapsed)
        job, err = fetch_upload_job(job_id)
        if err:
            if status_slot is not None:
                status_slot.error(err)
            return False, err

        status = (job or {}).get("status")
        _render_upload_job_status(status_slot, str(status), filename, elapsed, eta_seconds)

        if status == "succeeded":
            _record_upload_duration(elapsed)
            result = (job or {}).get("result") or {}
            done_filename = result.get("filename") or filename or "file"
            return True, f"✅ {done_filename} indexed successfully!"

        if status == "failed":
            error_msg = (job or {}).get("error") or "Unknown error"
            return False, f"❌ **Indexing failed:** {error_msg}"

        if time.monotonic() - started > UPLOAD_JOB_WAIT_SECONDS:
            return False, (
                "⏱️ Indexing is still running in the background. "
                "Please wait a moment, then click Refresh Data or reopen this page."
            )

        time.sleep(poll_interval)
        poll_interval = _next_poll_interval(poll_interval)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("InvenioAI")
    st.caption("AI · Document Intelligence")

    # Upload Section
    st.subheader("📤 Upload PDF")

    delete_after_index = (
        (os.getenv("INVENIOAI_DELETE_UPLOADED_PDFS") or os.getenv("DELETE_UPLOADED_PDFS") or "0").strip() == "1"
    )
    if delete_after_index:
        st.info('Uploaded PDFs will be deleted after indexing.')

    uploaded_file = st.file_uploader(
        f"Add documents to build your AI knowledge base (max {MAX_UPLOAD_SIZE_MB}MB)",
        key=f"pdf_uploader_{st.session_state.get('pdf_uploader_gen', 0)}",
        type=["pdf"],
    )
    if uploaded_file:
        upload_status_slot = st.empty()
        if uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            st.error(f"❌ File exceeds the {MAX_UPLOAD_SIZE_MB}MB upload limit.")
        elif st.button("⚡ Process & Index", width="stretch"):
            with st.spinner("Indexing document..."):
                job_id, err = create_upload_job(uploaded_file)
                if err or not job_id:
                    st.error(err or "❌ **Error:** Failed to create upload job.")
                else:
                    upload_status_slot.info(f"**{uploaded_file.name}** uploaded. Starting indexing job...")
                    ok, message = wait_for_upload_job(
                        job_id,
                        status_slot=upload_status_slot,
                        filename=uploaded_file.name,
                    )
                    if ok:
                        st.success(message)
                        if "docs_cache" in st.session_state:
                            del st.session_state.docs_cache
                        st.session_state.pdf_uploader_gen = st.session_state.get("pdf_uploader_gen", 0) + 1
                        st.rerun()
                    else:
                        if "still running in the background" in message:
                            st.info(message)
                        else:
                            st.error(message)



    # Knowledge Base
    st.subheader("🧠 Knowledge Base")
    indexed_files, backend_reachable = get_indexed_files()
    if indexed_files:
        for f in indexed_files:
            if st.session_state.get("confirm_delete_doc") == f:
                st.warning(f"Delete **{f}**? This cannot be undone.")
                confirm_col1, confirm_col2 = st.columns(2)
                with confirm_col1:
                    if st.button("✅ Yes", key=f"del_yes_{f}", width="stretch"):
                        try:
                            resp = requests.delete(
                                f"{API_BASE_URL}/documents/delete",
                                params={"filename": f},
                                headers=API_HEADERS,
                                timeout=30,
                            )
                            resp.raise_for_status()
                            if "docs_cache" in st.session_state:
                                del st.session_state.docs_cache
                            st.session_state.confirm_delete_doc = None
                            st.rerun()
                        except requests.exceptions.RequestException as e:
                            st.error(format_error_message(e))
                with confirm_col2:
                    if st.button("Cancel", key=f"del_cancel_{f}", width="stretch"):
                        st.session_state.confirm_delete_doc = None
                        st.rerun()
            else:
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.write(f"📄 {f}")
                with col2:
                    if st.button("🗑️", key=f"del_{f}"):
                        st.session_state.confirm_delete_doc = f
                        st.rerun()
    elif not backend_reachable:
        st.error(f"⚠️ Cannot reach backend at {API_BASE_URL}. Documents may still be indexed.")
    else:
        st.write("No documents yet.")
    # Force spacing to match Dashboard (cancel out list container padding)
    st.markdown('<div style="margin-top: -70px;"></div>', unsafe_allow_html=True)
    
    # Action Buttons (tightly coupled to Knowledge Base)
    # Delete-all is destructive and irreversible, so require an explicit
    # second confirmation click before it fires.
    if not st.session_state.get("confirm_delete_all"):
        if st.button("🗑️ Delete All Documents", width="stretch"):
            st.session_state.confirm_delete_all = True
            st.rerun()
    else:
        st.warning("This permanently deletes all documents and chat history. Are you sure?")
        confirm_col1, confirm_col2 = st.columns(2)
        with confirm_col1:
            if st.button("✅ Yes, delete all", width="stretch"):
                with st.spinner("Deleting..."):
                    try:
                        resp = requests.delete(f"{API_BASE_URL}/documents", headers=API_HEADERS, timeout=60)
                        resp.raise_for_status()
                        st.session_state.messages = []
                        if "docs_cache" in st.session_state:
                            del st.session_state.docs_cache
                        st.session_state.confirm_delete_all = False
                        st.session_state.pdf_uploader_gen = st.session_state.get("pdf_uploader_gen", 0) + 1
                        st.rerun()
                    except requests.exceptions.RequestException as e:
                        st.error(format_error_message(e))
        with confirm_col2:
            if st.button("Cancel", width="stretch"):
                st.session_state.confirm_delete_all = False
                st.rerun()

    if st.button("💬 Clear Chat History", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)



# ── Chat ──────────────────────────────────────────────────────────────────────
# Chat history lives only in st.session_state - scoped to this browser
# session by Streamlit itself, so it can never leak between users. It won't
# survive a full page reload, which is the correct tradeoff for a RAG demo
# without user accounts.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome screen when no messages
if _is_chat_active():
    if not st.session_state.messages:
        st.markdown("""
            <div class="welcome-container">
                <h1 class="welcome-title">🧠 InvenioAI</h1>
                <p class="welcome-subtitle">Ask anything about your Knowledge Base.</p>
            </div>
        """, unsafe_allow_html=True)

    # Render history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Display Thinking Process (if any)
            thoughts = message.get("thoughts")
            if thoughts:
                with st.expander("🧠 Thought Process", expanded=False):
                    if isinstance(thoughts, list):
                        st.markdown("\n".join(thoughts))
                    else:
                        # Ultra-aggressive cleanup: remove all stars anywhere near Step X
                        import re
                        # 1. Remove all stars first to get raw text
                        clean_thoughts = thoughts.replace("**", "")
                        # 2. Format Step X: with proper bolding and spacing
                        clean_thoughts = re.sub(r'(?i)(Step\s*\d+:)', r'\n\n**\1**', clean_thoughts)
                        st.markdown(clean_thoughts.strip())
            
            st.markdown(message["content"])
        
            # Interactive Sources for assistant messages
            sources = message.get("sources")
            if sources and isinstance(sources, list):
                # Group sources by filename
                from collections import defaultdict
                grouped = defaultdict(list)
                for s in sources:
                    grouped[s['file']].append({
                        "text": s.get('text', ''),
                        "page": s.get('page'),
                        "header": s.get('header'),
                        "score": s.get('score')
                    })

                source_items = list(grouped.items())
                if source_items:
                    with st.expander(f"📚 {len(source_items)} Sources", expanded=False):
                        for filename, snippets in source_items:
                            st.markdown(f"**📄 {filename}**")
                            for item in snippets:
                                text = item["text"]
                                page = item["page"]
                                header = item.get("header")
                                score = item.get("score")

                                meta_parts = []
                                if page:
                                    meta_parts.append(f"**Page {page}**")
                                if header:
                                    meta_parts.append(f"_{header}_")
                                if isinstance(score, (int, float)):
                                    meta_parts.append(f"Relevance: {score:.2f}")

                                if meta_parts:
                                    st.caption(" · ".join(meta_parts))
                                    
                                # Render as raw markdown (no blockquote) to ensure tables render correctly.
                                # Wrap in a div to allow horizontal scrolling if tables are wide.
                                st.markdown(f'<div style="overflow-x: auto;">\n{text.strip()}\n</div>', unsafe_allow_html=True)

def _stream_query_worker(
    prompt: str,
    history: list[str],
    state: dict,
    cancel_event: threading.Event,
    state_lock: threading.Lock,
) -> None:
    """Runs in a background thread so the main script stays free to render a
    Stop button and react to its click - a plain synchronous loop in the main
    thread cannot be interrupted mid-stream in Streamlit's execution model.

    All mutations to `state` go through `state_lock` so the polling fragment
    never reads a dict mid-mutation (clicking Stop mid-stream was racing the
    fragment's read against this thread's writes and hanging the tab)."""
    response = None
    try:
        response = requests.post(
            f"{API_BASE_URL}/query/stream",
            json={"question": prompt, "history": history},
            headers=API_HEADERS,
            stream=True,
            timeout=(5, QUERY_READ_TIMEOUT_SECONDS),  # 5s to connect
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if cancel_event.is_set():
                with state_lock:
                    state["cancelled"] = True
                break

            if not line:
                continue

            line_str = line.decode("utf-8")
            if not line_str.startswith("data: "):
                continue

            try:
                data = json.loads(line_str[6:])
                step = data.get("step")

                with state_lock:
                    if step == "cached":
                        state["label"] = "⚡ Serving from cache..."
                    elif step == "rewriting":
                        state["label"] = "🔍 Rewriting query for context..."
                    elif step == "retrieving":
                        state["label"] = "🛰️ Searching document library..."
                    elif step == "reranking":
                        state["label"] = "🎯 Ranking relevant chunks..."
                    elif step == "generating":
                        state["label"] = "🧠 Synthesizing answer..."
                    elif step == "thinking":
                        content = data.get("content", "")
                        state["thoughts"].append(content)
                        snippet = content.strip().replace("\n", " ")[:80]
                        if snippet:
                            state["label"] = f"⚙️ {snippet}" if "Step " in snippet else f"🧠 {snippet}..."
                    elif step == "token":
                        state["answer"] += data.get("content", "")
                        state["label"] = "✅ Reasoning Complete"
                        state["status_state"] = "complete"
                    elif step == "done":
                        state["answer"] = data.get("answer", state["answer"])
                        state["sources"] = data.get("sources", [])
                        backend_thoughts = data.get("thoughts")
                        state["thoughts"] = backend_thoughts if backend_thoughts else "".join(state["thoughts"])
                        state["label"] = "✅ Reasoning Complete"
                        state["status_state"] = "complete"
                    elif step == "error":
                        state["error"] = data.get("message", "Unknown backend error")
                        state["status_state"] = "error"
            except json.JSONDecodeError:
                continue

        if cancel_event.is_set() and response is not None:
            response.close()
    except requests.exceptions.RequestException as e:
        with state_lock:
            state["error"] = format_error_message(e)
    finally:
        with state_lock:
            state["done"] = True


@st.fragment(run_every=0.25)
def _render_streaming_answer() -> None:
    state = st.session_state.get("stream_state")
    if state is None:
        return

    state_lock = st.session_state.stream_state_lock
    with state_lock:
        state = dict(state)  # snapshot - render off a stable copy, not the live dict

    with st.chat_message("assistant"):
        status_state = "error" if state["error"] else state["status_state"]
        st.status(state["label"], state=status_state, expanded=(status_state == "running"))
        if not state["done"]:
            stopping = st.session_state.get("stream_stop_requested", False)
            if st.button("⏹️ Stop generating", key="stop_generation_btn", disabled=stopping):
                st.session_state.stream_stop_requested = True
                st.session_state.stream_cancel_event.set()
        if state["error"]:
            st.error(f"❌ **Pipeline Error:** {state['error']}")
        elif state["answer"]:
            st.markdown(state["answer"] + ("" if state["done"] else " ▌"))
        if state["done"] and state["cancelled"] and state["answer"]:
            st.caption("⏹️ Generation stopped early by user.")

    if state["done"]:
        if not state["error"] and state["answer"]:
            st.session_state.messages.append({
                "role": "assistant",
                "content": state["answer"],
                "sources": state["sources"],
                "thoughts": state["thoughts"],
            })
        st.session_state.stream_state = None
        st.session_state.pop("stream_cancel_event", None)
        st.session_state.pop("stream_state_lock", None)
        st.session_state.pop("stream_stop_requested", None)
        st.rerun()


# Input
raw_prompt = st.chat_input("Ask something about your documents...")
prompt = raw_prompt.strip() if raw_prompt else ""
if prompt:
    last_user_message = next(
        (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"),
        None,
    )
    if prompt == last_user_message:
        st.toast("You just asked that - scroll up to see the answer.")
    else:
        indexed_files, backend_reachable = get_indexed_files()
        if not indexed_files:
            with st.chat_message("assistant"):
                if backend_reachable:
                    st.warning("⚠️ No documents indexed yet. Please upload a PDF in the sidebar first.")
                else:
                    st.error(f"⚠️ Cannot reach the backend at {API_BASE_URL}. Please check the connection.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            formatted_history = [
                f"{m['role']}: {m['content']}"
                for m in st.session_state.messages[:-1]
            ]

            cancel_event = threading.Event()
            state_lock = threading.Lock()
            stream_state = {
                "answer": "",
                "sources": [],
                "thoughts": [],
                "label": "🧠 Thinking...",
                "status_state": "running",
                "error": None,
                "cancelled": False,
                "done": False,
            }
            st.session_state.stream_state = stream_state
            st.session_state.stream_cancel_event = cancel_event
            st.session_state.stream_state_lock = state_lock
            st.session_state.stream_stop_requested = False
            threading.Thread(
                target=_stream_query_worker,
                args=(prompt, formatted_history, stream_state, cancel_event, state_lock),
                daemon=True,
            ).start()

if st.session_state.get("stream_state") is not None:
    _render_streaming_answer()

