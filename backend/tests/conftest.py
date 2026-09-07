"""Shared pytest fixtures and test-time configuration."""
import os
import sys
import pytest

# Add project root to `sys.path` so `import app.*` works when tests are executed
# from the repository root.
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)


@pytest.fixture
def sample_text():
    """Small sample text used by multiple tests."""
    return "This is a sample text for testing purposes."


@pytest.fixture
def sample_documents():
    """Small list of short documents for tests."""
    return [
        "First document about Python programming.",
        "Second document about machine learning.",
        "Third document about data science."
    ]


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary directory for tests that need file I/O."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    return test_dir


@pytest.fixture(autouse=True)
def _isolate_metrics_file(tmp_path, monkeypatch):
    """Redirect the metrics store to a per-test tmp file.

    Without this, every test that touches app.metrics reads/writes the real
    backend/metrics.json - overwriting production analytics history with
    test data (e.g. a "Test Async" query polluting the live dashboard).
    """
    import app.metrics as metrics_module

    tmp_metrics_file = str(tmp_path / "metrics.json")
    monkeypatch.setattr(metrics_module, "METRICS_FILE", tmp_metrics_file)
    monkeypatch.setattr(metrics_module, "_METRICS_LOCKFILE", f"{tmp_metrics_file}.lock")
    yield


@pytest.fixture(autouse=True)
def _clear_singleton_caches():
    """Clear module-level caches between tests.

    Some app modules cache heavy objects (e.g. embedding model) for runtime
    performance. Unit tests often patch constructors and expect a fresh object
    per test.
    """

    try:
        from app.embeddings import get_embeddings

        get_embeddings.cache_clear()
    except Exception:
        # If imports fail for any reason, don't block unrelated tests.
        pass

    yield

    try:
        from app.embeddings import get_embeddings

        get_embeddings.cache_clear()
    except Exception:
        pass
