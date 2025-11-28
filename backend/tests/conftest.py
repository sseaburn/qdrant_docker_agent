import pytest


@pytest.fixture
def api_base_url():
    """Base URL for API tests."""
    return "http://localhost:8000"


@pytest.fixture
def qdrant_base_url():
    """Base URL for Qdrant tests."""
    return "http://localhost:6333"
