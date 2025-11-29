"""Phase 4 Tests: Chat API

Tests for the chat API endpoint.
These tests require:
- Backend to be running (docker-compose up backend)
- OPENAI_API_KEY to be set
"""

import sys
from pathlib import Path
import os

# Add backend to path for local testing
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import pytest
import httpx

# Skip all tests if OPENAI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


@pytest.fixture
def api_base_url():
    """Get the API base URL."""
    return os.getenv("API_BASE_URL", "http://localhost:8000")


class TestChatAPI:
    """Tests for the chat API endpoint."""

    @pytest.mark.asyncio
    async def test_chat_endpoint_exists(self, api_base_url):
        """Test chat endpoint is accessible."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_base_url}/api/chat",
                json={"message": "Hello"}
            )
            # Should not be 404
            assert response.status_code != 404

    @pytest.mark.asyncio
    async def test_chat_returns_response_format(self, api_base_url):
        """Test chat returns correct response format."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_base_url}/api/chat",
                json={"message": "What documents do you have?"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert isinstance(data["sources"], list)

    @pytest.mark.asyncio
    async def test_chat_empty_message_rejected(self, api_base_url):
        """Test chat rejects empty messages."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_base_url}/api/chat",
                json={"message": ""}
            )

            assert response.status_code == 400
            assert "empty" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_chat_whitespace_message_rejected(self, api_base_url):
        """Test chat rejects whitespace-only messages."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_base_url}/api/chat",
                json={"message": "   "}
            )

            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_chat_missing_message_field(self, api_base_url):
        """Test chat requires message field."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_base_url}/api/chat",
                json={}
            )

            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_chat_with_document_filter(self, api_base_url):
        """Test chat with document_id filter parameter."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_base_url}/api/chat",
                json={"message": "What is this about?"},
                params={"document_id": "nonexistent_doc"}
            )

            # Should return 200 with empty sources
            assert response.status_code == 200
            data = response.json()
            assert len(data["sources"]) == 0
