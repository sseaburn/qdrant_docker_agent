"""Phase 1 Tests: Project Setup Verification

These tests verify that all services are running and can communicate.
Run these tests after starting Docker Compose.
"""

import pytest
import httpx


@pytest.mark.asyncio
async def test_backend_health(api_base_url):
    """Test backend health endpoint is accessible."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_backend_root(api_base_url):
    """Test backend root endpoint returns API info."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_base_url}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "RAG API" in data["message"]


@pytest.mark.asyncio
async def test_backend_docs(api_base_url):
    """Test FastAPI docs are accessible."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_base_url}/docs")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_qdrant_connection(qdrant_base_url):
    """Test Qdrant is reachable."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{qdrant_base_url}/")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_qdrant_collections_endpoint(qdrant_base_url):
    """Test Qdrant collections endpoint is accessible."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{qdrant_base_url}/collections")
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
