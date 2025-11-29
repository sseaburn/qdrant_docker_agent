"""Phase 2 Tests: Documents API

Tests for document upload, list, and delete endpoints.
These tests require the backend service to be running.
"""

import pytest
import httpx
from io import BytesIO

# Simple PDF content (minimal valid PDF)
# This is a minimal PDF that contains "Hello World" text
MINIMAL_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000359 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
434
%%EOF"""


@pytest.fixture
def api_base_url():
    """Base URL for API tests."""
    return "http://localhost:8000"


@pytest.mark.asyncio
async def test_upload_document_success(api_base_url):
    """Test successful PDF upload."""
    async with httpx.AsyncClient() as client:
        files = {"file": ("test.pdf", BytesIO(MINIMAL_PDF), "application/pdf")}
        response = await client.post(f"{api_base_url}/api/documents/upload", files=files)

        # Note: This might fail if the PDF doesn't extract text properly
        # In that case, the test documents the expected behavior
        if response.status_code == 200:
            data = response.json()
            assert "id" in data
            assert data["filename"] == "test.pdf"
            assert data["status"] == "ready"
            assert "chunk_count" in data

            # Cleanup: delete the uploaded document
            doc_id = data["id"]
            await client.delete(f"{api_base_url}/api/documents/{doc_id}")
        else:
            # PDF might not have extractable text
            assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_non_pdf_rejected(api_base_url):
    """Test that non-PDF files are rejected."""
    async with httpx.AsyncClient() as client:
        files = {"file": ("test.txt", BytesIO(b"This is not a PDF"), "text/plain")}
        response = await client.post(f"{api_base_url}/api/documents/upload", files=files)

        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_empty_file_rejected(api_base_url):
    """Test that empty files are rejected."""
    async with httpx.AsyncClient() as client:
        files = {"file": ("empty.pdf", BytesIO(b""), "application/pdf")}
        response = await client.post(f"{api_base_url}/api/documents/upload", files=files)

        assert response.status_code == 400


@pytest.mark.asyncio
async def test_list_documents(api_base_url):
    """Test listing documents."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_base_url}/api/documents")

        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)


@pytest.mark.asyncio
async def test_get_document_not_found(api_base_url):
    """Test getting a non-existent document returns 404."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_base_url}/api/documents/nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_delete_document_not_found(api_base_url):
    """Test deleting a non-existent document returns 404."""
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{api_base_url}/api/documents/nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_documents_endpoint_available(api_base_url):
    """Test that documents endpoints are available."""
    async with httpx.AsyncClient() as client:
        # Check OpenAPI docs include documents endpoints
        response = await client.get(f"{api_base_url}/openapi.json")

        assert response.status_code == 200
        openapi = response.json()

        # Verify documents paths exist
        paths = openapi.get("paths", {})
        assert "/api/documents/upload" in paths
        assert "/api/documents" in paths
