"""Phase 3 Tests: Vector Store Service

Tests for Qdrant vector store operations.
These tests require:
- Qdrant to be running (docker-compose up qdrant)
- OPENAI_API_KEY to be set (for embedding generation)
"""

import sys
from pathlib import Path
import os

# Add backend to path for local testing
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import pytest

# Skip all tests if OPENAI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


@pytest.fixture
async def vector_store():
    """Create a test vector store with a unique collection."""
    from app.services.vector_store import VectorStore
    import uuid

    # Use unique collection name for test isolation
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    store = VectorStore(collection_name=collection_name)

    yield store

    # Cleanup
    try:
        store.delete_collection()
    except Exception:
        pass


class TestVectorStore:
    """Tests for the VectorStore class."""

    @pytest.mark.asyncio
    async def test_add_and_search_documents(self, vector_store):
        """Test adding documents and searching."""
        texts = [
            "Python is a versatile programming language",
            "JavaScript runs in web browsers",
            "Machine learning uses neural networks"
        ]

        # Add documents
        point_ids = await vector_store.add_documents(
            texts=texts,
            document_id="test_doc_1",
            filename="test.pdf"
        )

        assert len(point_ids) == 3

        # Search for programming
        results = await vector_store.search("programming languages", top_k=2)

        assert len(results) == 2
        # Python should be most relevant
        assert "Python" in results[0].text or "JavaScript" in results[0].text

    @pytest.mark.asyncio
    async def test_search_returns_metadata(self, vector_store):
        """Test that search results include correct metadata."""
        texts = ["Sample text for testing metadata"]

        await vector_store.add_documents(
            texts=texts,
            document_id="doc_123",
            filename="sample.pdf"
        )

        results = await vector_store.search("testing", top_k=1)

        assert len(results) == 1
        assert results[0].document_id == "doc_123"
        assert results[0].filename == "sample.pdf"
        assert results[0].chunk_index == 0
        assert results[0].text == texts[0]
        assert 0 <= results[0].score <= 1

    @pytest.mark.asyncio
    async def test_search_with_document_filter(self, vector_store):
        """Test searching within a specific document."""
        # Add documents from different sources
        await vector_store.add_documents(
            texts=["Python programming basics"],
            document_id="doc_1",
            filename="python.pdf"
        )

        await vector_store.add_documents(
            texts=["JavaScript programming basics"],
            document_id="doc_2",
            filename="javascript.pdf"
        )

        # Search only in doc_1
        results = await vector_store.search(
            "programming",
            top_k=5,
            document_id="doc_1"
        )

        assert len(results) == 1
        assert results[0].document_id == "doc_1"

    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, vector_store):
        """Test deleting all chunks for a document."""
        # Add documents
        await vector_store.add_documents(
            texts=["Chunk 1", "Chunk 2", "Chunk 3"],
            document_id="doc_to_delete",
            filename="delete_me.pdf"
        )

        await vector_store.add_documents(
            texts=["Keep this chunk"],
            document_id="doc_to_keep",
            filename="keep_me.pdf"
        )

        # Delete one document
        deleted_count = vector_store.delete_by_document_id("doc_to_delete")
        assert deleted_count == 3

        # Verify deletion
        results = await vector_store.search("chunk", top_k=10)

        # Only the kept document should remain
        assert len(results) == 1
        assert results[0].document_id == "doc_to_keep"

    @pytest.mark.asyncio
    async def test_similarity_ranking(self, vector_store):
        """Test that similar content ranks higher."""
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks for AI",
            "Cooking recipes for Italian pasta dishes"
        ]

        await vector_store.add_documents(
            texts=texts,
            document_id="test_doc",
            filename="test.pdf"
        )

        results = await vector_store.search("artificial intelligence", top_k=3)

        # AI-related texts should rank higher than cooking
        assert len(results) == 3
        # First result should be about AI/ML
        assert "intelligence" in results[0].text.lower() or "learning" in results[0].text.lower()
        # Cooking should be last
        assert "cooking" in results[2].text.lower() or "pasta" in results[2].text.lower()

    @pytest.mark.asyncio
    async def test_add_empty_documents(self, vector_store):
        """Test adding empty document list."""
        point_ids = await vector_store.add_documents(
            texts=[],
            document_id="empty_doc",
            filename="empty.pdf"
        )

        assert point_ids == []

    def test_get_collection_info(self, vector_store):
        """Test getting collection information."""
        info = vector_store.get_collection_info()

        assert "name" in info
        assert "vectors_count" in info
        assert "points_count" in info
        assert "status" in info

    @pytest.mark.asyncio
    async def test_recreate_collection(self, vector_store):
        """Test recreating a collection clears all data."""
        # Add some data
        await vector_store.add_documents(
            texts=["Some text"],
            document_id="doc_1",
            filename="test.pdf"
        )

        # Recreate collection
        vector_store.recreate_collection()

        # Collection should be empty
        info = vector_store.get_collection_info()
        assert info["points_count"] == 0
