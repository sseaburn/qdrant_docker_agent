"""Phase 4 Tests: RAG Agent

Tests for the RAG agent service.
These tests require:
- Qdrant to be running (docker-compose up qdrant)
- OPENAI_API_KEY to be set
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
async def rag_agent():
    """Create a test RAG agent with a unique collection."""
    from app.services.rag_agent import RAGAgent
    from app.services.vector_store import VectorStore
    import uuid

    # Use unique collection name for test isolation
    collection_name = f"test_rag_{uuid.uuid4().hex[:8]}"
    vector_store = VectorStore(collection_name=collection_name)

    # Create agent using this vector store
    agent = RAGAgent(top_k=3)
    agent.vector_store = vector_store

    yield agent

    # Cleanup
    try:
        vector_store.delete_collection()
    except Exception:
        pass


class TestRAGAgent:
    """Tests for the RAG agent."""

    @pytest.mark.asyncio
    async def test_query_with_context(self, rag_agent):
        """Test RAG agent returns answer from context."""
        # Add test documents
        await rag_agent.vector_store.add_documents(
            texts=[
                "The capital of France is Paris. Paris is known for the Eiffel Tower.",
                "France is a country in Western Europe.",
                "The Louvre Museum is located in Paris."
            ],
            document_id="france_doc",
            filename="france.pdf"
        )

        # Query the agent
        response = await rag_agent.query("What is the capital of France?")

        assert "Paris" in response.answer
        assert len(response.sources) > 0
        assert response.query == "What is the capital of France?"

    @pytest.mark.asyncio
    async def test_query_returns_sources(self, rag_agent):
        """Test RAG agent returns source citations."""
        await rag_agent.vector_store.add_documents(
            texts=["Python is a programming language created by Guido van Rossum."],
            document_id="python_doc",
            filename="python.pdf"
        )

        response = await rag_agent.query("Who created Python?")

        assert len(response.sources) > 0
        source = response.sources[0]
        assert source.document_id == "python_doc"
        assert source.filename == "python.pdf"
        assert source.chunk_index == 0
        assert 0 <= source.score <= 1

    @pytest.mark.asyncio
    async def test_query_no_documents(self, rag_agent):
        """Test RAG agent handles no documents gracefully."""
        response = await rag_agent.query("What is machine learning?")

        # Should indicate no information available
        assert "don't have enough information" in response.answer.lower() or "upload" in response.answer.lower()
        assert len(response.sources) == 0

    @pytest.mark.asyncio
    async def test_query_filters_by_document(self, rag_agent):
        """Test RAG agent can filter by document ID."""
        # Add documents from different sources
        await rag_agent.vector_store.add_documents(
            texts=["Python is great for data science."],
            document_id="python_doc",
            filename="python.pdf"
        )

        await rag_agent.vector_store.add_documents(
            texts=["JavaScript is great for web development."],
            document_id="js_doc",
            filename="javascript.pdf"
        )

        # Query with filter
        response = await rag_agent.query(
            "What is this language great for?",
            document_id="python_doc"
        )

        # Should only get Python source
        assert all(s.document_id == "python_doc" for s in response.sources)

    @pytest.mark.asyncio
    async def test_query_similarity_ranking(self, rag_agent):
        """Test that similar content ranks higher in sources."""
        await rag_agent.vector_store.add_documents(
            texts=[
                "Machine learning is a subset of artificial intelligence.",
                "Cooking recipes for Italian pasta dishes.",
                "Deep learning uses neural networks for AI tasks."
            ],
            document_id="test_doc",
            filename="test.pdf"
        )

        response = await rag_agent.query("What is artificial intelligence?")

        # AI-related content should rank higher
        assert len(response.sources) > 0
        # Top source should be about AI/ML, not cooking
        assert "intelligence" in response.sources[0].text.lower() or "learning" in response.sources[0].text.lower()

    def test_build_context(self, rag_agent):
        """Test context building from search results."""
        from app.services.vector_store import SearchResult

        results = [
            SearchResult(
                id="1",
                score=0.9,
                text="First chunk of text.",
                document_id="doc1",
                chunk_index=0,
                filename="file1.pdf"
            ),
            SearchResult(
                id="2",
                score=0.8,
                text="Second chunk of text.",
                document_id="doc1",
                chunk_index=1,
                filename="file1.pdf"
            )
        ]

        context = rag_agent._build_context(results)

        assert "[Source 1: file1.pdf]" in context
        assert "[Source 2: file1.pdf]" in context
        assert "First chunk of text." in context
        assert "Second chunk of text." in context

    def test_build_context_empty(self, rag_agent):
        """Test context building with empty results."""
        context = rag_agent._build_context([])
        assert context == ""


class TestRAGAgentFactory:
    """Tests for RAG agent factory function."""

    def test_get_rag_agent(self):
        """Test get_rag_agent returns singleton."""
        from app.services.rag_agent import get_rag_agent

        agent1 = get_rag_agent()
        agent2 = get_rag_agent()

        assert agent1 is agent2
