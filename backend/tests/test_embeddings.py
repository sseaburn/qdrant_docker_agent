"""Phase 3 Tests: Embeddings Service

Tests for OpenAI embeddings generation.
These tests require OPENAI_API_KEY to be set.
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


class TestEmbeddings:
    """Tests for the embeddings service."""

    @pytest.mark.asyncio
    async def test_generate_single_embedding(self):
        """Test generating a single embedding."""
        from app.services.embeddings import generate_embedding

        text = "This is a test sentence for embedding."
        embedding = await generate_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 1536  # text-embedding-3-small dimensions
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_generate_batch_embeddings(self):
        """Test generating embeddings for multiple texts."""
        from app.services.embeddings import generate_embeddings_batch

        texts = [
            "First text for embedding",
            "Second text for embedding",
            "Third text for embedding"
        ]
        embeddings = await generate_embeddings_batch(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 1536
            assert all(isinstance(x, float) for x in emb)

    @pytest.mark.asyncio
    async def test_generate_batch_empty_list(self):
        """Test that empty list returns empty list."""
        from app.services.embeddings import generate_embeddings_batch

        embeddings = await generate_embeddings_batch([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_embeddings_are_different(self):
        """Test that different texts produce different embeddings."""
        from app.services.embeddings import generate_embeddings_batch

        texts = [
            "The cat sat on the mat",
            "Machine learning is fascinating"
        ]
        embeddings = await generate_embeddings_batch(texts)

        # Embeddings should be different for different texts
        assert embeddings[0] != embeddings[1]

    @pytest.mark.asyncio
    async def test_similar_texts_have_similar_embeddings(self):
        """Test that similar texts have higher cosine similarity."""
        from app.services.embeddings import generate_embeddings_batch
        import math

        texts = [
            "The weather is sunny today",
            "Today the sun is shining brightly",
            "Python is a programming language"
        ]
        embeddings = await generate_embeddings_batch(texts)

        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot_product / (norm_a * norm_b)

        # Similar texts (0 and 1) should have higher similarity than dissimilar (0 and 2)
        sim_01 = cosine_similarity(embeddings[0], embeddings[1])
        sim_02 = cosine_similarity(embeddings[0], embeddings[2])

        assert sim_01 > sim_02

    def test_get_embedding_dimensions(self):
        """Test that embedding dimensions are correct."""
        from app.services.embeddings import get_embedding_dimensions

        assert get_embedding_dimensions() == 1536
