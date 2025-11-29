"""Embeddings service using OpenAI text-embedding-3-small."""

from typing import List
from openai import AsyncOpenAI

try:
    from app.config import settings
except ImportError:
    from backend.app.config import settings

# Initialize OpenAI client
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


async def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text string.

    Args:
        text: The text to generate an embedding for.

    Returns:
        List of floats representing the embedding vector (1536 dimensions).
    """
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in a single API call.

    More efficient than calling generate_embedding multiple times.

    Args:
        texts: List of texts to generate embeddings for.

    Returns:
        List of embedding vectors, one for each input text.
    """
    if not texts:
        return []

    # OpenAI supports batching up to 2048 texts
    # For larger batches, we'd need to chunk them
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    # Sort by index to ensure correct ordering
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


def get_embedding_dimensions() -> int:
    """
    Get the dimensions of the embedding model.

    Returns:
        Number of dimensions (1536 for text-embedding-3-small).
    """
    return EMBEDDING_DIMENSIONS
