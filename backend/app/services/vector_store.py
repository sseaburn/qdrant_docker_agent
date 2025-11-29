"""Vector store service using Qdrant for storing and searching embeddings."""

from typing import List, Optional
from dataclasses import dataclass
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

try:
    from app.config import settings
    from app.services.embeddings import (
        generate_embedding,
        generate_embeddings_batch,
        get_embedding_dimensions,
    )
except ImportError:
    from backend.app.config import settings
    from backend.app.services.embeddings import (
        generate_embedding,
        generate_embeddings_batch,
        get_embedding_dimensions,
    )


@dataclass
class SearchResult:
    """Represents a search result from the vector store."""
    id: str
    score: float
    text: str
    document_id: str
    chunk_index: int
    filename: str


class VectorStore:
    """
    Vector store for managing document embeddings in Qdrant.

    Handles collection management, document storage, and similarity search.
    """

    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the Qdrant collection. Defaults to settings value.
        """
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the collection exists, create if it doesn't."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.create_collection()

    def create_collection(self):
        """
        Create the Qdrant collection with appropriate vector configuration.

        Uses cosine distance and 1536 dimensions for OpenAI embeddings.
        """
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=get_embedding_dimensions(),
                distance=Distance.COSINE
            )
        )

    def delete_collection(self):
        """Delete the collection and all its data."""
        self.client.delete_collection(collection_name=self.collection_name)

    def recreate_collection(self):
        """Delete and recreate the collection (useful for testing)."""
        try:
            self.delete_collection()
        except Exception:
            pass  # Collection might not exist
        self.create_collection()

    async def add_documents(
        self,
        texts: List[str],
        document_id: str,
        filename: str = ""
    ) -> List[str]:
        """
        Add document chunks to the vector store.

        Generates embeddings and stores them with metadata.

        Args:
            texts: List of text chunks to add.
            document_id: ID of the parent document.
            filename: Original filename of the document.

        Returns:
            List of point IDs for the added chunks.
        """
        if not texts:
            return []

        # Generate embeddings for all chunks
        embeddings = await generate_embeddings_batch(texts)

        # Create points with metadata
        point_ids = []
        points = []

        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": text,
                        "document_id": document_id,
                        "chunk_index": i,
                        "filename": filename
                    }
                )
            )

        # Upsert points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return point_ids

    async def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents using a text query.

        Args:
            query: The search query text.
            top_k: Number of results to return.
            document_id: Optional filter to search within a specific document.

        Returns:
            List of SearchResult objects with matches.
        """
        # Generate embedding for the query
        query_embedding = await generate_embedding(query)

        # Build filter if document_id specified
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )

        # Search Qdrant using query_points
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k
        )

        # Convert to SearchResult objects
        search_results = []
        for point in response.points:
            search_results.append(
                SearchResult(
                    id=str(point.id),
                    score=point.score,
                    text=point.payload.get("text", ""),
                    document_id=point.payload.get("document_id", ""),
                    chunk_index=point.payload.get("chunk_index", 0),
                    filename=point.payload.get("filename", "")
                )
            )

        return search_results

    def delete_by_document_id(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.

        Args:
            document_id: ID of the document whose chunks should be deleted.

        Returns:
            Number of points deleted.
        """
        # Get count before deletion
        count_before = self.client.count(
            collection_name=self.collection_name,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        ).count

        # Delete points matching the document_id
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )

        return count_before

    def get_collection_info(self) -> dict:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection statistics.
        """
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.indexed_vectors_count or 0,
            "points_count": info.points_count or 0,
            "status": info.status.value
        }


# Global instance for convenience
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Get the global vector store instance.

    Returns:
        VectorStore instance.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
