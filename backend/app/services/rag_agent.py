"""RAG Agent service for retrieval-augmented generation."""

from dataclasses import dataclass
from typing import List, Optional

try:
    from app.services.vector_store import get_vector_store, SearchResult
    from app.services.llm import get_llm_provider
except ImportError:
    from backend.app.services.vector_store import get_vector_store, SearchResult
    from backend.app.services.llm import get_llm_provider


@dataclass
class Source:
    """Represents a source document used in the response."""
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


@dataclass
class RAGResponse:
    """Response from the RAG agent."""
    answer: str
    sources: List[Source]
    query: str


class RAGAgent:
    """
    RAG Agent that combines retrieval and generation.

    Handles:
    - Searching the vector store for relevant documents
    - Building context from retrieved chunks
    - Generating responses using the LLM
    - Formatting responses with source citations
    """

    def __init__(
        self,
        top_k: int = 5,
        min_score: float = 0.0,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the RAG agent.

        Args:
            top_k: Number of documents to retrieve for context.
            min_score: Minimum similarity score threshold.
            provider: Override for LLM provider.
            model: Override for LLM model.
        """
        self.top_k = top_k
        self.min_score = min_score
        self.vector_store = get_vector_store()
        self.llm = get_llm_provider(provider=provider, model=model)

    async def query(
        self,
        question: str,
        document_id: Optional[str] = None
    ) -> RAGResponse:
        """
        Process a user query and generate a response.

        Args:
            question: The user's question.
            document_id: Optional filter to search within a specific document.

        Returns:
            RAGResponse with answer and sources.
        """
        # 1. Search for relevant documents
        search_results = await self.vector_store.search(
            query=question,
            top_k=self.top_k,
            document_id=document_id
        )

        # 2. Filter by minimum score
        relevant_results = [
            r for r in search_results
            if r.score >= self.min_score
        ]

        # 3. Build context from retrieved chunks
        context = self._build_context(relevant_results)

        # 4. Generate response
        if not context.strip():
            answer = "I don't have enough information to answer that question. Please upload some documents first."
        else:
            answer = await self.llm.generate(prompt=question, context=context)

        # 5. Build sources list
        sources = [
            Source(
                document_id=r.document_id,
                filename=r.filename,
                chunk_index=r.chunk_index,
                text=r.text[:200] + "..." if len(r.text) > 200 else r.text,
                score=r.score
            )
            for r in relevant_results
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question
        )

    def _build_context(self, results: List[SearchResult]) -> str:
        """
        Build context string from search results.

        Args:
            results: List of search results.

        Returns:
            Formatted context string.
        """
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {result.filename}]\n{result.text}"
            )

        return "\n\n".join(context_parts)


# Global instance for convenience
_rag_agent: Optional[RAGAgent] = None


def get_rag_agent() -> RAGAgent:
    """
    Get the global RAG agent instance.

    Returns:
        RAGAgent instance.
    """
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = RAGAgent()
    return _rag_agent
