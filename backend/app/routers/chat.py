"""Chat router for RAG-based question answering."""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException

try:
    from app.services.rag_agent import get_rag_agent
    from app.models.schemas import ChatRequest, ChatResponse, Source
except ImportError:
    from backend.app.services.rag_agent import get_rag_agent
    from backend.app.models.schemas import ChatRequest, ChatResponse, Source

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, document_id: Optional[str] = None):
    """
    Send a message and get a RAG-based response.

    The response is generated using:
    1. Vector search to find relevant document chunks
    2. LLM generation with retrieved context
    3. Source citations from matched documents

    Args:
        request: Chat request with message.
        document_id: Optional filter to search within a specific document.

    Returns:
        ChatResponse with answer and sources.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        rag_agent = get_rag_agent()
        response = await rag_agent.query(
            question=request.message,
            document_id=document_id
        )

        # Convert RAG response to API schema
        sources = [
            Source(
                document_id=s.document_id,
                filename=s.filename,
                chunk_index=s.chunk_index,
                text=s.text,
                score=s.score
            )
            for s in response.sources
        ]

        return ChatResponse(
            answer=response.answer,
            sources=sources
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )
