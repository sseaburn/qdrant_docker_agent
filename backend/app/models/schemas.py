"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class DocumentBase(BaseModel):
    """Base document schema."""
    filename: str


class DocumentCreate(DocumentBase):
    """Schema for document creation."""
    pass


class Document(DocumentBase):
    """Schema for document response."""
    id: str
    status: str  # "processing", "ready", "error"
    chunk_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentList(BaseModel):
    """Schema for list of documents."""
    documents: List[Document]
    total: int


class ChunkMetadata(BaseModel):
    """Schema for chunk metadata."""
    document_id: str
    chunk_index: int
    filename: str
    text: str


class ChatRequest(BaseModel):
    """Schema for chat request."""
    message: str


class Source(BaseModel):
    """Schema for source citation."""
    filename: str
    chunk_index: int
    text: str
    score: float


class ChatResponse(BaseModel):
    """Schema for chat response."""
    answer: str
    sources: List[Source]


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str


class ErrorResponse(BaseModel):
    """Schema for error response."""
    detail: str
