"""Documents router for PDF upload and management."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import uuid
from datetime import datetime

from backend.app.services.pdf_processor import process_pdf
from backend.app.models.schemas import Document, DocumentList

router = APIRouter(prefix="/api/documents", tags=["documents"])

# In-memory storage (will be replaced with Qdrant in Phase 3)
documents_db: dict[str, Document] = {}
chunks_db: dict[str, List[str]] = {}


@router.post("/upload", response_model=Document)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.

    - Validates file is a PDF
    - Extracts text from PDF
    - Chunks text for embedding
    - Stores document metadata
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed. Please upload a .pdf file."
        )

    # Validate content type
    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Invalid content type. Expected application/pdf."
        )

    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    # Process PDF
    try:
        result = process_pdf(content)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process PDF: {str(e)}"
        )

    if not result["chunks"]:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any text from the PDF"
        )

    # Create document record
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename=file.filename,
        status="ready",
        chunk_count=result["chunk_count"],
        created_at=datetime.utcnow()
    )

    # Store document and chunks
    documents_db[doc_id] = doc
    chunks_db[doc_id] = result["chunks"]

    return doc


@router.get("", response_model=DocumentList)
async def list_documents():
    """
    List all uploaded documents.

    Returns a list of document metadata including filename, status, and chunk count.
    """
    docs = list(documents_db.values())
    return DocumentList(documents=docs, total=len(docs))


@router.get("/{doc_id}", response_model=Document)
async def get_document(doc_id: str):
    """
    Get a specific document by ID.

    Returns document metadata including filename, status, and chunk count.
    """
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")

    return documents_db[doc_id]


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document and its associated chunks.

    This will also remove any embeddings from the vector store (in Phase 3).
    """
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove document and chunks
    del documents_db[doc_id]
    if doc_id in chunks_db:
        del chunks_db[doc_id]

    return {"status": "deleted", "id": doc_id}


def get_chunks(doc_id: str) -> List[str]:
    """
    Get chunks for a document by ID.

    This is used internally by other services.
    """
    if doc_id not in chunks_db:
        return []
    return chunks_db[doc_id]


def get_all_chunks() -> List[dict]:
    """
    Get all chunks from all documents with metadata.

    Returns a list of dicts with document_id, chunk_index, filename, and text.
    """
    all_chunks = []
    for doc_id, chunks in chunks_db.items():
        doc = documents_db.get(doc_id)
        if doc:
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "document_id": doc_id,
                    "chunk_index": idx,
                    "filename": doc.filename,
                    "text": chunk
                })
    return all_chunks
