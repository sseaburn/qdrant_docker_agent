"""Documents router for PDF upload and management."""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
import uuid
from datetime import datetime

try:
    from app.services.pdf_processor import process_pdf
    from app.services.vector_store import get_vector_store
    from app.models.schemas import Document, DocumentList
except ImportError:
    from backend.app.services.pdf_processor import process_pdf
    from backend.app.services.vector_store import get_vector_store
    from backend.app.models.schemas import Document, DocumentList

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# In-memory storage for document metadata
documents_db: dict[str, Document] = {}
chunks_db: dict[str, List[str]] = {}


async def _store_embeddings(doc_id: str, chunks: List[str], filename: str):
    """Background task to generate and store embeddings."""
    try:
        vector_store = get_vector_store()
        await vector_store.add_documents(
            texts=chunks,
            document_id=doc_id,
            filename=filename
        )
        # Update document status
        if doc_id in documents_db:
            documents_db[doc_id].status = "ready"
        logger.info(f"Stored {len(chunks)} embeddings for document {doc_id}")
    except Exception as e:
        logger.error(f"Failed to store embeddings for document {doc_id}: {e}")
        if doc_id in documents_db:
            documents_db[doc_id].status = "error"


@router.post("/upload", response_model=Document)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process a PDF document.

    - Validates file is a PDF
    - Extracts text from PDF
    - Chunks text for embedding
    - Generates embeddings and stores in vector database
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
        status="processing",  # Will be updated after embeddings are stored
        chunk_count=result["chunk_count"],
        created_at=datetime.utcnow()
    )

    # Store document and chunks
    documents_db[doc_id] = doc
    chunks_db[doc_id] = result["chunks"]

    # Generate and store embeddings
    # For now, do it synchronously to ensure it's ready
    # Can switch to background_tasks.add_task() for async processing
    try:
        vector_store = get_vector_store()
        await vector_store.add_documents(
            texts=result["chunks"],
            document_id=doc_id,
            filename=file.filename
        )
        doc.status = "ready"
        documents_db[doc_id] = doc
    except Exception as e:
        logger.error(f"Failed to store embeddings: {e}")
        doc.status = "error"
        documents_db[doc_id] = doc
        # Don't raise - document is still usable, just not searchable

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
    Delete a document and its associated chunks and embeddings.

    Removes:
    - Document metadata
    - Text chunks
    - Vector embeddings from Qdrant
    """
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove embeddings from vector store
    try:
        vector_store = get_vector_store()
        deleted_count = vector_store.delete_by_document_id(doc_id)
        logger.info(f"Deleted {deleted_count} embeddings for document {doc_id}")
    except Exception as e:
        logger.error(f"Failed to delete embeddings for document {doc_id}: {e}")
        # Continue with deletion even if vector store fails

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
