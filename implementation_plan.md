# RAG Application Implementation Plan

This document provides step-by-step instructions for implementing the RAG application. Follow each phase in order, ensuring all tests pass before proceeding to the next phase.

---

## Pre-Implementation Checklist

Before starting, ensure you have:
- [ ] Docker and Docker Compose installed
- [ ] Python 3.11+ installed
- [ ] Node.js 18+ installed
- [ ] `uv` package manager installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [ ] OpenAI API key ready
- [ ] GitHub repository created

---

## Phase 1: Project Setup

### Step 1.1: Create Directory Structure

```bash
mkdir -p backend/app/{routers,services/llm,models}
mkdir -p backend/tests
mkdir -p frontend/src/{components,api}
mkdir -p frontend/public
mkdir -p frontend/tests/components
mkdir -p tests/integration
mkdir -p qdrant_data
touch backend/app/__init__.py
touch backend/app/routers/__init__.py
touch backend/app/services/__init__.py
touch backend/app/services/llm/__init__.py
touch backend/app/models/__init__.py
touch backend/tests/__init__.py
touch tests/__init__.py
touch tests/integration/__init__.py
```

### Step 1.2: Create .gitignore

Create `.gitignore` with Python, Node, Docker, and environment exclusions.

### Step 1.3: Create .env.example

```env
# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o

# OpenAI (required)
OPENAI_API_KEY=your-openai-key

# Anthropic (optional)
ANTHROPIC_API_KEY=your-anthropic-key

# Ollama (optional)
OLLAMA_HOST=http://ollama:11434

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=documents

# Backend
BACKEND_CORS_ORIGINS=http://localhost:3000

# Frontend
VITE_API_URL=http://localhost:8000
```

### Step 1.4: Initialize Python Project with uv

```bash
uv init --name rag-backend
```

Update `pyproject.toml`:
```toml
[project]
name = "rag-backend"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",
]
```

### Step 1.5: Create docker-compose.yml

```yaml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://localhost:8000
    depends_on:
      - backend
    volumes:
      - ./frontend:/app
      - /app/node_modules

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - LLM_PROVIDER=${LLM_PROVIDER:-openai}
      - LLM_MODEL=${LLM_MODEL:-gpt-4o}
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    depends_on:
      - qdrant
    volumes:
      - ./backend:/app

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
```

### Step 1.6: Create Backend Dockerfile

`backend/Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY uv.lock* .

# Install dependencies
RUN uv sync

# Copy application
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### Step 1.7: Create Basic FastAPI App

`backend/app/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Step 1.8: Create Frontend Dockerfile

`frontend/Dockerfile`:
```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host"]
```

### Step 1.9: Initialize React App with Vite

```bash
cd frontend
npm create vite@latest . -- --template react
npm install axios lucide-react
npm install -D tailwindcss postcss autoprefixer @testing-library/react @testing-library/jest-dom vitest
npx tailwindcss init -p
```

### Step 1.10: Create Phase 1 Tests

`backend/tests/test_setup.py`:
```python
import pytest
import httpx

@pytest.mark.asyncio
async def test_backend_health():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_qdrant_connection():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:6333/")
        assert response.status_code == 200
```

### Step 1.11: Verify and Commit

```bash
# Start services
docker-compose up --build -d

# Wait for services to be ready
sleep 10

# Run tests
uv run pytest backend/tests/test_setup.py -v

# If tests pass, commit
git add .
git commit -m "feat(phase-1): project setup with Docker Compose and folder structure"
git push origin main
```

### Phase 1 Acceptance Criteria
- [ ] `docker-compose up` starts all three services
- [ ] http://localhost:3000 shows React app
- [ ] http://localhost:8000/health returns `{"status": "healthy"}`
- [ ] http://localhost:6333 returns Qdrant info
- [ ] All tests pass

---

## Phase 2: Backend - PDF Processing

### Step 2.1: Add Dependencies

```bash
uv add fastapi uvicorn python-multipart pypdf tiktoken python-dotenv
```

### Step 2.2: Create Config Module

`backend/app/config.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "documents")

settings = Settings()
```

### Step 2.3: Create Schemas

`backend/app/models/schemas.py`:
```python
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class DocumentBase(BaseModel):
    filename: str

class DocumentCreate(DocumentBase):
    pass

class Document(DocumentBase):
    id: str
    status: str  # "processing", "ready", "error"
    chunk_count: int
    created_at: datetime

class ChatRequest(BaseModel):
    message: str

class Source(BaseModel):
    filename: str
    chunk_index: int
    text: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
```

### Step 2.4: Create PDF Processor Service

`backend/app/services/pdf_processor.py`:
```python
from pypdf import PdfReader
import tiktoken
from typing import List
import io

def extract_text(file_content: bytes) -> str:
    """Extract text from PDF file bytes."""
    pdf = PdfReader(io.BytesIO(file_content))
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    """Split text into overlapping chunks."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap

    return chunks
```

### Step 2.5: Create Documents Router

`backend/app/routers/documents.py`:
```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import uuid
from datetime import datetime

from app.services.pdf_processor import extract_text, chunk_text
from app.models.schemas import Document

router = APIRouter(prefix="/api/documents", tags=["documents"])

# In-memory storage (replace with database in production)
documents_db = {}
chunks_db = {}

@router.post("/upload", response_model=Document)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    doc_id = str(uuid.uuid4())
    content = await file.read()

    # Extract and chunk text
    text = extract_text(content)
    chunks = chunk_text(text)

    # Store document and chunks
    doc = Document(
        id=doc_id,
        filename=file.filename,
        status="ready",
        chunk_count=len(chunks),
        created_at=datetime.utcnow()
    )

    documents_db[doc_id] = doc
    chunks_db[doc_id] = chunks

    return doc

@router.get("", response_model=List[Document])
async def list_documents():
    return list(documents_db.values())

@router.get("/{doc_id}", response_model=Document)
async def get_document(doc_id: str):
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    return documents_db[doc_id]

@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    del documents_db[doc_id]
    del chunks_db[doc_id]
    return {"status": "deleted"}
```

### Step 2.6: Update main.py

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import documents

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Step 2.7: Create Test Fixtures

Create `backend/tests/fixtures/` directory and add a sample PDF for testing.

### Step 2.8: Create Phase 2 Tests

`backend/tests/test_pdf_processor.py` - implement tests as specified in planning.md

### Step 2.9: Verify and Commit

```bash
uv run pytest backend/tests/test_pdf_processor.py -v

git add .
git commit -m "feat(phase-2): PDF upload and text chunking"
git push origin main
```

---

## Phase 3: Backend - Embeddings & Vector Storage

### Step 3.1: Add Dependencies

```bash
uv add openai qdrant-client
```

### Step 3.2: Create Embeddings Service

`backend/app/services/embeddings.py`:
```python
from openai import AsyncOpenAI
from typing import List
from app.config import settings

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def generate_embedding(text: str) -> List[float]:
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]
```

### Step 3.3: Create Vector Store Service

`backend/app/services/vector_store.py`:
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List
import uuid

from app.config import settings
from app.services.embeddings import generate_embedding, generate_embeddings_batch

class VectorStore:
    def __init__(self, collection_name: str = None):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.collection_name = collection_name or settings.QDRANT_COLLECTION

    async def create_collection(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

    async def add_documents(
        self,
        texts: List[str],
        document_id: str,
        filename: str = ""
    ):
        embeddings = await generate_embeddings_batch(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": text,
                    "document_id": document_id,
                    "chunk_index": i,
                    "filename": filename
                }
            )
            for i, (text, embedding) in enumerate(zip(texts, embeddings))
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    async def search(self, query: str, top_k: int = 5):
        query_embedding = await generate_embedding(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        return results

    async def delete_by_document_id(self, document_id: str):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector={
                "filter": {
                    "must": [
                        {"key": "document_id", "match": {"value": document_id}}
                    ]
                }
            }
        )
```

### Step 3.4: Create Phase 3 Tests

Implement `backend/tests/test_embeddings.py` and `backend/tests/test_vector_store.py` as specified in planning.md

### Step 3.5: Verify and Commit

```bash
uv run pytest backend/tests/test_embeddings.py backend/tests/test_vector_store.py -v

git add .
git commit -m "feat(phase-3): OpenAI embeddings and Qdrant vector storage"
git push origin main
```

---

## Phase 4: Backend - RAG Agent

### Step 4.1: Add Dependencies

```bash
uv add anthropic httpx
```

### Step 4.2: Create LLM Provider Base Class

`backend/app/services/llm/base.py`:
```python
from abc import ABC, abstractmethod
from typing import AsyncIterator
import os

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, context: str) -> str:
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, context: str) -> AsyncIterator[str]:
        pass

def get_llm_provider():
    from app.services.llm.openai import OpenAIProvider
    from app.services.llm.anthropic import AnthropicProvider
    from app.services.llm.ollama import OllamaProvider

    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4o")

    if provider == "openai":
        return OpenAIProvider(model=model)
    elif provider == "anthropic":
        return AnthropicProvider(model=model)
    elif provider == "ollama":
        return OllamaProvider(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

### Step 4.3: Create OpenAI Provider

`backend/app/services/llm/openai.py`:
```python
from openai import AsyncOpenAI
from typing import AsyncIterator
from app.services.llm.base import LLMProvider
from app.config import settings

RAG_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer. If the answer is not in the
context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Answer:"""

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model

    async def generate(self, prompt: str, context: str) -> str:
        full_prompt = RAG_PROMPT.format(context=context, question=prompt)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}]
        )

        return response.choices[0].message.content

    async def generate_stream(self, prompt: str, context: str) -> AsyncIterator[str]:
        full_prompt = RAG_PROMPT.format(context=context, question=prompt)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

### Step 4.4: Create Anthropic Provider

`backend/app/services/llm/anthropic.py`:
```python
from anthropic import AsyncAnthropic
from typing import AsyncIterator
from app.services.llm.base import LLMProvider
from app.config import settings

class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = model

    async def generate(self, prompt: str, context: str) -> str:
        # Implementation
        pass

    async def generate_stream(self, prompt: str, context: str) -> AsyncIterator[str]:
        # Implementation
        pass
```

### Step 4.5: Create Ollama Provider

`backend/app/services/llm/ollama.py`:
```python
import httpx
from typing import AsyncIterator
from app.services.llm.base import LLMProvider
import os

class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3.2"):
        self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model

    async def generate(self, prompt: str, context: str) -> str:
        # Implementation using httpx to call Ollama API
        pass

    async def generate_stream(self, prompt: str, context: str) -> AsyncIterator[str]:
        # Implementation
        pass
```

### Step 4.6: Create RAG Agent Service

`backend/app/services/rag_agent.py`:
```python
from typing import List
from app.services.llm.base import get_llm_provider
from app.services.vector_store import VectorStore
from app.models.schemas import ChatResponse, Source

class RAGAgent:
    def __init__(self):
        self.llm = get_llm_provider()
        self.vector_store = VectorStore()

    async def query(self, question: str, top_k: int = 5) -> ChatResponse:
        # Search for relevant chunks
        results = await self.vector_store.search(question, top_k=top_k)

        if not results:
            return ChatResponse(
                answer="I don't have enough information to answer that question.",
                sources=[]
            )

        # Build context from results
        context = "\n\n".join([r.payload["text"] for r in results])

        # Generate response
        answer = await self.llm.generate(question, context)

        # Build sources
        sources = [
            Source(
                filename=r.payload.get("filename", ""),
                chunk_index=r.payload.get("chunk_index", 0),
                text=r.payload["text"][:200] + "...",
                score=r.score
            )
            for r in results
        ]

        return ChatResponse(answer=answer, sources=sources)
```

### Step 4.7: Create Chat Router

`backend/app/routers/chat.py`:
```python
from fastapi import APIRouter
from app.services.rag_agent import RAGAgent
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/chat", tags=["chat"])
rag_agent = RAGAgent()

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return await rag_agent.query(request.message)
```

### Step 4.8: Update main.py to Include Chat Router

### Step 4.9: Create Phase 4 Tests

Implement `backend/tests/test_llm_providers.py` and `backend/tests/test_rag_agent.py`

### Step 4.10: Verify and Commit

```bash
uv run pytest backend/tests/test_llm_providers.py backend/tests/test_rag_agent.py -v

git add .
git commit -m "feat(phase-4): RAG agent with LLM provider abstraction"
git push origin main
```

---

## Phase 5: Frontend - React Application

### Step 5.1: Configure Tailwind CSS

`frontend/tailwind.config.js`:
```javascript
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: { extend: {} },
  plugins: [],
}
```

### Step 5.2: Create API Client

`frontend/src/api/client.js`:
```javascript
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_URL,
});

export const uploadDocument = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/api/documents/upload', formData);
};

export const getDocuments = () => api.get('/api/documents');
export const deleteDocument = (id) => api.delete(`/api/documents/${id}`);
export const sendMessage = (message) => api.post('/api/chat', { message });
```

### Step 5.3: Create Components

Implement:
- `frontend/src/components/FileUpload.jsx`
- `frontend/src/components/FileList.jsx`
- `frontend/src/components/ChatInterface.jsx`
- `frontend/src/components/MessageBubble.jsx`

### Step 5.4: Create Main App Layout

`frontend/src/App.jsx`:
```jsx
import FileUpload from './components/FileUpload';
import FileList from './components/FileList';
import ChatInterface from './components/ChatInterface';

function App() {
  return (
    <div className="flex h-screen">
      {/* Left Panel - Documents */}
      <div className="w-1/3 border-r p-4">
        <h2 className="text-xl font-bold mb-4">Documents</h2>
        <FileUpload />
        <FileList />
      </div>

      {/* Right Panel - Chat */}
      <div className="w-2/3 p-4">
        <ChatInterface />
      </div>
    </div>
  );
}

export default App;
```

### Step 5.5: Create Frontend Tests

Implement tests in `frontend/tests/components/`

### Step 5.6: Verify and Commit

```bash
cd frontend
npm test

git add .
git commit -m "feat(phase-5): React frontend with upload and chat UI"
git push origin main
```

---

## Phase 6: Integration & Testing

### Step 6.1: Create Test Fixtures

Add a sample PDF to `tests/fixtures/sample.pdf`

### Step 6.2: Create Integration Tests

Implement `tests/integration/test_e2e.py` as specified in planning.md

### Step 6.3: Add Error Handling

- Ensure all API endpoints return appropriate error messages
- Add loading states in frontend
- Handle edge cases (empty responses, timeouts)

### Step 6.4: Verify and Commit

```bash
# Start all services
docker-compose up -d

# Run integration tests
uv run pytest tests/integration/ -v

git add .
git commit -m "feat(phase-6): integration testing and error handling"
git push origin main
```

---

## Phase 7: Deployment to Digital Ocean

### Step 7.1: Create Production Docker Compose

`docker-compose.prod.yml` with:
- No volume mounts
- Production environment variables
- Health checks
- Restart policies

### Step 7.2: Configure GitHub Actions (Optional)

Create `.github/workflows/deploy.yml` for CI/CD

### Step 7.3: Set Up Digital Ocean

1. Create Digital Ocean App Platform app or Droplet
2. Configure environment variables
3. Set up persistent storage for Qdrant
4. Configure domain and SSL

### Step 7.4: Deploy and Test

```bash
# Run production tests
uv run pytest tests/integration/test_deployment.py -v

git add .
git commit -m "feat(phase-7): Digital Ocean deployment configuration"
git push origin main
```

---

## Testing Commands Reference

```bash
# Phase 1
uv run pytest backend/tests/test_setup.py -v

# Phase 2
uv run pytest backend/tests/test_pdf_processor.py -v

# Phase 3
uv run pytest backend/tests/test_embeddings.py backend/tests/test_vector_store.py -v

# Phase 4
uv run pytest backend/tests/test_llm_providers.py backend/tests/test_rag_agent.py -v

# Phase 5
cd frontend && npm test

# Phase 6
uv run pytest tests/integration/ -v

# All backend tests
uv run pytest backend/tests/ -v
```

---

## Troubleshooting

### Docker Issues
```bash
# Rebuild containers
docker-compose down -v
docker-compose up --build

# Check logs
docker-compose logs -f backend
```

### Qdrant Connection Issues
- Ensure Qdrant container is running
- Check `QDRANT_HOST` is set to `qdrant` (service name) in Docker

### OpenAI API Issues
- Verify `OPENAI_API_KEY` is set in `.env`
- Check API key has sufficient credits

### Frontend Build Issues
```bash
cd frontend
rm -rf node_modules
npm install
npm run dev
```
