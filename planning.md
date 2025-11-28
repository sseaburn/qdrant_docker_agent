# RAG Application Development Plan

## Project Overview

Build a full-stack RAG (Retrieval-Augmented Generation) application with the following components:

- **Frontend**: React application with PDF upload and chat interface
- **Backend**: FastAPI Python application with AI agent for RAG
- **Vector Database**: Qdrant for storing and querying embeddings
- **Embedding Model**: OpenAI text-embedding-3-small
- **LLM**: OpenAI GPT-4o (default), with abstraction for Anthropic/Ollama
- **Deployment Target**: Digital Ocean

All services will run in Docker containers orchestrated with Docker Compose.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Compose                          │
├──────────────────┬──────────────────┬───────────────────────┤
│  React Frontend  │  FastAPI Backend │      Qdrant DB        │
│   (Port 3000)    │    (Port 8000)   │     (Port 6333)       │
│                  │                  │                       │
│  - PDF upload    │  - PDF chunking  │  - Vector storage     │
│  - Chat UI       │  - Embeddings    │  - Similarity search  │
│  - File list     │  - RAG agent     │                       │
│                  │  - LLM Provider  │                       │
│                  │    Abstraction   │                       │
└──────────────────┴──────────────────┴───────────────────────┘
```

---

## Project Structure

```
rag-app/
├── planning.md                 # This design document
├── implementation_plan.md      # Execution guide for Claude Code
├── docker-compose.yml
├── docker-compose.dev.yml      # Development overrides
├── .env.example
├── .env                        # gitignored
├── .gitignore
├── pyproject.toml              # uv managed
├── uv.lock
├── README.md
│
├── backend/
│   ├── Dockerfile
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py
│   │   │   └── chat.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── pdf_processor.py
│   │   │   ├── embeddings.py
│   │   │   ├── vector_store.py
│   │   │   ├── rag_agent.py
│   │   │   └── llm/
│   │   │       ├── __init__.py
│   │   │       ├── base.py         # LLMProvider ABC
│   │   │       ├── openai.py       # OpenAIProvider
│   │   │       ├── anthropic.py    # AnthropicProvider
│   │   │       └── ollama.py       # OllamaProvider
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── schemas.py
│   │       └── documents.py
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_pdf_processor.py
│       ├── test_embeddings.py
│       ├── test_vector_store.py
│       ├── test_llm_providers.py
│       └── test_rag_agent.py
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── index.css
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx
│   │   │   ├── FileUpload.jsx
│   │   │   ├── FileList.jsx
│   │   │   └── MessageBubble.jsx
│   │   └── api/
│   │       └── client.js
│   ├── public/
│   └── tests/
│       └── components/
│
├── tests/
│   └── integration/
│       ├── __init__.py
│       ├── conftest.py
│       └── test_e2e.py
│
└── qdrant_data/                # Persistent Qdrant storage
```

---

## LLM Provider Abstraction

The application uses an abstraction layer for LLM providers, allowing easy switching between:
- **OpenAI** (default): GPT-4o, GPT-4o-mini
- **Anthropic**: Claude Sonnet, Claude Haiku
- **Ollama**: llama3.2, mistral, etc. (local models)

### Provider Interface

```python
# backend/app/services/llm/base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, context: str) -> str:
        """Generate a response given prompt and context."""
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, context: str) -> AsyncIterator[str]:
        """Stream a response given prompt and context."""
        pass

def get_llm_provider() -> LLMProvider:
    """Factory function to get configured LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4o")

    if provider == "openai":
        return OpenAIProvider(model=model)
    elif provider == "anthropic":
        return AnthropicProvider(model=model)
    elif provider == "ollama":
        return OllamaProvider(model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
```

---

## Environment Variables

```env
# LLM Configuration
LLM_PROVIDER=openai       # openai | anthropic | ollama
LLM_MODEL=gpt-4o          # Model name for selected provider

# OpenAI (required for embeddings, optional LLM if using other provider)
OPENAI_API_KEY=your-openai-key

# Anthropic (optional - only if using anthropic provider)
ANTHROPIC_API_KEY=your-anthropic-key

# Ollama (optional - only if using ollama provider)
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

---

## Phase 1: Project Setup

### Tasks:
1. Create the folder structure as shown above
2. Initialize Python project with `uv init`
3. Create `docker-compose.yml` with three services:
   - `frontend`: React app on port 3000
   - `backend`: FastAPI app on port 8000
   - `qdrant`: Qdrant vector database on port 6333
4. Create `.env.example` with required environment variables
5. Create `.gitignore` for Python, Node, and environment files
6. Create Dockerfiles for frontend and backend
7. Verify all containers can start and communicate

### Dependencies (pyproject.toml):
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

### Unit Tests (Phase 1):
```python
# backend/tests/test_setup.py
import pytest
import httpx

@pytest.mark.asyncio
async def test_backend_health():
    """Test backend health endpoint is accessible."""
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/health")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_qdrant_connection():
    """Test Qdrant is reachable."""
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:6333/")
        assert response.status_code == 200
```

### Acceptance Criteria:
- [ ] `docker-compose up` starts all three services
- [ ] Frontend accessible at http://localhost:3000
- [ ] Backend accessible at http://localhost:8000/health
- [ ] Backend can connect to Qdrant
- [ ] All Phase 1 tests pass: `uv run pytest backend/tests/test_setup.py -v`

### Git Checkpoint:
```bash
git add .
git commit -m "feat(phase-1): project setup with Docker Compose and folder structure"
git push origin main
```

---

## Phase 2: Backend - PDF Processing

### Tasks:
1. Set up FastAPI application structure with routers
2. Create PDF upload endpoint (`POST /api/documents/upload`)
3. Implement PDF text extraction using `pypdf`
4. Implement text chunking with the following strategy:
   - Chunk size: 500-1000 tokens
   - Overlap: 100-200 tokens
   - Preserve paragraph boundaries where possible
5. Store document metadata (filename, upload date, chunk count)
6. Create endpoint to list uploaded documents (`GET /api/documents`)
7. Create endpoint to delete a document (`DELETE /api/documents/{id}`)

### Dependencies:
```toml
dependencies = [
    "fastapi>=0.109",
    "uvicorn[standard]>=0.27",
    "python-multipart>=0.0.9",
    "pypdf>=4.0",
    "tiktoken>=0.6",
    "python-dotenv>=1.0",
]
```

### Unit Tests (Phase 2):
```python
# backend/tests/test_pdf_processor.py
import pytest
from app.services.pdf_processor import extract_text, chunk_text

def test_extract_text_from_pdf():
    """Test PDF text extraction."""
    # Use a test PDF file
    text = extract_text("tests/fixtures/sample.pdf")
    assert len(text) > 0
    assert isinstance(text, str)

def test_chunk_text_size():
    """Test text chunking produces correct size chunks."""
    text = "Lorem ipsum " * 1000
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 600  # Allow some overflow

def test_chunk_text_overlap():
    """Test chunks have proper overlap."""
    text = "word1 word2 word3 word4 word5 " * 200
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    # Verify overlap exists between consecutive chunks
    for i in range(len(chunks) - 1):
        chunk_words = set(chunks[i].split()[-15:])
        next_words = set(chunks[i+1].split()[:15])
        assert len(chunk_words & next_words) > 0

def test_chunk_preserves_paragraphs():
    """Test chunking prefers paragraph boundaries."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    # Should not split mid-paragraph if possible
    for chunk in chunks:
        assert not chunk.startswith(" ")
```

### Acceptance Criteria:
- [ ] Can upload a PDF via API
- [ ] PDF text is extracted correctly
- [ ] Text is chunked into appropriate sizes with overlap
- [ ] Document metadata is stored and retrievable
- [ ] Can list and delete documents
- [ ] All Phase 2 tests pass: `uv run pytest backend/tests/test_pdf_processor.py -v`

### Git Checkpoint:
```bash
git add .
git commit -m "feat(phase-2): PDF upload and text chunking"
git push origin main
```

---

## Phase 3: Backend - Embeddings & Vector Storage

### Tasks:
1. Create embeddings service using OpenAI `text-embedding-3-small`
2. Set up Qdrant client and collection management
3. Create collection with appropriate vector configuration:
   - Vector size: 1536 (for text-embedding-3-small)
   - Distance metric: Cosine
4. After PDF chunking, generate embeddings for each chunk
5. Store embeddings in Qdrant with metadata:
   - Document ID
   - Chunk index
   - Original text
   - Source filename
6. Implement similarity search function

### Dependencies:
```toml
dependencies = [
    # ... previous dependencies
    "openai>=1.12",
    "qdrant-client>=1.7",
]
```

### Unit Tests (Phase 3):
```python
# backend/tests/test_embeddings.py
import pytest
from app.services.embeddings import generate_embedding, generate_embeddings_batch

@pytest.mark.asyncio
async def test_generate_single_embedding():
    """Test single embedding generation."""
    text = "This is a test sentence."
    embedding = await generate_embedding(text)
    assert len(embedding) == 1536
    assert all(isinstance(x, float) for x in embedding)

@pytest.mark.asyncio
async def test_generate_batch_embeddings():
    """Test batch embedding generation."""
    texts = ["First text", "Second text", "Third text"]
    embeddings = await generate_embeddings_batch(texts)
    assert len(embeddings) == 3
    for emb in embeddings:
        assert len(emb) == 1536

# backend/tests/test_vector_store.py
import pytest
from app.services.vector_store import VectorStore

@pytest.fixture
async def vector_store():
    store = VectorStore(collection_name="test_collection")
    await store.create_collection()
    yield store
    await store.delete_collection()

@pytest.mark.asyncio
async def test_store_and_retrieve(vector_store):
    """Test storing and retrieving vectors."""
    texts = ["Python is a programming language", "JavaScript runs in browsers"]
    await vector_store.add_documents(texts, document_id="test_doc")

    results = await vector_store.search("programming", top_k=1)
    assert len(results) == 1
    assert "Python" in results[0].text

@pytest.mark.asyncio
async def test_similarity_ranking(vector_store):
    """Test that similar content ranks higher."""
    texts = [
        "Machine learning is a subset of AI",
        "Cooking recipes for pasta",
        "Deep learning neural networks",
    ]
    await vector_store.add_documents(texts, document_id="test_doc")

    results = await vector_store.search("artificial intelligence", top_k=3)
    # ML and DL should rank higher than cooking
    assert "learning" in results[0].text.lower()
```

### Acceptance Criteria:
- [ ] Embeddings are generated for all chunks
- [ ] Embeddings are stored in Qdrant with metadata
- [ ] Can query Qdrant and retrieve relevant chunks
- [ ] Similarity search returns ranked results
- [ ] All Phase 3 tests pass: `uv run pytest backend/tests/test_embeddings.py backend/tests/test_vector_store.py -v`

### Git Checkpoint:
```bash
git add .
git commit -m "feat(phase-3): OpenAI embeddings and Qdrant vector storage"
git push origin main
```

---

## Phase 4: Backend - RAG Agent

### Tasks:
1. Create LLM provider abstraction (base class + implementations)
2. Implement OpenAI provider (default)
3. Create RAG agent service
4. Implement query endpoint (`POST /api/chat`)
5. RAG pipeline:
   a. Take user query
   b. Generate embedding for query using OpenAI
   c. Search Qdrant for top-k relevant chunks (k=5)
   d. Construct prompt with retrieved context
   e. Send to LLM for response generation
   f. Return response with source citations
6. Add streaming response support (optional)

### Dependencies:
```toml
dependencies = [
    # ... previous dependencies
    "anthropic>=0.18",  # Optional, for Anthropic provider
    "httpx>=0.27",      # For Ollama HTTP client
]
```

### Prompt Template:
```python
RAG_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer. If the answer is not in the
context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Answer:"""
```

### Unit Tests (Phase 4):
```python
# backend/tests/test_llm_providers.py
import pytest
from app.services.llm.base import get_llm_provider
from app.services.llm.openai import OpenAIProvider

@pytest.mark.asyncio
async def test_openai_provider_generate():
    """Test OpenAI provider can generate responses."""
    provider = OpenAIProvider(model="gpt-4o-mini")
    response = await provider.generate(
        prompt="What is 2+2?",
        context="Mathematics: 2+2=4"
    )
    assert "4" in response

@pytest.mark.asyncio
async def test_provider_factory():
    """Test provider factory returns correct provider."""
    import os
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["LLM_MODEL"] = "gpt-4o"

    provider = get_llm_provider()
    assert isinstance(provider, OpenAIProvider)

# backend/tests/test_rag_agent.py
import pytest
from app.services.rag_agent import RAGAgent

@pytest.fixture
async def rag_agent():
    return RAGAgent()

@pytest.mark.asyncio
async def test_rag_query_with_context(rag_agent):
    """Test RAG agent returns answer from context."""
    # First add some test documents
    await rag_agent.add_document("test.pdf", "The capital of France is Paris.")

    response = await rag_agent.query("What is the capital of France?")
    assert "Paris" in response.answer
    assert len(response.sources) > 0

@pytest.mark.asyncio
async def test_rag_query_no_context(rag_agent):
    """Test RAG agent handles missing context gracefully."""
    response = await rag_agent.query("What is quantum computing?")
    assert "don't have enough information" in response.answer.lower()
```

### Acceptance Criteria:
- [ ] LLM provider abstraction implemented
- [ ] OpenAI provider working
- [ ] Can send a query and receive a response
- [ ] Response is grounded in uploaded documents
- [ ] Sources are cited in the response
- [ ] All Phase 4 tests pass: `uv run pytest backend/tests/test_llm_providers.py backend/tests/test_rag_agent.py -v`

### Git Checkpoint:
```bash
git add .
git commit -m "feat(phase-4): RAG agent with LLM provider abstraction"
git push origin main
```

---

## Phase 5: Frontend - React Application

### Tasks:
1. Set up React application with Vite
2. Create main layout with two panels:
   - Left: Document management (upload, list)
   - Right: Chat interface
3. Implement FileUpload component:
   - Drag-and-drop support
   - File type validation (PDF only)
   - Upload progress indicator
4. Implement FileList component:
   - Show uploaded documents
   - Delete functionality
   - Processing status
5. Implement ChatInterface component:
   - Message input
   - Message history display
   - Loading states
   - Source citations display
6. Create API client for backend communication
7. Style with Tailwind CSS

### Dependencies (package.json):
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0",
    "lucide-react": "^0.330.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.0",
    "autoprefixer": "^10.4.17",
    "postcss": "^8.4.35",
    "tailwindcss": "^3.4.1",
    "vite": "^5.1.0",
    "@testing-library/react": "^14.2.0",
    "@testing-library/jest-dom": "^6.4.0",
    "vitest": "^1.3.0"
  }
}
```

### Unit Tests (Phase 5):
```javascript
// frontend/tests/components/FileUpload.test.jsx
import { render, screen, fireEvent } from '@testing-library/react';
import FileUpload from '../../src/components/FileUpload';

test('renders upload area', () => {
  render(<FileUpload onUpload={() => {}} />);
  expect(screen.getByText(/drag and drop/i)).toBeInTheDocument();
});

test('validates PDF files only', () => {
  const onUpload = vi.fn();
  render(<FileUpload onUpload={onUpload} />);

  const file = new File([''], 'test.txt', { type: 'text/plain' });
  // Attempt to upload non-PDF should show error
});

// frontend/tests/components/ChatInterface.test.jsx
import { render, screen, fireEvent } from '@testing-library/react';
import ChatInterface from '../../src/components/ChatInterface';

test('renders message input', () => {
  render(<ChatInterface />);
  expect(screen.getByPlaceholderText(/ask a question/i)).toBeInTheDocument();
});

test('sends message on submit', async () => {
  render(<ChatInterface />);
  const input = screen.getByPlaceholderText(/ask a question/i);
  const button = screen.getByRole('button', { name: /send/i });

  fireEvent.change(input, { target: { value: 'Test question' } });
  fireEvent.click(button);

  // Should show loading state
  expect(screen.getByText(/thinking/i)).toBeInTheDocument();
});
```

### Acceptance Criteria:
- [ ] Can upload PDFs through the UI
- [ ] Can see list of uploaded documents
- [ ] Can delete documents
- [ ] Can send chat messages and see responses
- [ ] Source citations are displayed
- [ ] UI is responsive and user-friendly
- [ ] All Phase 5 tests pass: `npm test`

### Git Checkpoint:
```bash
git add .
git commit -m "feat(phase-5): React frontend with upload and chat UI"
git push origin main
```

---

## Phase 6: Integration & Testing

### Tasks:
1. End-to-end testing of full workflow:
   - Upload PDF → Process → Embed → Query → Response
2. Error handling for all failure modes:
   - Invalid file types
   - API failures
   - Empty responses
3. Add loading states and user feedback
4. Optimize chunking and retrieval parameters
5. Add health check endpoints

### Integration Tests:
```python
# tests/integration/test_e2e.py
import pytest
import httpx

@pytest.mark.asyncio
async def test_full_rag_workflow():
    """Test complete RAG workflow end-to-end."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # 1. Upload a PDF
        with open("tests/fixtures/sample.pdf", "rb") as f:
            response = await client.post(
                "/api/documents/upload",
                files={"file": ("sample.pdf", f, "application/pdf")}
            )
        assert response.status_code == 200
        doc_id = response.json()["id"]

        # 2. Wait for processing (poll status)
        for _ in range(30):
            status = await client.get(f"/api/documents/{doc_id}")
            if status.json()["status"] == "ready":
                break
            await asyncio.sleep(1)

        # 3. Query the document
        response = await client.post(
            "/api/chat",
            json={"message": "What is this document about?"}
        )
        assert response.status_code == 200
        assert "answer" in response.json()
        assert len(response.json()["sources"]) > 0

        # 4. Cleanup
        await client.delete(f"/api/documents/{doc_id}")

@pytest.mark.asyncio
async def test_error_handling_invalid_file():
    """Test proper error handling for invalid files."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", b"not a pdf", "text/plain")}
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

@pytest.mark.asyncio
async def test_chat_without_documents():
    """Test chat responds appropriately with no documents."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post(
            "/api/chat",
            json={"message": "What is machine learning?"}
        )
        assert response.status_code == 200
        # Should indicate no relevant documents found
```

### Acceptance Criteria:
- [ ] Full workflow works end-to-end
- [ ] Errors are handled gracefully
- [ ] User receives appropriate feedback
- [ ] All integration tests pass: `uv run pytest tests/integration/ -v`

### Git Checkpoint:
```bash
git add .
git commit -m "feat(phase-6): integration testing and error handling"
git push origin main
```

---

## Phase 7: Deployment to Digital Ocean

### Tasks:
1. Create Digital Ocean account and project
2. Choose deployment method:
   - Option A: App Platform (easier, managed)
   - Option B: Droplet with Docker (more control)
3. Set up container registry (Digital Ocean or Docker Hub)
4. Configure production environment variables
5. Set up persistent storage for Qdrant data
6. Configure networking and domains
7. Set up HTTPS with SSL certificates
8. Create deployment documentation

### Production Considerations:
- Use production-grade ASGI server (gunicorn + uvicorn workers)
- Enable CORS properly for frontend domain
- Set up rate limiting
- Configure logging and monitoring
- Set up backup strategy for Qdrant data

### Deployment Tests:
```python
# tests/integration/test_deployment.py
import pytest
import httpx

PRODUCTION_URL = "https://your-app.ondigitalocean.app"

@pytest.mark.asyncio
async def test_production_health():
    """Test production health endpoints."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{PRODUCTION_URL}/health")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_production_https():
    """Test HTTPS is properly configured."""
    async with httpx.AsyncClient() as client:
        response = await client.get(PRODUCTION_URL, follow_redirects=False)
        # Should not redirect (already HTTPS) or redirect to HTTPS
        assert response.status_code in [200, 301, 308]
```

### Acceptance Criteria:
- [ ] Application is accessible on the internet
- [ ] HTTPS is enabled
- [ ] Data persists across deployments
- [ ] Health endpoints respond correctly
- [ ] Application performs well under load

### Git Checkpoint:
```bash
git add .
git commit -m "feat(phase-7): Digital Ocean deployment configuration"
git push origin main
```

---

## API Endpoints Summary

### Documents
- `POST /api/documents/upload` - Upload and process a PDF
- `GET /api/documents` - List all documents
- `GET /api/documents/{id}` - Get document details
- `DELETE /api/documents/{id}` - Delete a document and its embeddings

### Chat
- `POST /api/chat` - Send a message and get RAG response
- `GET /health` - Health check

---

## Development Workflow

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Node.js 18+
- uv (Python package manager)

### Commands Reference

```bash
# Initialize Python project
uv init
uv add fastapi uvicorn python-multipart pypdf tiktoken python-dotenv openai qdrant-client
uv add --dev pytest pytest-asyncio httpx

# Start all services
docker-compose up --build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f backend

# Run backend tests
uv run pytest backend/tests/ -v

# Run integration tests
uv run pytest tests/integration/ -v

# Run frontend tests
cd frontend && npm test

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Testing Commands

```bash
# Run all backend tests
uv run pytest backend/tests/ -v

# Run specific phase tests
uv run pytest backend/tests/test_pdf_processor.py -v  # Phase 2
uv run pytest backend/tests/test_embeddings.py -v      # Phase 3
uv run pytest backend/tests/test_vector_store.py -v    # Phase 3
uv run pytest backend/tests/test_llm_providers.py -v   # Phase 4
uv run pytest backend/tests/test_rag_agent.py -v       # Phase 4

# Run integration tests
uv run pytest tests/integration/ -v

# Run frontend tests
cd frontend && npm test
```

---

## Notes for Implementation

1. **Complete each phase fully before moving to the next**
2. **All tests must pass before committing**
3. Use `uv` for all Python package management
4. Keep the LLM provider abstraction clean for easy switching
5. Use async/await for all I/O operations in FastAPI
6. Add docstrings to all functions
7. Include error handling from the start
8. Keep the frontend simple and functional first, then enhance
