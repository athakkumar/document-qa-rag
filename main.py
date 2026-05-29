"""
FastAPI Backend — Document Q&A System (RAG)
Endpoints:
    POST /ingest  — upload and embed a PDF
    POST /ask     — ask a question about the document
    GET  /health  — health check
"""

import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import ingest_pdf, answer_question

# App setup
app = FastAPI(
    title="Document Q&A — RAG API",
    description=(
        "A Retrieval-Augmented Generation (RAG) API that answers questions about any uploaded PDF. "
        "Upload a document, then ask questions; answers are grounded in the document content. "
        "Built with LangChain (LCEL), Gemini 2.5 Flash, ChromaDB, and FastAPI."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / response schemas 
class AskRequest(BaseModel):
    question: str
    gemini_api_key: str
    top_k: int = 4              # number of chunks to retrieve

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the main topic of this document?",
                "gemini_api_key": "gemini-api-key",
                "top_k": 4,
            }
        }


class IngestResponse(BaseModel):
    status: str
    pages_processed: int
    chunks_created: int
    message: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources_used: list[str]
    chunks_retrieved: int


# Routes 

@app.get("/health", tags=["System"])
def health_check():
    """Check if the API is running."""
    return {
        "status": "healthy",
        "service": "Document Q&A RAG API",
        "llm": "Google Gemini 2.5 Flash",
        "vector_store": "ChromaDB (local)",
        "embeddings": "all-MiniLM-L6-v2 (HuggingFace)",
    }


@app.post("/ingest", response_model=IngestResponse, tags=["RAG Pipeline"])
async def ingest(file: UploadFile = File(...)):
    """
    Upload any PDF and ingest it into the vector store.
    
    - Extracts text using PyMuPDF
    - Splits into overlapping chunks (800 chars, 100 overlap)
    - Embeds using HuggingFace all-MiniLM-L6-v2
    - Stores in local ChromaDB
    
    **Usage:** Upload your PDF here first, then use POST /ask to query it.
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = ingest_pdf(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        os.unlink(tmp_path)     # clean up temp file

    return IngestResponse(
        status=result["status"],
        pages_processed=result["pages_processed"],
        chunks_created=result["chunks_created"],
        message=(
            f"Successfully ingested {result['pages_processed']} pages into "
            f"{result['chunks_created']} chunks. You can now use POST /ask."
        ),
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG Pipeline"])
def ask(request: AskRequest):
    """
    Ask a question about the ingested document.
    
    - Embeds the question and retrieves top-k similar chunks from ChromaDB
    - Passes retrieved context + question to Gemini 2.5 Flash
    - Returns a grounded answer with source snippets
    
    **Note:** Run POST /ingest first to load your PDF into the vector store.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not request.gemini_api_key.strip():
        raise HTTPException(status_code=400, detail="Gemini API key is required.")

    try:
        result = answer_question(
            question=request.question,
            api_key=request.gemini_api_key,
            top_k=request.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")

    return AskResponse(**result)
