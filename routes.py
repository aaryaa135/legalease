"""
app/api/routes.py
All FastAPI route handlers.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.models.schemas import (
    DocumentsResponse,
    DocumentEntry,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.rag_engine import rag_engine

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_MIME = {"application/pdf"}
MAX_FILE_MB = 50


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    try:
        stats = rag_engine.collection_stats()
        return HealthResponse(
            status="ok",
            chroma_ok=True,
            total_chunks=stats["total_chunks"],
        )
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return HealthResponse(status="degraded", chroma_ok=False, total_chunks=0)


# ── Ingest ─────────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED, tags=["documents"])
async def ingest_pdf(
    file: UploadFile = File(...),
    doc_title: str = Form(..., min_length=2, max_length=200),
    category: str = Form("Law"),
):
    """
    Upload a PDF legal document and index it into ChromaDB.
    Provide a human-readable doc_title (e.g. "Indian Penal Code 1860")
    and a category (e.g. "Criminal Law", "Civil Rights", "Constitutional").
    """
    # Validate file type
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only PDF files are accepted. Got: {file.content_type}",
        )

    # Read and size-check
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_FILE_MB} MB.",
        )

    # Write to temp file for pdfplumber
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        chunks_added = rag_engine.ingest_pdf(
            pdf_path=tmp_path,
            doc_title=doc_title,
            category=category,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingest failed for %s", file.filename)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        status="success",
        filename=file.filename or doc_title,
        chunks_added=chunks_added,
        collection=rag_engine._chroma_collection.name,
        message=f"Successfully indexed '{doc_title}' — {chunks_added} chunks added.",
    )


# ── Query ──────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["query"])
async def query(req: QueryRequest):
    """
    Ask a legal question. Returns a cited answer + ranked source chunks.

    - HyDE generates a hypothetical expert answer first (improves retrieval).
    - Cohere reranker filters top_k_retrieval → top_k_rerank most relevant chunks.
    - Answer always cites sections like [1], [2].
    """
    try:
        answer, sources, hyde_doc, stats = rag_engine.query(
            question=req.question,
            top_k=req.top_k,
            hyde_override=req.hyde_enabled,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.exception("Query failed: %s", req.question)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return QueryResponse(
        answer=answer,
        sources=sources,
        hypothetical_doc=hyde_doc,
        retrieval_stats=stats,
    )


# ── Documents list ─────────────────────────────────────────────────────────────

@router.get("/documents", response_model=DocumentsResponse, tags=["documents"])
async def list_documents():
    """List all indexed documents with chunk counts."""
    try:
        stats = rag_engine.collection_stats()
        docs = [
            DocumentEntry(
                title=d["title"],
                filename=d["filename"],
                chunks=d["chunks"],
                category=d["category"],
            )
            for d in stats["documents"]
        ]
        return DocumentsResponse(documents=docs, total_chunks=stats["total_chunks"])
    except Exception as exc:
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Delete document ────────────────────────────────────────────────────────────

@router.delete("/documents/{doc_title}", tags=["documents"])
async def delete_document(doc_title: str):
    """Remove all chunks for a given document title from the vector store."""
    try:
        collection = rag_engine._chroma_collection
        results = collection.get(where={"doc_title": doc_title}, include=["metadatas"])
        ids = results.get("ids", [])
        if not ids:
            raise HTTPException(status_code=404, detail=f"No document found with title '{doc_title}'")
        collection.delete(ids=ids)
        return JSONResponse({"status": "deleted", "doc_title": doc_title, "chunks_removed": len(ids)})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
