"""
app/services/rag_engine.py

The heart of LegalEase.  Three-stage retrieval pipeline:
  1. HyDE  — generate a hypothetical answer, embed it (better semantic match)
  2. Retrieve — pull top_k_retrieval chunks from ChromaDB
  3. Rerank — Cohere cross-encoder reranker cuts list to top_k_rerank

Section-aware metadata is stored at ingest time so every retrieved chunk
carries exact section numbers (e.g. "IPC Section 420") for citations.
"""
from __future__ import annotations

import re
import time
import logging
from pathlib import Path
from typing import Optional

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings as LISettings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank

from app.core.config import get_settings
from app.models.schemas import SourceNode, RetrievalStats

logger = logging.getLogger(__name__)
settings = get_settings()


# ── helpers ───────────────────────────────────────────────────────────────────

_SECTION_PATTERNS = [
    # IPC / BNS style  — "Section 302", "Sec. 420", "S. 144"
    r"\bS(?:ection|ec\.?)\s+(\d+[A-Za-z]?(?:\s*to\s*\d+)?)",
    # Article style    — "Article 21", "Art. 19"
    r"\bArt(?:icle|\.)\s+(\d+[A-Za-z]?)",
    # Rule style       — "Rule 8", "Rule 14(1)"
    r"\bRule\s+(\d+(?:\(\d+\))?)",
    # Clause style     — "Clause (a)", "Clause 3"
    r"\bClause\s+(\d+|[a-z])",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _SECTION_PATTERNS]


def _extract_section(text: str) -> str:
    """Return the first section reference found in chunk text, else 'General'."""
    for pattern in _COMPILED:
        m = pattern.search(text)
        if m:
            return m.group(0).strip()
    return "General"


def _score_to_float(score) -> float:
    """Normalise reranker relevance score to 0-1."""
    try:
        s = float(score)
        # Cohere scores are in [0, 1] already; clamp just in case
        return round(max(0.0, min(1.0, s)), 4)
    except Exception:
        return 0.0


# ── RAG Engine ────────────────────────────────────────────────────────────────

class RAGEngine:
    """Singleton RAG engine — initialised once at app startup."""

    def __init__(self) -> None:
        self._ready = False
        self._index: Optional[VectorStoreIndex] = None
        self._chroma_collection = None

    # ── setup ─────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Configure LlamaIndex global settings and connect to ChromaDB."""
        LISettings.llm = OpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        )
        LISettings.embed_model = OpenAIEmbedding(
            model=settings.embed_model,
            api_key=settings.openai_api_key,
        )
        LISettings.node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=64,
        )

        # ChromaDB — persistent local store
        chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._chroma_collection = chroma_client.get_or_create_collection(
            settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

        self._index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_ctx
        )
        self._ready = True
        logger.info(
            "RAGEngine ready — collection '%s', %d chunks",
            settings.chroma_collection,
            self._chroma_collection.count(),
        )

    # ── ingest ────────────────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: Path, doc_title: str, category: str = "Law") -> int:
        """
        Parse a PDF, split into chunks, embed and store in ChromaDB.
        Returns number of chunks added.
        """
        self._assert_ready()
        import pdfplumber

        raw_pages: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                raw_pages.append(text)

        # Build LlamaIndex Documents — one per page with rich metadata
        documents: list[Document] = []
        for page_num, page_text in enumerate(raw_pages, start=1):
            if not page_text.strip():
                continue
            documents.append(
                Document(
                    text=page_text,
                    metadata={
                        "doc_title": doc_title,
                        "category": category,
                        "page": page_num,
                        "filename": pdf_path.name,
                        "section": _extract_section(page_text),
                    },
                )
            )

        if not documents:
            raise ValueError(f"No extractable text found in {pdf_path.name}")

        # Insert into index (auto-embeds via LISettings.embed_model)
        before = self._chroma_collection.count()
        for doc in documents:
            self._index.insert(doc)
        after = self._chroma_collection.count()

        added = after - before
        logger.info("Ingested '%s': %d chunks added", doc_title, added)
        return added

    # ── query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        hyde_override: Optional[bool] = None,
    ) -> tuple[str, list[SourceNode], Optional[str], RetrievalStats]:
        """
        Full RAG pipeline.
        Returns (answer, source_nodes, hypothetical_doc, stats).
        """
        self._assert_ready()
        t0 = time.time()

        k_retrieve = top_k or settings.top_k_retrieval
        use_hyde = hyde_override if hyde_override is not None else settings.hyde_enabled

        # ── Stage 1: HyDE ─────────────────────────────────────────────────────
        hypothetical_doc: Optional[str] = None
        embed_query = question

        if use_hyde:
            hypothetical_doc = self._generate_hypothetical_doc(question)
            # We embed the hypothetical answer instead of the raw question.
            # This pulls chunks that look like an expert answer, not a question.
            embed_query = hypothetical_doc
            logger.debug("HyDE doc: %s", hypothetical_doc[:120])

        # ── Stage 2: Semantic retrieval ───────────────────────────────────────
        retriever = self._index.as_retriever(similarity_top_k=k_retrieve)
        # Temporarily swap the query text for embedding
        raw_nodes: list[NodeWithScore] = retriever.retrieve(embed_query)
        chunks_retrieved = len(raw_nodes)

        if not raw_nodes:
            stats = RetrievalStats(
                chunks_retrieved=0,
                chunks_after_rerank=0,
                hyde_used=use_hyde,
                latency_ms=int((time.time() - t0) * 1000),
            )
            return (
                "I couldn't find relevant information in the loaded legal documents. "
                "Please ensure the relevant law PDFs have been ingested.",
                [],
                hypothetical_doc,
                stats,
            )

        # ── Stage 3: Cohere Reranker ──────────────────────────────────────────
        reranker = CohereRerank(
            api_key=settings.cohere_api_key,
            top_n=settings.top_k_rerank,
            model="rerank-english-v3.0",
        )
        reranked: list[NodeWithScore] = reranker.postprocess_nodes(
            raw_nodes, query_str=question  # rerank against ORIGINAL question
        )
        chunks_after_rerank = len(reranked)

        # ── Stage 4: Generate answer with citations ───────────────────────────
        context_parts: list[str] = []
        for i, node in enumerate(reranked, 1):
            meta = node.node.metadata
            section = meta.get("section", _extract_section(node.node.get_content()))
            context_parts.append(
                f"[{i}] {meta.get('doc_title', 'Unknown')} — {section}\n"
                f"{node.node.get_content()}"
            )

        context_str = "\n\n---\n\n".join(context_parts)

        SYSTEM_PROMPT = (
            "You are LegalEase, an expert assistant on Indian law. "
            "Answer the user's question using ONLY the provided legal document excerpts. "
            "Always cite sources using [1], [2] notation matching the numbered excerpts. "
            "If the answer spans multiple sections, cite each one. "
            "End with a 'Key Sections' bullet list showing cited section numbers. "
            "Never speculate beyond the provided text. "
            "If information is insufficient, say so clearly."
        )

        llm = LISettings.llm
        full_prompt = (
            f"Legal document excerpts:\n\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Provide a clear, structured answer with citations:"
        )

        response = llm.complete(
            full_prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        answer = str(response)

        # ── Build source nodes for response ───────────────────────────────────
        source_nodes: list[SourceNode] = []
        for node in reranked:
            meta = node.node.metadata
            content = node.node.get_content()
            source_nodes.append(
                SourceNode(
                    text=content[:500] + ("…" if len(content) > 500 else ""),
                    score=_score_to_float(node.score),
                    section=meta.get("section", _extract_section(content)),
                    doc_title=meta.get("doc_title", "Unknown"),
                    page=meta.get("page"),
                )
            )

        stats = RetrievalStats(
            chunks_retrieved=chunks_retrieved,
            chunks_after_rerank=chunks_after_rerank,
            hyde_used=use_hyde,
            latency_ms=int((time.time() - t0) * 1000),
        )

        return answer, source_nodes, hypothetical_doc, stats

    # ── HyDE helper ───────────────────────────────────────────────────────────

    def _generate_hypothetical_doc(self, question: str) -> str:
        """
        HyDE: ask the LLM to write what an expert answer WOULD look like,
        then embed THAT instead of the raw question.
        Dense retrieval works better when query and documents are in the
        same semantic space (answers match answers, not questions).
        """
        hyde_prompt = (
            f"You are an expert in Indian law. "
            f"Write a concise, authoritative paragraph that directly answers this legal question, "
            f"as if it appeared in an Indian law textbook or commentary. "
            f"Include relevant section numbers where possible.\n\n"
            f"Question: {question}\n\nHypothetical answer:"
        )
        response = LISettings.llm.complete(hyde_prompt)
        return str(response).strip()

    # ── collection metadata ───────────────────────────────────────────────────

    def collection_stats(self) -> dict:
        self._assert_ready()
        count = self._chroma_collection.count()
        # Get distinct doc titles from metadata
        try:
            results = self._chroma_collection.get(include=["metadatas"])
            metas = results.get("metadatas") or []
            docs: dict[str, dict] = {}
            for m in metas:
                title = m.get("doc_title", "Unknown")
                if title not in docs:
                    docs[title] = {
                        "title": title,
                        "filename": m.get("filename", ""),
                        "category": m.get("category", "Law"),
                        "chunks": 0,
                    }
                docs[title]["chunks"] += 1
        except Exception:
            docs = {}
        return {"total_chunks": count, "documents": list(docs.values())}

    def is_ready(self) -> bool:
        return self._ready

    def _assert_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("RAGEngine.setup() has not been called.")


# Singleton instance
rag_engine = RAGEngine()
