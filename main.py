"""
app/main.py
FastAPI application factory — startup, middleware, routers.
"""
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.services.rag_engine import rag_engine

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("legalease")


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot-time initialisation — runs before the first request."""
    logger.info("🚀 LegalEase starting up …")
    try:
        rag_engine.setup()
        logger.info("✅ RAG engine ready")
    except Exception as exc:
        logger.error("❌ RAG engine failed to initialise: %s", exc)
        raise
    yield
    logger.info("👋 LegalEase shutting down")


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="LegalEase API",
        description=(
            "RAG-powered Indian law assistant. "
            "Three-stage pipeline: HyDE → ChromaDB retrieval → Cohere reranker → GPT-4o-mini answer."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — allow React dev server + production frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount all routes under /api/v1
    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
