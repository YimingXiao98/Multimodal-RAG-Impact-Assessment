"""FastAPI entrypoint for Harvey RAG."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .core.dataio.schemas import RAGAnswer, RAGQuery
from .core.dataio.storage import DataLocator
from .core.models.vlm_client import VLMClient
from .core.retrieval.context_packager import package_context
from .core.retrieval.retriever import Retriever
from .core.retrieval.selector import select_candidates
from .routers import health
from .routers.query import router as query_router

app = FastAPI(title="Harvey RAG API")
app.include_router(health.router)
app.include_router(query_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    settings = get_settings()
    app.state.locator = DataLocator(Path(settings.data_dir))
    app.state.retriever = Retriever(app.state.locator)
    app.state.vlm_client = VLMClient(settings.model_provider)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Harvey RAG API"}
