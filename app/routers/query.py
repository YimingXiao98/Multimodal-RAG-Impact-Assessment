"""Query endpoint."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, Request

from ..config import Settings
from ..core.dataio.schemas import RAGAnswer, RAGQuery
from ..core.retrieval.context_packager import package_context
from ..core.retrieval.selector import select_candidates
from ..deps import SettingsDep

router = APIRouter(prefix="", tags=["query"])


@router.post("/query", response_model=RAGAnswer)
async def query_endpoint(payload: RAGQuery, request: Request, settings: Settings = SettingsDep) -> Dict[str, Any]:
    """Handle RAG inference request."""

    return run_rag_pipeline(payload, request)


def run_rag_pipeline(payload: RAGQuery, request: Request) -> Dict[str, Any]:
    """Execute the retrieval + inference pipeline for a structured RAG query."""

    retriever = request.app.state.retriever
    vlm_client = request.app.state.vlm_client
    result = retriever.retrieve(payload)
    candidates = select_candidates(result, payload.k_tiles, payload.n_text)
    context = package_context(candidates)
    time_window = {"start": str(payload.start), "end": str(payload.end)}
    answer = vlm_client.infer(
        zip_code=payload.zip,
        time_window=time_window,
        imagery_tiles=context.get("imagery_tiles", []),
        text_snippets=context.get("text_snippets", []),
        sensor_table=context.get("sensor_table", ""),
        kb_summary=context.get("kb_summary", ""),
        tweets=context.get("tweets", []),
        calls=context.get("calls", []),
        sensors=context.get("sensors", []),
        fema=context.get("fema", []),
    )
    return answer
