"""Hybrid-only retriever facade."""
from __future__ import annotations

from typing import Optional

from ...config import get_settings
from ..dataio.schemas import RAGQuery
from ..dataio.storage import DataLocator
from .hybrid_text_retriever import HybridTextRetriever
from .types import RetrievalResult


class Retriever:
    """Entry point used by the API/tests. Always backed by the hybrid retriever."""

    def __init__(self, locator: DataLocator, settings=None) -> None:
        self.locator = locator
        self.settings = settings or get_settings()
        self._retriever = HybridTextRetriever(locator, self.settings)

    def retrieve(self, query: RAGQuery) -> RetrievalResult:
        return self._retriever.retrieve(query)
