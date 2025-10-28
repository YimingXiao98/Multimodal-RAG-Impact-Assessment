"""Model client adapters for multimodal inference."""
from __future__ import annotations

import json
import os
from typing import Dict, List

from loguru import logger

from .prompts import SYSTEM_PROMPT, build_user_prompt


class VLMClient:
    """Interface wrapping different multimodal providers."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider or os.getenv("MODEL_PROVIDER", "mock")

    def infer(
        self,
        zip_code: str,
        time_window: Dict[str, str],
        imagery_tiles: List[dict],
        text_snippets: List[str],
        sensor_table: str,
        kb_summary: str,
    ) -> Dict[str, object]:
        """Generate RAG answer using configured provider."""

        context = {
            "imagery_tiles": imagery_tiles,
            "text_snippets": text_snippets,
            "sensor_table": sensor_table,
            "kb_summary": kb_summary,
        }
        if self.provider == "mock":
            return self._mock_response(zip_code, time_window, context)
        if self.provider == "openai":
            return self._openai_stub(zip_code, time_window, context)
        if self.provider == "gemini":
            return self._gemini_stub(zip_code, time_window, context)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _mock_response(self, zip_code: str, time_window: Dict[str, str], context: Dict[str, object]) -> Dict[str, object]:
        """Return deterministic pseudo-output for tests."""

        tile_ids = [tile["tile_id"] for tile in context.get("imagery_tiles", [])]
        tweet_ids = []
        call_ids = []
        if context.get("text_snippets"):
            tweet_ids = [f"tw_{i}" for i in range(len(context["text_snippets"]))]
            call_ids = [f"311_{i}" for i in range(len(context["text_snippets"]))]
        sensor_ids = [tile.get("sensor_id", f"sensor_{i}") for i, tile in enumerate(context.get("imagery_tiles", []))]
        kb_refs = [f"fema_zip_{zip_code}_2010_2016"] if context.get("kb_summary") else []
        return {
            "zip": zip_code,
            "time_window": time_window,
            "estimates": {
                "structural_damage_pct": round(min(len(tile_ids) * 0.05, 0.9), 2),
                "roads_impacted": [f"Road_{i}" for i in range(min(2, len(tile_ids)))],
                "confidence": 0.75,
            },
            "evidence_refs": {
                "imagery_tile_ids": tile_ids,
                "tweet_ids": tweet_ids[:2],
                "call_311_ids": call_ids[:2],
                "sensor_ids": sensor_ids[:2],
                "kb_refs": kb_refs,
            },
        }

    def _openai_stub(self, zip_code: str, time_window: Dict[str, str], context: Dict[str, object]) -> Dict[str, object]:
        """Placeholder for OpenAI GPT-4V integration."""

        logger.info("OpenAI provider selected but not implemented; falling back to mock")
        return self._mock_response(zip_code, time_window, context)

    def _gemini_stub(self, zip_code: str, time_window: Dict[str, str], context: Dict[str, object]) -> Dict[str, object]:
        """Placeholder for Google Gemini integration."""

        logger.info("Gemini provider selected but not implemented; falling back to mock")
        return self._mock_response(zip_code, time_window, context)
