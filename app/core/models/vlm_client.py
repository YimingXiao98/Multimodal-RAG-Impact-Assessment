"""Model client adapters for multimodal inference."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

from loguru import logger

from .prompts import (
    QUERY_PARSING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_query_parsing_prompt,
    build_user_prompt,
)


class VLMClient:
    """Interface wrapping different multimodal providers."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider or os.getenv("MODEL_PROVIDER", "mock")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv(
            "GEMINI_MODEL", "models/gemini-1.5-flash")
        self.gemini_max_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "3"))
        self._gemini_client = None

    def infer(
        self,
        zip_code: str,
        time_window: Dict[str, str],
        imagery_tiles: List[dict],
        text_snippets: List[str],
        sensor_table: str,
        kb_summary: str,
        tweets: List[dict] = None,
        calls: List[dict] = None,
        sensors: List[dict] = None,
        fema: List[dict] = None,
    ) -> Dict[str, object]:
        """Generate RAG answer using configured provider."""

        context = {
            "imagery_tiles": imagery_tiles,
            "text_snippets": text_snippets,
            "sensor_table": sensor_table,
            "kb_summary": kb_summary,
            "tweets": tweets or [],
            "calls": calls or [],
            "sensors": sensors or [],
            "fema": fema or [],
        }
        if self.provider == "mock":
            return self._mock_response(zip_code, time_window, context)
        if self.provider == "openai":
            return self._openai_stub(zip_code, time_window, context)
        if self.provider == "gemini":
            return self._gemini_response(zip_code, time_window, context)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def parse_query(self, message: str) -> Dict[str, str | None]:
        """Parse natural language query into structured parameters using LLM."""
        if self.provider == "mock":
            # Mock parser just returns None to let regex fallback handle it,
            # or we could implement a simple mock parser here.
            return {"zip": None, "start": None, "end": None}

        if self.provider == "gemini":
            return self._gemini_parse_query(message)

        logger.warning(
            f"Query parsing not implemented for provider {self.provider}")
        return {"zip": None, "start": None, "end": None}

    def _gemini_parse_query(self, message: str) -> Dict[str, str | None]:
        """Use Gemini to parse the query."""
        if not self.gemini_api_key:
            logger.warning(
                "GEMINI_API_KEY not set, skipping LLM query parsing")
            return {"zip": None, "start": None, "end": None}

        try:
            import google.generativeai as genai
        except ImportError:
            return {"zip": None, "start": None, "end": None}

        # We use a separate client or reconfigure?
        # Ideally we reuse the client but with different system prompt.
        # The current client is initialized with a specific system prompt.
        # We can create a new GenerativeModel instance for parsing.

        parser_model = genai.GenerativeModel(
            self.gemini_model,
            system_instruction=QUERY_PARSING_SYSTEM_PROMPT.strip()
        )

        user_prompt = build_query_parsing_prompt(message)
        messages = [{"role": "user", "parts": [{"text": user_prompt}]}]

        try:
            response = parser_model.generate_content(
                messages,
                generation_config={"response_mime_type": "application/json"},
            )
            payload = self._extract_response_text(response)
            parsed = json.loads(payload)
            return parsed
        except Exception as exc:
            logger.error(f"Gemini query parsing failed: {exc}")
            return {"zip": None, "start": None, "end": None}

    def _mock_response(self, zip_code: str, time_window: Dict[str, str], context: Dict[str, object]) -> Dict[str, object]:
        """Return deterministic pseudo-output for tests."""

        tile_ids = [tile["tile_id"]
                    for tile in context.get("imagery_tiles", [])]
        tweet_ids = []
        call_ids = []
        if context.get("text_snippets"):
            tweet_ids = [f"tw_{i}" for i in range(
                len(context["text_snippets"]))]
            call_ids = [f"311_{i}" for i in range(
                len(context["text_snippets"]))]
        sensor_ids = [tile.get("sensor_id", f"sensor_{i}") for i, tile in enumerate(
            context.get("imagery_tiles", []))]
        kb_refs = [f"fema_zip_{zip_code}_2010_2016"] if context.get(
            "kb_summary") else []
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
            "natural_language_summary": f"Based on {len(tile_ids)} imagery tiles and {len(tweet_ids)} tweets, we estimate moderate structural damage in zip {zip_code}.",
        }

    def _openai_stub(self, zip_code: str, time_window: Dict[str, str], context: Dict[str, object]) -> Dict[str, object]:
        """Placeholder for OpenAI GPT-4V integration."""

        logger.info(
            "OpenAI provider selected but not implemented; falling back to mock")
        return self._mock_response(zip_code, time_window, context)

    def _gemini_response(self, zip_code: str, time_window: Dict[str, str], context: Dict[str, object]) -> Dict[str, object]:
        """Call the Gemini API and parse the structured JSON response."""

        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set; required for Gemini provider.")
        try:
            import google.generativeai as genai  # type: ignore import-not-found
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "google-generativeai package is required for Gemini provider") from exc

        if self._gemini_client is None:
            genai.configure(api_key=self.gemini_api_key)
            self._gemini_client = genai.GenerativeModel(
                self.gemini_model, system_instruction=SYSTEM_PROMPT.strip())

        # Load images for true multimodal context
        from PIL import Image
        images = []
        image_ids = []
        for tile in context.get("imagery_tiles", []):
            uri = tile.get("uri")
            if uri and os.path.exists(uri):
                try:
                    img = Image.open(uri)
                    images.append(img)
                    image_ids.append(tile.get("tile_id"))
                except Exception as e:
                    logger.warning(f"Failed to load image {uri}: {e}")

        user_prompt = build_user_prompt(zip_code, time_window, context)
        
        # Construct multimodal message parts
        parts = [{"text": user_prompt}]
        for img, img_id in zip(images, image_ids):
            parts.append({"text": f"Image ID: {img_id}"})
            parts.append(img)
            
        messages = [
            {"role": "user", "parts": parts},
        ]

        for attempt in range(1, self.gemini_max_attempts + 1):
            start_ts = time.perf_counter()
            try:
                response = self._gemini_client.generate_content(
                    messages,
                    generation_config={
                        "response_mime_type": "application/json"},
                )
                latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)
                payload = self._extract_response_text(response)
                parsed = json.loads(payload)
                usage = getattr(response, "usage_metadata", None)
                usage_info = getattr(usage, "__dict__", usage)
                logger.info("Gemini call succeeded", attempt=attempt,
                            latency_ms=latency_ms, usage=usage_info)
                return parsed
            except json.JSONDecodeError:
                logger.warning(
                    "Gemini returned non-JSON response; falling back to mock output",
                    snippet=payload[:200] if 'payload' in locals() else '',
                )
                return self._mock_response(zip_code, time_window, context)
            except Exception as exc:  # pragma: no cover - network errors
                latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)
                logger.error("Gemini API call failed", attempt=attempt,
                             latency_ms=latency_ms, error=str(exc))
                if attempt == self.gemini_max_attempts:
                    raise
                time.sleep(min(2 ** (attempt - 1), 4))

        # Should not reach here because loop either returns or raises.
        return self._mock_response(zip_code, time_window, context)

    def _extract_response_text(self, response) -> str:
        """Best-effort extraction of text from Gemini responses."""

        text = getattr(response, "text", None)
        if text:
            return text.strip()
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                value = getattr(part, "text", None)
                if value:
                    return value.strip()
        logger.warning(
            "Gemini response had no text content; returning empty payload")
        return ""
