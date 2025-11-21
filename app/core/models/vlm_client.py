"""Model client adapters for multimodal inference."""
from __future__ import annotations

import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from loguru import logger
from PIL import Image

from .prompts import (
    QUERY_PARSING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_query_parsing_prompt,
    build_user_prompt,
)


class VLMClient:
    """Interface wrapping different multimodal providers."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider or os.getenv("MODEL_PROVIDER")
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

        if not self.provider:
            return self._warning_response(zip_code, time_window, context, "No model provider configured.")

        if self.provider == "gemini":
            return self._gemini_response(zip_code, time_window, context)
        
        # If provider is set but not supported (e.g. "openai" which was a stub)
        return self._warning_response(zip_code, time_window, context, f"Provider '{self.provider}' is not supported or implemented.")

    def parse_query(self, message: str) -> Dict[str, str | None]:
        """Parse natural language query into structured parameters using LLM."""
        if not self.provider:
            return {"zip": None, "start": None, "end": None}

        if self.provider == "gemini":
            return self._gemini_parse_query(message)

        logger.warning(f"Query parsing not implemented for provider {self.provider}")
        return {"zip": None, "start": None, "end": None}

    def _gemini_parse_query(self, message: str) -> Dict[str, str | None]:
        """Use Gemini to parse the query."""
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY not set, skipping LLM query parsing")
            return {"zip": None, "start": None, "end": None}

        try:
            import google.generativeai as genai
        except ImportError:
            return {"zip": None, "start": None, "end": None}

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

    def _warning_response(self, zip_code: str, time_window: Dict[str, str], context: Dict[str, object], message: str) -> Dict[str, object]:
        """Return a response indicating that generation was skipped."""
        tile_ids = [tile["tile_id"] for tile in context.get("imagery_tiles", [])]
        tweet_ids = []
        call_ids = []
        if context.get("text_snippets"):
            # This is a bit loose, assuming text_snippets align with tweets/calls if passed separately
            # But for the warning response, we just want to show what we found.
            # Better to use the explicit lists if available.
            pass
            
        tweet_ids = [t.get("tweet_id") for t in context.get("tweets", [])]
        call_ids = [c.get("record_id") for c in context.get("calls", [])]
        sensor_ids = [s.get("sensor_id") for s in context.get("sensors", [])]
        kb_refs = [f"fema_zip_{zip_code}_2010_2016"] if context.get("kb_summary") else []

        return {
            "zip": zip_code,
            "time_window": time_window,
            "estimates": {
                "structural_damage_pct": 0.0,
                "roads_impacted": [],
                "confidence": 0.0,
            },
            "evidence_refs": {
                "imagery_tile_ids": tile_ids,
                "tweet_ids": tweet_ids,
                "call_311_ids": call_ids,
                "sensor_ids": sensor_ids,
                "kb_refs": kb_refs,
            },
            "natural_language_summary": f"[WARNING] {message} Evidence retrieved: {len(tile_ids)} tiles, {len(tweet_ids)} tweets, {len(call_ids)} calls.",
        }

    def _fetch_and_process_image(self, uri: str) -> Optional[bytes]:
        """Fetch image, resize, and compress."""
        if not uri:
            return None
        
        try:
            data = None
            # Handle local files
            if not uri.startswith("http"):
                path = Path(uri)
                if path.exists():
                    data = path.read_bytes()
                else:
                    # Try relative to project root if needed, or just fail
                    # Assuming uri is relative to cwd or absolute
                    pass
            
            # Handle HTTP
            if data is None:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(uri)
                    resp.raise_for_status()
                    data = resp.content
            
            if not data:
                return None

            # Process with PIL
            with Image.open(io.BytesIO(data)) as img:
                img = img.convert("RGB")
                # Resize if too large (max 1024 dim)
                max_dim = 1024
                if max(img.size) > max_dim:
                    img.thumbnail((max_dim, max_dim))
                
                out_io = io.BytesIO()
                img.save(out_io, format="JPEG", quality=85)
                return out_io.getvalue()

        except Exception as exc:
            logger.warning(f"Failed to fetch/process image {uri}: {exc}")
            return None

    def _gemini_response(self, zip_code: str, time_window: Dict[str, str], context: Dict[str, object]) -> Dict[str, object]:
        """Call the Gemini API and parse the structured JSON response."""

        if not self.gemini_api_key:
            return self._warning_response(zip_code, time_window, context, "GEMINI_API_KEY not set.")

        try:
            import google.generativeai as genai
        except ImportError as exc:
            return self._warning_response(zip_code, time_window, context, "google-generativeai package missing.")

        if self._gemini_client is None:
            genai.configure(api_key=self.gemini_api_key)
            self._gemini_client = genai.GenerativeModel(
                self.gemini_model, system_instruction=SYSTEM_PROMPT.strip())

        user_prompt = build_user_prompt(zip_code, time_window, context)
        
        parts = [{"text": user_prompt}]
        
        # Parallel fetch images
        imagery_tiles = context.get("imagery_tiles", [])
        # Limit to 16 images max
        imagery_tiles = imagery_tiles[:16]
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_uri = {
                executor.submit(self._fetch_and_process_image, tile.get("uri")): tile.get("uri")
                for tile in imagery_tiles if tile.get("uri")
            }
            
            for future in as_completed(future_to_uri):
                image_data = future.result()
                if image_data:
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_data
                        }
                    })

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
                    "Gemini returned non-JSON response",
                    snippet=payload[:200] if 'payload' in locals() else '',
                )
                return self._warning_response(zip_code, time_window, context, "Gemini returned invalid JSON.")
            except Exception as exc:
                latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)
                logger.error("Gemini API call failed", attempt=attempt,
                             latency_ms=latency_ms, error=str(exc))
                if attempt == self.gemini_max_attempts:
                    return self._warning_response(zip_code, time_window, context, f"Gemini API failed: {exc}")
                time.sleep(min(2 ** (attempt - 1), 4))

        return self._warning_response(zip_code, time_window, context, "Gemini API retries exhausted.")

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
