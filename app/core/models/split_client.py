"""Client orchestrating the split pipeline with text, visual, and fusion components."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from loguru import logger

from .text_client import TextAnalysisClient
from .verifier import Verifier


class SplitPipelineClient:
    """
    Orchestrates the split pipeline for multimodal disaster impact assessment.

    Pipeline stages:
    1. Text Analysis - Analyze tweets, 311 calls, sensors (TextAnalysisClient)
    2. Visual Analysis - Analyze imagery with GPT-4 Vision (VisualAnalysisClient)
    3. Fusion - Combine text and visual insights (FusionEngine)
    4. Verification - Validate evidence citations (Verifier)
    """

    def __init__(
        self,
        provider: str = None,
        enable_visual: bool = True,
        use_llm_fusion: bool = True,
    ):
        """
        Initialize the split pipeline client.

        Args:
            provider: LLM provider ("openai" or "gemini")
            enable_visual: Whether to enable visual analysis (requires OpenAI)
            use_llm_fusion: Whether to use LLM for fusion (vs heuristics)
        """
        if not provider:
            provider = os.getenv("MODEL_PROVIDER", "openai")

        self.provider = provider
        self.text_client = TextAnalysisClient(provider=provider)
        self.verifier = Verifier()

        # Visual pipeline components (lazy-loaded)
        self.enable_visual = enable_visual
        self.use_llm_fusion = use_llm_fusion
        self._visual_client: Optional[Any] = None
        self._fusion_engine: Optional[Any] = None

    def _init_visual_components(self) -> None:
        """Lazy-initialize visual pipeline components."""
        if self._visual_client is not None:
            return

        try:
            from .visual_client import VisualAnalysisClient
            from .fusion_engine import FusionEngine

            self._visual_client = VisualAnalysisClient()
            self._fusion_engine = FusionEngine(use_llm_fusion=self.use_llm_fusion)
            logger.info("Visual pipeline components initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize visual components: {e}")
            self.enable_visual = False

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
        project_root: Path = None,
    ) -> Dict[str, object]:
        """
        Run the full multimodal inference pipeline.

        Args:
            zip_code: Target ZIP code
            time_window: Dict with 'start' and 'end' dates
            imagery_tiles: List of imagery tile metadata
            text_snippets: Legacy text snippets (deprecated)
            sensor_table: Formatted sensor data table
            kb_summary: FEMA knowledge base summary
            tweets: List of tweet documents
            calls: List of 311 call documents
            sensors: List of sensor observations
            fema: List of FEMA records
            project_root: Project root for resolving image paths

        Returns:
            Unified assessment with estimates and evidence references
        """
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

        # Stage 1: Text Analysis
        logger.debug(f"Running text analysis for ZIP {zip_code}")
        text_result = self.text_client.analyze(zip_code, time_window, context)

        # Stage 2: Visual Analysis (if enabled and imagery available)
        visual_result = None
        if self.enable_visual and imagery_tiles:
            self._init_visual_components()
            if self._visual_client:
                logger.debug(f"Running visual analysis on {len(imagery_tiles)} tiles")
                visual_result = self._visual_client.analyze(
                    zip_code,
                    time_window,
                    imagery_tiles,
                    project_root or Path.cwd(),
                )

        # Stage 3: Fusion (if visual results available)
        if visual_result and self._fusion_engine:
            logger.debug("Fusing text and visual results")
            fused_result = self._fusion_engine.fuse(
                text_result, visual_result, zip_code, time_window
            )
        else:
            fused_result = text_result

        # Stage 4: Verification
        verified_response = self.verifier.verify(fused_result, context)

        # Ensure expected output format
        self._normalize_response(verified_response)

        return verified_response

    def infer_text_only(
        self,
        zip_code: str,
        time_window: Dict[str, str],
        context: Dict[str, Any],
    ) -> Dict[str, object]:
        """
        Run text-only inference (no visual analysis).

        This is the original split pipeline behavior.
        """
        text_result = self.text_client.analyze(zip_code, time_window, context)
        verified_response = self.verifier.verify(text_result, context)
        self._normalize_response(verified_response)
        return verified_response

    def _normalize_response(self, response: Dict[str, Any]) -> None:
        """Ensure response has all expected fields."""
        # Map reasoning to natural_language_summary
        if "natural_language_summary" not in response:
            response["natural_language_summary"] = response.get("reasoning", "")

        # Ensure estimates exist
        if "estimates" not in response:
            response["estimates"] = {
                "structural_damage_pct": 0.0,
                "confidence": 0.0,
            }

        # Ensure evidence_refs exist
        if "evidence_refs" not in response:
            response["evidence_refs"] = {
                "tweet_ids": [],
                "call_311_ids": [],
                "imagery_tile_ids": [],
            }

    def generate_text(self, prompt: str) -> str:
        """
        Generate text for evaluation purposes (LLM-as-a-judge).

        Delegates to the text client's generate_text method.
        """
        return self.text_client.generate_text(prompt)
