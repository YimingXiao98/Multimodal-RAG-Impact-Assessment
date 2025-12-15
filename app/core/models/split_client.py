"""Client orchestrating the split pipeline with text, visual, and fusion components."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from loguru import logger

from .text_client import TextAnalysisClient
from .verifier import Verifier
from .temporal_matcher import apply_temporal_weighting_to_visual
from datetime import datetime


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
        visual_provider: str = None,
    ):
        """
        Initialize the split pipeline client.

        Args:
            provider: LLM provider for text analysis ("openai" or "gemini")
            enable_visual: Whether to enable visual analysis
            use_llm_fusion: Whether to use LLM for fusion (vs heuristics)
            visual_provider: Provider for visual analysis ("openai" or "gemini", defaults to provider)
        """
        if not provider:
            provider = os.getenv("MODEL_PROVIDER", "openai")

        self.provider = provider
        self.text_client = TextAnalysisClient(provider=provider)
        self.verifier = Verifier()

        # Visual pipeline components (lazy-loaded)
        self.enable_visual = enable_visual
        self.use_llm_fusion = use_llm_fusion
        self.visual_provider = visual_provider or os.getenv(
            "VISUAL_MODEL_PROVIDER", provider
        )
        self._visual_client: Optional[Any] = None
        self._fusion_engine: Optional[Any] = None

    def _init_visual_components(self) -> None:
        """Lazy-initialize visual pipeline components."""
        if self._visual_client is not None:
            return

        try:
            from .visual_client import VisualAnalysisClient
            from .fusion_engine import FusionEngine

            self._visual_client = VisualAnalysisClient(provider=self.visual_provider)
            self._fusion_engine = FusionEngine(use_llm_fusion=self.use_llm_fusion)
            logger.info(
                f"Visual pipeline components initialized (provider: {self.visual_provider})"
            )
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
        captions: List[dict] = None,
        gauges: List[dict] = None,
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
            captions: List of image caption documents
            gauges: List of rainfall gauge documents
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
            "captions": captions or [],
            "gauges": gauges or [],
        }

        # Stage 1: Text Analysis
        logger.debug(f"Running text analysis for ZIP {zip_code}")
        text_result = self.text_client.analyze(zip_code, time_window, context)

        # Stage 2: Visual Analysis (if enabled and imagery available)
        # NEW: Pass text summary to guide visual analysis (text-guided visual)
        visual_result = None
        if self.enable_visual and imagery_tiles:
            self._init_visual_components()
            if self._visual_client:
                # Extract text summary to guide visual search
                text_summary = self._build_text_summary_for_visual(text_result)

                logger.debug(
                    f"Running text-guided visual analysis on {len(imagery_tiles)} tiles"
                )
                visual_result = self._visual_client.analyze(
                    zip_code,
                    time_window,
                    imagery_tiles,
                    project_root or Path.cwd(),
                    text_summary=text_summary,  # Pass text context to visual
                )

                # Apply temporal weighting to visual results (non-prompt-based temporal matching)
                try:
                    event_start = datetime.fromisoformat(time_window["start"])
                    event_end = datetime.fromisoformat(time_window["end"])

                    # Apply temporal weighting for flood extent (rapid decay for post-event imagery)
                    visual_result = apply_temporal_weighting_to_visual(
                        visual_result,
                        imagery_tiles,
                        event_start,
                        event_end,
                        weighting_method="linear_decay",
                        metric_type="flood_extent",
                    )
                    logger.debug(
                        f"Applied temporal weighting: weight={visual_result.get('overall_assessment', {}).get('temporal_weight', 'N/A')}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to apply temporal weighting: {e}")

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

    def _build_text_summary_for_visual(self, text_result: Dict[str, Any]) -> str:
        """Build a concise summary of text analysis to guide visual search."""
        parts = []

        # Get estimates
        estimates = text_result.get("estimates", {})
        flood_pct = estimates.get("flood_extent_pct", 0)
        damage_pct = estimates.get(
            "damage_severity_pct", estimates.get("structural_damage_pct", 0)
        )

        if flood_pct > 0:
            parts.append(
                f"Text reports estimate {flood_pct:.0f}% flood extent in this area."
            )
        if damage_pct > 0:
            parts.append(f"Text reports estimate {damage_pct:.0f}% structural damage.")

        # Get key evidence
        reasoning = text_result.get("reasoning", "")
        if reasoning:
            # Take first 300 chars of reasoning
            parts.append(f"Key reports: {reasoning[:300]}...")

        # Get natural language summary
        summary = text_result.get("natural_language_summary", "")
        if summary and len(summary) < 500:
            parts.append(f"Summary: {summary}")

        if not parts:
            return "No text evidence available for this area."

        return "\n".join(parts)

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
