"""Fusion engine to combine text and visual analysis results."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from loguru import logger


FUSION_PROMPT = """
You are a disaster impact analyst combining evidence from multiple sources.

You have two independent assessments:
1. TEXT ANALYSIS: Based on tweets, 311 calls, and sensor data
2. VISUAL ANALYSIS: Based on aerial/satellite imagery

Your task is to:
1. Compare the two assessments for consistency
2. Resolve any conflicts using the most reliable evidence
3. Produce a final, unified damage estimate

CONFLICT RESOLUTION RULES:
- If text reports flooding but images show dry conditions, note the discrepancy and consider:
  - Images may be from before/after the flooding peak
  - Ground-level reports can capture conditions not visible from above
- If images show damage but text has no reports, consider:
  - The area may be sparsely populated or inaccessible
  - Visual evidence is direct and generally reliable
- When in doubt, favor the evidence with higher confidence scores

Respond with valid JSON only.
"""


class FusionEngine:
    """
    Combines text and visual analysis results into a unified assessment.

    Handles conflict resolution between different evidence sources.
    """

    def __init__(
        self,
        model_name: str = None,
        api_key: Optional[str] = None,
        use_llm_fusion: bool = True,
    ):
        """
        Initialize the fusion engine.

        Args:
            model_name: OpenAI model for LLM-based fusion
            api_key: OpenAI API key
            use_llm_fusion: If True, use LLM to resolve conflicts; else use heuristics
        """
        self.use_llm_fusion = use_llm_fusion

        if use_llm_fusion:
            try:
                from openai import OpenAI

                self.api_key = api_key or os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    logger.warning("OPENAI_API_KEY not set, falling back to heuristic fusion")
                    self.use_llm_fusion = False
                else:
                    self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                    self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("OpenAI not available, falling back to heuristic fusion")
                self.use_llm_fusion = False

    def fuse(
        self,
        text_analysis: Dict[str, Any],
        visual_analysis: Dict[str, Any],
        zip_code: str,
        time_window: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Fuse text and visual analysis results.

        Args:
            text_analysis: Results from TextAnalysisClient
            visual_analysis: Results from VisualAnalysisClient
            zip_code: Target ZIP code
            time_window: Analysis time window

        Returns:
            Unified assessment with combined evidence
        """
        if self.use_llm_fusion:
            return self._llm_fusion(text_analysis, visual_analysis, zip_code, time_window)
        else:
            return self._heuristic_fusion(text_analysis, visual_analysis, zip_code, time_window)

    def _heuristic_fusion(
        self,
        text_analysis: Dict[str, Any],
        visual_analysis: Dict[str, Any],
        zip_code: str,
        time_window: Dict[str, str],
    ) -> Dict[str, Any]:
        """Fuse results using simple heuristics."""
        # Extract estimates
        text_estimates = text_analysis.get("estimates", {})
        visual_overall = visual_analysis.get("overall_assessment", {})

        text_damage = text_estimates.get("structural_damage_pct", 0.0)
        visual_damage = visual_overall.get("structural_damage_pct", 0.0)
        visual_flood = visual_overall.get("flood_severity_pct", 0.0)

        text_confidence = text_estimates.get("confidence", 0.5)
        visual_confidence = visual_overall.get("confidence", 0.5)

        # Weighted average based on confidence
        total_confidence = text_confidence + visual_confidence
        if total_confidence > 0:
            weight_text = text_confidence / total_confidence
            weight_visual = visual_confidence / total_confidence
        else:
            weight_text = weight_visual = 0.5

        # Fused damage estimate
        fused_damage = text_damage * weight_text + visual_damage * weight_visual

        # Detect conflicts
        conflicts = []
        if abs(text_damage - visual_damage) > 30:
            conflicts.append(
                f"Text reports {text_damage:.0f}% damage, visual shows {visual_damage:.0f}%"
            )

        # Combine evidence refs
        text_refs = text_analysis.get("evidence_refs", {})
        visual_refs = visual_analysis.get("evidence_refs", {})

        combined_refs = {
            "tweet_ids": text_refs.get("tweet_ids", []),
            "call_311_ids": text_refs.get("call_311_ids", []),
            "imagery_tile_ids": visual_refs.get("imagery_tile_ids", []),
            "sensor_ids": text_refs.get("sensor_ids", []),
        }

        # Build summary
        text_reasoning = text_analysis.get("reasoning", "")
        visual_observations = visual_overall.get("key_observations", [])

        summary_parts = []
        if text_reasoning:
            summary_parts.append(f"Text evidence: {text_reasoning}")
        if visual_observations:
            summary_parts.append(f"Visual evidence: {'; '.join(visual_observations)}")
        if conflicts:
            summary_parts.append(f"Conflicts noted: {'; '.join(conflicts)}")

        return {
            "zip": zip_code,
            "time_window": time_window,
            "estimates": {
                "structural_damage_pct": round(fused_damage, 1),
                "flood_severity_pct": round(visual_flood, 1),
                "confidence": round((text_confidence + visual_confidence) / 2, 2),
            },
            "text_analysis": {
                "damage_pct": text_damage,
                "confidence": text_confidence,
            },
            "visual_analysis": {
                "damage_pct": visual_damage,
                "flood_pct": visual_flood,
                "confidence": visual_confidence,
            },
            "conflicts": conflicts,
            "evidence_refs": combined_refs,
            "reasoning": " | ".join(summary_parts),
            "fusion_method": "heuristic",
        }

    def _llm_fusion(
        self,
        text_analysis: Dict[str, Any],
        visual_analysis: Dict[str, Any],
        zip_code: str,
        time_window: Dict[str, str],
    ) -> Dict[str, Any]:
        """Fuse results using LLM for intelligent conflict resolution."""
        prompt = f"""
Combine these two independent damage assessments for ZIP {zip_code} ({time_window['start']} to {time_window['end']}):

## TEXT ANALYSIS (from tweets, 311 calls, sensors):
{json.dumps(text_analysis, indent=2)}

## VISUAL ANALYSIS (from aerial/satellite imagery):
{json.dumps(visual_analysis, indent=2)}

Produce a unified assessment. Identify and resolve any conflicts.

Respond with JSON:
{{
    "zip": "{zip_code}",
    "time_window": {json.dumps(time_window)},
    "estimates": {{
        "structural_damage_pct": float,  // 0-100, your best unified estimate
        "flood_severity_pct": float,  // 0-100
        "confidence": float  // 0.0-1.0
    }},
    "conflicts": list[str],  // any discrepancies between text and visual
    "conflict_resolution": str,  // how you resolved conflicts
    "evidence_refs": {{
        "tweet_ids": list[str],
        "call_311_ids": list[str],
        "imagery_tile_ids": list[str],
        "sensor_ids": list[str]
    }},
    "reasoning": str,  // your chain of thought
    "natural_language_summary": str
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": FUSION_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=1500,
            )

            result = json.loads(response.choices[0].message.content)
            result["fusion_method"] = "llm"
            return result

        except Exception as e:
            logger.error(f"LLM fusion failed: {e}, falling back to heuristic")
            return self._heuristic_fusion(
                text_analysis, visual_analysis, zip_code, time_window
            )


class MultimodalPipelineClient:
    """
    Complete multimodal pipeline combining text and visual analysis.

    This is the main entry point for the Split-Pipeline Architecture.
    """

    def __init__(
        self,
        text_client=None,
        visual_client=None,
        fusion_engine=None,
        verifier=None,
    ):
        """
        Initialize the multimodal pipeline.

        Args:
            text_client: TextAnalysisClient instance
            visual_client: VisualAnalysisClient instance
            fusion_engine: FusionEngine instance
            verifier: Verifier instance for post-processing
        """
        from .text_client import TextAnalysisClient
        from .visual_client import VisualAnalysisClient

        self.text_client = text_client or TextAnalysisClient(provider="openai")
        self.visual_client = visual_client or VisualAnalysisClient()
        self.fusion_engine = fusion_engine or FusionEngine()
        self.verifier = verifier

    def analyze(
        self,
        zip_code: str,
        time_window: Dict[str, str],
        context: Dict[str, Any],
        project_root=None,
    ) -> Dict[str, Any]:
        """
        Run the complete multimodal analysis pipeline.

        Args:
            zip_code: Target ZIP code
            time_window: Dict with 'start' and 'end' dates
            context: Retrieved context (tweets, calls, imagery, etc.)
            project_root: Project root for resolving image paths

        Returns:
            Fused analysis results
        """
        from pathlib import Path

        project_root = project_root or Path.cwd()

        # Step 1: Text Analysis
        logger.info(f"Running text analysis for ZIP {zip_code}...")
        text_result = self.text_client.analyze(zip_code, time_window, context)

        # Step 2: Visual Analysis
        imagery_tiles = context.get("imagery_tiles", [])
        if imagery_tiles:
            logger.info(f"Running visual analysis on {len(imagery_tiles)} tiles...")
            visual_result = self.visual_client.analyze(
                zip_code, time_window, imagery_tiles, project_root
            )
        else:
            logger.warning("No imagery tiles for visual analysis")
            visual_result = self.visual_client._empty_result()

        # Step 3: Fusion
        logger.info("Fusing text and visual results...")
        fused_result = self.fusion_engine.fuse(
            text_result, visual_result, zip_code, time_window
        )

        # Step 4: Verification (optional)
        if self.verifier:
            logger.info("Verifying evidence citations...")
            fused_result = self.verifier.verify(fused_result, context)

        return fused_result

