"""Fusion engine to combine text and visual analysis results."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from loguru import logger


FUSION_PROMPT = """
You are a disaster impact analyst combining evidence from multiple sources.

You have two independent assessments:
1. TEXT ANALYSIS: Based on tweets, 311 calls, and sensor data (REAL-TIME during event)
2. VISUAL ANALYSIS: Based on aerial/satellite imagery (POST-EVENT snapshot, Aug 31, 2017)

## CRITICAL INSIGHT:
Satellite imagery was captured 3-4 days AFTER peak flooding. Floodwaters had RECEDED.
This creates CONFLICTS between text (flooding reported) and visual (no flooding visible).

## VISUAL CONFIRMATION ONLY STRATEGY:
To avoid errors from conflicting signals, use visual ONLY when it CONFIRMS text:

### For FLOOD EXTENT:
- TEXT is your PRIMARY source (real-time reports during event)
- If text says flooding AND visual shows flooding → CONFIRMED, use average
- If text says flooding BUT visual shows dry → Use TEXT only (water receded)
- If CONFLICT between text and visual → IGNORE visual, use TEXT

### For DAMAGE:
- TEXT is your baseline (internal damage not visible from aerial)
- If text says damage AND visual shows damage → CONFIRMED, use higher value
- If text says damage BUT visual shows none → Use TEXT (internal damage)
- Visual can BOOST damage if it shows debris/destruction text missed

### Key Rule: When in CONFLICT, default to TEXT. Visual is confirmatory only.

| Text | Visual | Action |
|------|--------|--------|
| High | High   | CONFIRM: Average/boost |
| High | Low    | CONFLICT: Use TEXT only |
| Low  | High   | Use TEXT (don't trust visual alone for flood) |

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
                    logger.warning(
                        "OPENAI_API_KEY not set, falling back to heuristic fusion"
                    )
                    self.use_llm_fusion = False
                else:
                    self.model_name = model_name or os.getenv(
                        "OPENAI_MODEL", "gpt-4o-mini"
                    )
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
            return self._llm_fusion(
                text_analysis, visual_analysis, zip_code, time_window
            )
        else:
            return self._heuristic_fusion(
                text_analysis, visual_analysis, zip_code, time_window
            )

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

        # Get damage estimates (try both old and new schema names)
        text_damage = text_estimates.get(
            "damage_severity_pct", text_estimates.get("structural_damage_pct", 0.0)
        )
        text_flood = text_estimates.get("flood_extent_pct", 0.0)

        visual_damage = visual_overall.get(
            "damage_severity_pct", visual_overall.get("structural_damage_pct", 0.0)
        )
        # Support new schema: flood_evidence_pct (post-flood indicators)
        visual_flood = visual_overall.get(
            "flood_evidence_pct",
            visual_overall.get(
                "flood_extent_pct", visual_overall.get("flood_severity_pct", 0.0)
            ),
        )

        # NEW: Get text confirmation level from visual analysis
        text_confirmation = visual_overall.get("text_confirmation_level", "unknown")

        text_confidence = text_estimates.get("confidence", 0.5)
        visual_confidence = visual_overall.get("confidence", 0.5)

        # TEXT-GUIDED VISUAL CONFIRMATION FUSION:
        # Visual analysis now explicitly reports whether it confirms text.
        # Use this to make smarter fusion decisions.

        # Flood extent:
        # - Text is always the baseline (real-time reports)
        # - Visual can BOOST if it confirms (shows flood evidence/debris)
        # - Visual cannot reduce (water receded, but damage happened)

        if text_confirmation == "strong":
            # Visual strongly confirms text - boost confidence, slight increase
            fused_flood = text_flood * 1.1  # 10% boost
            fused_flood = min(fused_flood, 100)  # Cap at 100
            fusion_note = "visual_confirms_strong"
        elif text_confirmation == "partial":
            # Partial confirmation - use text as-is with confidence boost
            fused_flood = text_flood
            fusion_note = "visual_confirms_partial"
        elif text_confirmation == "contradicts":
            # Visual contradicts - but we trust text (temporal mismatch)
            fused_flood = text_flood
            fusion_note = "visual_contradicts_trust_text"
        elif text_flood > 20 and visual_flood > 20:
            # Fallback: both show evidence - average
            fused_flood = text_flood * 0.6 + visual_flood * 0.4
            fusion_note = "both_show_evidence"
        elif text_flood > 0:
            # Only text shows flooding
            fused_flood = text_flood
            fusion_note = "text_only"
        else:
            # No text signal - use visual as fallback
            fused_flood = visual_flood
            fusion_note = "visual_fallback"

        # Damage:
        # - Text is baseline (internal damage not visible)
        # - Visual can BOOST if it confirms (shows debris/damage)
        # - Visual cannot REDUCE (may miss internal damage)
        if text_damage > 20 and visual_damage > 20:
            # CONFIRMATION: Both show damage - use the higher one
            fused_damage = max(text_damage, visual_damage)
        elif visual_damage > text_damage + 20:
            # Visual shows significant damage text missed - boost slightly
            fused_damage = text_damage + (visual_damage - text_damage) * 0.3
        else:
            # Default to text (visual can't veto)
            fused_damage = text_damage

        # Detect conflicts (for logging only, not used in decision)
        AGREEMENT_THRESHOLD = 30  # percentage points
        conflicts = []
        text_visual_flood_diff = abs(text_flood - visual_flood)
        text_visual_damage_diff = abs(text_damage - visual_damage)

        if text_visual_flood_diff > AGREEMENT_THRESHOLD:
            conflicts.append(
                f"Flood CONFLICT: Text={text_flood:.0f}%, Visual={visual_flood:.0f}% → Used {fusion_note}"
            )
        if text_visual_damage_diff > AGREEMENT_THRESHOLD:
            conflicts.append(
                f"Damage CONFLICT: Text={text_damage:.0f}%, Visual={visual_damage:.0f}%"
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
                "flood_extent_pct": round(fused_flood, 1),
                "damage_severity_pct": round(fused_damage, 1),
                "confidence": round((text_confidence + visual_confidence) / 2, 2),
            },
            "text_analysis": {
                "flood_pct": text_flood,
                "damage_pct": text_damage,
                "confidence": text_confidence,
            },
            "visual_analysis": {
                "flood_pct": visual_flood,
                "damage_pct": visual_damage,
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

CRITICAL FUSION RULES:
- For FLOOD EXTENT: TRUST TEXT. If text reports flooding but visual shows dry, the water receded before image capture. Use text estimate.
- For DAMAGE: Visual is ADDITIVE. If visual shows damage, boost the estimate. If visual shows "no damage" but text reports damage, trust text (internal damage hidden).

Respond with JSON:
{{
    "zip": "{zip_code}",
    "time_window": {json.dumps(time_window)},
    "estimates": {{
        "flood_extent_pct": float,  // 0-100, HAZARD: % of area flooded. TRUST TEXT over visual.
        "damage_severity_pct": float,  // 0-100, CONSEQUENCE: structural damage. Visual can boost, not veto.
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
