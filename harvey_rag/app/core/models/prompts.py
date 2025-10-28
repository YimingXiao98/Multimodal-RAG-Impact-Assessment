"""Prompt templates for multimodal models."""
from __future__ import annotations

from textwrap import dedent


SYSTEM_PROMPT = dedent(
    """
    You are an assistant estimating post-disaster impact for Harris County, TX during Hurricane Harvey.
    Use provided imagery IDs, text snippets, sensor summaries, and FEMA priors to produce JSON output.
    Respond with valid JSON matching the schema provided by the user.
    """
)


def build_user_prompt(zip_code: str, time_window: dict[str, str], context: dict[str, object]) -> str:
    """Compose the user prompt."""

    snippets = "\n".join(context.get("text_snippets", []))
    return dedent(
        f"""
        ZIP: {zip_code}
        Time window: {time_window['start']} to {time_window['end']}
        Imagery IDs: {[tile['tile_id'] for tile in context.get('imagery_tiles', [])]}
        Sensor summary (Markdown table):
        {context.get('sensor_table', '')}
        FEMA prior summary:
        {context.get('kb_summary', '')}

        Text snippets:
        {snippets}

        Respond with JSON matching schema:
        {{
          "zip": str,
          "time_window": {{"start": str, "end": str}},
          "estimates": {{"structural_damage_pct": float, "roads_impacted": list[str], "confidence": float}},
          "evidence_refs": {{
            "imagery_tile_ids": list[str],
            "tweet_ids": list[str],
            "call_311_ids": list[str],
            "sensor_ids": list[str],
            "kb_refs": list[str]
          }}
        }}
        """
    )
