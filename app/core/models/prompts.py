"""Prompt templates for multimodal models."""
from __future__ import annotations

from textwrap import dedent


SYSTEM_PROMPT = dedent(
    """
    You are an assistant estimating post-disaster impact for Harris County, TX during Hurricane Harvey.
    You will be provided with satellite imagery, text snippets, sensor data, and FEMA priors.
    
    CRITICAL INSTRUCTION:
    - You MUST analyze the provided images to visually confirm flooding (e.g., water on roads, submerged structures).
    - Do NOT rely solely on the text or metadata.
    - Cite specific Image IDs when you see visual evidence of damage.
    
    Respond with valid JSON matching the schema provided by the user.
    """
)

QUERY_PARSING_SYSTEM_PROMPT = dedent(
    """
    You are a query parser for a disaster impact assessment system.
    Your goal is to extract structured parameters from a natural language user request.
    The user is asking about Hurricane Harvey impact in a specific location and time.
    
    Extract the following fields:
    - zip: The 5-digit ZIP code (e.g., "77096"). If missing, return null.
    - start: The start date in YYYY-MM-DD format.
    - end: The end date in YYYY-MM-DD format.
    
    If only one date is mentioned, use it for both start and end.
    If no year is mentioned but "Harvey" is implied, assume 2017.
    If no date is mentioned, return null for dates.
    
    Respond with valid JSON only.
    """
)


def build_query_parsing_prompt(message: str) -> str:
    """Compose the prompt for parsing a natural language query."""
    return dedent(
        f"""
        User Message: "{message}"
        
        Respond with JSON matching schema:
        {{
          "zip": str | null,
          "start": str | null,
          "end": str | null
        }}
        """
    )


def build_user_prompt(zip_code: str, time_window: dict[str, str], context: dict[str, object]) -> str:
    """Compose the user prompt."""

    # Format tweets with IDs
    tweets = context.get("tweets", [])
    tweet_lines = []
    for t in tweets:
        tid = t.get("doc_id") or t.get("tweet_id") or "unknown"
        text = (t.get("text") or "").replace("\n", " ")
        tweet_lines.append(f"- [{tid}] {text}")
    tweet_section = "\n".join(tweet_lines)

    # Format 311 calls with IDs
    calls = context.get("calls", [])
    call_lines = []
    for c in calls:
        cid = c.get("doc_id") or c.get("record_id") or "unknown"
        desc = (c.get("description") or "").replace("\n", " ")
        call_lines.append(f"- [{cid}] {desc}")
    call_section = "\n".join(call_lines)

    return dedent(
        f"""
        ZIP: {zip_code}
        Time window: {time_window['start']} to {time_window['end']}
        Imagery IDs: {[tile['tile_id'] for tile in context.get('imagery_tiles', [])]}
        
        Sensor summary (Markdown table):
        {context.get('sensor_table', '')}
        
        FEMA prior summary:
        {context.get('kb_summary', '')}

        Tweets:
        {tweet_section}

        311 Calls:
        {call_section}

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
          }},
          "natural_language_summary": str
        }}
        """
    )
