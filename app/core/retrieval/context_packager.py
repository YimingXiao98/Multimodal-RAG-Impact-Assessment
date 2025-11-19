"""Create structured context for model prompting."""
from __future__ import annotations

from typing import Dict, List


def _list_to_markdown(records: List[dict], columns: List[str]) -> str:
    if not records:
        return ""
    header = " | ".join(columns)
    separator = " | ".join(["---"] * len(columns))
    rows = []
    for record in records[:5]:
        rows.append(" | ".join(str(record.get(col, "")) for col in columns))
    return "\n".join([header, separator, *rows])


def package_context(candidates: Dict[str, object]) -> Dict[str, object]:
    imagery = candidates.get("imagery", [])
    tweets: List[dict] = candidates.get("tweets", [])
    calls: List[dict] = candidates.get("calls", [])
    sensors: List[dict] = candidates.get("sensors", [])
    fema: List[dict] = candidates.get("fema", [])

    text_snippets: List[str] = []
    text_snippets.extend((tweet.get("text") or "")[:400] for tweet in tweets)
    text_snippets.extend((call.get("description") or "")
                         [:400] for call in calls)

    sensor_table = _list_to_markdown(
        sensors, ["sensor_id", "timestamp", "value", "unit"]) if sensors else ""
    kb_summary = _list_to_markdown(fema, ["year", "loss_mean"]) if fema else ""

    return {
        "imagery_tiles": imagery,
        "tweets": tweets,
        "calls": calls,
        "sensors": sensors,
        "fema": fema,
        "text_snippets": [snippet for snippet in text_snippets if snippet],
        "sensor_table": sensor_table,
        "kb_summary": kb_summary,
    }
