from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from .dataio.schemas import RAGQuery

LOG_FILE = Path("conversation_history.md")


def log_conversation(message: str, query: RAGQuery, answer: Dict[str, Any]):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"## Interaction at {timestamp}\n\n")
        f.write(f"**User Question:** {message}\n\n")

        f.write("**Interpreted Query:**\n")
        f.write(f"- **ZIP:** {query.zip}\n")
        f.write(f"- **Time Window:** {query.start} to {query.end}\n\n")

        f.write("**RAG Answer:**\n")
        summary = answer.get("natural_language_summary")
        if summary:
            f.write(f"> {summary}\n\n")

        estimates = answer.get("estimates", {})
        f.write("**Estimates:**\n")
        f.write(
            f"- Structural Damage: {estimates.get('structural_damage_pct', 0)}%\n")
        f.write(f"- Confidence: {estimates.get('confidence', 0)}\n\n")

        f.write("**Evidence Used:**\n")
        refs = answer.get("evidence_refs", {})
        if refs:
            for key, items in refs.items():
                if items:
                    f.write(f"- **{key}:** {', '.join(items[:5])}")
                    if len(items) > 5:
                        f.write(f" (+{len(items)-5} more)")
                    f.write("\n")
        else:
            f.write("No specific evidence references returned.\n")

        f.write("\n---\n\n")
