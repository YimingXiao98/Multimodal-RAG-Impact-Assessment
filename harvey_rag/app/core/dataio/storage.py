"""Convenience wrappers around processed data locations."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .schemas import RAGQuery


class DataLocator:
    """Resolve processed data locations from configured base directory."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.processed = base_dir / "processed"
        self.examples = base_dir / "examples"
        self.indexes = base_dir / "indexes"

    def table_path(self, name: str, example: bool = False) -> Path:
        root = self.examples if example else self.processed
        return root / f"{name}.parquet"

    def faiss_index_path(self, name: str) -> Path:
        return self.indexes / f"{name}.faiss"


def resolve_example_query(locator: DataLocator) -> Optional[RAGQuery]:
    """Load a canned query JSON from examples directory."""

    import json
    from datetime import date

    query_path = locator.examples / "query.json"
    if not query_path.exists():
        return None
    payload = json.loads(query_path.read_text())
    return RAGQuery(zip=payload["zip"], start=date.fromisoformat(payload["start"]), end=date.fromisoformat(payload["end"]))
