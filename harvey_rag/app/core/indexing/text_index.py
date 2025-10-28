"""Text indexing with deterministic random scoring."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from typing import List, Tuple

from ..dataio.loaders import load_parquet_table


@dataclass
class TextIndex:
    table: List[dict]

    @classmethod
    def build_mock(cls, table_path: Path, seed: int = 42) -> "TextIndex":
        return cls(load_parquet_table(table_path))

    def search_text(self, zip_code: str, start: datetime, end: datetime, n: int) -> List[Tuple[str, float]]:
        records = [
            r
            for r in self.table
            if r.get("zip") == zip_code and start.isoformat() <= r.get("timestamp", "") <= end.isoformat()
        ]
        if not records:
            records = self.table
        rng = Random(hash(zip_code))
        scored = [(rec.get("tweet_id") or rec.get("record_id"), rng.random()) for rec in records]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]
