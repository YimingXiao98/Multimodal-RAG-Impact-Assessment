"""Core retrieval orchestrator using in-memory lists."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List


def _coerce_date(value):
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "date"):
        return value.date()
    return datetime.fromisoformat(str(value)).date()


from ..dataio.loaders import load_parquet_table
from ..dataio.schemas import RAGQuery
from ..dataio.storage import DataLocator
from ..indexing.spatial_index import SpatialIndex
from ..indexing.text_index import TextIndex
from .query_planner import build_plan


@dataclass
class RetrievalResult:
    imagery: List[dict]
    tweets: List[dict]
    calls: List[dict]
    sensors: List[dict]
    fema: List[dict]


class Retriever:
    def __init__(self, locator: DataLocator) -> None:
        self.locator = locator
        self.spatial_index = SpatialIndex.from_parquet(
            locator.table_path("imagery_tiles", example=True), locator.table_path("gauges", example=True)
        )
        self.tweet_index = TextIndex.build_mock(locator.table_path("tweets", example=True))
        self.calls_table = load_parquet_table(locator.table_path("311", example=True))
        self.fema_table = load_parquet_table(locator.table_path("fema_kb", example=True))

    def retrieve(self, query: RAGQuery) -> RetrievalResult:
        plan = build_plan(query)
        start_dt = datetime.combine(_coerce_date(query.start), datetime.min.time())
        end_dt = datetime.combine(_coerce_date(query.end), datetime.max.time())
        imagery = self.spatial_index.get_tiles_by_zip(query.zip, start_dt, end_dt, plan.imagery_k)
        tweet_ids = self.tweet_index.search_text(query.zip, start_dt, end_dt, plan.text_n)
        tweets = self._filter_records(self.tweet_index.table, tweet_ids, key="tweet_id")
        calls = self._filter_calls(query.zip, start_dt, end_dt, plan.text_n)
        sensors = self.spatial_index.nearest_sensors_by_zip(query.zip, n=3)
        fema = [row for row in self.fema_table if row.get("zip") == query.zip]
        return RetrievalResult(imagery=imagery, tweets=tweets, calls=calls, sensors=sensors, fema=fema)

    def _filter_records(self, table: List[dict], ids_scores: List, key: str) -> List[dict]:
        ids = {doc_id for doc_id, _ in ids_scores if doc_id}
        return [row for row in table if row.get(key) in ids]

    def _filter_calls(self, zip_code: str, start: datetime, end: datetime, n: int) -> List[dict]:
        records = [
            row
            for row in self.calls_table
            if row.get("zip") == zip_code and start.isoformat() <= row.get("timestamp", "") <= end.isoformat()
        ]
        records.sort(key=lambda r: r.get("timestamp", ""))
        return records[:n]
