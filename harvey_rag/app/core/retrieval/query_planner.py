"""Plan retrieval sub-queries."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ..dataio.schemas import RAGQuery


@dataclass
class RetrievalPlan:
    """Structured plan of resources to retrieve."""

    imagery_k: int
    text_n: int
    sensor_window_hours: int
    include_fema: bool = True


def build_plan(query: RAGQuery) -> RetrievalPlan:
    """Create a retrieval plan from the user query."""

    start_date = query.start if isinstance(query.start, datetime) else datetime.fromisoformat(str(query.start))
    end_date = query.end if isinstance(query.end, datetime) else datetime.fromisoformat(str(query.end))
    start_dt = datetime.combine(start_date.date(), datetime.min.time()) if isinstance(start_date, datetime) else datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date.date(), datetime.max.time()) if isinstance(end_date, datetime) else datetime.combine(end_date, datetime.max.time())
    _ = (end_dt - start_dt).days
    return RetrievalPlan(imagery_k=query.k_tiles, text_n=query.n_text, sensor_window_hours=48)
