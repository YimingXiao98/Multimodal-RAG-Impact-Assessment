"""Candidate selection heuristics."""
from __future__ import annotations

from typing import Dict, List

from .retriever import RetrievalResult


def select_candidates(results: RetrievalResult, k_tiles: int, n_text: int, n_sensors: int = 3) -> Dict[str, object]:
    imagery = results.imagery[:k_tiles]
    tweets = sorted(results.tweets, key=lambda r: r.get("timestamp", ""))[:n_text]
    calls = sorted(results.calls, key=lambda r: r.get("timestamp", ""))[:n_text]
    sensors = results.sensors[:n_sensors]
    return {
        "imagery": imagery,
        "tweets": tweets,
        "calls": calls,
        "sensors": sensors,
        "fema": results.fema,
    }
