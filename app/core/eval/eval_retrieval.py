"""Retrieval evaluation metrics."""
from __future__ import annotations

from typing import Dict, List, Set

from ..retrieval.types import RetrievalResult


class RetrievalEvaluator:
    """Computes IR metrics for retrieval results against ground truth."""

    def __init__(self, ground_truth: Dict[str, Dict[str, List[str]]]) -> None:
        # ground_truth: { query_id: { "imagery": [id1, ...], "text": [id2, ...] } }
        self.ground_truth = ground_truth

    def evaluate(self, query_id: str, result: RetrievalResult) -> Dict[str, float]:
        gt = self.ground_truth.get(query_id)
        if not gt:
            return {}

        metrics = {}
        
        # Evaluate Imagery
        retrieved_imagery = [t["tile_id"] for t in result.imagery]
        relevant_imagery = set(gt.get("imagery", []))
        metrics.update(self._compute_modality_metrics("imagery", retrieved_imagery, relevant_imagery))

        # Evaluate Text (Tweets + Calls + FEMA + Claims)
        retrieved_text = []
        if result.tweets:
            retrieved_text.extend([f"tweet_{t['tweet_id']}" for t in result.tweets])
        if result.calls:
            retrieved_text.extend([f"311_{c['record_id']}" for c in result.calls])
        # Note: FEMA and Claims IDs might need consistent formatting with ground truth
        
        relevant_text = set(gt.get("text", []))
        metrics.update(self._compute_modality_metrics("text", retrieved_text, relevant_text))

        return metrics

    def _compute_modality_metrics(self, prefix: str, retrieved: List[str], relevant: Set[str]) -> Dict[str, float]:
        if not relevant:
            return {f"{prefix}_recall": 0.0, f"{prefix}_precision": 0.0}

        retrieved_set = set(retrieved)
        hits = len(retrieved_set & relevant)
        
        precision = hits / len(retrieved) if retrieved else 0.0
        recall = hits / len(relevant) if relevant else 0.0
        
        return {
            f"{prefix}_recall": round(recall, 3),
            f"{prefix}_precision": round(precision, 3),
            f"{prefix}_hits": hits,
            f"{prefix}_total_relevant": len(relevant)
        }
