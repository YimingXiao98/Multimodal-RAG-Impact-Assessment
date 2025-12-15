"""Temporal matching methods for aligning text and visual evidence by timestamp."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger


def compute_temporal_weight(
    imagery_timestamp: datetime,
    event_start: datetime,
    event_end: datetime,
    method: str = "linear_decay",
) -> float:
    """
    Compute temporal weight for visual evidence based on how close imagery is to event.

    Args:
        imagery_timestamp: When the imagery was captured
        event_start: Start of the disaster event
        event_end: End of the disaster event
        method: Weighting method ("linear_decay", "exponential_decay", "window", "peak_centered")

    Returns:
        Weight between 0.0 and 1.0 (higher = more temporally relevant)
    """
    if method == "linear_decay":
        # Linear decay: weight decreases linearly with time from event
        if imagery_timestamp < event_start:
            # Pre-event: weight based on how close to event start
            days_before = (event_start - imagery_timestamp).days
            return max(0.0, 1.0 - days_before * 0.1)  # 10% per day before
        elif imagery_timestamp <= event_end:
            # During event: full weight
            return 1.0
        else:
            # Post-event: decay based on days after
            days_after = (imagery_timestamp - event_end).days
            # For flood extent: rapid decay (water recedes quickly)
            # For damage: slower decay (damage persists)
            return max(0.0, 1.0 - days_after * 0.2)  # 20% per day after

    elif method == "exponential_decay":
        # Exponential decay: faster decay for post-event imagery
        if imagery_timestamp < event_start:
            days_before = (event_start - imagery_timestamp).days
            return max(0.0, 0.9 ** days_before)
        elif imagery_timestamp <= event_end:
            return 1.0
        else:
            days_after = (imagery_timestamp - event_end).days
            return max(0.0, 0.7 ** days_after)  # Faster decay

    elif method == "window":
        # Hard window: only use imagery within ±2 days of event
        window_days = 2
        if abs((imagery_timestamp - event_start).days) <= window_days:
            return 1.0
        elif abs((imagery_timestamp - event_end).days) <= window_days:
            return 1.0
        else:
            return 0.0

    elif method == "peak_centered":
        # Weight highest at peak event time (midpoint)
        peak_time = event_start + (event_end - event_start) / 2
        time_diff = abs((imagery_timestamp - peak_time).total_seconds() / 86400)  # days
        # Gaussian-like decay
        sigma = 1.0  # days
        return max(0.0, 1.0 - (time_diff / sigma) ** 2)

    else:
        logger.warning(f"Unknown temporal weighting method: {method}, using linear_decay")
        return compute_temporal_weight(imagery_timestamp, event_start, event_end, "linear_decay")


def match_text_to_imagery_temporal(
    text_docs: List[Dict],
    imagery_tiles: List[Dict],
    event_start: datetime,
    event_end: datetime,
    max_time_diff_hours: float = 24.0,
) -> Dict[str, List[Dict]]:
    """
    Match text documents to imagery tiles based on temporal proximity.

    Args:
        text_docs: List of text documents (tweets, calls) with timestamps
        imagery_tiles: List of imagery tiles with timestamps
        event_start: Event start time
        event_end: Event end time
        max_time_diff_hours: Maximum time difference for matching

    Returns:
        Dict mapping tile_id -> list of temporally-matched text docs
    """
    matches = {}

    for tile in imagery_tiles:
        tile_id = tile.get("tile_id", "unknown")
        tile_timestamp = _parse_timestamp(tile.get("timestamp"))
        if not tile_timestamp:
            continue

        matched_docs = []
        for doc in text_docs:
            doc_timestamp = _parse_timestamp(doc.get("timestamp"))
            if not doc_timestamp:
                continue

            # Compute time difference
            time_diff = abs((tile_timestamp - doc_timestamp).total_seconds() / 3600)  # hours

            if time_diff <= max_time_diff_hours:
                matched_docs.append({
                    "doc": doc,
                    "time_diff_hours": time_diff,
                    "temporal_weight": 1.0 - (time_diff / max_time_diff_hours) * 0.5,  # 0.5 to 1.0
                })

        if matched_docs:
            # Sort by temporal proximity
            matched_docs.sort(key=lambda x: x["time_diff_hours"])
            matches[tile_id] = matched_docs

    return matches


def apply_temporal_weighting_to_visual(
    visual_analysis: Dict[str, Any],
    imagery_tiles: List[Dict],
    event_start: datetime,
    event_end: datetime,
    weighting_method: str = "linear_decay",
    metric_type: str = "flood_extent",  # "flood_extent" or "damage"
) -> Dict[str, Any]:
    """
    Apply temporal weighting to visual analysis results based on imagery timestamps.

    Args:
        visual_analysis: Visual analysis results
        imagery_tiles: List of imagery tiles with timestamps
        event_start: Event start time
        event_end: Event end time
        weighting_method: Temporal weighting method
        metric_type: "flood_extent" (rapid decay) or "damage" (slow decay)

    Returns:
        Modified visual analysis with temporally-weighted estimates
    """
    if not imagery_tiles:
        return visual_analysis

    # Compute average temporal weight across all tiles
    weights = []
    for tile in imagery_tiles:
        tile_timestamp = _parse_timestamp(tile.get("timestamp"))
        if tile_timestamp:
            weight = compute_temporal_weight(
                tile_timestamp, event_start, event_end, weighting_method
            )
            # Adjust decay rate based on metric type
            if metric_type == "flood_extent":
                # Flood extent: imagery captured days after event is less reliable
                # 30% decay per day after event
                if tile_timestamp > event_end:
                    days_after = (tile_timestamp - event_end).days
                    weight *= max(0.1, 1.0 - days_after * 0.30)  # 30% per day
            elif metric_type == "damage":
                # Damage: persists longer, slower decay
                # 5% decay per day after event
                if tile_timestamp > event_end:
                    days_after = (tile_timestamp - event_end).days
                    weight *= max(0.3, 1.0 - days_after * 0.05)  # 5% per day

            weights.append(weight)

    avg_weight = sum(weights) / len(weights) if weights else 0.5

    # Apply temporal weighting to estimates
    overall = visual_analysis.get("overall_assessment", {})
    
    # Create weighted version
    weighted_overall = overall.copy()
    
    # Scale estimates by temporal weight
    if "flood_evidence_pct" in overall:
        weighted_overall["flood_evidence_pct"] = overall["flood_evidence_pct"] * avg_weight
    if "flood_extent_pct" in overall:
        weighted_overall["flood_extent_pct"] = overall["flood_extent_pct"] * avg_weight
    if "flood_severity_pct" in overall:
        weighted_overall["flood_severity_pct"] = overall["flood_severity_pct"] * avg_weight
    
    # Damage decays slower
    damage_weight = min(1.0, avg_weight * 1.2)  # Boost damage weight slightly
    if "structural_damage_pct" in overall:
        weighted_overall["structural_damage_pct"] = overall["structural_damage_pct"] * damage_weight
    if "damage_severity_pct" in overall:
        weighted_overall["damage_severity_pct"] = overall["damage_severity_pct"] * damage_weight

    # Store temporal metadata
    weighted_overall["temporal_weight"] = avg_weight
    weighted_overall["temporal_weighting_method"] = weighting_method

    result = visual_analysis.copy()
    result["overall_assessment"] = weighted_overall
    result["temporal_metadata"] = {
        "avg_temporal_weight": avg_weight,
        "num_tiles": len(imagery_tiles),
        "weighting_method": weighting_method,
    }

    return result


def _parse_timestamp(ts: Any) -> Optional[datetime]:
    """Parse timestamp from various formats."""
    if not ts:
        return None
    
    if isinstance(ts, datetime):
        return ts
    
    if isinstance(ts, str):
        try:
            # Try ISO format
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            try:
                # Try common formats
                for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        return datetime.strptime(ts, fmt)
                    except ValueError:
                        continue
            except Exception:
                pass
    
    return None

