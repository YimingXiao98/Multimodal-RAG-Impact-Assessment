"""Ground-truth helpers for flood and damage assessment."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict

from ..dataio.loaders import load_parquet_table


class FloodDepthGroundTruth:
    """Ground truth from FEMA flood depth grid (flood_depth_by_zip.json)."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Flood depth file not found: {path}")
        self.data = json.loads(path.read_text())

    def score(self, zip_code: str, start_date=None, end_date=None) -> Dict[str, float]:
        """Return flood metrics for a ZIP code.
        
        Note: start_date and end_date are accepted for API compatibility
        but ignored since flood depth is a static snapshot.
        """
        entry = self.data.get(str(zip_code), {})
        return {
            "mean_depth_m": entry.get("mean_depth_m", 0.0),
            "max_depth_m": entry.get("max_depth_m", 0.0),
            "flooded_pct": entry.get("flooded_pct", 0.0),
        }


class PDEGroundTruth:
    """Ground truth from Point Damage Estimates (pde_by_zip.json)."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"PDE file not found: {path}")
        self.data = json.loads(path.read_text())
        # Compute max for normalization
        self.max_pde = max(
            (entry.get("mean_pde", 0.0) for entry in self.data.values()),
            default=1.0
        )

    def score(self, zip_code: str) -> Dict[str, float]:
        """Return damage metrics for a ZIP code (normalized to 0-100%)."""
        entry = self.data.get(str(zip_code), {})
        mean_pde = entry.get("mean_pde", 0.0)
        # Normalize to percentage (0-100)
        damage_pct = (mean_pde / self.max_pde) * 100.0 if self.max_pde > 0 else 0.0
        return {
            "damage_pct": round(damage_pct, 2),
            "mean_pde": mean_pde,
            "point_count": entry.get("point_count", 0),
        }


class ClaimsGroundTruth:
    """Aggregate insurance claims to estimate structural damage intensity."""

    def __init__(self, claims_path: Path) -> None:
        if not claims_path.exists():
            raise FileNotFoundError(f"Claims table not found: {claims_path}")
        raw = load_parquet_table(claims_path)
        self.by_zip: Dict[str, list[dict]] = defaultdict(list)
        totals: Dict[str, float] = defaultdict(float)
        for record in raw:
            zip_code = str(record.get("zip") or "").zfill(5)
            if not zip_code.strip("0"):
                continue
            timestamp = self._parse_ts(record.get("timestamp"))
            if not timestamp:
                continue
            amount = float(record.get("amount") or 0.0)
            self.by_zip[zip_code].append({"timestamp": timestamp, "amount": amount})
            totals[zip_code] += amount
        self.max_total = max(totals.values()) if totals else 0.0

    def score(self, zip_code: str, start_date, end_date) -> Dict[str, float]:
        records = self.by_zip.get(str(zip_code).zfill(5), [])
        if not records:
            return {"damage_pct": 0.0, "claim_count": 0, "total_amount": 0.0}

        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        total = 0.0
        count = 0
        for record in records:
            if start_dt <= record["timestamp"] <= end_dt:
                total += record["amount"]
                count += 1
        if total <= 0.0 or self.max_total <= 0:
            return {"damage_pct": 0.0, "claim_count": count, "total_amount": total}
        pct = min((total / self.max_total) * 100.0, 100.0)
        return {"damage_pct": round(pct, 2), "claim_count": count, "total_amount": total}

    @staticmethod
    def _parse_ts(value) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace('Z', ''))
        except ValueError:
            return None
