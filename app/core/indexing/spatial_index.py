"""Spatial indexing utilities using simple list filters."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger

from ..dataio.loaders import load_parquet_table


@dataclass
class SpatialIndex:
    """In-memory spatial index for imagery and sensors."""

    imagery: List[dict]
    sensors: List[dict]

    @classmethod
    def from_parquet(cls, imagery_path: Path, sensors_path: Path) -> "SpatialIndex":
        return cls(load_parquet_table(imagery_path), load_parquet_table(sensors_path))

    def get_tiles_by_zip(self, zip_code: str, start: datetime, end: datetime, k: int) -> List[dict]:
        """Return imagery tiles filtered by ZIP/time with graceful fallback."""

        if k <= 0:
            return []

        tiles = self._filter_tiles(zip_code=zip_code, start=start, end=end)
        if tiles:
            return tiles[:k]

        if zip_code:
            logger.warning("No imagery tiles for requested ZIP/time window; falling back to time-only filter", zip=zip_code)
        time_only = self._filter_tiles(zip_code=None, start=start, end=end)
        if time_only:
            return time_only[:k]

        logger.warning("No imagery tiles within time window; returning most recent globally", zip=zip_code)
        latest = sorted(
            [
                (self._parse_timestamp(tile.get("timestamp")), tile)
                for tile in self.imagery
                if self._parse_timestamp(tile.get("timestamp"))
            ],
            key=lambda item: item[0],
            reverse=True,
        )
        return [tile for _, tile in latest[:k]]

    def nearest_sensors_by_zip(self, zip_code: str, n: int = 3) -> List[dict]:
        sensors = [s for s in self.sensors if s.get("zip") == zip_code]
        sensors.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
        return sensors[:n]

    def _filter_tiles(self, zip_code: Optional[str], start: datetime, end: datetime) -> List[dict]:
        matches = []
        for tile in self.imagery:
            if zip_code and tile.get("zip") != zip_code:
                continue
            ts = self._parse_timestamp(tile.get("timestamp"))
            if ts is None or ts < start or ts > end:
                continue
            matches.append((ts, tile))
        matches.sort(key=lambda item: item[0])
        return [tile for _, tile in matches]

    @staticmethod
    def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
