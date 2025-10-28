"""Spatial indexing utilities using simple list filters."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

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
        """Return imagery tiles filtered by ZIP/time."""

        tiles = [
            tile
            for tile in self.imagery
            if tile.get("zip") == zip_code and start.isoformat() <= tile.get("timestamp", "") <= end.isoformat()
        ]
        tiles.sort(key=lambda t: t.get("timestamp", ""))
        return tiles[:k]

    def nearest_sensors_by_zip(self, zip_code: str, n: int = 3) -> List[dict]:
        sensors = [s for s in self.sensors if s.get("zip") == zip_code]
        sensors.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
        return sensors[:n]
