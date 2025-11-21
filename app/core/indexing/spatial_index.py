"""Spatial indexing utilities using R-tree for efficient geospatial querying."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from rtree import index

try:
    import pgeocode
except ImportError:
    pgeocode = None

from ..dataio.loaders import load_parquet_table


@dataclass
class SpatialIndex:
    """In-memory spatial index for imagery and sensors using R-tree."""

    imagery: List[dict]
    sensors: List[dict]
    _zip_index: Dict[str, List[dict]] = field(init=False, default_factory=dict)
    _sensor_zip_index: Dict[str, List[dict]] = field(init=False, default_factory=dict)
    _imagery_rtree: index.Index = field(init=False)
    _sensor_rtree: index.Index = field(init=False)
    _nomi: Optional[object] = field(init=False, default=None)

    def __post_init__(self):
        if pgeocode:
            self._nomi = pgeocode.Nominatim("us")
        else:
            self._nomi = None
            logger.warning("pgeocode not installed; spatial radius fallback will be disabled.")

        self._imagery_rtree = index.Index()
        self._sensor_rtree = index.Index()

        # Index imagery
        for i, tile in enumerate(self.imagery):
            # 1. ZIP index
            z = tile.get("zip")
            if z:
                if z not in self._zip_index:
                    self._zip_index[z] = []
                self._zip_index[z].append(tile)
            
            # 2. R-tree index
            bbox = tile.get("bbox")
            if bbox is not None and len(bbox) == 4:
                # [minx, miny, maxx, maxy]
                self._imagery_rtree.insert(i, tuple(bbox))

        # Index sensors
        for i, sensor in enumerate(self.sensors):
            # 1. ZIP index
            z = sensor.get("zip")
            if z:
                if z not in self._sensor_zip_index:
                    self._sensor_zip_index[z] = []
                self._sensor_zip_index[z].append(sensor)
            
            # 2. R-tree index
            lat, lon = sensor.get("lat"), sensor.get("lon")
            if lat is not None and lon is not None:
                # Point as small bbox
                self._sensor_rtree.insert(i, (lon, lat, lon, lat))

    @classmethod
    def from_parquet(cls, imagery_path: Path, sensors_path: Path) -> "SpatialIndex":
        return cls(load_parquet_table(imagery_path), load_parquet_table(sensors_path))

    def get_tiles_by_zip(self, zip_code: str, start: datetime, end: datetime, k: int) -> List[dict]:
        """Return imagery tiles filtered by ZIP/time, falling back to spatial radius if needed."""
        if k <= 0:
            return []

        # 1. Try exact ZIP match
        candidates = self._zip_index.get(zip_code, [])
        tiles = self._filter_by_time(candidates, start, end)
        
        if len(tiles) < k and zip_code:
            # 2. Fallback: Spatial query around ZIP centroid
            logger.info("Insufficient exact ZIP matches, trying spatial radius", zip=zip_code, found=len(tiles))
            spatial_candidates = self._query_near_zip(zip_code, radius_km=5.0)
            # Exclude ones we already found
            seen_ids = {t["tile_id"] for t in tiles}
            for cand in spatial_candidates:
                if cand["tile_id"] not in seen_ids:
                    tiles.append(cand)
            
            # Re-filter by time just in case
            tiles = self._filter_by_time(tiles, start, end)

        if not tiles:
            # 3. Fallback: Global time search
            logger.warning("No imagery tiles in spatial range; falling back to global time search", zip=zip_code)
            tiles = self._filter_by_time(self.imagery, start, end)

        if not tiles:
             # 4. Last resort: Most recent globally
            logger.warning("No imagery tiles in time window; returning most recent globally", zip=zip_code)
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

        return tiles[:k]

    def nearest_sensors_by_zip(self, zip_code: str, n: int = 3) -> List[dict]:
        # 1. Try exact ZIP
        sensors = self._sensor_zip_index.get(zip_code, [])
        
        # 2. If few, try spatial
        if len(sensors) < n and zip_code:
            spatial_sensors = self._query_sensors_near_zip(zip_code, radius_km=10.0)
            seen_ids = {s["sensor_id"] for s in sensors}
            for s in spatial_sensors:
                if s["sensor_id"] not in seen_ids:
                    sensors.append(s)
        
        if not sensors:
            sensors = self.sensors

        # Sort by timestamp descending (assuming we want recent readings)
        # Note: In a real system, we'd want readings CLOSEST to the query time window.
        # For now, we keep the existing logic of "just return sensors", but maybe sorted by proximity?
        # The original code sorted by timestamp. Let's stick to that but maybe prioritize proximity if we had a query point.
        sensors_sorted = sorted(
            sensors, 
            key=lambda s: s.get("timestamp", ""), 
            reverse=True
        )
        return sensors_sorted[:n]

    def get_tiles_by_point(self, lat: float, lon: float, radius_km: float, start: datetime, end: datetime, k: int) -> List[dict]:
        """Return imagery tiles near a specific point."""
        if k <= 0:
            return []
        
        # Approx degrees (1 deg ~ 111km)
        delta_deg = radius_km / 111.0
        bbox = (lon - delta_deg, lat - delta_deg, lon + delta_deg, lat + delta_deg)
        
        indices = list(self._imagery_rtree.intersection(bbox))
        candidates = [self.imagery[i] for i in indices]
        
        tiles = self._filter_by_time(candidates, start, end)
        return tiles[:k]

    def get_sensors_by_point(self, lat: float, lon: float, radius_km: float, n: int = 3) -> List[dict]:
        """Return sensors near a specific point."""
        delta_deg = radius_km / 111.0
        bbox = (lon - delta_deg, lat - delta_deg, lon + delta_deg, lat + delta_deg)
        
        indices = list(self._sensor_rtree.intersection(bbox))
        sensors = [self.sensors[i] for i in indices]
        
        # Sort by distance would be better, but for now sort by timestamp or just return
        sensors_sorted = sorted(
            sensors, 
            key=lambda s: s.get("timestamp", ""), 
            reverse=True
        )
        return sensors_sorted[:n]

    def _query_near_zip(self, zip_code: str, radius_km: float) -> List[dict]:
        lat, lon = self._resolve_zip_centroid(zip_code)
        if lat is None or lon is None:
            return []
        
        # Reuse the point query logic but without time filtering (caller handles it)
        # Actually, _query_near_zip was used by get_tiles_by_zip which does time filtering later.
        # So we just return candidates.
        delta_deg = radius_km / 111.0
        bbox = (lon - delta_deg, lat - delta_deg, lon + delta_deg, lat + delta_deg)
        indices = list(self._imagery_rtree.intersection(bbox))
        return [self.imagery[i] for i in indices]

    def _query_sensors_near_zip(self, zip_code: str, radius_km: float) -> List[dict]:
        lat, lon = self._resolve_zip_centroid(zip_code)
        if lat is None or lon is None:
            return []
        
        delta_deg = radius_km / 111.0
        bbox = (lon - delta_deg, lat - delta_deg, lon + delta_deg, lat + delta_deg)
        indices = list(self._sensor_rtree.intersection(bbox))
        return [self.sensors[i] for i in indices]

    def _resolve_zip_centroid(self, zip_code: str) -> Tuple[Optional[float], Optional[float]]:
        if self._nomi is None:
            return None, None
        try:
            res = self._nomi.query_postal_code(zip_code)
            if res.empty or str(res.latitude) == 'nan':
                return None, None
            return float(res.latitude), float(res.longitude)
        except Exception:
            return None, None

    def _filter_by_time(self, candidates: List[dict], start: datetime, end: datetime) -> List[dict]:
        matches = []
        for tile in candidates:
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
