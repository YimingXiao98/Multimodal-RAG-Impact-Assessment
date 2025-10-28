"""Geospatial helper utilities for ZIP/H3 derivation."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from loguru import logger

try:  # pragma: no cover - optional dependency
    import geopandas as gpd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    gpd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import h3  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    h3 = None  # type: ignore

H3_RESOLUTION = 8


def to_h3(lat: Optional[float], lon: Optional[float], res: int = H3_RESOLUTION) -> Optional[str]:
    """Convert latitude/longitude to an H3 index string."""

    if lat is None or lon is None or h3 is None:
        return None
    try:
        return h3.geo_to_h3(lat, lon, res)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to compute H3 index: {exc}", exc=exc)
        return None


def add_zip_from_shapes(df, zip_shapes, zip_column: str = "ZIP"):
    """Spatially join input records with ZIP code polygons (requires GeoPandas)."""

    if gpd is None:
        raise ImportError("geopandas is required for spatial joins")
    if df.empty:
        return df
    joined = gpd.sjoin(df, zip_shapes[[zip_column, "geometry"]], how="left", predicate="intersects")
    return joined.rename(columns={zip_column: "zip"})


def localize_timestamp(ts: datetime) -> datetime:
    """Placeholder for timezone localization (naive passthrough in mock mode)."""

    return ts
