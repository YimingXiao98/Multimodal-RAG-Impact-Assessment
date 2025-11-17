"""Extract imagery metadata from GeoTIFF scenes for the RAG pipeline."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import h3
import rasterio
from loguru import logger
from pyproj import Transformer
import numpy as np
import pgeocode

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.dataio.loaders import save_parquet

DATE_PATTERNS = [
    (re.compile(r"(\d{1,2})-(\d{1,2})-(\d{4})"), "%m-%d-%Y"),
    (re.compile(r"(\d{4})_(\d{2})_(\d{2})"), "%Y_%m_%d"),
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})"), "%Y-%m-%d"),
    (re.compile(r"(\d{8})"), "%Y%m%d"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate imagery metadata from GeoTIFF files")
    parser.add_argument("--imagery-dir", default="data/raw/imagery", help="Folder containing GeoTIFF scenes")
    parser.add_argument("--pattern", default="*.tif", help="Glob pattern for imagery files")
    parser.add_argument("--output", default="data/processed/imagery_tiles.parquet", help="Output metadata path")
    parser.add_argument("--imagery-type", default="sat", help="Imagery type label (sat/aerial/uav)")
    parser.add_argument("--default-zip", help="ZIP code to assign when no mapping exists")
    parser.add_argument("--zip-map", help="JSON file mapping filename -> ZIP code")
    parser.add_argument("--timestamp-map", help="JSON file mapping filename -> ISO timestamp")
    parser.add_argument("--uri-prefix", help="Optional URI/prefix to store instead of absolute paths")
    parser.add_argument("--h3-res", type=int, default=8, help="H3 resolution for centroids (default: 8)")
    parser.add_argument("--auto-zip", action="store_true", help="Infer ZIP via centroid coordinates (requires pgeocode data)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imagery_path = Path(args.imagery_dir)
    if not imagery_path.exists():
        raise SystemExit(f"Imagery directory not found: {imagery_path}")

    files = sorted(imagery_path.glob(args.pattern))
    if not files:
        raise SystemExit("No GeoTIFF files matched the pattern")

    zip_map = _load_optional_json(args.zip_map)
    ts_map = _load_optional_json(args.timestamp_map)

    docs: List[dict] = []
    for tif_path in files:
        row = _build_metadata(
            tif_path=tif_path,
            imagery_type=args.imagery_type,
            default_zip=args.default_zip,
            zip_map=zip_map,
            timestamp_map=ts_map,
            uri_prefix=args.uri_prefix,
            h3_res=args.h3_res,
            auto_zip=args.auto_zip,
        )
        if row:
            docs.append(row)

    if not docs:
        raise SystemExit("No metadata rows were generated")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(docs, output_path)
    logger.info("Imagery metadata written", path=str(output_path), rows=len(docs))


def _build_metadata(
    *,
    tif_path: Path,
    imagery_type: str,
    default_zip: Optional[str],
    zip_map: Optional[Dict[str, str]],
    timestamp_map: Optional[Dict[str, str]],
    uri_prefix: Optional[str],
    h3_res: int,
    auto_zip: bool,
) -> Optional[dict]:
    filename = tif_path.name
    zip_code = (zip_map or {}).get(filename, default_zip)

    timestamp = _resolve_timestamp(filename, timestamp_map)
    if not timestamp:
        logger.warning("Could not infer timestamp for %s; skipping", filename)
        return None

    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs = src.crs
        if crs is None:
            logger.warning("%s has no CRS; skipping", filename)
            return None
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
        max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
        center_lon, center_lat = transformer.transform(
            (bounds.left + bounds.right) / 2,
            (bounds.bottom + bounds.top) / 2,
        )
        bbox = [min_lon, min_lat, max_lon, max_lat]
        h3_index = h3.latlng_to_cell(center_lat, center_lon, h3_res)
        pixel_size_x = abs(src.transform.a)
        pixel_size_y = abs(src.transform.e)

    if not zip_code and auto_zip:
        zip_code = _lookup_zip(center_lat, center_lon)

    if not zip_code:
        logger.warning("No ZIP provided or inferred for %s; skipping", filename)
        return None

    tile_id = tif_path.stem
    uri = _build_uri(uri_prefix, tif_path)

    return {
        "tile_id": tile_id,
        "type": imagery_type,
        "bbox": bbox,
        "timestamp": timestamp,
        "zip": zip_code,
        "h3": h3_index,
        "resolution_m": max(pixel_size_x, pixel_size_y),
        "uri": uri,
    }


def _build_uri(uri_prefix: Optional[str], tif_path: Path) -> str:
    if uri_prefix:
        return str(Path(uri_prefix) / tif_path.name)
    return str(tif_path)


def _resolve_timestamp(filename: str, ts_map: Optional[Dict[str, str]]) -> Optional[str]:
    if ts_map and filename in ts_map:
        return ts_map[filename]
    stem = Path(filename).stem
    for pattern, fmt in DATE_PATTERNS:
        match = pattern.search(stem)
        if not match:
            continue
        try:
            dt = datetime.strptime(match.group(0), fmt)
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    return None


def _load_optional_json(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return {Path(k).name: str(v) for k, v in data.items()}


_ZIP_SEARCH = None
_ZIP_CODES = None
_ZIP_COORDS = None


def _lookup_zip(lat: float, lon: float) -> Optional[str]:
    global _ZIP_SEARCH, _ZIP_CODES, _ZIP_COORDS
    if _ZIP_SEARCH is None:
        logger.info("Loading ZIP centroids via pgeocode")
        _ZIP_SEARCH = pgeocode.Nominatim('us')
        data = _ZIP_SEARCH._data[['postal_code', 'latitude', 'longitude']].dropna()
        if data.empty:
            logger.error("pgeocode dataset is empty; cannot infer ZIP codes")
            return None
        _ZIP_CODES = data['postal_code'].astype(str).tolist()
        coords = data[['latitude', 'longitude']].to_numpy(dtype='float64')
        _ZIP_COORDS = np.radians(coords)
    if _ZIP_COORDS is None:
        return None
    target = np.radians([lat, lon])
    dlat = _ZIP_COORDS[:, 0] - target[0]
    dlon = _ZIP_COORDS[:, 1] - target[1]
    a = np.sin(dlat / 2) ** 2 + np.cos(target[0]) * np.cos(_ZIP_COORDS[:, 0]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = 6371.0 * c
    idx = int(np.argmin(distances))
    return _ZIP_CODES[idx]


if __name__ == "__main__":
    main()
