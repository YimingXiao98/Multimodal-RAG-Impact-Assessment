"""Dataset loading and normalization utilities with JSON-backed storage."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from loguru import logger

from .utils_geo import to_h3


def _parse_datetime(value: str) -> datetime:
    """Parse various timestamp formats produced by partner exports."""

    if not value:
        raise ValueError("Missing datetime value")
    value = value.strip()

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized datetime format: {value}")



def _normalize_common(records: List[dict], lat_key: str | None, lon_key: str | None) -> List[dict]:
    if lat_key and lon_key:
        for record in records:
            lat = record.get(lat_key)
            lon = record.get(lon_key)
            try:
                lat_f = float(lat) if lat is not None else None
                lon_f = float(lon) if lon is not None else None
            except (TypeError, ValueError):
                lat_f = lon_f = None
            record["h3"] = to_h3(lat_f, lon_f)
    return records


def load_311(input_path: Path) -> List[dict]:
    """Normalize Houston 311 CSV into list of dicts."""

    logger.info("Loading 311 records", path=input_path)
    records = _read_csv(input_path)
    for record in records:
        record["timestamp"] = _parse_datetime(record["timestamp"]).isoformat()
    return _normalize_common(records, "lat", "lon")


def load_tweets(input_path: Path) -> List[dict]:
    """Load tweets from CSV or JSON."""

    logger.info("Loading tweets", path=input_path)
    if input_path.suffix == ".json":
        records = json.loads(input_path.read_text())
    else:
        records = _read_csv(input_path)
    for record in records:
        record["timestamp"] = _parse_datetime(record["timestamp"]).isoformat()
    return _normalize_common(records, "lat", "lon")


def load_sensors(input_path: Path) -> List[dict]:
    """Load sensor observations."""

    logger.info("Loading sensors", path=input_path)
    records = _read_csv(input_path)
    for record in records:
        record["timestamp"] = _parse_datetime(record["timestamp"]).isoformat()
    return _normalize_common(records, "lat", "lon")


def load_fema_kb(input_path: Path) -> List[dict]:
    """Load FEMA KB from CSV."""

    logger.info("Loading FEMA KB", path=input_path)
    return _read_csv(input_path)


def load_claims(input_path: Path) -> List[dict]:
    """Load claims from CSV/GeoJSON."""

    logger.info("Loading claims", path=input_path)
    if input_path.suffix == ".geojson":
        records = json.loads(input_path.read_text()).get("features", [])
        normalized = []
        for feature in records:
            props = feature.get("properties", {})
            geom = feature.get("geometry", {}).get("coordinates", [None, None])
            normalized.append(
                {
                    "claim_id": props.get("claim_id"),
                    "lat": geom[1],
                    "lon": geom[0],
                    "timestamp": props.get("timestamp"),
                    "severity": props.get("severity"),
                    "zip": props.get("zip"),
                    "amount": props.get("amount"),
                }
            )
        records = normalized
    else:
        records = _read_csv(input_path)
    for record in records:
        record["timestamp"] = _parse_datetime(record["timestamp"]).isoformat()
    return _normalize_common(records, "lat", "lon")


def load_roads(input_path: Path) -> List[dict]:
    """Load road status records."""

    logger.info("Loading road status", path=input_path)
    records = _read_csv(input_path)
    for record in records:
        record["start_time"] = _parse_datetime(record["start_time"]).isoformat()
        end_time = record.get("end_time")
        if end_time:
            record["end_time"] = _parse_datetime(end_time).isoformat()
    return records


def save_parquet(records: List[dict], output_path: Path) -> None:
    """Persist records as JSON array stored in a .parquet file for mock mode."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, default=str))
    logger.info("Saved records", path=output_path, rows=len(records))


def load_parquet_table(path: Path) -> List[dict]:
    """Load records from JSON-backed parquet file."""

    if not path.exists():
        logger.warning("Missing table", path=path)
        return []
    return json.loads(path.read_text())


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]
