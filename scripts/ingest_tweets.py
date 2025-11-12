#!/usr/bin/env python
"""Convert GNIP Harvey tweet archive into the pipeline's processed format."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Optional

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.dataio.loaders import save_parquet

INPUT_DIR = Path("data/raw/twitter/GNIPHarvey")
OUTPUT_PATH = Path("data/processed/tweets.parquet")


def iter_activity_files() -> list[Path]:
    if not INPUT_DIR.exists():
        raise SystemExit(f"Missing input dir: {INPUT_DIR}")
    return [p for p in sorted(INPUT_DIR.glob("*.json.gz")) if not p.name.startswith(".")]


def extract_record(activity: dict) -> Optional[dict]:
    posted = activity.get("postedTime")
    body = (activity.get("long_object") or {}).get(
        "body") or activity.get("body")
    if not posted or not body:
        return None

    tweet_id = activity.get("id")
    if tweet_id:
        tweet_id = tweet_id.split(":")[-1]

    media_uri = None
    entities = activity.get("twitter_entities") or {}
    media_list = entities.get("media")
    if not media_list and activity.get("long_object"):
        media_list = activity["long_object"].get(
            "twitter_entities", {}).get("media")
    if media_list:
        media_uri = media_list[0].get(
            "media_url_https") or media_list[0].get("media_url")

    lat = lon = None
    zip_code: Optional[str] = None

    profile_locs = (activity.get("gnip") or {}).get("profileLocations") or []
    if profile_locs:
        loc = profile_locs[0]
        coords = (loc.get("geo") or {}).get("coordinates")
        if isinstance(coords, list) and len(coords) == 2:
            lon, lat = coords
        addr = loc.get("address") or {}
        zip_code = addr.get("postalCode") or addr.get("postal_code")

    twitter_geo = activity.get("twitter_geo") or {}
    geo_coords = twitter_geo.get("coordinates")
    if isinstance(geo_coords, dict):
        geo_coords = geo_coords.get("coordinates")
    if isinstance(geo_coords, list) and len(geo_coords) == 2:
        lon, lat = geo_coords

    if not zip_code:
        place = activity.get("place") or {}
        if isinstance(place, dict):
            zip_code = place.get("postal_code") or place.get("postalCode")

    return {
        "tweet_id": tweet_id,
        "timestamp": posted,
        "text": body,
        "lat": lat,
        "lon": lon,
        "zip": zip_code,
        "media_uri": media_uri,
    }


def main() -> None:
    files = iter_activity_files()
    if not files:
        raise SystemExit(f"No GNIP files found under {INPUT_DIR}")

    records: list[dict] = []
    for idx, gz_path in enumerate(files, start=1):
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        activity = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    record = extract_record(activity)
                    if record:
                        records.append(record)
        except (OSError, EOFError):
            print(f"Skipping corrupt file: {gz_path.name}")
            continue

        if idx % 250 == 0:
            print(
                f"[{idx}/{len(files)}] files processed ({idx/len(files):.1%}), tweets={len(records):,}")

    print(f"Total tweets parsed: {len(records):,}")
    save_parquet(records, OUTPUT_PATH)
    print(f"Wrote {len(records):,} rows → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
