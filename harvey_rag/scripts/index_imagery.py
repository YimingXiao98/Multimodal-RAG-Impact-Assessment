"""Index imagery metadata."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from loguru import logger

from app.core.dataio.loaders import save_parquet
from app.core.dataio.utils_geo import to_h3


def main() -> None:
    parser = argparse.ArgumentParser(description="Index imagery tiles")
    parser.add_argument("--input", required=True, help="Input CSV with imagery metadata")
    parser.add_argument("--output", required=True, help="Output path")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        records = []
        for row in reader:
            lat = row.get("lat")
            lon = row.get("lon")
            h3_index = to_h3(float(lat), float(lon)) if lat and lon else None
            row["h3"] = h3_index
            records.append(row)
    save_parquet(records, Path(args.output))
    logger.info("Imagery index built", rows=len(records))


if __name__ == "__main__":
    main()
