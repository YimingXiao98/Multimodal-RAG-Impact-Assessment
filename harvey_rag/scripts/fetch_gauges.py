"""Normalize rainfall and stage gauges."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from loguru import logger

from app.core.dataio.loaders import load_sensors, save_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize gauge observations")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    records = load_sensors(Path(args.input))
    if args.start:
        start = datetime.fromisoformat(args.start)
        records = [r for r in records if datetime.fromisoformat(r["timestamp"]) >= start]
    if args.end:
        end = datetime.fromisoformat(args.end)
        records = [r for r in records if datetime.fromisoformat(r["timestamp"]) <= end]
    save_parquet(records, Path(args.output))
    logger.info("Gauge normalization complete", rows=len(records))


if __name__ == "__main__":
    main()
