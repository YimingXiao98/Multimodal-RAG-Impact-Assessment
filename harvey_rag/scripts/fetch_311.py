"""Normalize Houston 311 data."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from loguru import logger

from app.core.dataio.loaders import load_311, save_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Houston 311 dataset")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--start", help="Start date filter", default=None)
    parser.add_argument("--end", help="End date filter", default=None)
    args = parser.parse_args()

    records = load_311(Path(args.input))
    if args.start:
        start = datetime.fromisoformat(args.start)
        records = [r for r in records if datetime.fromisoformat(r["timestamp"]) >= start]
    if args.end:
        end = datetime.fromisoformat(args.end)
        records = [r for r in records if datetime.fromisoformat(r["timestamp"]) <= end]
    save_parquet(records, Path(args.output))
    logger.info("311 normalization complete", rows=len(records))


if __name__ == "__main__":
    main()
