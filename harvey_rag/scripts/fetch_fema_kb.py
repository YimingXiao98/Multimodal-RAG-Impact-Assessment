"""Fetch historical FEMA losses."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List

import requests
from loguru import logger

from app.core.dataio.loaders import save_parquet

OPEN_FEMA_ENDPOINT = "https://www.fema.gov/api/open/v2/IndividualAssistanceHousingRegistrations"


def fetch_data(url: str) -> List[dict]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("IndividualAssistanceHousingRegistrations", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FEMA KB data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=2010)
    parser.add_argument("--end", type=int, default=2016)
    args = parser.parse_args()

    try:
        records = fetch_data(OPEN_FEMA_ENDPOINT)
    except Exception as exc:  # pragma: no cover - network optional
        logger.warning("Failed to fetch FEMA data; writing placeholder", error=str(exc))
        records = []

    aggregates: dict[tuple[str, int], List[float]] = defaultdict(list)
    for record in records:
        zip_code = record.get("zipCode")
        year = record.get("fyDeclared")
        loss = record.get("totalDamage")
        if zip_code and isinstance(year, int) and args.start <= year <= args.end:
            try:
                loss_val = float(loss)
            except (TypeError, ValueError):
                continue
            aggregates[(zip_code, year)].append(loss_val)

    output_records = []
    for (zip_code, year), values in aggregates.items():
        mean_loss = sum(values) / len(values)
        output_records.append({"zip": zip_code, "year": year, "loss_mean": round(mean_loss, 2)})

    save_parquet(output_records, Path(args.output))
    logger.info("FEMA KB download complete", rows=len(output_records))


if __name__ == "__main__":
    main()
