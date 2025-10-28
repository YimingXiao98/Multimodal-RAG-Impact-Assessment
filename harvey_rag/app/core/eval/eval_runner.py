"""Evaluation runner for batch assessments."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from loguru import logger

from ..dataio.schemas import RAGQuery
from ..dataio.storage import DataLocator
from ..models.vlm_client import VLMClient
from ..retrieval.context_packager import package_context
from ..retrieval.retriever import Retriever
from ..retrieval.selector import select_candidates


@dataclass
class EvalConfig:
    queries: List[RAGQuery]
    mode: str = "full"


class EvalRunner:
    def __init__(self, locator: DataLocator, provider: str = "mock") -> None:
        self.locator = locator
        self.retriever = Retriever(locator)
        self.client = VLMClient(provider)

    def run(self, queries: List[RAGQuery]) -> Dict[str, Path]:
        metrics_rows = []
        for query in queries:
            logger.info("Evaluating query", zip=query.zip)
            result = self.retriever.retrieve(query)
            candidates = select_candidates(result, query.k_tiles, query.n_text)
            context = package_context(candidates)
            answer = self.client.infer(
                zip_code=query.zip,
                time_window={"start": str(query.start), "end": str(query.end)},
                imagery_tiles=context["imagery_tiles"],
                text_snippets=context["text_snippets"],
                sensor_table=context["sensor_table"],
                kb_summary=context["kb_summary"],
            )
            metrics_rows.append(
                {
                    "zip": query.zip,
                    "structural_damage_pct": answer["estimates"]["structural_damage_pct"],
                    "confidence": answer["estimates"]["confidence"],
                }
            )
        output_path = self.locator.processed / "eval_metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics_rows, indent=2))
        return {"metrics": output_path}


def main(config_path: str) -> None:
    config = json.loads(Path(config_path).read_text())
    queries = [RAGQuery(**q) for q in config["queries"]]
    locator = DataLocator(Path(config.get("data_dir", "data")))
    runner = EvalRunner(locator)
    outputs = runner.run(queries)
    logger.info("Evaluation complete", outputs={k: str(v) for k, v in outputs.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
