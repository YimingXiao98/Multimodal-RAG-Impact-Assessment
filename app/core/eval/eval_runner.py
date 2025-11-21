"""Evaluation runner for batch assessments."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from ..dataio.schemas import RAGQuery
from ..dataio.storage import DataLocator
from ..models.vlm_client import VLMClient
from ..retrieval.context_packager import package_context
from ..retrieval.retriever import Retriever
from ..retrieval.selector import select_candidates
from .eval_retrieval import RetrievalEvaluator
from .ground_truth import ClaimsGroundTruth


@dataclass
class EvalConfig:
    queries: List[RAGQuery]
    provider: Optional[str] = None
    ground_truth: Optional[str] = "claims"
    retrieval_gt: Optional[str] = None


class EvalRunner:
    def __init__(self, locator: DataLocator, provider: Optional[str] = None, ground_truth: Optional[str] = "claims", retrieval_gt: Optional[str] = None) -> None:
        self.locator = locator
        self.retriever = Retriever(locator)
        self.client = VLMClient(provider)
        self.gt = ClaimsGroundTruth(locator.table_path("claims")) if ground_truth == "claims" else None
        
        self.retrieval_evaluator = None
        if retrieval_gt:
            gt_path = Path(retrieval_gt)
            if gt_path.exists():
                self.retrieval_evaluator = RetrievalEvaluator(json.loads(gt_path.read_text()))
            else:
                logger.warning("Retrieval ground truth file not found", path=retrieval_gt)

    def run(self, queries: List[RAGQuery]) -> Dict[str, Path]:
        metrics_rows = []
        abs_errors: List[float] = []
        sq_errors: List[float] = []
        for query in queries:
            logger.info("Evaluating query", zip=query.zip)
            result = self.retriever.retrieve(query)
            
            retrieval_metrics = {}
            if self.retrieval_evaluator:
                retrieval_metrics = self.retrieval_evaluator.evaluate(query.zip, result)
            
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
            row = {
                "zip": query.zip,
                "start": str(query.start),
                "end": str(query.end),
                "pred_damage_pct": float(answer["estimates"].get("structural_damage_pct", 0.0)),
                "confidence": float(answer["estimates"].get("confidence", 0.0)),
            }
            row.update(retrieval_metrics)
            if self.gt:
                truth = self.gt.score(query.zip, query.start, query.end)
                row.update(
                    actual_damage_pct=truth["damage_pct"],
                    claim_count=truth["claim_count"],
                    total_claim_amount=round(truth["total_amount"], 2),
                )
                diff = row["pred_damage_pct"] - truth["damage_pct"]
                abs_errors.append(abs(diff))
                sq_errors.append(diff ** 2)
                row["abs_error"] = round(abs(diff), 2)
            metrics_rows.append(row)
        summary = {}
        if abs_errors:
            summary = {
                "mae": round(sum(abs_errors) / len(abs_errors), 3),
                "rmse": round(sqrt(sum(sq_errors) / len(sq_errors)), 3),
                "count": len(abs_errors),
            }
        output = {"summary": summary, "records": metrics_rows}
        output_path = self.locator.processed / "eval_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))
        return {"results": output_path}


def main(config_path: str) -> None:
    config = json.loads(Path(config_path).read_text())
    queries = [RAGQuery(**q) for q in config["queries"]]
    locator = DataLocator(Path(config.get("data_dir", "data")))
    runner = EvalRunner(
        locator, 
        provider=config.get("provider"), 
        ground_truth=config.get("ground_truth", "claims"),
        retrieval_gt=config.get("retrieval_gt")
    )
    outputs = runner.run(queries)
    logger.info("Evaluation complete", outputs={k: str(v) for k, v in outputs.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
