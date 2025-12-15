"""Run Exp0 Baseline experiment for retrieval evaluation.

This script runs the current RAG pipeline (with BGE reranker if enabled) on
validation queries and saves results as exp0_baseline.json.

Reuses components from run_multi_judge_eval.py but focuses on baseline capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from loguru import logger

from app.config import get_settings
from app.core.dataio.schemas import RAGQuery
from app.core.dataio.storage import DataLocator
from app.core.models.split_client import SplitPipelineClient
from app.core.retrieval.context_packager import package_context
from app.core.retrieval.retriever import Retriever
from app.core.retrieval.selector import select_candidates
from app.core.retrieval.selector import select_candidates
from app.core.eval.ground_truth import (
    ClaimsGroundTruth,
    FloodDepthGroundTruth,
    PDEGroundTruth,
)


def compute_config_hash(config: dict) -> str:
    """Compute a short hash of the config for reproducibility tracking."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


def format_answer_text(answer: Dict[str, Any]) -> str:
    """Format the VLM answer dict as readable text."""
    parts = []

    if summary := answer.get("natural_language_summary"):
        parts.append(f"Summary: {summary}")

    if estimates := answer.get("estimates"):
        parts.append(f"Damage Estimate: {estimates.get('flood_impact_pct', 'N/A')}%")
        parts.append(f"Confidence: {estimates.get('confidence', 'N/A')}")
        if roads := estimates.get("roads_affected"):
            parts.append(f"Roads Affected: {roads}")

    if refs := answer.get("evidence_refs"):
        if tweet_ids := refs.get("tweet_ids"):
            parts.append(f"Cited Tweets: {', '.join(tweet_ids[:5])}")
        if call_ids := refs.get("call_311_ids"):
            parts.append(f"Cited 311 Calls: {', '.join(call_ids[:5])}")

    if reasoning := answer.get("reasoning"):
        parts.append(f"Reasoning: {reasoning}")

    return "\n".join(parts) if parts else str(answer)


def extract_retrieval_metadata(context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata about retrieval results for analysis."""
    tweets = context.get("tweets", [])
    calls = context.get("calls", [])
    gauges = context.get("gauges", [])

    # Extract IDs with fallback chain
    tweet_ids = []
    for t in tweets:
        tid = t.get("doc_id") or t.get("tweet_id") or t.get("id") or ""
        tweet_ids.append(tid)

    call_ids = []
    for c in calls:
        cid = c.get("doc_id") or c.get("record_id") or c.get("id") or ""
        call_ids.append(cid)

    # Extract sensor IDs from gauges
    sensor_ids = []
    for g in gauges:
        sid = (
            g.get("sensor_id")
            or g.get("doc_id", "").replace("gauge_", "").split("_")[0]
            or ""
        )
        if sid and sid not in sensor_ids:
            sensor_ids.append(sid)

    return {
        "tweet_count": len(tweets),
        "tweet_ids": tweet_ids,
        "gauge_count": len(gauges),
        "sensor_ids": sensor_ids,
        "call_311_count": len(calls),
        "call_311_ids": call_ids,
        "imagery_tile_count": len(context.get("imagery_tiles", [])),
        "has_sensor_data": len(gauges) > 0 or context.get("sensor_table") is not None,
    }


def run_baseline_experiment(
    config_path: str,
    output_path: str,
    experiment_name: str = "baseline",
    limit: Optional[int] = None,
    run_judge: bool = False,
    no_captions: bool = False,
    no_visual: bool = False,
    judge_name: str = "gpt-4o-mini",
    shuffle: bool = False,
    seed: int = 42,
) -> None:
    """Run baseline retrieval experiment and save results."""

    # Load config
    config = json.loads(Path(config_path).read_text())
    if no_visual:
        logger.info("Override: Disabling visual analysis")
        config["enable_visual"] = False

    queries = [RAGQuery(**q) for q in config["queries"]]

    if shuffle:
        logger.info(f"Shuffling queries with seed {seed}")
        random.seed(seed)
        random.shuffle(queries)

    if limit:
        queries = queries[:limit]
        logger.info(f"Limiting to first {limit} queries (after shuffle={shuffle})")

    # Get current settings
    settings = get_settings()

    logger.info(f"Running {experiment_name} with {len(queries)} queries")
    logger.info(f"Reranker enabled: {settings.enable_reranker}")
    logger.info(f"Geo-boost enabled: {settings.enable_geo_boost}")
    logger.info(f"Filter negative captions: {settings.filter_negative_captions}")

    # Initialize components
    locator = DataLocator(Path(config.get("data_dir", "data")))
    retriever = Retriever(locator)
    # Ensure consistent model usage: default to gemini if not specified
    # This ensures all experiments use the same model regardless of config
    provider = config.get("provider", "gemini")
    if provider != "gemini" and provider != "openai":
        logger.warning(f"Unknown provider '{provider}', defaulting to 'gemini'")
        provider = "gemini"
    
    client = SplitPipelineClient(
        provider=provider,
        enable_visual=config.get("enable_visual", True),
        use_llm_fusion=False,  # Use heuristic fusion with confirmation logic
    )
    logger.info(f"Using provider: {provider} for {experiment_name}")
    # Use flood depth as ground truth (more accurate than claims)
    flood_depth_path = locator.base_dir / "processed" / "flood_depth_by_zip.json"
    if flood_depth_path.exists():
        gt = FloodDepthGroundTruth(flood_depth_path)
        logger.info("Using FloodDepthGroundTruth (meters)")
    else:
        gt = ClaimsGroundTruth(locator.table_path("claims"))
        logger.warning("Flood depth not found, falling back to ClaimsGroundTruth")

    # Initialize PDE Ground Truth for Damage Severity
    pde_path = (
        locator.base_dir / "processed" / "damage_by_zip.json"
    )  # verifying file path?
    # Actually, previous interaction said data/processed/pde_by_zip.json
    pde_path = locator.base_dir / "processed" / "pde_by_zip.json"
    pde_gt = None
    if pde_path.exists():
        pde_gt = PDEGroundTruth(pde_path)
        logger.info("Using PDEGroundTruth for Damage Severity")
    else:
        logger.warning(
            f"PDE data not found at {pde_path}, Damage Severity metric will be empty"
        )

    # Optional: Initialize judge for quality check
    judge_client = None
    if run_judge:
        try:
            from app.core.eval.multi_judge import JudgeClient

            judge_client = JudgeClient(judge_name)
            logger.info(f"✓ Initialized judge: {judge_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize judge: {e}")

    # Experiment metadata
    import os
    experiment_meta = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "config_path": str(config_path),
        "config_hash": compute_config_hash(config),
        "settings": {
            "enable_reranker": settings.enable_reranker,
            "enable_geo_boost": settings.enable_geo_boost,
            "filter_negative_captions": settings.filter_negative_captions,
            "dense_top_k": settings.dense_top_k,
            # Model configuration for reproducibility
            "model_provider": provider,
            "gemini_model": os.getenv("GEMINI_MODEL", "gemini-1.5-flash") if provider == "gemini" else None,
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini") if provider == "openai" else None,
            "visual_model_provider": client.visual_provider if hasattr(client, 'visual_provider') else provider,
            "gemini_vision_model": os.getenv("GEMINI_VISION_MODEL") if provider == "gemini" else None,
            "openai_vision_model": os.getenv("OPENAI_VISION_MODEL") if provider == "openai" else None,
        },
        "total_queries": len(queries),
    }

    # Process queries
    records = []
    for i, query in enumerate(queries, 1):
        logger.info(
            f"[{i}/{len(queries)}] Processing ZIP {query.zip} ({query.start} to {query.end})"
        )

        try:
            # Step 1: Retrieve context
            result = retriever.retrieve(query)
            candidates = select_candidates(result, query.k_tiles, query.n_text)
            context = package_context(candidates)

            if no_captions:
                context["captions"] = []

            # Step 2: Generate VLM response
            answer = client.infer(
                zip_code=query.zip,
                time_window={"start": str(query.start), "end": str(query.end)},
                imagery_tiles=context["imagery_tiles"],
                text_snippets=context["text_snippets"],
                sensor_table=context["sensor_table"],
                kb_summary=context["kb_summary"],
                tweets=context.get("tweets"),
                calls=context.get("calls"),
                captions=context.get("captions"),
                gauges=context.get("gauges"),
                project_root=locator.base_dir.parent,
            )

            # Step 3: Get ground truth
            truth = gt.score(query.zip, query.start, query.end)

            # The provided `evaluate` method seems to be intended for a class like `Evaluator`.
            # Since it's provided as a replacement for the existing evaluation logic,
            # and the original code calculates metrics at the end, I will assume
            # the user wants to replace the final metric calculation with this new logic,
            # adapted to fit the existing `run_baseline_experiment` function's flow.
            # The `self` and `results` parameters indicate it's meant for a class method
            # that processes a list of results.
            # I will integrate the logic for calculating extent and damage metrics
            # into the summary statistics section, using the `records` list.

            # Build record
            record = {
                "query_id": i,
                "query": {
                    "zip": query.zip,
                    "start_date": str(query.start),
                    "end_date": str(query.end),
                    "comment": getattr(query, "comment", ""),
                },
                "retrieval_metadata": extract_retrieval_metadata(context),
                "model_response": {
                    "raw": answer,
                    "formatted": format_answer_text(answer),
                    "flood_impact_pct": float(
                        answer.get("estimates", {}).get("flood_impact_pct", 0.0)
                    ),
                    "confidence": float(
                        answer.get("estimates", {}).get("confidence", 0.0)
                    ),
                    # Add new fields for dual metrics if they exist in the answer
                    "flood_extent_pct": float(
                        answer.get("estimates", {}).get("flood_extent_pct", 0.0)
                    ),
                    "damage_severity_pct": float(
                        answer.get("estimates", {}).get("damage_severity_pct", 0.0)
                    ),
                },
                "ground_truth": {
                    "mean_depth_m": truth.get("mean_depth_m", 0.0),
                    "max_depth_m": truth.get("max_depth_m", 0.0),
                    "flooded_pct": truth.get("flooded_pct", 0.0),
                    "claim_count": truth.get("claim_count", 0),
                    "total_claim_amount": round(truth.get("total_amount", 0.0), 2),
                    "pde_damage_score": (
                        pde_gt.score(query.zip).get("mean_pde", 0.0) if pde_gt else 0.0
                    ),  # Raw mean_pde (0-1): average damage per building in the ZIP
                },
            }

            # Optional: Add judge score
            if judge_client:
                try:
                    query_text = f"Assess impact for {query.zip} from {query.start} to {query.end}"
                    eval_result = judge_client.evaluate(
                        query=query_text,
                        context=context,
                        answer=format_answer_text(answer),
                        query_params={
                            "zip": query.zip,
                            "start": str(query.start),
                            "end": str(query.end),
                        },
                    )
                    record["judge_score"] = {
                        "judge": judge_name,
                        "faithfulness": eval_result["faithfulness"],
                        "relevance": eval_result["relevance"],
                    }
                except Exception as e:
                    logger.warning(f"Judge evaluation failed: {e}")
                    record["judge_score"] = {"error": str(e)}

            records.append(record)

            # Log progress - compare flood_impact_pct vs flooded_pct (both %)
            pred_impact = record["model_response"]["flood_impact_pct"]
            actual_flooded = truth.get("flooded_pct", 0.0)
            logger.info(
                f"  Predicted: {pred_impact:.1f}%, Ground Truth: {actual_flooded:.1f}%, "
                f"Diff: {abs(pred_impact - actual_flooded):.1f}%"
            )

        except Exception as e:
            logger.error(f"Failed to process query {i}: {e}")
            records.append(
                {
                    "query_id": i,
                    "query": {
                        "zip": query.zip,
                        "start_date": str(query.start),
                        "end_date": str(query.end),
                    },
                    "error": str(e),
                }
            )

        # Save intermediate results
        intermediate_path = Path(output_path).with_suffix(".intermediate.json")
        intermediate_path.write_text(
            json.dumps(
                {"metadata": experiment_meta, "records": records},
                indent=2,
                default=str,
            )
        )

    # Calculate summary statistics
    successful_records = [r for r in records if "error" not in r]
    if successful_records:
        # 1. Flood Extent (Hazard) vs FEMA Depth Grid (flooded_pct)
        extent_preds = [
            r["model_response"]["flood_extent_pct"] for r in successful_records
        ]
        extent_targets = [r["ground_truth"]["flooded_pct"] for r in successful_records]
        extent_errors = [abs(p - a) for p, a in zip(extent_preds, extent_targets)]

        # 2. Damage Severity (Consequence) vs PDE (normalized 0-100%)
        damage_preds = [
            r["model_response"]["damage_severity_pct"] for r in successful_records
        ]
        # pde_damage_score is stored as normalized damage_pct (0-100%), already in percentage
        damage_targets = [
            r["ground_truth"].get("pde_damage_score", 0.0)
            * 100.0  # Convert from 0-1 to 0-100
            for r in successful_records
        ]
        damage_errors = [abs(p - a) for p, a in zip(damage_preds, damage_targets)]

        n = len(successful_records)

        # Helper metrics
        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        experiment_meta["summary_stats"] = {
            "successful_queries": n,
            "failed_queries": len(records) - n,
            # Dual Metrics
            "extent_mae": round(safe_mean(extent_errors), 2),
            "damage_mae": round(safe_mean(damage_errors), 2),
            # Legacy for compatibility
            "mae": round(safe_mean(extent_errors), 2),
        }

    # Save final results
    output_data = {
        "metadata": experiment_meta,
        "records": records,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_data, indent=2, default=str))

    logger.success(f"✓ Saved experiment results to {output_file}")
    if "summary_stats" in experiment_meta:
        stats = experiment_meta["summary_stats"]
        logger.info(
            f"  Extent MAE: {stats.get('extent_mae', 'N/A')}% | Damage MAE: {stats.get('damage_mae', 'N/A')}%"
        )
        logger.info(
            f"  Successful: {stats.get('successful_queries', len(records))}/{len(records)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run baseline retrieval experiment (Exp0)"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to query config JSON",
    )
    parser.add_argument(
        "--output",
        default="data/experiments/exp0_baseline.json",
        help="Output path for experiment results",
    )
    parser.add_argument(
        "--name",
        default="exp0_baseline",
        help="Experiment name for metadata",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of queries to run",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run LLM-as-a-judge evaluation",
    )
    parser.add_argument(
        "--no_captions",
        action="store_true",
        help="Exclude image captions from text context",
    )
    parser.add_argument(
        "--no_visual",
        action="store_true",
        help="Disable visual analysis (override config)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle queries before limiting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )

    args = parser.parse_args()

    run_baseline_experiment(
        config_path=args.config,
        output_path=args.output,
        limit=args.limit,
        experiment_name=args.name,
        run_judge=args.judge,
        no_captions=args.no_captions,
        no_visual=args.no_visual,
        shuffle=args.shuffle,
        seed=args.seed,
    )
