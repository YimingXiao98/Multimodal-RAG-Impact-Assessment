#!/usr/bin/env python
"""Test script for visual analysis and multimodal fusion pipeline."""

import sys
from pathlib import Path
from datetime import date

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from loguru import logger


def test_visual_client():
    """Test the VisualAnalysisClient with a sample image."""
    logger.info("=" * 60)
    logger.info("Testing VisualAnalysisClient")
    logger.info("=" * 60)

    from app.core.models.visual_client import VisualAnalysisClient

    client = VisualAnalysisClient()

    # Find a sample image to test
    imagery_dir = Path("data/raw/imagery")
    sample_images = list(imagery_dir.glob("tiles/*.tif"))[:1]

    if not sample_images:
        sample_images = list(imagery_dir.glob("0831-noaa/*.tif"))[:1]

    if not sample_images:
        logger.warning("No sample images found for testing")
        return False

    sample_path = sample_images[0]
    logger.info(f"Testing with image: {sample_path.name}")

    result = client.analyze_single(sample_path, tile_id=sample_path.stem)

    logger.info(f"\nSingle Image Analysis Result:")
    logger.info(f"  Flooding visible: {result.get('flooding_visible', 'N/A')}")
    logger.info(f"  Damage visible: {result.get('structural_damage_visible', 'N/A')}")
    logger.info(f"  Severity: {result.get('severity_pct', 0):.1f}%")
    logger.info(f"  Confidence: {result.get('confidence', 0):.2f}")

    if result.get("flooding_description"):
        logger.info(f"  Flooding: {result['flooding_description']}")
    if result.get("damage_description"):
        logger.info(f"  Damage: {result['damage_description']}")

    return True


def test_batch_analysis():
    """Test batch imagery analysis."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Batch Imagery Analysis")
    logger.info("=" * 60)

    from app.core.models.visual_client import VisualAnalysisClient

    client = VisualAnalysisClient(max_images=3)

    # Get a few sample images
    imagery_dir = Path("data/raw/imagery")
    sample_tiles = []

    for pattern in ["tiles/*.tif", "0831-noaa/*.tif"]:
        for path in list(imagery_dir.glob(pattern))[:2]:
            sample_tiles.append({
                "tile_id": path.stem,
                "uri": str(path),  # Use full path string
            })
        if len(sample_tiles) >= 3:
            break

    if not sample_tiles:
        logger.warning("No sample tiles found")
        return False

    logger.info(f"Analyzing {len(sample_tiles)} tiles...")

    result = client.analyze(
        zip_code="77002",
        time_window={"start": "2017-08-25", "end": "2017-09-05"},
        imagery_tiles=sample_tiles,
        project_root=Path.cwd(),
    )

    overall = result.get("overall_assessment", {})
    logger.info(f"\nOverall Assessment:")
    logger.info(f"  Flood severity: {overall.get('flood_severity_pct', 0):.1f}%")
    logger.info(f"  Structural damage: {overall.get('structural_damage_pct', 0):.1f}%")
    logger.info(f"  Confidence: {overall.get('confidence', 0):.2f}")

    observations = overall.get("key_observations", [])
    if observations:
        logger.info(f"  Key observations:")
        for obs in observations[:3]:
            logger.info(f"    - {obs}")

    return True


def test_fusion_engine():
    """Test the FusionEngine with mock data."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing FusionEngine")
    logger.info("=" * 60)

    from app.core.models.fusion_engine import FusionEngine

    # Mock text analysis result
    text_analysis = {
        "reasoning": "Multiple tweets report flooding in residential areas. 311 calls mention water damage.",
        "estimates": {
            "structural_damage_pct": 35.0,
            "confidence": 0.7,
        },
        "evidence_refs": {
            "tweet_ids": ["T123", "T456"],
            "call_311_ids": ["C789"],
            "sensor_ids": ["S001"],
        },
    }

    # Mock visual analysis result
    visual_analysis = {
        "overall_assessment": {
            "flood_severity_pct": 45.0,
            "structural_damage_pct": 25.0,
            "key_observations": [
                "Standing water visible on streets",
                "Some roof damage apparent",
            ],
            "confidence": 0.8,
        },
        "evidence_refs": {
            "imagery_tile_ids": ["IMG_001", "IMG_002"],
        },
    }

    # Test heuristic fusion
    logger.info("\nTesting heuristic fusion...")
    engine = FusionEngine(use_llm_fusion=False)
    result = engine.fuse(
        text_analysis,
        visual_analysis,
        zip_code="77002",
        time_window={"start": "2017-08-25", "end": "2017-09-05"},
    )

    logger.info(f"Fused Result (heuristic):")
    logger.info(f"  Damage: {result['estimates']['structural_damage_pct']:.1f}%")
    logger.info(f"  Confidence: {result['estimates']['confidence']:.2f}")
    logger.info(f"  Conflicts: {result.get('conflicts', [])}")

    # Test LLM fusion
    logger.info("\nTesting LLM fusion...")
    engine_llm = FusionEngine(use_llm_fusion=True)
    result_llm = engine_llm.fuse(
        text_analysis,
        visual_analysis,
        zip_code="77002",
        time_window={"start": "2017-08-25", "end": "2017-09-05"},
    )

    logger.info(f"Fused Result (LLM):")
    logger.info(f"  Damage: {result_llm['estimates']['structural_damage_pct']:.1f}%")
    logger.info(f"  Confidence: {result_llm['estimates']['confidence']:.2f}")
    if result_llm.get("conflict_resolution"):
        logger.info(f"  Conflict resolution: {result_llm['conflict_resolution']}")

    return True


def test_full_pipeline():
    """Test the complete multimodal pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Full Multimodal Pipeline")
    logger.info("=" * 60)

    from app.core.dataio.storage import DataLocator
    from app.core.dataio.schemas import RAGQuery
    from app.core.retrieval.retriever import Retriever
    from app.core.models.fusion_engine import MultimodalPipelineClient

    locator = DataLocator(Path("data"))
    retriever = Retriever(locator)

    # Create a test query
    query = RAGQuery(
        zip="77002",
        start=date(2017, 8, 25),
        end=date(2017, 9, 5),
        k_tiles=3,
        text_query="What is the flood damage in this area?",
        enable_visual_search=True,
    )

    logger.info(f"Query: ZIP {query.zip}, {query.start} to {query.end}")

    # Retrieve context
    logger.info("Retrieving context...")
    result = retriever.retrieve(query)

    context = {
        "imagery_tiles": result.imagery,
        "tweets": result.tweets,
        "calls": result.calls,
        "sensors": result.sensors,
        "fema": result.fema,
        "sensor_table": "",
        "kb_summary": "",
    }

    logger.info(f"Retrieved: {len(result.imagery)} images, {len(result.tweets)} tweets, {len(result.calls)} calls")

    # Run multimodal pipeline
    logger.info("\nRunning multimodal analysis...")
    pipeline = MultimodalPipelineClient()

    final_result = pipeline.analyze(
        zip_code=query.zip,
        time_window={"start": str(query.start), "end": str(query.end)},
        context=context,
        project_root=Path.cwd(),
    )

    logger.info(f"\nFinal Fused Result:")
    estimates = final_result.get("estimates", {})
    logger.info(f"  Structural damage: {estimates.get('structural_damage_pct', 0):.1f}%")
    logger.info(f"  Flood severity: {estimates.get('flood_severity_pct', 0):.1f}%")
    logger.info(f"  Confidence: {estimates.get('confidence', 0):.2f}")

    if final_result.get("conflicts"):
        logger.info(f"  Conflicts: {final_result['conflicts']}")

    if final_result.get("reasoning"):
        logger.info(f"  Reasoning: {final_result['reasoning'][:200]}...")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test visual analysis pipeline")
    parser.add_argument("--single", action="store_true", help="Test single image only")
    parser.add_argument("--batch", action="store_true", help="Test batch analysis only")
    parser.add_argument("--fusion", action="store_true", help="Test fusion only")
    parser.add_argument("--full", action="store_true", help="Test full pipeline")
    args = parser.parse_args()

    # If no specific test selected, run all
    run_all = not (args.single or args.batch or args.fusion or args.full)

    try:
        if args.single or run_all:
            test_visual_client()

        if args.batch or run_all:
            test_batch_analysis()

        if args.fusion or run_all:
            test_fusion_engine()

        if args.full or run_all:
            test_full_pipeline()

        logger.success("\n✅ All tests passed!")

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

