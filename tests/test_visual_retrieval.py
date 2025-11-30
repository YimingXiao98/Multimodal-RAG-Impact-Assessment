#!/usr/bin/env python
"""Test script for visual retrieval integration."""

import sys
from pathlib import Path
from datetime import date

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from app.core.dataio.storage import DataLocator
from app.core.dataio.schemas import RAGQuery
from app.core.retrieval.retriever import Retriever
from app.core.retrieval.visual_retriever import VisualRetriever, infer_visual_queries


def test_visual_retriever_standalone():
    """Test the visual retriever in isolation."""
    logger.info("=" * 60)
    logger.info("Testing VisualRetriever (standalone)")
    logger.info("=" * 60)

    locator = DataLocator(Path("data"))
    visual = VisualRetriever(locator, lazy_load=False)

    if not visual.is_available:
        logger.error("Visual index not available!")
        return False

    # Test basic search
    test_queries = [
        "flooded streets",
        "damaged buildings",
        "aerial view of flooding",
    ]

    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        results = visual.search(query, top_k=3)
        for r in results:
            logger.info(
                f"  {r['tile_id']}: score={r['similarity_score']:.3f}, "
                f"source={r.get('source', 'unknown')}"
            )

    return True


def test_visual_query_inference():
    """Test automatic visual query inference from text."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Visual Query Inference")
    logger.info("=" * 60)

    test_texts = [
        "What is the flood damage in this area?",
        "Are there any damaged roofs visible?",
        "Hurricane Harvey impact assessment",
        "Show me destroyed buildings and debris",
    ]

    for text in test_texts:
        inferred = infer_visual_queries(text)
        logger.info(f"\n  Input: '{text}'")
        logger.info(f"  Inferred: {inferred}")


def test_integrated_retriever():
    """Test the full integrated retriever with visual search."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Integrated Retriever (Text + Visual)")
    logger.info("=" * 60)

    locator = DataLocator(Path("data"))
    retriever = Retriever(locator)

    logger.info(f"Visual search enabled: {retriever.visual_search_enabled}")

    # Create a test query
    query = RAGQuery(
        zip="77002",
        start=date(2017, 8, 25),
        end=date(2017, 9, 5),
        k_tiles=6,
        text_query="What is the flood damage in this area?",
        enable_visual_search=True,
    )

    logger.info(f"\nQuery: zip={query.zip}, dates={query.start} to {query.end}")
    logger.info(f"Text query: '{query.text_query}'")

    # Perform retrieval
    result = retriever.retrieve(query)

    logger.info(f"\nResults:")
    logger.info(f"  Imagery: {len(result.imagery)} tiles")
    logger.info(f"  Tweets: {len(result.tweets)}")
    logger.info(f"  311 Calls: {len(result.calls)}")
    logger.info(f"  Sensors: {len(result.sensors)}")
    logger.info(f"  FEMA: {len(result.fema)}")

    # Show imagery details
    if result.imagery:
        logger.info("\nImagery tiles:")
        for i, tile in enumerate(result.imagery[:5]):
            visual_match = tile.get("visual_match", False)
            score = tile.get("similarity_score", "N/A")
            matched_query = tile.get("matched_query", "spatial")
            logger.info(
                f"  {i+1}. {tile.get('tile_id', 'unknown')}: "
                f"visual={visual_match}, score={score}, match={matched_query}"
            )

    return True


def test_explicit_visual_query():
    """Test with an explicit visual query."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Explicit Visual Query")
    logger.info("=" * 60)

    locator = DataLocator(Path("data"))
    retriever = Retriever(locator)

    # Query with explicit visual search term
    query = RAGQuery(
        zip="77002",
        start=date(2017, 8, 25),
        end=date(2017, 9, 5),
        k_tiles=6,
        visual_query="damaged rooftops aerial view",  # Explicit visual query
        enable_visual_search=True,
    )

    logger.info(f"Visual query: '{query.visual_query}'")

    result = retriever.retrieve(query)

    logger.info(f"\nImagery returned: {len(result.imagery)} tiles")
    for tile in result.imagery[:3]:
        logger.info(
            f"  - {tile.get('tile_id')}: score={tile.get('similarity_score', 'N/A')}"
        )

    return True


if __name__ == "__main__":
    logger.info("Testing Visual Retrieval Integration")
    logger.info("=" * 60)

    try:
        test_visual_retriever_standalone()
        test_visual_query_inference()
        test_integrated_retriever()
        test_explicit_visual_query()

        logger.success("\n✅ All tests passed!")
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

