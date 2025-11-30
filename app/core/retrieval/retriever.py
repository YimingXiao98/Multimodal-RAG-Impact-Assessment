"""Hybrid retriever facade with optional visual (CLIP) search."""
from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict

from loguru import logger

from ...config import get_settings
from ..dataio.schemas import RAGQuery
from ..dataio.storage import DataLocator
from .hybrid_text_retriever import HybridTextRetriever
from .visual_retriever import VisualRetriever, infer_visual_queries
from .types import RetrievalResult


class Retriever:
    """
    Entry point for retrieval. Combines text-based hybrid search with
    optional visual (CLIP-based) semantic search.
    """

    def __init__(self, locator: DataLocator, settings=None) -> None:
        self.locator = locator
        self.settings = settings or get_settings()
        self._text_retriever = HybridTextRetriever(locator, self.settings)

        # Initialize visual retriever (lazy-loaded)
        self._visual_retriever: Optional[VisualRetriever] = None
        self._init_visual_retriever()

    def _init_visual_retriever(self) -> None:
        """Initialize visual retriever if index is available."""
        try:
            visual = VisualRetriever(self.locator, lazy_load=True)
            if visual.is_available:
                self._visual_retriever = visual
                logger.info("Visual retriever enabled")
            else:
                logger.info(
                    "Visual index not found; visual search disabled. "
                    "Run scripts/build_visual_index.py to enable."
                )
        except Exception as e:
            logger.warning(f"Failed to initialize visual retriever: {e}")

    def retrieve(self, query: RAGQuery) -> RetrievalResult:
        """
        Retrieve relevant context for the query.

        Performs hybrid text retrieval, and optionally adds
        semantically-matched imagery via CLIP if enabled.
        """
        # Get base results from text retriever (includes spatial imagery)
        result = self._text_retriever.retrieve(query)

        # Optionally enhance with visual search
        if query.enable_visual_search and self._visual_retriever:
            visual_imagery = self._get_visual_imagery(query, result.imagery)
            if visual_imagery:
                # Merge visual results with spatial results, deduplicating
                result.imagery = self._merge_imagery(result.imagery, visual_imagery)

        return result

    def _get_visual_imagery(
        self, query: RAGQuery, existing_imagery: List[Dict]
    ) -> List[Dict]:
        """Get semantically-matched imagery using CLIP."""
        if not self._visual_retriever:
            return []

        # Determine visual queries
        visual_queries = []

        if query.visual_query:
            # Explicit visual query provided
            visual_queries = [query.visual_query]
        elif query.text_query:
            # Auto-infer visual concepts from text query
            visual_queries = infer_visual_queries(query.text_query)

        if not visual_queries:
            # Default disaster-related visual search
            visual_queries = ["flood damage", "flooded streets"]

        # Build time range for filtering
        start_dt = datetime.combine(query.start, datetime.min.time())
        end_dt = datetime.combine(query.end, datetime.max.time())

        # Perform visual search
        try:
            results = self._visual_retriever.search_multi(
                queries=visual_queries,
                top_k=query.k_tiles,
                start=start_dt,
                end=end_dt,
                zip_code=query.zip,
            )
            logger.debug(
                f"Visual search returned {len(results)} results for queries: {visual_queries}"
            )
            return results
        except Exception as e:
            logger.error(f"Visual search failed: {e}")
            return []

    def _merge_imagery(
        self, spatial: List[Dict], visual: List[Dict]
    ) -> List[Dict]:
        """
        Merge spatial and visual imagery results, deduplicating by tile_id.

        Visual results are marked with 'visual_match=True' for transparency.
        """
        seen_ids = set()
        merged = []

        # Add spatial results first (they're from the query's specific location)
        for tile in spatial:
            tile_id = tile.get("tile_id")
            if tile_id:
                seen_ids.add(tile_id)
            merged.append(tile)

        # Add visual results that aren't duplicates
        for tile in visual:
            tile_id = tile.get("tile_id")
            if tile_id and tile_id not in seen_ids:
                seen_ids.add(tile_id)
                merged.append(tile)

        return merged

    @property
    def visual_search_enabled(self) -> bool:
        """Check if visual search is available."""
        return self._visual_retriever is not None and self._visual_retriever.is_available
