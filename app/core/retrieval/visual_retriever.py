"""Visual retriever using CLIP-based semantic image search."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from loguru import logger

from ..dataio.storage import DataLocator
from ..indexing.visual_index import VisualIndex
from ..indexing.clip_indexer import CLIPIndexer


class VisualRetriever:
    """
    Retriever for semantic image search using CLIP embeddings.

    Complements the spatial-based imagery lookup with semantic understanding.
    For example, can find "flooded streets" or "damaged roofs" across the dataset.
    """

    def __init__(
        self,
        locator: DataLocator,
        model_name: str = "ViT-B/32",
        lazy_load: bool = True,
    ):
        """
        Initialize visual retriever.

        Args:
            locator: Data locator for finding index files
            model_name: CLIP model variant to use
            lazy_load: If True, defer loading CLIP until first query
        """
        self.locator = locator
        self.model_name = model_name

        # Index file paths
        self.index_dir = locator.processed / "indexes"
        self.embeddings_path = self.index_dir / "visual_embeddings.npy"
        self.metadata_path = self.index_dir / "visual_metadata.json"

        self._index: Optional[VisualIndex] = None
        self._clip: Optional[CLIPIndexer] = None
        self._loaded = False

        if not lazy_load:
            self._load()

    def _load(self) -> None:
        """Load the visual index and CLIP model."""
        if self._loaded:
            return

        if not self.embeddings_path.exists() or not self.metadata_path.exists():
            logger.warning(
                f"Visual index not found at {self.index_dir}. "
                "Run scripts/build_visual_index.py to create it."
            )
            return

        logger.info("Loading visual index and CLIP model...")
        self._index = VisualIndex.load(self.embeddings_path, self.metadata_path)
        self._clip = CLIPIndexer(model_name=self.model_name)
        self._loaded = True
        logger.info(f"Visual retriever ready with {len(self._index.metadata)} tiles")

    @property
    def is_available(self) -> bool:
        """Check if visual search is available (index exists)."""
        return self.embeddings_path.exists() and self.metadata_path.exists()

    def search(
        self,
        query: str,
        top_k: int = 10,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        zip_code: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search for images semantically matching the query.

        Args:
            query: Natural language description (e.g., "flooded streets", "damaged roof")
            top_k: Maximum number of results to return
            start: Optional start datetime to filter results
            end: Optional end datetime to filter results
            zip_code: Optional zip code to filter results

        Returns:
            List of tile metadata dicts with similarity scores
        """
        self._load()

        if self._index is None or self._clip is None:
            logger.warning("Visual index not loaded, returning empty results")
            return []

        # Perform semantic search
        results = self._index.search(query, self._clip, top_k=top_k * 3)  # Over-fetch for filtering

        # Filter by temporal/spatial constraints if provided
        filtered = []
        for tile_id, score, metadata in results:
            # Apply temporal filter
            if start or end:
                ts = metadata.get("timestamp")
                if ts:
                    try:
                        tile_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if start and tile_time < start:
                            continue
                        if end and tile_time > end:
                            continue
                    except (ValueError, TypeError):
                        pass  # Skip temporal filtering if timestamp is invalid

            # Apply zip filter
            if zip_code:
                tile_zip = metadata.get("zip")
                if tile_zip and str(tile_zip) != str(zip_code):
                    continue

            # Add score to metadata
            result = dict(metadata)
            result["similarity_score"] = score
            result["visual_match"] = True  # Flag as visual search result
            filtered.append(result)

            if len(filtered) >= top_k:
                break

        return filtered

    def search_multi(
        self,
        queries: List[str],
        top_k: int = 10,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        zip_code: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search with multiple query terms and aggregate results.

        Args:
            queries: List of search terms (e.g., ["flooding", "structural damage"])
            top_k: Maximum total results to return
            start: Optional start datetime
            end: Optional end datetime
            zip_code: Optional zip filter

        Returns:
            Deduplicated list of tiles matching any query
        """
        if not queries:
            return []

        seen_tile_ids = set()
        all_results = []

        # Search with each query and aggregate
        per_query_k = max(top_k // len(queries), 5)

        for query in queries:
            results = self.search(
                query=query,
                top_k=per_query_k,
                start=start,
                end=end,
                zip_code=zip_code,
            )

            for result in results:
                tile_id = result.get("tile_id")
                if tile_id and tile_id not in seen_tile_ids:
                    seen_tile_ids.add(tile_id)
                    result["matched_query"] = query
                    all_results.append(result)

        # Sort by similarity score and limit
        all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return all_results[:top_k]


def infer_visual_queries(text_query: str) -> List[str]:
    """
    Extract visual search concepts from a natural language query.

    This is a simple heuristic-based extractor. For production,
    consider using an LLM to identify visual concepts.

    Args:
        text_query: Natural language query

    Returns:
        List of visual search terms
    """
    # Disaster/damage-related visual concepts
    visual_keywords = {
        "flood": ["flooded streets", "standing water", "submerged"],
        "water": ["flooded streets", "standing water"],
        "damage": ["damaged buildings", "structural damage", "destroyed"],
        "roof": ["damaged roof", "roof damage"],
        "debris": ["debris on streets", "fallen trees"],
        "destroyed": ["destroyed buildings", "collapsed structures"],
        "submerged": ["submerged vehicles", "flooded area"],
        "hurricane": ["storm damage", "wind damage"],
        "collapsed": ["collapsed building", "structural failure"],
        "inundation": ["flooded streets", "water inundation"],
    }

    query_lower = text_query.lower()
    visual_queries = []

    for keyword, concepts in visual_keywords.items():
        if keyword in query_lower:
            visual_queries.extend(concepts)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for q in visual_queries:
        if q not in seen:
            seen.add(q)
            result.append(q)

    return result[:3]  # Limit to top 3 concepts

