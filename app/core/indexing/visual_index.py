"""Visual index for storing and querying CLIP embeddings."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from loguru import logger


class VisualIndex:
    """
    Storage and search index for CLIP image embeddings.

    Supports fast nearest-neighbor search for semantic image retrieval.
    """

    def __init__(self):
        """Initialize empty visual index."""
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []
        self.tile_id_to_idx: Dict[str, int] = {}

    def build_from_tiles(
        self, tiles: List[Dict], clip_indexer, image_dir: Path, batch_size: int = 32
    ) -> None:
        """
        Build index from imagery tiles metadata.

        Args:
            tiles: List of tile dicts with 'tile_id' and 'image_path' or 'file_path'
            clip_indexer: CLIPIndexer instance for embedding generation
            image_dir: Base directory for image files
            batch_size: Batch size for encoding
        """
        logger.info(f"Building visual index from {len(tiles)} tiles...")

        valid_tiles = []
        image_paths = []

        for tile in tiles:
            tile_id = tile.get("tile_id")
            # Try different possible path fields (uri is the actual field in our data)
            rel_path = (
                tile.get("uri")
                or tile.get("image_path")
                or tile.get("file_path")
                or tile.get("path")
            )

            if not tile_id or not rel_path:
                logger.warning(f"Skipping tile with missing ID or path: {tile}")
                continue

            # Handle different path types:
            # 1. Absolute paths - use as-is
            # 2. Paths starting with "data/raw/imagery/" - strip prefix, use relative to image_dir
            # 3. Paths starting with "data/" - resolve relative to project root
            # 4. Other paths - relative to image_dir
            path_obj = Path(rel_path)
            if path_obj.is_absolute():
                full_path = path_obj
            elif str(rel_path).startswith("data/raw/imagery/"):
                # Strip the common prefix since image_dir already points there
                relative_part = str(rel_path).replace("data/raw/imagery/", "", 1)
                full_path = image_dir / relative_part
            elif str(rel_path).startswith("data/"):
                # Path is relative to project root
                # Resolve image_dir to absolute path first for reliable navigation
                abs_image_dir = image_dir.resolve()
                # Go up 3 levels: data/raw/imagery -> data/raw -> data -> project_root
                project_root = abs_image_dir.parent.parent.parent
                full_path = project_root / rel_path
            else:
                full_path = image_dir / rel_path

            if not full_path.exists():
                logger.warning(f"Image not found: {full_path} (from {rel_path})")
                continue

            valid_tiles.append(tile)
            image_paths.append(full_path)

        if not valid_tiles:
            logger.error("No valid tiles to index!")
            return

        logger.info(f"Encoding {len(valid_tiles)} images in batches of {batch_size}...")
        self.embeddings = clip_indexer.encode_batch(image_paths, batch_size=batch_size)

        self.metadata = valid_tiles
        self.tile_id_to_idx = {
            tile["tile_id"]: idx for idx, tile in enumerate(valid_tiles)
        }

        logger.info(f"Visual index built with {len(self.metadata)} tiles")

    def search(
        self, query_text: str, clip_indexer, top_k: int = 10
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for images semantically similar to a text query.

        Args:
            query_text: Natural language description (e.g., "flooded streets")
            clip_indexer: CLIPIndexer instance for query encoding
            top_k: Number of top results to return

        Returns:
            List of (tile_id, similarity_score, metadata) tuples, sorted by relevance
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            logger.warning("Visual index is empty!")
            return []

        # Encode query
        query_emb = clip_indexer.encode_text(query_text)

        # Compute similarities (dot product, since embeddings are normalized)
        similarities = self.embeddings @ query_emb

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            tile_id = self.metadata[idx]["tile_id"]
            score = float(similarities[idx])
            metadata = self.metadata[idx]
            results.append((tile_id, score, metadata))

        return results

    def search_by_embedding(
        self, query_emb: np.ndarray, top_k: int = 10
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search using a pre-computed embedding vector.

        Args:
            query_emb: Query embedding vector
            top_k: Number of top results

        Returns:
            List of (tile_id, similarity_score, metadata) tuples
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        similarities = self.embeddings @ query_emb
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            tile_id = self.metadata[idx]["tile_id"]
            score = float(similarities[idx])
            metadata = self.metadata[idx]
            results.append((tile_id, score, metadata))

        return results

    def save(self, embeddings_path: Path, metadata_path: Path) -> None:
        """
        Save index to disk.

        Args:
            embeddings_path: Path to save embeddings (.npy)
            metadata_path: Path to save metadata (.json)
        """
        if self.embeddings is None:
            logger.error("Cannot save empty index!")
            return

        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(embeddings_path, self.embeddings)

        # Convert any numpy arrays in metadata to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        clean_metadata = [convert_numpy(m) for m in self.metadata]

        with open(metadata_path, "w") as f:
            json.dump(clean_metadata, f, indent=2)

        logger.info(f"Visual index saved to {embeddings_path} and {metadata_path}")

    @classmethod
    def load(cls, embeddings_path: Path, metadata_path: Path) -> "VisualIndex":
        """
        Load index from disk.

        Args:
            embeddings_path: Path to embeddings file (.npy)
            metadata_path: Path to metadata file (.json)

        Returns:
            Loaded VisualIndex instance
        """
        index = cls()

        index.embeddings = np.load(embeddings_path)

        with open(metadata_path, "r") as f:
            index.metadata = json.load(f)

        index.tile_id_to_idx = {
            tile["tile_id"]: idx for idx, tile in enumerate(index.metadata)
        }

        logger.info(f"Visual index loaded with {len(index.metadata)} tiles")
        return index
