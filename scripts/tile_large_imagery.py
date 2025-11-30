#!/usr/bin/env python
"""Tile large orthomosaic GeoTIFFs into smaller patches for CLIP indexing.

Large imagery files (>100MB) are ineffective for CLIP because:
1. CLIP input is 224x224 - massive downsampling loses all detail
2. Loading multi-GB files is slow and memory-intensive
3. One embedding per huge image provides poor spatial resolution

This script:
- Identifies large GeoTIFF files
- Tiles them into patches (default 1024x1024 pixels)
- Preserves geospatial metadata (each tile knows its coordinates)
- Moves originals to originals/ subdirectory to avoid reprocessing

Usage:
    python scripts/tile_large_imagery.py
    python scripts/tile_large_imagery.py --tile-size 512 --min-size-mb 50
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import re

import numpy as np
from loguru import logger

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import from_bounds
except ImportError:
    raise ImportError("rasterio is required: pip install rasterio")

# Size threshold for tiling (in MB)
DEFAULT_MIN_SIZE_MB = 100

# Default tile size in pixels
DEFAULT_TILE_SIZE = 1024

# Overlap between tiles (to avoid edge artifacts)
DEFAULT_OVERLAP = 128


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in megabytes."""
    return filepath.stat().st_size / (1024 * 1024)


def find_large_tiffs(imagery_dir: Path, min_size_mb: float) -> List[Path]:
    """Find TIFF files larger than the threshold."""
    large_files = []

    for pattern in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
        for filepath in imagery_dir.glob(pattern):
            # Skip files already in originals/ or tiles/ subdirectories
            if "originals" in filepath.parts or "tiles" in filepath.parts:
                continue

            size_mb = get_file_size_mb(filepath)
            if size_mb >= min_size_mb:
                large_files.append(filepath)
                logger.info(f"Found large file: {filepath.name} ({size_mb:.1f} MB)")

    return large_files


def parse_date_from_filename(filename: str) -> Optional[str]:
    """Extract date from various filename patterns."""
    # Pattern 1: 9-10-2017_Ortho_ColorBalance.tif -> 2017-09-10
    match = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", filename)
    if match:
        month, day, year = match.groups()
        try:
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Pattern 2: Area1_post, Area1_pre (default to Harvey dates)
    if "post" in filename.lower():
        return "2017-09-01"
    elif "pre" in filename.lower():
        return "2017-08-24"

    return None


def tile_geotiff(
    src_path: Path,
    output_dir: Path,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[dict]:
    """
    Tile a large GeoTIFF into smaller patches.

    Args:
        src_path: Path to source GeoTIFF
        output_dir: Directory to save tiles
        tile_size: Size of each tile in pixels
        overlap: Overlap between tiles in pixels

    Returns:
        List of metadata dicts for each created tile
    """
    logger.info(f"Tiling {src_path.name}...")

    output_dir.mkdir(parents=True, exist_ok=True)
    tile_metadata = []

    # Extract base info from filename
    base_name = src_path.stem
    date_str = parse_date_from_filename(src_path.name)

    with rasterio.open(src_path) as src:
        # Get image dimensions
        width = src.width
        height = src.height

        logger.info(f"  Image size: {width} x {height} pixels")
        logger.info(f"  CRS: {src.crs}")
        logger.info(f"  Bounds: {src.bounds}")

        # Calculate number of tiles
        step = tile_size - overlap
        n_cols = max(
            1, (width - overlap) // step + (1 if (width - overlap) % step else 0)
        )
        n_rows = max(
            1, (height - overlap) // step + (1 if (height - overlap) % step else 0)
        )

        total_tiles = n_cols * n_rows
        logger.info(f"  Creating {n_cols} x {n_rows} = {total_tiles} tiles")

        tile_count = 0
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                # Calculate window position
                col_off = col_idx * step
                row_off = row_idx * step

                # Adjust for edge tiles
                win_width = min(tile_size, width - col_off)
                win_height = min(tile_size, height - row_off)

                # Skip tiny edge tiles
                if win_width < tile_size // 4 or win_height < tile_size // 4:
                    continue

                window = Window(col_off, row_off, win_width, win_height)

                # Read the tile data
                tile_data = src.read(window=window)

                # Skip mostly empty tiles (more than 50% nodata/zero)
                if np.sum(tile_data == 0) > 0.5 * tile_data.size:
                    continue

                # Calculate georeferenced bounds for this tile
                tile_transform = src.window_transform(window)
                tile_bounds = rasterio.windows.bounds(window, src.transform)

                # Create tile filename
                tile_name = f"{base_name}_tile_{row_idx:03d}_{col_idx:03d}.tif"
                tile_path = output_dir / tile_name

                # Calculate center coordinates
                center_x = (tile_bounds[0] + tile_bounds[2]) / 2
                center_y = (tile_bounds[1] + tile_bounds[3]) / 2

                # Write the tile
                profile = src.profile.copy()
                profile.update(
                    width=win_width,
                    height=win_height,
                    transform=tile_transform,
                    compress="lzw",  # Add compression to reduce file size
                )

                with rasterio.open(tile_path, "w", **profile) as dst:
                    dst.write(tile_data)

                # Record metadata
                metadata = {
                    "tile_id": tile_path.stem,
                    "uri": f"data/raw/imagery/tiles/{tile_name}",
                    "timestamp": f"{date_str}T12:00:00" if date_str else None,
                    "lat": center_y,
                    "lon": center_x,
                    "bbox": list(tile_bounds),
                    "source": base_name,
                    "source_file": src_path.name,
                    "row": row_idx,
                    "col": col_idx,
                    "filename": tile_name,
                }
                tile_metadata.append(metadata)
                tile_count += 1

                if tile_count % 50 == 0:
                    logger.info(f"  Created {tile_count} tiles...")

        logger.info(f"  Created {tile_count} tiles from {src_path.name}")

    return tile_metadata


def move_to_originals(filepath: Path, originals_dir: Path) -> None:
    """Move a file to the originals directory."""
    originals_dir.mkdir(parents=True, exist_ok=True)
    dest = originals_dir / filepath.name

    if dest.exists():
        logger.warning(f"  Original already exists: {dest}")
        return

    logger.info(f"  Moving {filepath.name} to originals/")
    shutil.move(str(filepath), str(dest))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--imagery-dir",
        type=Path,
        default=Path("data/raw/imagery"),
        help="Directory containing imagery files",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"Tile size in pixels (default: {DEFAULT_TILE_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Overlap between tiles in pixels (default: {DEFAULT_OVERLAP})",
    )
    parser.add_argument(
        "--min-size-mb",
        type=float,
        default=DEFAULT_MIN_SIZE_MB,
        help=f"Minimum file size to tile in MB (default: {DEFAULT_MIN_SIZE_MB})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it",
    )
    args = parser.parse_args()

    imagery_dir = args.imagery_dir
    if not imagery_dir.exists():
        logger.error(f"Imagery directory not found: {imagery_dir}")
        return 1

    # Find large files
    large_files = find_large_tiffs(imagery_dir, args.min_size_mb)

    if not large_files:
        logger.info(
            f"No files larger than {args.min_size_mb} MB found in {imagery_dir}"
        )
        return 0

    logger.info(f"\nFound {len(large_files)} large files to tile")

    if args.dry_run:
        logger.info("\n[DRY RUN] Would tile the following files:")
        for f in large_files:
            logger.info(f"  - {f.name} ({get_file_size_mb(f):.1f} MB)")
        return 0

    # Create output directories
    tiles_dir = imagery_dir / "tiles"
    originals_dir = imagery_dir / "originals"

    all_tile_metadata = []

    for filepath in large_files:
        logger.info(f"\nProcessing: {filepath.name}")

        try:
            # Tile the file
            tile_metadata = tile_geotiff(
                filepath,
                tiles_dir,
                tile_size=args.tile_size,
                overlap=args.overlap,
            )
            all_tile_metadata.extend(tile_metadata)

            # Move original to originals/
            move_to_originals(filepath, originals_dir)

        except Exception as e:
            logger.error(f"Failed to process {filepath.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TILING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {len(large_files)}")
    logger.info(f"Tiles created: {len(all_tile_metadata)}")
    logger.info(f"Tiles directory: {tiles_dir}")
    logger.info(f"Originals directory: {originals_dir}")
    logger.info("=" * 60)

    logger.info("\nNext steps:")
    logger.info("1. Run: python scripts/build_imagery_index.py")
    logger.info("   This will regenerate imagery_tiles.parquet with the new tiles")
    logger.info("2. Run: python scripts/build_visual_index.py --force")
    logger.info("   This will build the CLIP embeddings for all tiles")

    return 0


if __name__ == "__main__":
    exit(main())
