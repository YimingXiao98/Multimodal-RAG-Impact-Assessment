#!/usr/bin/env python
"""Extract metadata from imagery TIFF files and create parquet index.

This script scans imagery directories, extracts geospatial metadata from TIFF files,
and creates a properly formatted parquet file for the spatial index.

Usage:
    python scripts/build_imagery_index.py
    python scripts/build_imagery_index.py --imagery-dir data/raw/imagery --output data/processed/imagery_tiles.parquet
"""
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from loguru import logger

try:
    import rasterio
    from rasterio.errors import RasterioIOError
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.warning("rasterio not installed - using filename-based metadata extraction only")


def parse_date_from_filename(filename: str) -> Optional[str]:
    """Extract date from various filename patterns."""
    # Pattern 1: 9-10-2017_Ortho_ColorBalance.tif -> 2017-09-10
    match = re.search(r'(\d{1,2})-(\d{1,2})-(\d{4})', filename)
    if match:
        month, day, year = match.groups()
        try:
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass
    
    # Pattern 2: 20170831bC0953600w294630n.tif -> 2017-08-31
    match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if match:
        year, month, day = match.groups()
        try:
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass
    
    # Pattern 3: Area1_post, Area1_pre (default to Harvey dates)
    if 'post' in filename.lower():
        return '2017-09-01'  # After Harvey
    elif 'pre' in filename.lower():
        return '2017-08-24'  # Before Harvey
    
    return None


def extract_bbox_from_tif(filepath: Path) -> Optional[tuple]:
    """Extract bounding box from TIFF file using rasterio."""
    if not HAS_RASTERIO:
        return None
    
    try:
        with rasterio.open(filepath) as src:
            bounds = src.bounds
            # Return as (min_lon, min_lat, max_lon, max_lat)
            return (bounds.left, bounds.bottom, bounds.right, bounds.top)
    except Exception as e:
        logger.debug(f"Could not read {filepath.name}: {e}")
        return None


def extract_centroid_from_filename(filename: str) -> Optional[tuple]:
    """Extract approximate lat/lon from NOAA filename pattern."""
    # Pattern: 20170831bC0953600w294630n.tif
    # C = lon prefix, w/e for west/east
    # Second number with n/s for lat north/south
    
    match = re.search(r'C(\d+)(w|e)(\d+)(n|s)', filename.lower())
    if match:
        lon_str, lon_dir, lat_str, lat_dir = match.groups()
        
        # Convert to decimal degrees
        lon = float(lon_str) / 10000.0
        if lon_dir == 'w':
            lon = -lon
        
        lat = float(lat_str) / 10000.0
        if lat_dir == 's':
            lat = -lat
        
        return (lat, lon)
    
    return None


def extract_metadata(filepath: Path, imagery_dir: Path, data_root: Path) -> Dict:
    """Extract all available metadata from an imagery file."""
    filename = filepath.name
    relative_path = filepath.relative_to(imagery_dir)
    
    # Generate tile_id
    tile_id = filepath.stem
    
    # Extract date
    date_str = parse_date_from_filename(filename)
    timestamp = f"{date_str}T12:00:00" if date_str else None
    
    # Extract bounding box (prefer rasterio, fallback to filename)
    bbox = extract_bbox_from_tif(filepath)
    
    # Extract centroid
    if bbox:
        lat = (bbox[1] + bbox[3]) / 2
        lon = (bbox[0] + bbox[2]) / 2
    else:
        centroid = extract_centroid_from_filename(filename)
        if centroid:
            lat, lon = centroid
            # Create approximate bbox (0.01 degree ~ 1km)
            bbox = (lon - 0.005, lat - 0.005, lon + 0.005, lat + 0.005)
        else:
            # Default to Houston area if no coordinates found
            lat, lon = 29.7604, -95.3698
            bbox = (lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01)
    
    # Infer ZIP from coordinates (approximate Houston ZIPs)
    zip_code = infer_zip_from_coords(lat, lon)
    
    # Generate URI relative to project root (e.g., "data/raw/imagery/tiles/file.tif")
    # We need paths that start with "data/" for the visual index to resolve correctly
    try:
        rel_to_data = filepath.relative_to(data_root)
        uri = f"data/{rel_to_data}"
    except ValueError:
        # Fallback if filepath is not under data_root
        uri = f"data/raw/imagery/{relative_path}"
    
    metadata = {
        'tile_id': tile_id,
        'uri': uri,
        'timestamp': timestamp,
        'lat': lat,
        'lon': lon,
        'bbox': list(bbox) if bbox else None,
        'zip': zip_code,
        'source': str(relative_path.parent) if relative_path.parent != Path('.') else 'root',
        'filename': filename
    }
    
    return metadata


def infer_zip_from_coords(lat: float, lon: float) -> str:
    """Rough approximation of Houston ZIP codes from coordinates."""
    # This is a very rough approximation - you should use pgeocode for accuracy
    # Houston downtown area
    if 29.7 < lat < 29.8 and -95.4 < lon < -95.3:
        return '77002'
    # Clear Lake area
    elif 29.5 < lat < 29.6 and -95.1 < lon < -95.0:
        return '77058'
    # South Houston
    elif 29.6 < lat < 29.7 and -95.3 < lon < -95.2:
        return '77089'
    # Default
    else:
        return '77002'


def scan_imagery_directory(imagery_dir: Path, data_root: Path) -> List[Dict]:
    """Scan directory for TIFF files and extract metadata.
    
    Skips files in 'originals/' subdirectory (large files moved during tiling).
    """
    logger.info(f"Scanning {imagery_dir} for imagery files...")
    
    # Find all TIFF files
    all_tif_files = list(imagery_dir.glob('**/*.tif')) + list(imagery_dir.glob('**/*.tiff'))
    
    # Filter out files in 'originals' directory (these are the large untiled originals)
    tif_files = [f for f in all_tif_files if 'originals' not in f.parts]
    
    skipped = len(all_tif_files) - len(tif_files)
    if skipped > 0:
        logger.info(f"Skipping {skipped} files in originals/ directory")
    
    logger.info(f"Found {len(tif_files)} TIFF files to index")
    
    metadata_list = []
    for i, filepath in enumerate(tif_files, 1):
        if i % 100 == 0:
            logger.info(f"Processing {i}/{len(tif_files)}...")
        
        try:
            metadata = extract_metadata(filepath, imagery_dir, data_root)
            metadata_list.append(metadata)
        except Exception as e:
            logger.warning(f"Failed to process {filepath.name}: {e}")
    
    return metadata_list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imagery-dir', type=Path, 
                        default=Path('data/raw/imagery'),
                        help='Directory containing imagery TIFF files')
    parser.add_argument('--data-root', type=Path,
                        default=Path('data'),
                        help='Root data directory (for generating relative URIs)')
    parser.add_argument('--output', type=Path,
                        default=Path('data/processed/imagery_tiles.parquet'),
                        help='Output parquet file path')
    args = parser.parse_args()
    
    if not args.imagery_dir.exists():
        logger.error(f"Imagery directory not found: {args.imagery_dir}")
        return 1
    
    # Scan and extract metadata
    metadata_list = scan_imagery_directory(args.imagery_dir, args.data_root)
    
    if not metadata_list:
        logger.error("No imagery files found or processed")
        return 1
    
    # Create DataFrame
    df = pd.DataFrame(metadata_list)
    
    # Summary
    logger.info("="*60)
    logger.info("IMAGERY INDEX SUMMARY")
    logger.info("="*60)
    logger.info(f"Total tiles: {len(df)}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Unique sources: {df['source'].nunique()}")
    logger.info(f"ZIP codes: {sorted(df['zip'].unique())}")
    logger.info("="*60)
    
    # Save to parquet
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    
    logger.success(f"✅ Imagery index saved to {args.output}")
    
    # Also save a CSV for easy inspection
    csv_path = args.output.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"📄 Also saved CSV to {csv_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
