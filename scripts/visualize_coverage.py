#!/usr/bin/env python
"""Visualize spatial coverage of sensors and imagery data.

Usage:
    python scripts/visualize_coverage.py
    python scripts/visualize_coverage.py --output NOTES/coverage_map.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_sensor_data(sensor_path: Path):
    """Load sensor data from CSV."""
    df = pd.read_csv(sensor_path)
    
    # Get unique sensor locations (one point per sensor, not per reading)
    sensors = df.groupby(['sensor_id', 'zipcode']).first().reset_index()
    
    # For this CSV, we don't have lat/lon, so we'll need to get ZIP centroids
    # This is a simplified approach - you may want to add actual coordinates
    return sensors


def load_imagery_data(imagery_dir: Path):
    """Load imagery tile data."""
    import glob
    
    parquet_files = glob.glob(str(imagery_dir / "imagery_tiles*.parquet"))
    
    if not parquet_files:
        print("⚠️  No imagery parquet files found")
        return None
    
    print(f"Loading {len(parquet_files)} imagery file(s)...")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    
    return df


def get_zip_centroids():
    """Approximate Houston area ZIP code centroids."""
    # These are approximate centroids for Houston area ZIP codes
    # You can replace with exact values from pgeocode or other sources
    zip_coords = {
        '77002': (29.7604, -95.3698),  # Downtown
        '77058': (29.5586, -95.0919),  # Clear Lake
        '77089': (29.6116, -95.2699),  # South Houston
        '77096': (29.7002, -95.5347),  # Westchase
        '77546': (29.5327, -95.1788),  # Friendswood
        '77573': (29.4989, -95.4383),  # League City
        '77581': (29.4547, -95.3083),  # Pearland
        '77584': (29.4241, -95.4024),  # Pearland South
        '77586': (29.5952, -95.1508),  # Seabrook
    }
    return zip_coords


def create_coverage_map(sensors, imagery, output_path):
    """Create a map showing sensor and imagery coverage."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    zip_coords = get_zip_centroids()
    
    # Plot imagery tiles
    if imagery is not None and len(imagery) > 0:
        imagery_zips = imagery['zip'].unique() if 'zip' in imagery.columns else []
        imagery_coords = [(zip_coords.get(str(z).zfill(5), (None, None))) 
                          for z in imagery_zips]
        imagery_coords = [c for c in imagery_coords if c[0] is not None]
        
        if imagery_coords:
            img_lats, img_lons = zip(*imagery_coords)
            ax.scatter(img_lons, img_lats, s=500, c='blue', alpha=0.3, 
                      marker='s', label=f'Imagery Coverage ({len(imagery_coords)} ZIPs)', 
                      edgecolors='darkblue', linewidths=2)
            
            # Annotate imagery ZIPs
            for lat, lon in imagery_coords:
                zip_code = [k for k, v in zip_coords.items() if v == (lat, lon)][0]
                ax.annotate(zip_code, (lon, lat), fontsize=8, ha='center', 
                           va='bottom', color='darkblue', weight='bold')
    
    # Plot sensors
    if sensors is not None and len(sensors) > 0:
        sensor_zips = sensors['zipcode'].unique()
        sensor_coords = [(zip_coords.get(str(z).zfill(5), (None, None))) 
                         for z in sensor_zips]
        sensor_coords = [c for c in sensor_coords if c[0] is not None]
        
        # Count sensors per ZIP
        sensor_counts = sensors.groupby('zipcode')['sensor_id'].nunique().to_dict()
        
        if sensor_coords:
            sens_lats, sens_lons = zip(*sensor_coords)
            
            # Size markers by number of sensors
            sizes = [sensor_counts.get(int([k for k, v in zip_coords.items() 
                                            if v == (lat, lon)][0]), 1) * 100 
                     for lat, lon in sensor_coords]
            
            ax.scatter(sens_lons, sens_lats, s=sizes, c='red', alpha=0.6, 
                      marker='o', label=f'Sensors ({len(sensor_coords)} ZIPs)', 
                      edgecolors='darkred', linewidths=2)
            
            # Annotate sensor ZIPs with counts
            for i, (lat, lon) in enumerate(sensor_coords):
                zip_code = [k for k, v in zip_coords.items() if v == (lat, lon)][0]
                count = sensor_counts.get(int(zip_code), 1)
                ax.annotate(f'{zip_code}\n({count} sensors)', (lon, lat), 
                           fontsize=7, ha='center', va='top', color='darkred')
    
    # Find overlaps
    if imagery is not None and sensors is not None:
        imagery_zips_set = set(str(z).zfill(5) for z in imagery['zip'].unique()) if 'zip' in imagery.columns else set()
        sensor_zips_set = set(str(z).zfill(5) for z in sensors['zipcode'].unique())
        
        overlap_zips = imagery_zips_set & sensor_zips_set
        overlap_coords = [zip_coords.get(z, (None, None)) for z in overlap_zips]
        overlap_coords = [c for c in overlap_coords if c[0] is not None]
        
        if overlap_coords:
            ovr_lats, ovr_lons = zip(*overlap_coords)
            ax.scatter(ovr_lons, ovr_lats, s=800, c='green', alpha=0.2, 
                      marker='*', label=f'Both Sensors & Imagery ({len(overlap_coords)} ZIPs)',
                      edgecolors='darkgreen', linewidths=3)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Spatial Coverage: Sensors vs. Imagery\nHurricane Harvey Impact Assessment', 
                 fontsize=14, weight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add Houston label
    ax.annotate('Houston', (-95.3698, 29.7604), fontsize=12, 
               ha='center', va='bottom', color='black', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved coverage map to: {output_path}")
    
    return fig


def print_coverage_summary(sensors, imagery):
    """Print text summary of coverage."""
    print("\n" + "="*60)
    print("COVERAGE SUMMARY")
    print("="*60)
    
    if sensors is not None:
        sensor_zips = set(str(z).zfill(5) for z in sensors['zipcode'].unique())
        sensor_count = sensors.groupby('zipcode')['sensor_id'].nunique().to_dict()
        print(f"\n📍 Sensors: {len(sensor_zips)} ZIP codes, {len(sensors)} total readings")
        print("   ZIP codes:", sorted(sensor_zips))
        print("   Sensors per ZIP:", {str(k).zfill(5): v for k, v in sensor_count.items()})
    
    if imagery is not None and 'zip' in imagery.columns:
        imagery_zips = set(str(z).zfill(5) for z in imagery['zip'].unique())
        print(f"\n🛰️  Imagery: {len(imagery_zips)} ZIP codes, {len(imagery)} tiles")
        print("   ZIP codes:", sorted(imagery_zips))
    
    if sensors is not None and imagery is not None and 'zip' in imagery.columns:
        sensor_zips = set(str(z).zfill(5) for z in sensors['zipcode'].unique())
        imagery_zips = set(str(z).zfill(5) for z in imagery['zip'].unique())
        
        overlap = sensor_zips & imagery_zips
        sensor_only = sensor_zips - imagery_zips
        imagery_only = imagery_zips - sensor_zips
        
        print(f"\n✅ Overlap: {len(overlap)} ZIP codes have BOTH sensors and imagery")
        if overlap:
            print("   ", sorted(overlap))
        
        print(f"\n⚠️  Sensor-only: {len(sensor_only)} ZIP codes")
        if sensor_only:
            print("   ", sorted(sensor_only))
        
        print(f"\n⚠️  Imagery-only: {len(imagery_only)} ZIP codes")
        if imagery_only:
            print("   ", sorted(imagery_only))
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, 
                        default=Path('NOTES/coverage_map.png'),
                        help='Output path for the visualization')
    parser.add_argument('--sensors', type=Path,
                        default=Path('data/raw/gauges.csv'),
                        help='Path to sensor CSV file')
    parser.add_argument('--imagery', type=Path,
                        default=Path('data/processed'),
                        help='Path to imagery data directory')
    args = parser.parse_args()
    
    print("Loading data...")
    sensors = load_sensor_data(args.sensors)
    imagery = load_imagery_data(args.imagery)
    
    print_coverage_summary(sensors, imagery)
    
    create_coverage_map(sensors, imagery, args.output)
    
    print(f"\n💡 Tip: Open {args.output} to see the visualization")


if __name__ == '__main__':
    main()
