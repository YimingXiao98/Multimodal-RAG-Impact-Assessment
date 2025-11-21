#!/usr/bin/env python
"""Rebuild all data indexes from raw sources into valid Parquet files.

This script handles:
1. 311 Calls (CSV -> Parquet)
2. Claims (Parquet/CSV -> Parquet)
3. FEMA KB (CSV -> Parquet)
4. Gauges (CSV -> Parquet)
5. Tweets (JSON.GZ -> Parquet) - PARALLELIZED
"""
import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

def process_311(raw_path: Path, output_path: Path):
    """Process 311 calls from CSV."""
    logger.info(f"Processing 311 calls from {raw_path}...")
    try:
        df = pd.read_csv(raw_path, low_memory=False)
        if 'SR TYPE' in df.columns:
            df = df.rename(columns={
                'SR TYPE': 'sr_type_name',
                'SR CREATE DATE': 'timestamp',
                'LATITUDE': 'lat',
                'LONGITUDE': 'lon',
                'ZIP CODE': 'zip'
            })
        df['zip'] = df['zip'].astype(str).str.extract(r'(\d{5})')[0]
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.to_parquet(output_path, index=False)
        logger.success(f"✅ Saved {len(df)} 311 records to {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to process 311: {e}")

def process_gauges(raw_path: Path, output_path: Path):
    """Process gauges from CSV."""
    logger.info(f"Processing gauges from {raw_path}...")
    try:
        df = pd.read_csv(raw_path)
        if 'zipcode' in df.columns:
            df['zip'] = df['zipcode'].astype(str).str.zfill(5)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.to_parquet(output_path, index=False)
        logger.success(f"✅ Saved {len(df)} gauge records to {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to process gauges: {e}")

def process_fema_kb(raw_path: Path, output_path: Path):
    """Process FEMA KB from CSV."""
    logger.info(f"Processing FEMA KB from {raw_path}...")
    try:
        df = pd.read_csv(raw_path)
        if 'zipCode' in df.columns:
            df['zip'] = df['zipCode'].astype(str).str.zfill(5)
        df.to_parquet(output_path, index=False)
        logger.success(f"✅ Saved {len(df)} FEMA KB records to {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to process FEMA KB: {e}")

def process_claims(raw_dir: Path, output_path: Path):
    """Process claims from raw directory."""
    logger.info(f"Processing claims from {raw_dir}...")
    try:
        files = list(raw_dir.glob("**/*.parquet")) + list(raw_dir.glob("**/*.csv"))
        if not files:
            logger.warning("No claims files found")
            return

        dfs = []
        for f in tqdm(files, desc="Processing claims files"):
            try:
                if f.suffix == '.parquet':
                    try:
                        df = pd.read_parquet(f)
                    except:
                        with open(f, 'r') as fh:
                            data = json.load(fh)
                        df = pd.DataFrame(data)
                elif f.suffix == '.csv':
                    df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Skipping {f}: {e}")
        
        if not dfs:
            logger.error("No valid claims data loaded")
            return

        full_df = pd.concat(dfs, ignore_index=True)
        if 'zipCode' in full_df.columns:
            full_df['zip'] = full_df['zipCode'].astype(str).str.zfill(5)
        
        full_df.to_parquet(output_path, index=False)
        logger.success(f"✅ Saved {len(full_df)} claims records to {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to process claims: {e}")

def process_tweet_batch(files: List[Path]) -> pd.DataFrame:
    """Process a batch of tweet files and return a DataFrame."""
    records = []
    for f in files:
        try:
            with gzip.open(f, 'rt', encoding='utf-8') as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        tweet = json.loads(line)
                        record = {
                            'tweet_id': tweet.get('id_str') or str(tweet.get('id')),
                            'text': tweet.get('text') or tweet.get('body'),
                            'timestamp': tweet.get('created_at') or tweet.get('postedTime'),
                            'zip': None,
                            'lat': None,
                            'lon': None
                        }
                        geo = tweet.get('geo') or tweet.get('location')
                        if geo and 'coordinates' in geo:
                            record['lat'] = geo['coordinates'][0]
                            record['lon'] = geo['coordinates'][1]
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue
    
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)

def process_tweets_parallel(raw_dir: Path, output_path: Path):
    """Process tweets in parallel."""
    logger.info(f"Processing tweets from {raw_dir} in PARALLEL...")
    
    files = sorted(list(raw_dir.glob("**/*.json.gz")))
    if not files:
        logger.warning("No tweet files found")
        return

    if output_path.exists():
        logger.warning(f"Removing existing {output_path}...")
        output_path.unlink()

    # Batch files
    batch_size = 50  # Adjust based on file size and memory
    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    
    logger.info(f"Found {len(files)} files, processing in {len(batches)} batches...")
    
    writer = None
    total_records = 0
    
    # Use ProcessPoolExecutor
    # max_workers defaults to number of processors
    with ProcessPoolExecutor() as executor:
        # Submit all batches
        futures = {executor.submit(process_tweet_batch, batch): batch for batch in batches}
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(batches), desc="Processing tweet batches"):
            try:
                df = future.result()
                if df.empty:
                    continue
                
                # Convert to PyArrow table
                table = pa.Table.from_pandas(df)
                
                # Initialize writer with schema from first non-empty batch
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                
                # Write batch
                writer.write_table(table)
                total_records += len(df)
                
            except Exception as e:
                logger.error(f"Batch failed: {e}")
    
    if writer:
        writer.close()
        logger.success(f"✅ Saved {total_records} tweets to {output_path}")
    else:
        logger.warning("⚠️ No tweet records found!")

def main():
    parser = argparse.ArgumentParser(description="Rebuild data indexes")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    process_311(raw_dir / "houston_311.csv", processed_dir / "311.parquet")
    process_gauges(raw_dir / "gauges.csv", processed_dir / "gauges.parquet")
    process_fema_kb(raw_dir / "fema_kb.csv", processed_dir / "fema_kb.parquet")
    process_claims(raw_dir / "FEMA_NFIP_floodClaims_V3", processed_dir / "claims.parquet")
    process_tweets_parallel(raw_dir / "twitter/GNIPHarvey", processed_dir / "tweets.parquet")

if __name__ == "__main__":
    main()
