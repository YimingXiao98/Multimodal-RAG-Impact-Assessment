#!/usr/bin/env python
"""Validate integrity and schema of processed parquet files."""
import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

def validate_parquet(file_path: Path):
    logger.info(f"Checking {file_path.name}...")
    
    if not file_path.exists():
        logger.error(f"❌ File not found: {file_path}")
        return

    try:
        # Use pyarrow to read metadata without loading the whole file (important for 6.5GB tweets)
        parquet_file = pq.ParquetFile(file_path)
        schema = parquet_file.schema
        num_rows = parquet_file.metadata.num_rows
        num_row_groups = parquet_file.num_row_groups
        
        logger.info(f"  ✅ Valid Parquet file")
        logger.info(f"  Rows: {num_rows}")
        logger.info(f"  Row Groups: {num_row_groups}")
        logger.info(f"  Columns: {schema.names}")
        
        # Check for empty file
        if num_rows == 0:
            logger.warning(f"  ⚠️  File is empty (0 rows)")
            
        # Sample read (first 5 rows) to ensure data is readable
        df_head = pd.read_parquet(file_path, engine='pyarrow', filters=[('index', '<', 5)] if 'index' in schema.names else None)
        # Just read head normally if filters tricky, pd.read_parquet usually reads whole file or uses memory map. 
        # Better: use pyarrow table read with limit
        table_head = parquet_file.read_row_group(0, columns=schema.names)
        df_head = table_head.to_pandas().head(5)
        
        logger.info(f"  ✅ Successfully read sample data")
        # print(df_head.head(2))

    except Exception as e:
        logger.error(f"  ❌ CORRUPTED or Invalid: {e}")

def main():
    data_dir = Path("data/processed")
    files = sorted(list(data_dir.glob("*.parquet")))
    
    if not files:
        logger.warning("No parquet files found in data/processed/")
        return

    logger.info(f"Found {len(files)} parquet files to validate.")
    
    for p in files:
        print("-" * 60)
        validate_parquet(p)
    print("-" * 60)

if __name__ == "__main__":
    main()
