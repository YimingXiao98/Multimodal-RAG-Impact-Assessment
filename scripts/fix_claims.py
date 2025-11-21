import pandas as pd
from pathlib import Path
from loguru import logger
import json

def process_claims(raw_dir: Path, output_path: Path):
    """Process claims from raw directory."""
    logger.info(f"Processing claims from {raw_dir}...")
    try:
        files = list(raw_dir.glob("**/*.parquet")) + list(raw_dir.glob("**/*.csv"))
        if not files:
            logger.warning("No claims files found")
            return

        dfs = []
        for f in files:
            logger.info(f"Reading {f}...")
            try:
                if f.suffix == '.parquet':
                    try:
                        df = pd.read_parquet(f)
                    except:
                        with open(f, 'r') as fh:
                            data = json.load(fh)
                        df = pd.DataFrame(data)
                elif f.suffix == '.csv':
                    df = pd.read_csv(f, low_memory=False)
                
                logger.info(f"  Loaded {len(df)} rows from {f.name}")
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Skipping {f}: {e}")
        
        if not dfs:
            logger.error("No valid claims data loaded")
            return

        full_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total rows before processing: {len(full_df)}")
        
        # Normalize columns
        if 'zipCode' in full_df.columns:
            full_df['zip'] = full_df['zipCode'].astype(str).str.zfill(5)
        
        # Ensure timestamp is datetime
        if 'dateOfLoss' in full_df.columns:
             full_df['timestamp'] = pd.to_datetime(full_df['dateOfLoss'], errors='coerce')
        elif 'timestamp' in full_df.columns:
             full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], errors='coerce')
             
        # Ensure lat/lon
        if 'latitude' in full_df.columns and 'longitude' in full_df.columns:
            full_df['lat'] = full_df['latitude']
            full_df['lon'] = full_df['longitude']
            
        # Filter valid
        full_df = full_df.dropna(subset=['timestamp'])
        
        # Convert object columns to string to avoid parquet type errors
        for col in full_df.columns:
            if full_df[col].dtype == 'object':
                full_df[col] = full_df[col].astype(str)
        
        if output_path.exists():
            output_path.unlink()
            
        full_df.to_parquet(output_path, index=False)
        logger.success(f"✅ Saved {len(full_df)} claims records to {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to process claims: {e}")

if __name__ == "__main__":
    process_claims(
        Path("data/raw/FEMA_NFIP_floodClaims_V3"),
        Path("data/processed/claims.parquet")
    )
