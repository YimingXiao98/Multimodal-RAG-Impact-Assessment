#!/usr/bin/env python3
"""Generate held-out test configuration from ZIPs not in development set."""

import json
from pathlib import Path
from datetime import datetime

def main():
    # Load all ZIPs with damage ground truth
    pde_path = Path('data/processed/pde_by_zip.json')
    pde_data = json.loads(pde_path.read_text())
    all_zips = set(pde_data.keys())
    
    # Load current dev set (50 queries)
    config_path = Path('config/queries_50_mixed.json')
    config = json.loads(config_path.read_text())
    dev_zips = set(q['zip'] for q in config.get('queries', []))
    
    # Calculate held-out test set
    test_zips = sorted(all_zips - dev_zips)
    
    print(f"Development ZIPs: {len(dev_zips)}")
    print(f"Held-out test ZIPs: {len(test_zips)}")
    print(f"Split: {len(dev_zips)}/{len(test_zips)} ({len(dev_zips)/(len(dev_zips)+len(test_zips))*100:.1f}% / {len(test_zips)/(len(dev_zips)+len(test_zips))*100:.1f}%)")
    
    # Generate test queries with standard time window (Hurricane Harvey peak)
    # Using Aug 25 - Aug 31 (6 days covering peak flooding)
    queries = []
    for zip_code in test_zips:
        queries.append({
            "zip": zip_code,
            "start_date": "2017-08-25",
            "end_date": "2017-08-31",
            "comment": "held-out test"
        })
    
    # Create config
    test_config = {
        "name": "held_out_test_99",
        "description": "Held-out test set: 99 ZIPs not seen during development",
        "created": datetime.now().isoformat(),
        "split_info": {
            "dev_set": "queries_50_mixed.json",
            "dev_size": len(dev_zips),
            "test_size": len(test_zips),
            "overlap": 0
        },
        "queries": queries
    }
    
    # Save
    output_path = Path('config/queries_heldout_test_99.json')
    output_path.write_text(json.dumps(test_config, indent=2))
    print(f"\nSaved to: {output_path}")
    
    # Also generate full config (all 139 ZIPs for reference)
    all_queries = config.get('queries', []).copy()
    for zip_code in test_zips:
        all_queries.append({
            "zip": zip_code,
            "start_date": "2017-08-25", 
            "end_date": "2017-08-31",
            "comment": "held-out test"
        })
    
    full_config = {
        "name": "full_139_zips",
        "description": "Full dataset: 50 dev + 99 held-out test ZIPs",
        "created": datetime.now().isoformat(),
        "queries": all_queries
    }
    
    full_path = Path('config/queries_full_139.json')
    full_path.write_text(json.dumps(full_config, indent=2))
    print(f"Saved full config to: {full_path}")

if __name__ == "__main__":
    main()

