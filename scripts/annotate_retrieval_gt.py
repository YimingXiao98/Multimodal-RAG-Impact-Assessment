#!/usr/bin/env python
"""Interactive annotation tool for creating retrieval ground truth.

Usage:
    python scripts/annotate_retrieval_gt.py --output data/examples/retrieval_gt.json
"""
import argparse
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from app.core.dataio.schemas import RAGQuery
from app.core.dataio.storage import DataLocator
from app.core.retrieval.retriever import Retriever


def display_query(query: RAGQuery):
    """Display query details."""
    print("\n" + "="*80)
    print(f"QUERY: ZIP {query.zip}, {query.start} to {query.end}")
    print("="*80)


def display_imagery(tiles: list):
    """Display imagery tiles for annotation."""
    print(f"\nFound {len(tiles)} imagery tiles:")
    for i, tile in enumerate(tiles, 1):
        print(f"  [{i}] {tile['tile_id']} - {tile.get('timestamp', 'N/A')}")
        print(f"      URI: {tile.get('uri', 'N/A')}")


def display_text(docs: list, doc_type: str):
    """Display text documents for annotation."""
    print(f"\nFound {len(docs)} {doc_type}:")
    for i, doc in enumerate(docs[:20], 1):  # Show first 20
        if doc_type == "tweets":
            print(f"  [{i}] Tweet {doc.get('tweet_id', 'N/A')}")
            print(f"      Time: {doc.get('timestamp', 'N/A')}")
            # Show full text (or at least 200 chars)
            text = doc.get('text', 'N/A')
            if len(text) > 200:
                print(f"      {text[:200]}...")
            else:
                print(f"      {text}")
        elif doc_type == "311 calls":
            print(f"  [{i}] Call {doc.get('record_id', 'N/A')}")
            print(f"      Time: {doc.get('timestamp', 'N/A')}")
            print(f"      Category: {doc.get('category', 'N/A')}")
            # Show description
            desc = doc.get('description', 'N/A')
            if len(desc) > 150:
                print(f"      {desc[:150]}...")
            else:
                print(f"      {desc}")


def get_user_selections(items: list, item_type: str) -> list:
    """Prompt user to select relevant items."""
    if not items:
        return []
    
    print(f"\nEnter the numbers of relevant {item_type} (comma-separated, or 'all'/'none'):")
    user_input = input("> ").strip().lower()
    
    if user_input == "none" or user_input == "":
        return []
    
    if user_input == "all":
        return list(range(len(items)))
    
    try:
        indices = [int(x.strip()) - 1 for x in user_input.split(",")]
        return [i for i in indices if 0 <= i < len(items)]
    except ValueError:
        print("Invalid input. Skipping...")
        return []


def annotate_query(query: RAGQuery, retriever: Retriever) -> dict:
    """Annotate a single query."""
    display_query(query)
    
    # Run retrieval
    print("\nRunning retrieval...")
    result = retriever.retrieve(query)
    
    ground_truth = {"imagery": [], "text": []}
    
    # Annotate imagery
    display_imagery(result.imagery)
    selected = get_user_selections(result.imagery, "imagery tiles")
    ground_truth["imagery"] = [result.imagery[i]["tile_id"] for i in selected]
    
    # Annotate tweets
    display_text(result.tweets, "tweets")
    selected = get_user_selections(result.tweets, "tweets")
    ground_truth["text"].extend([str(result.tweets[i].get("tweet_id")) for i in selected])
    
    # Annotate 311 calls
    display_text(result.calls, "311 calls")
    selected = get_user_selections(result.calls, "311 calls")
    ground_truth["text"].extend([str(result.calls[i].get("record_id")) for i in selected])
    
    print(f"\n✅ Annotated: {len(ground_truth['imagery'])} imagery, {len(ground_truth['text'])} text docs")
    
    return ground_truth


def main():
    parser = argparse.ArgumentParser(description="Annotate retrieval ground truth")
    parser.add_argument("--output", default="data/examples/retrieval_gt.json", 
                        help="Output path for ground truth JSON")
    parser.add_argument("--queries", help="Path to queries JSON file (optional)")
    args = parser.parse_args()
    
    # Initialize
    locator = DataLocator(Path("data"))
    retriever = Retriever(locator)
    
    # Define or load queries
    if args.queries:
        # User explicitly specified a query file
        with open(args.queries) as f:
            query_dicts = json.load(f)
        queries = [RAGQuery(**q) for q in query_dicts]
    else:
        # Try to load default annotation queries file
        default_queries_path = Path("data/examples/annotation_queries.json")
        if default_queries_path.exists():
            print(f"Loading queries from {default_queries_path}")
            with open(default_queries_path) as f:
                query_dicts = json.load(f)
            queries = [RAGQuery(**q) for q in query_dicts]
        else:
            # Fall back to hardcoded queries
            print("Using default hardcoded queries (annotation_queries.json not found)")
            queries = [
                RAGQuery(zip="77002", start=date(2017, 8, 26), end=date(2017, 8, 30), k_tiles=10, n_text=20),
                RAGQuery(zip="77096", start=date(2017, 8, 25), end=date(2017, 9, 5), k_tiles=10, n_text=20),
            ]
    
    # Load existing ground truth if it exists
    output_path = Path(args.output)
    if output_path.exists():
        print(f"Loading existing ground truth from {output_path}")
        ground_truth = json.loads(output_path.read_text())
    else:
        ground_truth = {}
    
    # Annotate each query
    print(f"\nStarting annotation for {len(queries)} queries...")
    print("(Press Ctrl+C to save and exit early)\n")
    
    try:
        for i, query in enumerate(queries, 1):
            query_id = f"{query.zip}_{query.start}_{query.end}"
            
            if query_id in ground_truth:
                print(f"\nSkipping {query_id} (already annotated)")
                continue
            
            print(f"\n[{i}/{len(queries)}] Annotating {query_id}")
            annotations = annotate_query(query, retriever)
            ground_truth[query_id] = annotations
            
            # Save after each query
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(ground_truth, indent=2))
            print(f"💾 Saved to {output_path}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    # Final save
    output_path.write_text(json.dumps(ground_truth, indent=2))
    print(f"\n✅ Final ground truth saved to {output_path}")
    print(f"   Total queries annotated: {len(ground_truth)}")


if __name__ == "__main__":
    main()
