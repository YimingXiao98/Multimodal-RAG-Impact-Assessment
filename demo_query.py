"""Demo script to run a RAG query without starting the API server."""
import json
import os
from datetime import date
from pathlib import Path

from app.core.dataio.schemas import RAGQuery
from app.core.dataio.storage import DataLocator
from app.core.models.vlm_client import VLMClient
from app.core.retrieval.context_packager import package_context
from app.core.retrieval.retriever import Retriever
from app.core.retrieval.selector import select_candidates


def main():
    # 1. Setup
    print("Initializing system...")
    base_dir = Path("data")
    if not base_dir.exists():
        print(f"Error: {base_dir} not found. Run from repository root.")
        return

    locator = DataLocator(base_dir)
    retriever = Retriever(locator)
    vlm_client = VLMClient() # Will use env var or default to warning

    # 2. Define Query
    # Harvey made landfall around Aug 25, 2017.
    query = RAGQuery(
        zip="77002", # Downtown Houston
        start=date(2017, 8, 26),
        end=date(2017, 8, 30),
        k_tiles=4,
        n_text=10,
        text_query="Describe flood impacts in downtown Houston"
    )
    
    print(f"\nRunning query for ZIP {query.zip} between {query.start} and {query.end}...")
    
    # 3. Retrieve
    print("Retrieving evidence...")
    result = retriever.retrieve(query)
    print(f"  - Found {len(result.imagery)} imagery tiles")
    print(f"  - Found {len(result.tweets)} tweets")
    print(f"  - Found {len(result.calls)} 311 calls")
    print(f"  - Found {len(result.sensors)} sensors")

    # 4. Select & Package
    candidates = select_candidates(result, query.k_tiles, query.n_text)
    context = package_context(candidates)

    # 5. Generate Answer
    print("Generating answer...")
    time_window = {"start": str(query.start), "end": str(query.end)}
    answer = vlm_client.infer(
        zip_code=query.zip,
        time_window=time_window,
        imagery_tiles=context.get("imagery_tiles", []),
        text_snippets=context.get("text_snippets", []),
        sensor_table=context.get("sensor_table", ""),
        kb_summary=context.get("kb_summary", ""),
        tweets=context.get("tweets", []),
        calls=context.get("calls", []),
        sensors=context.get("sensors", []),
        fema=context.get("fema", []),
    )

    # 6. Output
    print("\n" + "="*50)
    print("RAG ANSWER")
    print("="*50)
    print(json.dumps(answer, indent=2))
    print("="*50)


if __name__ == "__main__":
    main()
