# DisasterImpactRAG

A multimodal retrieval-augmented generation (RAG) framework for post-disaster impact assessment. This system integrates aerial imagery, social media reports, emergency calls, and sensor data to estimate flood extent and structural damage at the ZIP-code level.

**Case Study**: Hurricane Harvey (2017) in Harris County, TX.

## Features

- **Multimodal Retrieval**: Hybrid dense-sparse text retrieval with cross-encoder reranking, plus CLIP-based visual retrieval
- **Split-Pipeline Architecture**: Separate Text and Visual Analyst modules with heuristic fusion
- **Multiple LLM Backends**: Support for Gemini, OpenAI, Llama, and Qwen models
- **Temporal Reasoning**: Handles asynchronous evidence (e.g., real-time tweets vs. post-event imagery)

## Quickstart

### 1. Configure your environment

```bash
cp .env.example .env
```

Edit `.env` to add your API keys. Set `MODEL_PROVIDER=mock` for offline testing.

### 2. Install dependencies

```bash
conda create -n harvey-rag python=3.11 pip
conda activate harvey-rag
pip install -e .
```

### 3. Prepare data

Place raw data files in `data/raw/` and run the processing scripts:

```bash
# Process 311 emergency calls
python scripts/fetch_311.py --input data/raw/houston_311.csv --output data/processed/311.parquet

# Process rainfall gauges
python scripts/fetch_gauges.py --input data/raw/gauges.csv --output data/processed/gauges.parquet

# Build text corpus and embeddings
python scripts/build_text_corpus.py --data-dir data --output data/processed/text_corpus.jsonl
python scripts/index_embeddings.py \
  --corpus data/processed/text_corpus.jsonl \
  --index data/processed/text_embeddings.faiss \
  --ids data/processed/text_embeddings_ids.json \
  --meta data/processed/text_embeddings_meta.json \
  --model sentence-transformers/all-MiniLM-L6-v2
```

### 4. Run tests

```bash
pytest
```

### 5. Launch the API

```bash
uvicorn app.main:app --reload
```

### 6. Submit a query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"zip":"77002","start":"2017-08-28","end":"2017-09-03","k_tiles":4,"n_text":10}'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Query Input                            │
│              (ZIP code + time window)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Text Retrieval │     │ Visual Retrieval│
│  (BM25 + Dense  │     │   (CLIP-based)  │
│   + Reranker)   │     │                 │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Text Analyst   │     │ Visual Analyst  │
│  (LLM: Gemini/  │     │ (LLM: GPT-4o/   │
│   Llama/Qwen)   │     │  Gemini Flash)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
          ┌─────────────────┐
          │  Fusion Engine  │
          │ (Temporal-aware │
          │   heuristics)   │
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │  Final Output   │
          │ (flood_extent,  │
          │ damage_severity)│
          └─────────────────┘
```

## Project Structure

```
app/
├── core/
│   ├── dataio/       # Data loading utilities
│   ├── eval/         # Evaluation framework
│   ├── indexing/     # Spatial and temporal indexing
│   ├── models/       # LLM clients and fusion logic
│   ├── nlp/          # Query parsing
│   └── retrieval/    # Hybrid text and visual retrieval
├── routers/          # FastAPI endpoints
└── main.py           # Application entry point

scripts/              # Data processing and evaluation scripts
tests/                # Unit tests
config/               # Query configurations
```

## Data Sources

| Source | Description |
|--------|-------------|
| Aerial Imagery | NOAA post-event imagery tiles (Aug 31, 2017) |
| Twitter/X | ~458K filtered tweets during Hurricane Harvey |
| 311 Calls | ~26K emergency service requests |
| Rain Gauges | Harris County Flood Control District sensors |
| FEMA NFIP Claims | Ground truth for damage assessment |

## Model Providers

Configure via `MODEL_PROVIDER` in `.env`:

- **gemini**: Google Gemini (default for production)
- **openai**: OpenAI GPT-4o
- **mock**: Deterministic offline mode for testing

## Evaluation

Run the baseline experiment:

```bash
python scripts/run_baseline_experiment.py \
  --config config/queries_complete_map.json \
  --output data/experiments/results.json
```

## Citation

If you use this code, please cite:

```bibtex
@article{disasterimpactrag2025,
  title={Multimodal Retrieval-Augmented Generation for Disaster Impact Assessment},
  author={...},
  year={2025}
}
```

## License

MIT License
