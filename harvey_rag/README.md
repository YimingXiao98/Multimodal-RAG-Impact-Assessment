# Harvey Multimodal RAG Skeleton

This repository provides a production-ready skeleton for a multimodal retrieval-augmented generation (RAG) pipeline focused on Hurricane Harvey impact assessment in Harris County, TX.

## Quickstart

1. Copy `.env.example` to `.env` and update values as needed.
2. Install dependencies:

```bash
pip install -e .
```

3. Start the API:

```bash
uvicorn app.main:app --reload
```

4. Query the API:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"zip":"77002","start":"2017-08-28","end":"2017-09-03","k_tiles":4,"n_text":10}'
```

## Data Workflow

Place raw files in `data/raw/`. Use the scripts in `scripts/` to normalize into Parquet/GeoParquet outputs stored in `data/processed/`. Index artifacts (FAISS, H3 references) are stored in `data/indexes/`.

Example datasets with 10 toy rows are available under `data/examples/`; tests rely on these fixtures.

## Scripts

Scripts accept `--input`, `--output`, `--start`, and `--end` flags and log progress with `loguru`:

- `scripts/fetch_311.py`
- `scripts/fetch_gauges.py`
- `scripts/fetch_fema_kb.py`
- `scripts/index_imagery.py`

## Model Providers

The system supports mock, OpenAI, and Gemini providers. Configure via the `MODEL_PROVIDER` environment variable (`mock` by default). API keys are read from environment variables loaded via `.env`.

## Evaluation

Run end-to-end evaluation on example data:

```bash
python -m app.core.eval.eval_runner --config data/examples/eval_config.json
```

Outputs include metrics CSVs and confusion matrices saved under `data/processed/`.

## Testing

```bash
pytest
```

## Repository Layout

```
harvey_rag/
  app/
  data/
  scripts/
  tests/
```

Each module contains docstrings and TODOs indicating where production integrations should be implemented.
