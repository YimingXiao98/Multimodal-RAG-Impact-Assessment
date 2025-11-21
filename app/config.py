"""Application configuration utilities."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Runtime configuration loaded from environment variables."""

    model_provider: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    default_time_window_days: int = 7
    data_dir: Path = Path("./data")
    hybrid_corpus_path: Path = Path("data/processed/text_corpus.jsonl")
    hybrid_index_path: Path = Path("data/processed/text_embeddings.faiss")
    hybrid_ids_path: Path = Path("data/processed/text_embeddings_ids.json")
    hybrid_meta_path: Path = Path("data/processed/text_embeddings_meta.json")
    enable_reranker: bool = False
    dense_top_k: int = 1000

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        data: dict[str, object] = {}

        def _assign(field: str, env_name: str, parser):
            raw = os.getenv(env_name)
            if raw is None or raw == "":
                return
            data[field] = parser(raw)

        _assign("model_provider", "MODEL_PROVIDER", str)
        _assign("openai_api_key", "OPENAI_API_KEY", str)
        _assign("gemini_api_key", "GEMINI_API_KEY", str)
        _assign("default_time_window_days", "DEFAULT_TIME_WINDOW_DAYS", int)
        _assign("data_dir", "DATA_DIR", lambda v: Path(v))
        _assign("hybrid_corpus_path", "HARVEY_CORPUS_PATH", Path)
        _assign("hybrid_index_path", "HARVEY_EMBED_INDEX_PATH", Path)
        _assign("hybrid_ids_path", "HARVEY_EMBED_IDS_PATH", Path)
        _assign("hybrid_meta_path", "HARVEY_EMBED_META_PATH", Path)
        _assign("enable_reranker", "HARVEY_ENABLE_RERANKER",
                lambda v: v.lower() in {"1", "true", "yes", "on"})
        _assign("dense_top_k", "DENSE_TOP_K", int)

        return cls(**data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load environment variables and return cached settings instance."""

    return Settings.from_env()
