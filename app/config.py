"""Application configuration utilities."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_provider: str = Field("mock", env="MODEL_PROVIDER")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    default_time_window_days: int = Field(7, env="DEFAULT_TIME_WINDOW_DAYS")
    data_dir: Path = Field(Path("./data"), env="DATA_DIR")
    hybrid_corpus_path: Path = Field(Path("data/processed/text_corpus.jsonl"), env="HARVEY_CORPUS_PATH")
    hybrid_index_path: Path = Field(Path("data/processed/text_embeddings.faiss"), env="HARVEY_EMBED_INDEX_PATH")
    hybrid_ids_path: Path = Field(Path("data/processed/text_embeddings_ids.json"), env="HARVEY_EMBED_IDS_PATH")
    hybrid_meta_path: Path = Field(Path("data/processed/text_embeddings_meta.json"), env="HARVEY_EMBED_META_PATH")
    enable_reranker: bool = Field(False, env="HARVEY_ENABLE_RERANKER")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load environment variables via python-dotenv and return cached settings instance."""

    load_dotenv()
    return Settings()
