"""Hybrid sparse+dense retriever with optional reranking."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import torch
from dateutil import parser as date_parser
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from ..dataio.schemas import RAGQuery
from ..dataio.storage import DataLocator
from ..indexing.spatial_index import SpatialIndex
from .query_planner import build_plan
from .types import RetrievalResult


class HybridTextRetriever:
    """Retrieves textual evidence via BM25 + dense search over a prepared corpus."""

    def __init__(self, locator: DataLocator, settings) -> None:
        self.locator = locator
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.corpus_path = self._resolve_path(settings.hybrid_corpus_path)
        self.index_path = self._resolve_path(settings.hybrid_index_path)
        self.ids_path = self._resolve_path(settings.hybrid_ids_path)
        self.meta_path = self._resolve_path(settings.hybrid_meta_path)

        self.docs = self._load_docs(self.corpus_path)
        self.doc_map: Dict[str, dict] = {doc["doc_id"]: doc for doc in self.docs}
        if not self.doc_map:
            raise RuntimeError("Hybrid corpus is empty; rerun scripts/build_text_corpus.py")

        self.embedding_doc_ids = self._load_ids(self.ids_path)
        self.meta = self._load_meta(self.meta_path)

        self.embedder = SentenceTransformer(self.meta["model"], device=self.device)
        self.faiss_index = faiss.read_index(str(self.index_path))

        bm25_tokens = [self.doc_map[doc_id]["tokens"] for doc_id in self.embedding_doc_ids if doc_id in self.doc_map]
        self.bm25_docs = [self.doc_map[doc_id] for doc_id in self.embedding_doc_ids if doc_id in self.doc_map]
        if not bm25_tokens:
            raise RuntimeError("No overlapping docs between corpus and embedding index")
        self.bm25 = BM25Okapi(bm25_tokens)

        imagery_path = self._resolve_table_path("imagery_tiles")
        gauges_path = self._resolve_table_path("gauges")
        self.spatial_index = SpatialIndex.from_parquet(imagery_path, gauges_path)

        self.enable_reranker = bool(getattr(settings, "enable_reranker", False))
        self.cross_encoder: Optional[CrossEncoder] = None
        if self.enable_reranker:
            rerank_model = getattr(settings, "hybrid_reranker_model", "BAAI/bge-reranker-base")
            logger.info("Loading reranker", model=rerank_model)
            self.cross_encoder = CrossEncoder(rerank_model, device=self.device)

        self.bm25_weight = 0.4
        self.dense_weight = 0.6
        self.bm25_top_k = 100
        self.dense_top_k = 100
        self.rerank_k = 20

    # ------------------------------------------------------------------
    def retrieve(self, query: RAGQuery) -> RetrievalResult:
        plan = build_plan(query)
        start_date = self._coerce_date(query.start)
        end_date = self._coerce_date(query.end)
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        imagery = self.spatial_index.get_tiles_by_zip(query.zip, start_dt, end_dt, plan.imagery_k)
        sensors = self.spatial_index.nearest_sensors_by_zip(query.zip, n=3)

        text_docs = self._hybrid_search(
            query_text=f"Harvey impact summary for zip {query.zip} between {start_date} and {end_date}.",
            zip_code=query.zip,
            start=start_date,
            end=end_date,
            limit=max(plan.text_n * 3, 50),
        )

        tweets = [doc for doc in text_docs if doc.get("source") == "tweet"][: plan.text_n]
        calls = [doc for doc in text_docs if doc.get("source") == "311"][: plan.text_n]
        fema = [doc for doc in text_docs if doc.get("source") in {"fema_kb", "claim"}][: plan.text_n]

        return RetrievalResult(
            imagery=imagery,
            tweets=tweets,
            calls=calls,
            sensors=sensors,
            fema=fema,
        )

    # ------------------------------------------------------------------
    def _hybrid_search(self, query_text: str, zip_code: str, start: date, end: date, limit: int) -> List[dict]:
        allowed_ids = self._filter_doc_ids(zip_code, start, end)
        if not allowed_ids:
            logger.warning("Hybrid retriever found no docs in time/zip window; falling back to time-only filter", zip=zip_code)
            allowed_ids = self._filter_doc_ids(None, start, end)
        if not allowed_ids:
            logger.warning("Hybrid retriever found no docs in time window; falling back to full corpus", zip=zip_code)
            allowed_ids = [doc["doc_id"] for doc in self.bm25_docs]
        if not allowed_ids:
            return []

        bm25_scores = self._bm25_scores(query_text, allowed_ids)
        dense_scores = self._dense_scores(query_text, allowed_ids)
        combined = self._combine_scores(bm25_scores, dense_scores)
        if not combined:
            return []

        doc_ids = sorted(combined, key=combined.get, reverse=True)
        if self.cross_encoder:
            reranked = self._rerank(query_text, doc_ids[: self.rerank_k])
            doc_ids = reranked + doc_ids[self.rerank_k :]

        results: List[dict] = []
        for doc_id in doc_ids:
            doc = self.doc_map.get(doc_id)
            if not doc:
                continue
            payload = dict(doc.get("payload") or {})
            payload["source"] = doc.get("source")
            results.append(payload)
            if len(results) >= limit:
                break
        return results

    def _resolve_path(self, path_value: Path | str) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise RuntimeError(f"Required file not found: {path}. Run the corpus/index scripts first.")
        return path

    def _load_docs(self, path: Path) -> List[dict]:
        docs: List[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                doc = json.loads(line)
                doc.setdefault("tokens", (doc.get("text") or "").lower().split())
                ts = doc.get("timestamp")
                doc["timestamp_dt"] = self._parse_datetime(ts) if ts else None
                docs.append(doc)
        return docs

    def _load_ids(self, path: Path) -> List[str]:
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise RuntimeError(f"Invalid embeddings id file: {path}")
        return [str(doc_id) for doc_id in data]

    def _load_meta(self, path: Path) -> dict:
        meta = json.loads(path.read_text())
        if "model" not in meta:
            raise RuntimeError(f"Embedding metadata missing model entry: {path}")
        return meta

    def _filter_doc_ids(self, zip_code: Optional[str], start: date, end: date) -> List[str]:
        allowed: List[str] = []
        for doc in self.bm25_docs:
            if zip_code and doc.get("zip") and doc.get("zip") != zip_code:
                continue
            ts = doc.get("timestamp_dt")
            if ts is None:
                continue
            doc_date = ts.date()
            if doc_date < start or doc_date > end:
                continue
            allowed.append(doc["doc_id"])
        return allowed

    def _bm25_scores(self, query_text: str, allowed_ids: List[str]) -> Dict[str, float]:
        if not allowed_ids:
            return {}
        scores = self.bm25.get_scores(query_text.lower().split())
        allowed_set = set(allowed_ids)
        result: Dict[str, float] = {}
        for idx, doc in enumerate(self.bm25_docs):
            doc_id = doc["doc_id"]
            if doc_id in allowed_set:
                result[doc_id] = float(scores[idx])
        return result

    def _dense_scores(self, query_text: str, allowed_ids: List[str]) -> Dict[str, float]:
        if not allowed_ids:
            return {}
        query_vec = self.embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        distances, indices = self.faiss_index.search(query_vec, self.dense_top_k)
        allowed_set = set(allowed_ids)
        scores: Dict[str, float] = {}
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.embedding_doc_ids):
                continue
            doc_id = self.embedding_doc_ids[idx]
            if doc_id in allowed_set:
                scores[doc_id] = float(dist)
        return scores

    def _combine_scores(self, bm25: Dict[str, float], dense: Dict[str, float]) -> Dict[str, float]:
        combined: Dict[str, float] = {}
        all_ids = set(bm25.keys()) | set(dense.keys())
        for doc_id in all_ids:
            score = self.bm25_weight * bm25.get(doc_id, 0.0) + self.dense_weight * dense.get(doc_id, 0.0)
            if score > 0:
                combined[doc_id] = score
        return combined

    def _rerank(self, query_text: str, doc_ids: List[str]) -> List[str]:
        if not doc_ids or not self.cross_encoder:
            return doc_ids
        pairs = [(query_text, self.doc_map[doc_id]["text"]) for doc_id in doc_ids if doc_id in self.doc_map]
        if not pairs:
            return doc_ids
        scores = self.cross_encoder.predict(pairs)
        scored = sorted(zip([doc_id for doc_id in doc_ids if doc_id in self.doc_map], scores), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scored]

    def _parse_datetime(self, value: str) -> Optional[datetime]:
        try:
            return date_parser.parse(value)
        except (ValueError, TypeError):
            return None

    def _resolve_table_path(self, name: str) -> Path:
        processed = self.locator.table_path(name)
        if processed.exists():
            return processed
        example = self.locator.table_path(name, example=True)
        if example.exists():
            logger.warning("Falling back to example %s table", name)
            return example
        raise RuntimeError(f"Required table '{name}' not found in processed or example directories.")

    def _coerce_date(self, value) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                parsed = date_parser.parse(value)
                return parsed.date()
        raise TypeError(f"Unsupported date value: {value!r}")
