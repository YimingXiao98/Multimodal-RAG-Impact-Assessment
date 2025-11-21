"""Pydantic schemas for normalized datasets."""
from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ImageryTile(BaseModel):
    """Metadata describing a single imagery tile."""

    tile_id: str
    type: Literal["sat", "aerial"]
    bbox: List[float] = Field(..., description="[minx, miny, maxx, maxy]")
    timestamp: datetime
    zip: str
    h3: str
    uri: str


class TweetDoc(BaseModel):
    """Social media post containing text and optional media."""

    tweet_id: str
    text: str
    media_uri: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    timestamp: datetime
    zip: Optional[str]
    h3: Optional[str]


class Call311(BaseModel):
    """311 call log record."""

    record_id: str
    category: str
    description: str
    lat: Optional[float]
    lon: Optional[float]
    timestamp: datetime
    zip: Optional[str]
    h3: Optional[str]


class SensorObs(BaseModel):
    """Observation from rainfall or stage gauge."""

    sensor_id: str
    kind: Literal["rain", "stage"]
    lat: float
    lon: float
    zip: Optional[str]
    timestamp: datetime
    value: float
    unit: str


class FemaKBZip(BaseModel):
    """Historical FEMA losses aggregated by ZIP and year."""

    zip: str
    year: int
    loss_stats: Dict[str, float]


class ClaimPoint(BaseModel):
    """Ground-truth insurance claim point."""

    claim_id: str
    lat: float
    lon: float
    timestamp: datetime
    severity: Optional[str]
    zip: Optional[str]
    amount: Optional[float]


class RoadStatus(BaseModel):
    """Road segment status observation."""

    segment_id: str
    status: Literal["open", "closed", "flooded", "unknown"]
    start_time: datetime
    end_time: Optional[datetime]
    zip: str


class RAGQuery(BaseModel):
    """Request payload for RAG inference."""

    zip: str
    start: date
    end: date
    k_tiles: int = 6
    n_text: int = 20
    text_query: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class RAGAnswer(BaseModel):
    """Structured response produced by the RAG pipeline."""

    zip: str
    time_window: Dict[str, str]
    estimates: Dict[str, object]
    evidence_refs: Dict[str, List[str]]
    natural_language_summary: Optional[str] = None


class ChatRequest(BaseModel):
    """Natural-language chat prompt converted into a structured query."""

    message: str
    zip: Optional[str] = None
    start: Optional[date] = None
    end: Optional[date] = None
    k_tiles: Optional[int] = None
    n_text: Optional[int] = None


class ChatResponse(BaseModel):
    """Chat-style response bundling the interpreted query and the RAG answer."""

    query: RAGQuery
    answer: RAGAnswer
