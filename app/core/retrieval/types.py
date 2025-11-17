"""Shared types for retrieval results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RetrievalResult:
    """Container returned by retrievers."""

    imagery: List[dict] = field(default_factory=list)
    tweets: List[dict] = field(default_factory=list)
    calls: List[dict] = field(default_factory=list)
    sensors: List[dict] = field(default_factory=list)
    fema: List[dict] = field(default_factory=list)
