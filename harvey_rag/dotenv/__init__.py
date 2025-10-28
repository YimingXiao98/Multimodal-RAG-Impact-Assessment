"""Minimal dotenv stub."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_dotenv(path: str | None = None) -> bool:  # pragma: no cover - noop
    return True
