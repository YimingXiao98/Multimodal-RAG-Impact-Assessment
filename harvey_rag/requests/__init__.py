"""Minimal requests stub."""
from __future__ import annotations

import json
from typing import Any, Dict


class Response:
    def __init__(self, data: Dict[str, Any] | None = None, status_code: int = 200) -> None:
        self._data = data or {}
        self.status_code = status_code

    def json(self) -> Dict[str, Any]:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP error {self.status_code}")


def get(url: str, timeout: int = 30) -> Response:  # pragma: no cover - offline stub
    return Response({}, 200)
