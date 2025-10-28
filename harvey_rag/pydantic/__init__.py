"""Minimal pydantic BaseModel stub."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


class Field:
    def __init__(self, default: Any = None, **kwargs: Any) -> Any:
        self.default = default
        self.metadata = kwargs


class BaseModel:
    def __init__(self, **data: Any) -> None:
        annotations = getattr(self, "__annotations__", {})
        for name, annotation in annotations.items():
            if name in data:
                value = data[name]
            else:
                value = getattr(self.__class__, name, None)
                if isinstance(value, Field):
                    value = value.default
            setattr(self, name, value)

    def dict(self) -> Dict[str, Any]:
        annotations = getattr(self, "__annotations__", {})
        return {name: getattr(self, name) for name in annotations}

    def model_dump(self) -> Dict[str, Any]:
        return self.dict()

    def __iter__(self):  # pragma: no cover - convenience
        for key, value in self.dict().items():
            yield key, value


class BaseSettings(BaseModel):
    class Config:
        env_file = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
