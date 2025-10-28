"""Common dependency injections for FastAPI routers."""
from __future__ import annotations

from typing import Generator

from fastapi import Depends

from .config import Settings, get_settings


def get_app_settings() -> Settings:
    """Return cached application settings."""

    return get_settings()


def get_settings_dep() -> Generator[Settings, None, None]:
    """Provide settings for request lifetime."""

    settings = get_settings()
    try:
        yield settings
    finally:
        # Placeholder for cleanup logic if needed.
        return


SettingsDep = Depends(get_settings_dep)
