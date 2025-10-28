"""Health check endpoint."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Return basic health information."""

    return {"status": "ok"}
