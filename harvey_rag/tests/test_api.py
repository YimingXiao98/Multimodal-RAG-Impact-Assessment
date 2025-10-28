from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.core.dataio.storage import DataLocator
from app.core.retrieval.retriever import Retriever
from app.core.models.vlm_client import VLMClient


def setup_module(module):
    locator = DataLocator(Path(__file__).resolve().parents[1] / "data")
    app.state.locator = locator
    app.state.retriever = Retriever(locator)
    app.state.vlm_client = VLMClient("mock")


def test_healthz():
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_endpoint():
    client = TestClient(app)
    payload = {"zip": "77002", "start": "2017-08-28", "end": "2017-09-03", "k_tiles": 4, "n_text": 5}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["zip"] == "77002"
    assert data["evidence_refs"]["imagery_tile_ids"], "Imagery references required"
