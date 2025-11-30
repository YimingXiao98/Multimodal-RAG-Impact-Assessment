"""Unit tests for TextAnalysisClient."""

import json
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_genai():
    """Mock the google.generativeai module."""
    with patch.dict("sys.modules", {"google.generativeai": MagicMock()}):
        import google.generativeai as genai

        yield genai


@pytest.fixture
def client(mock_genai):
    """Create a TextAnalysisClient with mocked Gemini."""
    with patch("google.generativeai.configure"):
        with patch("google.generativeai.GenerativeModel") as MockModel:
            mock_model = MagicMock()
            MockModel.return_value = mock_model

            from app.core.models.text_client import TextAnalysisClient

            client = TextAnalysisClient(api_key="fake_key", provider="gemini")
            client.gemini_model = mock_model
            return client


def test_analyze_success(client):
    """Test successful analysis."""
    mock_response = MagicMock()
    mock_response.text = json.dumps(
        {
            "reasoning": "Test reasoning",
            "estimates": {"structural_damage_pct": 10.0},
            "evidence_refs": {"tweet_ids": ["123"]},
        }
    )
    client.gemini_model.generate_content.return_value = mock_response

    context = {
        "tweets": [{"tweet_id": "123", "text": "Flood"}],
        "imagery_tiles": [{"uri": "img.jpg"}],  # Should be stripped
    }

    result = client.analyze(
        "77002", {"start": "2017-08-25", "end": "2017-08-30"}, context
    )

    assert result["reasoning"] == "Test reasoning"
    assert result["evidence_refs"]["tweet_ids"] == ["123"]


def test_analyze_failure(client):
    """Test analysis failure handling."""
    client.gemini_model.generate_content.side_effect = Exception("API Error")

    context = {}
    result = client.analyze(
        "77002", {"start": "2017-08-25", "end": "2017-08-30"}, context
    )

    assert "Analysis failed" in result["reasoning"]
    assert result["estimates"]["structural_damage_pct"] == 0.0
