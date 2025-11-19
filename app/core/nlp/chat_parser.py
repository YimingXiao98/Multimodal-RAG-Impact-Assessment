"""Utilities for turning natural-language chat prompts into structured queries."""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import List, Optional

from dateutil import parser as date_parser

from ..dataio.schemas import ChatRequest, RAGQuery
from ..dataio.storage import DataLocator, resolve_example_query

ZIP_PATTERN = re.compile(r"\b(\d{5})\b")
MONTH_NAMES = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
DATE_PATTERN = re.compile(
    rf"(\b\d{{4}}-\d{{2}}-\d{{2}}\b)|(\b\d{{1,2}}/\d{{1,2}}/\d{{2,4}}\b)|(\b(?:{MONTH_NAMES})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{2,4}})?)",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b\d{4}\b")
HARVEY_YEAR = 2017


def build_chat_query(
    chat: ChatRequest,
    *,
    default_window_days: int,
    locator: Optional[DataLocator] = None,
    vlm_client: Optional[object] = None,
) -> RAGQuery:
    """Convert the chat request into a concrete `RAGQuery`."""

    message = chat.message or ""

    # 1. Try LLM parsing if available
    llm_zip = None
    llm_start = None
    llm_end = None

    if vlm_client and hasattr(vlm_client, "parse_query"):
        try:
            parsed = vlm_client.parse_query(message)
            llm_zip = parsed.get("zip")
            if parsed.get("start"):
                llm_start = date.fromisoformat(parsed["start"])
            if parsed.get("end"):
                llm_end = date.fromisoformat(parsed["end"])
        except Exception:
            # Ignore LLM errors and fall back to regex
            pass

    # 2. Resolve ZIP
    zip_code = chat.zip or llm_zip or _extract_zip(message)
    if not zip_code:
        raise ValueError(
            "Please include a 5-digit ZIP code in your prompt or payload.")

    # 3. Resolve Dates
    dates: List[date] = []
    if chat.start:
        dates.append(chat.start)
    if chat.end:
        dates.append(chat.end)

    if llm_start:
        dates.append(llm_start)
    if llm_end:
        dates.append(llm_end)

    # Only use regex extraction if we don't have enough dates from explicit/LLM sources
    if not dates:
        extracted = _extract_dates(message)
        dates.extend(extracted)

    unique_dates = sorted({d for d in dates})

    if len(unique_dates) >= 2:
        start_date, end_date = unique_dates[0], unique_dates[-1]
    elif len(unique_dates) == 1:
        end_date = unique_dates[0]
        start_date = end_date - timedelta(days=max(1, default_window_days - 1))
    else:
        start_date, end_date = _default_dates(locator, default_window_days)

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    k_tiles = chat.k_tiles or RAGQuery.model_fields["k_tiles"].default
    n_text = chat.n_text or RAGQuery.model_fields["n_text"].default

    return RAGQuery(
        zip=zip_code,
        start=start_date,
        end=end_date,
        k_tiles=k_tiles,
        n_text=n_text,
        text_query=message
    )


def _extract_zip(message: str) -> Optional[str]:
    match = ZIP_PATTERN.search(message)
    return match.group(1) if match else None


def _extract_dates(message: str) -> List[date]:
    tokens: List[str] = []
    for match in DATE_PATTERN.finditer(message):
        tokens.append(match.group(0))

    dates: List[date] = []
    seen = set()
    lower_msg = message.lower()
    for token in tokens:
        parsed = _parse_token(token, lower_msg)
        if parsed and parsed not in seen:
            seen.add(parsed)
            dates.append(parsed)
    return dates


def _parse_token(token: str, message_lower: str) -> Optional[date]:
    try:
        parsed = date_parser.parse(token, dayfirst=False, fuzzy=True).date()
    except (ValueError, OverflowError):
        return None

    if not YEAR_PATTERN.search(token) and "harvey" in message_lower:
        parsed = parsed.replace(year=HARVEY_YEAR)
    return parsed


def _default_dates(locator: Optional[DataLocator], window_days: int) -> tuple[date, date]:
    if locator is not None:
        sample = resolve_example_query(locator)
        if sample:
            return sample.start, sample.end
    end = date.today()
    start = end - timedelta(days=max(1, window_days - 1))
    return start, end
