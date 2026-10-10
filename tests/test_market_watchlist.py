from __future__ import annotations

from datetime import date, datetime, timezone
import pandas as pd

from market_watchlist import (
    build_alias_map,
    build_stock_watchlist,
    extract_mentions,
    normalize_text,
    target_session_date,
)


def _universe() -> pd.DataFrame:
    return pd.DataFrame({
        "SYMBOL": ["RELIANCE", "TCS", "INFY", "SBIN"],
        "NAME OF COMPANY": [
            "Reliance Industries Limited",
            "Tata Consultancy Services Limited",
            "Infosys Limited",
            "State Bank of India",
        ],
    })


def test_alias_map_matches_official_symbols_and_company_names():
    aliases = build_alias_map(_universe())
    assert "RELIANCE INDUSTRIES" in aliases
    assert aliases["RELIANCE INDUSTRIES"]["symbol"] == "RELIANCE"
    assert "TATA CONSULTANCY SERVICES" in aliases
    assert aliases["TATA CONSULTANCY SERVICES"]["symbol"] == "TCS"
    assert extract_mentions("TCS and Reliance Industries in focus", aliases) == [
        "RELIANCE", "TCS"
    ]


def test_normalization_removes_markup_and_punctuation():
    assert normalize_text("Tata Consultancy Services Ltd.") == "TATA CONSULTANCY SERVICES LTD"


def test_watchlist_ranks_by_independent_publisher_support_and_links_articles():
    universe = _universe()
    articles = [
        {
            "title": "TCS, Reliance Industries among stocks to watch",
            "text": "TCS, Reliance Industries among stocks to watch",
            "url": "https://example.com/a",
            "publisher": "Moneycontrol",
            "domain": "moneycontrol.com",
            "published_utc": "2026-10-10T09:00:00+00:00",
            "published_at": datetime(2026, 10, 10, 9, 0, tzinfo=timezone.utc),
        },
        {
            "title": "Tata Consultancy Services in focus before results",
            "text": "Tata Consultancy Services in focus before results",
            "url": "https://example.org/b",
            "publisher": "Economic Times",
            "domain": "economictimes.indiatimes.com",
            "published_utc": "2026-10-10T10:00:00+00:00",
            "published_at": datetime(2026, 10, 10, 10, 0, tzinfo=timezone.utc),
        },
    ]
    result = build_stock_watchlist(
        universe,
        {"articles": articles, "sources": [], "checked_at_utc": "2026-10-10T10:01:00+00:00"},
        max_stocks=5,
    )
    assert result["status"] == "PASS"
    assert result["stocks"][0]["symbol"] == "TCS"
    assert result["stocks"][0]["source_count"] == 2
    assert result["stocks"][0]["mention_count"] == 2
    assert {a["url"] for a in result["stocks"][0]["headlines"]} == {
        "https://example.com/a", "https://example.org/b"
    }


def test_no_source_mentions_does_not_invent_default_stocks():
    result = build_stock_watchlist(
        _universe(),
        {"articles": [], "sources": [], "checked_at_utc": "2026-10-10T10:01:00+00:00"},
    )
    assert result["status"] == "NO_SOURCE_MATCHES"
    assert result["stocks"] == []
    assert "No fallback stocks were invented" in result["message"]


def test_target_session_respects_weekends_and_exchange_holidays():
    saturday = datetime(2026, 10, 10, 11, 0, tzinfo=timezone.utc)
    assert target_session_date(saturday).isoformat() == "2026-10-12"

    # Monday is explicitly holiday; the next watchlist target becomes Tuesday.
    assert target_session_date(
        saturday, holidays={date(2026, 10, 12)}
    ).isoformat() == "2026-10-13"

    monday_preopen = datetime(2026, 10, 12, 1, 0, tzinfo=timezone.utc)
    assert target_session_date(monday_preopen).isoformat() == "2026-10-12"
