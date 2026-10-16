from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

import live_news_scraper


class FakeDriveStore:
    def __init__(self):
        self.uploads = []

    def upload_or_update(self, local_path, remote_path, mime_type=None):
        self.uploads.append((remote_path, mime_type, local_path.read_bytes()))
        return {"status": "updated", "path": remote_path}


def _universe():
    return pd.DataFrame({
        "SYMBOL": ["RELIANCE", "TCS"],
        "NAME OF COMPANY": ["Reliance Industries Limited", "Tata Consultancy Services Limited"],
    })


def _source_result():
    now = datetime(2026, 10, 16, 10, 0, tzinfo=timezone.utc)
    return {
        "checked_at_utc": now.isoformat(),
        "sources": [
            {"publisher": "Moneycontrol", "domain": "moneycontrol.com", "status": "PASS", "article_count": 1},
            {"publisher": "Economic Times", "domain": "economictimes.indiatimes.com", "status": "NO_RECENT_ARTICLES", "article_count": 0},
            {"publisher": "Mint", "domain": "livemint.com", "status": "PASS", "article_count": 1},
            {"publisher": "Business Standard", "domain": "business-standard.com", "status": "PASS", "article_count": 1},
            {"publisher": "CNBC-TV18", "domain": "cnbctv18.com", "status": "ERROR", "article_count": 0},
        ],
        "articles": [
            {
                "title": "Reliance Industries shares rise",
                "text": "Reliance Industries shares rise on volume",
                "url": "https://moneycontrol.com/article/1",
                "publisher": "Moneycontrol",
                "domain": "moneycontrol.com",
                "published_utc": "2026-10-16T09:59:00+00:00",
            },
            {
                "title": "Indian market opens higher",
                "text": "Indian market opens higher",
                "url": "https://livemint.com/article/2",
                "publisher": "Mint",
                "domain": "livemint.com",
                "published_utc": "2026-10-16T09:58:00+00:00",
            },
        ],
    }


def test_live_news_scraper_writes_pit_rows_manifest_and_drive_uploads(tmp_path, monkeypatch):
    store = FakeDriveStore()
    monkeypatch.setattr(live_news_scraper, "_sentiment_analyzer", lambda: type(
        "Analyzer", (), {"polarity_scores": lambda self, text: {"compound": 0.25}}
    )())
    monkeypatch.setattr(
        "market_watchlist.fetch_watchlist_articles",
        lambda now=None: _source_result(),
    )
    pit = tmp_path / "website_news_pit_2026.parquet"
    manifest_path = tmp_path / "website_news_manifest_2026.json"

    manifest = live_news_scraper.refresh_live_news_archive(
        universe=_universe(),
        pit_path=pit,
        manifest_path=manifest_path,
        drive_store=store,
    )

    assert manifest["collection_status"] == "PASS"
    assert manifest["successful_feed_count"] == 4
    assert manifest["article_count_fetched"] == 2
    frame = pd.read_parquet(pit)
    assert set(frame["symbol"]) == {"RELIANCE", "MARKET"}
    assert frame["news_sentiment"].eq(0.25).all()
    assert frame["first_seen_at_utc"].notna().all()
    assert {upload[0] for upload in store.uploads} == {
        live_news_scraper.PIT_REMOTE_PATH,
        live_news_scraper.MANIFEST_REMOTE_PATH,
    }
    uploaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert uploaded_manifest["pit_rows_total"] == len(frame)


def test_live_news_scraper_deduplicates_same_symbol_and_url(tmp_path, monkeypatch):
    store = FakeDriveStore()
    monkeypatch.setattr(live_news_scraper, "_sentiment_analyzer", lambda: type(
        "Analyzer", (), {"polarity_scores": lambda self, text: {"compound": 0.0}}
    )())
    monkeypatch.setattr(
        "market_watchlist.fetch_watchlist_articles",
        lambda now=None: _source_result(),
    )
    pit = tmp_path / "website_news_pit_2026.parquet"
    manifest_path = tmp_path / "website_news_manifest_2026.json"

    first = live_news_scraper.refresh_live_news_archive(
        universe=_universe(), pit_path=pit, manifest_path=manifest_path, drive_store=store
    )
    second = live_news_scraper.refresh_live_news_archive(
        universe=_universe(), pit_path=pit, manifest_path=manifest_path, drive_store=store
    )

    frame = pd.read_parquet(pit)
    assert first["pit_rows_total"] == second["pit_rows_total"]
    assert not frame.duplicated(["symbol", "news_url"]).any()


def test_failed_sources_are_reported_and_do_not_mark_news_fresh(tmp_path, monkeypatch):
    store = FakeDriveStore()
    monkeypatch.setattr(
        "market_watchlist.fetch_watchlist_articles",
        lambda now=None: {
            "sources": [
                {"publisher": "Moneycontrol", "domain": "moneycontrol.com", "status": "ERROR", "error": "timeout"}
            ],
            "articles": [],
        },
    )
    pit = tmp_path / "website_news_pit_2026.parquet"
    manifest_path = tmp_path / "website_news_manifest_2026.json"

    manifest = live_news_scraper.refresh_live_news_archive(
        universe=_universe(), pit_path=pit, manifest_path=manifest_path, drive_store=store
    )

    assert manifest["collection_status"] == "ERROR"
    assert manifest["successful_feed_count"] == 0
    assert manifest["source_status"][0]["status"] == "ERROR"
    assert not pit.exists()
    assert any(upload[0] == live_news_scraper.MANIFEST_REMOTE_PATH for upload in store.uploads)
