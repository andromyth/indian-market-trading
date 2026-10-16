from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def test_live_news_collector_publishes_pit_rows_and_non_exhaustive_manifest(tmp_path, monkeypatch):
    from intraday_news_pit_collector import collect_and_publish_live_news

    uploaded = []

    class FakeDriveStore:
        root_id = "fake-root"

        def _find_child(self, parent, name, mime_type=None):
            return None

        def upload_or_update(self, local_path, remote_path, mime_type=None):
            uploaded.append({
                "remote_path": remote_path,
                "mime_type": mime_type,
                "bytes": Path(local_path).read_bytes(),
            })
            return {"status": "OK", "path": remote_path}

    source_result = {
        "checked_at_utc": "2026-10-16T10:30:00+00:00",
        "sources": [
            {"publisher": "Economic Times", "domain": "economictimes.indiatimes.com", "status": "PASS", "article_count": 1},
            {"publisher": "Moneycontrol", "domain": "moneycontrol.com", "status": "NO_RECENT_ARTICLES", "article_count": 0},
            {"publisher": "Mint", "domain": "livemint.com", "status": "PASS", "article_count": 1},
            {"publisher": "Business Standard", "domain": "business-standard.com", "status": "NO_RECENT_ARTICLES", "article_count": 0},
            {"publisher": "CNBC-TV18", "domain": "cnbctv18.com", "status": "ERROR", "article_count": 0},
        ],
        "articles": [{
            "title": "Reliance Industries announces quarterly results",
            "text": "Reliance Industries announces quarterly results and strong profit",
            "url": "https://example.com/reliance-results",
            "publisher": "Economic Times",
            "domain": "economictimes.indiatimes.com",
            "published_utc": "2026-10-16T10:00:00+00:00",
            "published_at": datetime(2026, 10, 16, 10, 0, tzinfo=timezone.utc),
        }],
    }
    monkeypatch.setattr(
        "market_watchlist.fetch_watchlist_articles",
        lambda now=None: source_result,
    )
    monkeypatch.setattr(
        "delivery_training_universe.load_official_nse_universe",
        lambda force_refresh=False: pd.DataFrame({
            "SYMBOL": ["RELIANCE", "TCS"],
            "NAME OF COMPANY": ["Reliance Industries Limited", "Tata Consultancy Services Limited"],
        }),
    )
    monkeypatch.setitem(
        sys.modules,
        "persistent_store",
        SimpleNamespace(GoogleDriveStore=FakeDriveStore),
    )

    result = collect_and_publish_live_news(
        now=datetime(2026, 10, 16, 10, 30, tzinfo=timezone.utc),
        root=tmp_path,
        source_result=source_result,
        universe=pd.DataFrame({
            "SYMBOL": ["RELIANCE", "TCS"],
            "NAME OF COMPANY": ["Reliance Industries Limited", "Tata Consultancy Services Limited"],
        }),
    )

    assert result["collection_status"] == "PASS"
    assert result["successful_feed_count"] == 4
    assert result["new_articles_observed"] == 1
    assert result["coverage_complete"] is False
    assert "not exhaustive" in result["coverage_note"]
    assert result["archive_rows"] == 2
    assert {x["remote_path"] for x in uploaded} == {
        "processed/live_news/website_news_pit_2026.parquet",
        "processed/live_news/website_news_manifest_2026.json",
    }
    archive = pd.read_parquet(tmp_path / "website_news_pit_2026.parquet")
    assert set(archive["symbol"]) == {"MARKET", "RELIANCE"}
    assert set(archive["news_title"]) == {"Reliance Industries announces quarterly results"}
    assert archive["decision_timestamp"].notna().all()
    manifest_upload = next(x for x in uploaded if x["remote_path"].endswith(".json"))
    manifest = json.loads(manifest_upload["bytes"])
    assert manifest["pit_timestamp_rule"].startswith("New rows use collector first-seen time")
