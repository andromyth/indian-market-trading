"""Live, point-in-time market-news collector for the autonomous intraday worker.

Queries the existing multi-publisher Google News RSS integration, matches headlines
only to the official NSE universe, computes sentiment, appends a deduplicated PIT
archive, and persists the archive plus a source-health manifest to Google Drive.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PIT_REMOTE_PATH = "processed/live_news/website_news_pit_2026.parquet"
MANIFEST_REMOTE_PATH = "processed/live_news/website_news_manifest_2026.json"
MIN_HEALTHY_FEEDS = 3


def _sentiment_analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def _article_rows(universe: pd.DataFrame, source_result: dict, fetched_at: pd.Timestamp) -> list[dict]:
    from market_watchlist import build_alias_map, extract_mentions

    aliases = build_alias_map(universe)
    analyzer = _sentiment_analyzer()
    output = []
    for article in source_result.get("articles", []) or []:
        title = str(article.get("title") or "").strip()
        text = str(article.get("text") or title).strip()
        url = str(article.get("url") or "").strip()
        if not title or not url:
            continue
        symbols = extract_mentions(f"{title} {text}", aliases) or ["MARKET"]
        published = pd.to_datetime(article.get("published_utc"), errors="coerce", utc=True)
        if pd.isna(published) or published > fetched_at + pd.Timedelta(hours=2):
            continue
        # Point-in-time knowledge is the collector's first-seen time, not an
        # unverified publisher timestamp. This prevents historical look-ahead.
        sentiment = analyzer.polarity_scores(f"{title}. {text}")["compound"]
        publisher = str(article.get("publisher") or article.get("domain") or "UNKNOWN")
        domain = str(article.get("domain") or "")
        for symbol in symbols:
            output.append({
                "symbol": str(symbol).upper(),
                "decision_timestamp": fetched_at.isoformat(),
                "published_at_utc": published.isoformat(),
                "first_seen_at_utc": fetched_at.isoformat(),
                "news_title": title[:1000],
                "news_text": text[:3000],
                "news_url": url,
                "news_source": publisher,
                "news_domain": domain,
                "news_sentiment": float(sentiment),
                "news_article_count": 1,
                "collector_version": "live-news-pit-v1",
            })
    return output


def refresh_live_news_archive(
    *,
    universe: pd.DataFrame,
    pit_path: Path,
    manifest_path: Path,
    drive_store: Any,
) -> dict[str, Any]:
    """Scrape publishers, append deduplicated PIT rows, and upload archive + manifest."""
    from market_watchlist import fetch_watchlist_articles

    pit_path = Path(pit_path)
    manifest_path = Path(manifest_path)
    pit_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fetched_at = pd.Timestamp.now(tz="UTC")
    source_result = fetch_watchlist_articles(now=fetched_at.to_pydatetime())
    sources = source_result.get("sources", []) or []
    healthy_sources = [
        source for source in sources
        if str(source.get("status") or "").upper() != "ERROR"
    ]
    successful_feed_count = len(healthy_sources)
    collection_status = "PASS" if successful_feed_count >= MIN_HEALTHY_FEEDS else (
        "PARTIAL" if successful_feed_count else "ERROR"
    )

    existing = pd.DataFrame()
    if pit_path.exists():
        try:
            existing = pd.read_parquet(pit_path)
        except Exception as exc:
            raise RuntimeError(
                f"Existing live news PIT archive is unreadable: {type(exc).__name__}: {exc}"
            ) from exc

    new_rows = _article_rows(universe, source_result, fetched_at)
    new = pd.DataFrame(new_rows)
    if not existing.empty and not new.empty:
        for column in new.columns:
            if column not in existing.columns:
                existing[column] = np.nan
        for column in existing.columns:
            if column not in new.columns:
                new[column] = np.nan
        combined = pd.concat([existing, new], ignore_index=True)
    elif not new.empty:
        combined = new
    else:
        combined = existing

    if not combined.empty:
        combined["symbol"] = combined["symbol"].astype(str).str.strip().str.upper()
        combined["news_url"] = combined.get("news_url", pd.Series("", index=combined.index)).fillna("").astype(str)
        combined["news_title"] = combined.get("news_title", pd.Series("", index=combined.index)).fillna("").astype(str)
        # URL is preferred; legacy rows without a URL dedupe by symbol + title.
        combined["_dedupe_key"] = np.where(
            combined["news_url"].str.len().gt(0),
            combined["symbol"] + "|" + combined["news_url"],
            combined["symbol"] + "|" + combined["news_title"],
        )
        combined = combined.drop_duplicates("_dedupe_key", keep="first").drop(columns=["_dedupe_key"])
        combined["decision_timestamp"] = pd.to_datetime(
            combined["decision_timestamp"], errors="coerce", utc=True
        )
        combined = combined.dropna(subset=["decision_timestamp"])
        combined = combined.sort_values("decision_timestamp").reset_index(drop=True)
        combined.to_parquet(pit_path, index=False)

    manifest = {
        "schema_version": "live-news-pit-v1",
        "checked_at_utc": fetched_at.isoformat(),
        "collection_status": collection_status,
        "successful_feed_count": successful_feed_count,
        "configured_feed_count": len(sources),
        "article_count_fetched": int(len(source_result.get("articles", []) or [])),
        "pit_rows_added_before_deduplication": int(len(new_rows)),
        "pit_rows_total": int(len(combined)),
        "symbol_rows_added_before_deduplication": int(
            sum(1 for row in new_rows if row.get("symbol") != "MARKET")
        ),
        "market_fallback_rows_added_before_deduplication": int(
            sum(1 for row in new_rows if row.get("symbol") == "MARKET")
        ),
        "source_status": sources,
        "source_domains": sorted({
            str(source.get("domain") or "")
            for source in sources if source.get("domain")
        }),
        "sentiment_method": "VADER compound score; context only, not a model feature unless model contract supports it",
        "pit_timestamp_policy": "decision_timestamp and first_seen_at_utc are collector first-seen time; published_at_utc is retained separately",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")

    # Keep the existing archive if all feeds fail; still publish the failed
    # health manifest so the worker cannot mistake stale headlines for fresh data.
    if pit_path.exists():
        drive_store.upload_or_update(
            pit_path, PIT_REMOTE_PATH, mime_type="application/octet-stream"
        )
    drive_store.upload_or_update(
        manifest_path, MANIFEST_REMOTE_PATH, mime_type="application/json"
    )
    print(json.dumps({
        "event": "live_news_pit_refresh",
        "status": collection_status,
        "successful_feed_count": successful_feed_count,
        "configured_feed_count": len(sources),
        "fetched_articles": manifest["article_count_fetched"],
        "new_symbol_article_rows": manifest["symbol_rows_added_before_deduplication"],
        "pit_rows_total": manifest["pit_rows_total"],
        "drive_archive_path": PIT_REMOTE_PATH if pit_path.exists() else None,
        "drive_manifest_path": MANIFEST_REMOTE_PATH,
    }, sort_keys=True))
    return manifest
