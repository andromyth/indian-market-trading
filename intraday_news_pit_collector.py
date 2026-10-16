"""Incremental 15-minute, point-in-time publisher-news collector for live intraday forecasts.

The archive is a publisher-RSS sample, not a guarantee of exhaustive news coverage
for every NSE-listed security. Published timestamps are retained as PIT availability
times; the trained classifier's feature contract is not changed by this module.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
FOLDER_MIME = "application/vnd.google-apps.folder"
PIT_REMOTE = "processed/live_news/website_news_pit_2026.parquet"
MANIFEST_REMOTE = "processed/live_news/website_news_manifest_2026.json"
MAX_ARCHIVE_DAYS = 7

# Independent broad-market queries complement the narrower "stocks to watch"
# queries used by the dashboard watchlist. All are publisher-restricted.
BROAD_MARKET_SOURCES = (
    ("Economic Times Market News", "economictimes.indiatimes.com",
     '("Nifty" OR "Sensex" OR "stock market" OR "shares" OR "earnings" OR "RBI" OR "SEBI")'),
    ("Moneycontrol Market News", "moneycontrol.com",
     '("Nifty" OR "Sensex" OR "stock market" OR "shares" OR "earnings" OR "RBI" OR "SEBI")'),
    ("Financial Express Market News", "financialexpress.com",
     '("Nifty" OR "Sensex" OR "stock market" OR "shares" OR "earnings" OR "RBI" OR "SEBI")'),
    ("Mint Market News", "livemint.com",
     '("Nifty" OR "Sensex" OR "stock market" OR "shares" OR "earnings" OR "RBI" OR "SEBI")'),
    ("CNBC-TV18 Market News", "cnbctv18.com",
     '("Nifty" OR "Sensex" OR "stock market" OR "shares" OR "earnings" OR "RBI" OR "SEBI")'),
)


def _download_if_present(store: Any, remote_path: str, local_path: Path) -> bool:
    parts = [p for p in remote_path.replace("\\", "/").split("/") if p]
    if not parts:
        return False
    parent = store.root_id
    for part in parts[:-1]:
        folder = store._find_child(parent, part, FOLDER_MIME)
        if not folder:
            return False
        parent = folder["id"]
    item = store._find_child(parent, parts[-1])
    if not item:
        return False
    from googleapiclient.http import MediaIoBaseDownload
    local_path.parent.mkdir(parents=True, exist_ok=True)
    request = store.service.files().get_media(fileId=item["id"])
    with local_path.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return True


def _sentiment(text: str) -> float | None:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return float(SentimentIntensityAnalyzer().polarity_scores(text).get("compound", 0.0))
    except Exception:
        return None


def _article_id(article: dict[str, Any]) -> str:
    url = str(article.get("url") or "").strip()
    fallback = "|".join((
        str(article.get("publisher") or ""),
        str(article.get("title") or ""),
        str(article.get("published_utc") or ""),
    ))
    return hashlib.sha256((url or fallback).encode("utf-8")).hexdigest()


def collect_and_publish_live_news(
    *,
    now: datetime | None = None,
    root: Path = Path("data/autonomous_intraday/news_pit"),
    source_result: dict[str, Any] | None = None,
    universe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Fetch publisher RSS search results, merge PIT rows, and publish archive+manifest.

    Callers may pass the source result already fetched for the same watchlist refresh
    so publisher endpoints are not scraped twice in a single 15-minute cycle.
    """
    from delivery_training_universe import load_official_nse_universe
    from market_watchlist import (
        fetch_watchlist_articles, build_alias_map, extract_mentions, _fetch_source
    )
    from persistent_store import GoogleDriveStore

    current = now or datetime.now(IST)
    current = current.replace(tzinfo=IST) if current.tzinfo is None else current.astimezone(IST)
    checked_utc = current.astimezone(timezone.utc)
    root.mkdir(parents=True, exist_ok=True)
    pit_path = root / "website_news_pit_2026.parquet"
    manifest_path = root / "website_news_manifest_2026.json"

    store = GoogleDriveStore()
    existing = _download_if_present(store, PIT_REMOTE, pit_path)
    if existing:
        old = pd.read_parquet(pit_path)
    else:
        old = pd.DataFrame()

    source_result = source_result or fetch_watchlist_articles(now=checked_utc)
    # Add broader publisher-restricted market, macro, earnings and regulatory news.
    # The already-fetched watchlist feeds are reused; only these five additional
    # queries are requested, once per 15-minute worker refresh.
    broad_results = []
    with ThreadPoolExecutor(max_workers=len(BROAD_MARKET_SOURCES)) as pool:
        futures = {
            pool.submit(_fetch_source, source, checked_utc): source
            for source in BROAD_MARKET_SOURCES
        }
        for future in as_completed(futures):
            publisher, domain, _ = futures[future]
            try:
                broad_results.append(future.result())
            except Exception as exc:
                broad_results.append({
                    "publisher": publisher,
                    "domain": domain,
                    "status": "ERROR",
                    "articles": [],
                    "error": f"{type(exc).__name__}: {exc}",
                })
    source_result = dict(source_result)
    source_result["sources"] = list(source_result.get("sources", []) or []) + [
        {
            key: source.get(key)
            for key in ("publisher", "domain", "status", "error")
        } | {"article_count": len(source.get("articles", []) or [])}
        for source in broad_results
    ]
    source_result["articles"] = list(source_result.get("articles", []) or []) + [
        article
        for source in broad_results
        for article in (source.get("articles", []) or [])
    ]
    source_status = source_result.get("sources", []) or []
    # NO_RECENT_ARTICLES means the publisher query succeeded but returned no fresh items;
    # only ERROR indicates a failed source request. Do not mislabel healthy quiet feeds.
    successful_feeds = sum(
        str(x.get("status", "")).upper() in {"PASS", "NO_RECENT_ARTICLES"}
        for x in source_status
    )
    universe = universe if universe is not None else load_official_nse_universe(force_refresh=False)
    aliases = build_alias_map(universe)
    rows: list[dict[str, Any]] = []
    article_ids_seen: set[str] = set()
    latest_published = None

    for article in source_result.get("articles", []) or []:
        title = str(article.get("title") or "").strip()
        url = str(article.get("url") or "").strip()
        publisher = str(article.get("publisher") or "Unknown").strip()
        domain = str(article.get("domain") or "").strip()
        published = pd.to_datetime(article.get("published_utc"), errors="coerce", utc=True)
        if not title or pd.isna(published) or published > pd.Timestamp(checked_utc):
            continue
        article_id = _article_id(article)
        if article_id in article_ids_seen:
            continue
        article_ids_seen.add(article_id)
        latest_published = max(latest_published, published) if latest_published is not None else published
        text = str(article.get("text") or title)
        sentiment = _sentiment(text)
        base = {
            "article_id": article_id,
            "decision_timestamp": pd.Timestamp(checked_utc),
            "published_at_utc": published.isoformat(),
            "news_article_count": 1,
            "news_source": f"{publisher};{domain}",
            "news_sentiment": sentiment,
            "news_title": title[:500],
            "article_url": url,
            "collected_at_utc": checked_utc.isoformat(),
            "publisher_domain": domain,
        }
        # MARKET row gives the runtime a transparent broad-market news context.
        rows.append({"symbol": "MARKET", **base})
        for symbol in extract_mentions(text, aliases):
            rows.append({"symbol": str(symbol).strip().upper(), **base})

    fresh = pd.DataFrame(rows)
    if not old.empty:
        old = old.copy()
        if "decision_timestamp" in old:
            old["decision_timestamp"] = pd.to_datetime(old["decision_timestamp"], errors="coerce", utc=True)
        # Legacy archives may lack a separate collection timestamp. Preserve their
        # existing decision timestamp as the best available first-seen time.
        if "collected_at_utc" not in old:
            old["collected_at_utc"] = old["decision_timestamp"].astype(str)
        else:
            old["collected_at_utc"] = old["collected_at_utc"].where(
                old["collected_at_utc"].notna(),
                old["decision_timestamp"].astype(str),
            )
        if "article_id" not in old:
            old["article_id"] = old.apply(
                lambda row: hashlib.sha256(
                    "|".join(str(row.get(k, "")) for k in
                             ("symbol", "decision_timestamp", "news_title", "news_source")).encode("utf-8")
                ).hexdigest(),
                axis=1,
            )
        combined = pd.concat([old, fresh], ignore_index=True, sort=False)
    else:
        combined = fresh

    if not combined.empty:
        combined["decision_timestamp"] = pd.to_datetime(
            combined["decision_timestamp"], errors="coerce", utc=True
        )
        cutoff = pd.Timestamp(checked_utc) - pd.Timedelta(days=MAX_ARCHIVE_DAYS)
        combined = combined.dropna(subset=["symbol", "decision_timestamp"])
        combined = combined.loc[
            combined["decision_timestamp"].le(pd.Timestamp(checked_utc))
            & combined["decision_timestamp"].ge(cutoff)
        ].copy()
        # One record per symbol/article. Repeated 15-minute RSS observations do not
        # inflate article counts or move the first-seen decision timestamp forward.
        if "collected_at_utc" in combined:
            combined = combined.sort_values("collected_at_utc", ascending=True, na_position="last")
        combined = combined.drop_duplicates(["symbol", "article_id"], keep="first")
        combined = combined.sort_values(["symbol", "decision_timestamp"]).reset_index(drop=True)

    status = "PASS" if successful_feeds >= 3 else ("PARTIAL" if successful_feeds else "ERROR")
    manifest = {
        "schema_version": "website-news-pit-v1",
        "collector": "intraday_news_pit_collector",
        "checked_at_utc": checked_utc.isoformat(),
        "checked_at_ist": current.isoformat(),
        "collection_status": status,
        "successful_feed_count": int(successful_feeds),
        "configured_feed_count": int(len(source_status)),
        "source_status": source_status,
        "new_articles_observed": int(len(article_ids_seen)),
        "new_symbol_article_rows": int(len(fresh)),
        "archive_rows": int(len(combined)),
        "archive_symbol_count": int(combined["symbol"].nunique()) if not combined.empty else 0,
        "latest_published_at_utc": latest_published.isoformat() if latest_published is not None else None,
        "archive_window_days": MAX_ARCHIVE_DAYS,
        "coverage_complete": False,
        "coverage_note": "Publisher RSS/search coverage is sampled and is not exhaustive for every NSE symbol; missing mentions must not be interpreted as confirmed no-news.",
        "pit_timestamp_rule": "New rows use collector first-seen time in UTC and retain published_at_utc separately; legacy rows keep their original decision_timestamp because first-seen time cannot be reconstructed; future-dated items are excluded",
        "archive_remote_path": PIT_REMOTE,
        "manifest_remote_path": MANIFEST_REMOTE,
    }

    if status != "ERROR":
        combined.to_parquet(pit_path, index=False)
        store.upload_or_update(
            pit_path, PIT_REMOTE, mime_type="application/vnd.apache.parquet"
        )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    store.upload_or_update(manifest_path, MANIFEST_REMOTE, mime_type="application/json")

    print(json.dumps({
        "event": "live_news_pit_published",
        "status": status,
        "successful_feed_count": successful_feeds,
        "configured_feed_count": len(source_status),
        "new_articles_observed": len(article_ids_seen),
        "archive_rows": len(combined),
        "archive_symbol_count": manifest["archive_symbol_count"],
        "archive_remote_path": PIT_REMOTE,
        "manifest_remote_path": MANIFEST_REMOTE,
    }, sort_keys=True), flush=True)
    return manifest


if __name__ == "__main__":
    result = collect_and_publish_live_news()
    if result.get("collection_status") == "ERROR":
        raise SystemExit("No configured news feed succeeded; previous PIT archive was preserved.")
