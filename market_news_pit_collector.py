#!/usr/bin/env python3
"""Collect licensed/officially exposed financial-news RSS headlines into a PIT archive.

The collector uses publisher RSS endpoints where available and publisher-filtered
Google News RSS only where a direct feed is not dependable. It stores headline,
short feed summary, source URL, publication time, first-seen time and sentiment;
it does not scrape full articles or bypass paywalls. It never produces trade signals.

Outputs (local beneath --drive-root; uploaded to Google Drive when credentials exist):
  processed/live_news/website_news_articles_2026.parquet
  processed/live_news/website_news_pit_2026.parquet
  processed/live_news/website_news_manifest_2026.json
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse
from zoneinfo import ZoneInfo

import feedparser
import numpy as np
import pandas as pd
import requests

IST = ZoneInfo("Asia/Kolkata")
UTC = timezone.utc
LIVE_NEWS_DIR = Path("processed/live_news")
ARTICLES_REMOTE = "processed/live_news/website_news_articles_2026.parquet"
PIT_REMOTE = "processed/live_news/website_news_pit_2026.parquet"
MANIFEST_REMOTE = "processed/live_news/website_news_manifest_2026.json"
MAX_ARTICLE_AGE_HOURS = 24 * 30
PIT_FRESHNESS_DAYS = 7
MIN_HEALTHY_FEEDS = 3
USER_AGENT = "IndianMarketTradingNewsCollector/1.0 (+RSS headline attribution; personal research)"


def _google_news_feed(domain: str, terms: str = "stock market OR shares OR NSE OR BSE") -> str:
    query = f"site:{domain} ({terms}) when:7d"
    return "https://news.google.com/rss/search?" + urlencode({
        "q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en",
    })


# Direct RSS URLs come from the publishers' RSS listings. Legacy Moneycontrol
# RSS endpoints were omitted because independent checks report that many no
# longer publish. Publisher-filtered Google News RSS is used as a fallback.
SOURCES: tuple[dict[str, Any], ...] = (
    {
        "source_id": "economic_times_stocks",
        "publisher": "Economic Times",
        "domain": "economictimes.indiatimes.com",
        "url": "https://economictimes.indiatimes.com/markets/stocks/rss.cms",
        "method": "publisher_rss",
        "source_weight": 0.95,
        "required": True,
    },
    {
        "source_id": "economic_times_markets",
        "publisher": "Economic Times",
        "domain": "economictimes.indiatimes.com",
        "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "method": "publisher_rss",
        "source_weight": 0.95,
        "required": True,
    },
    {
        "source_id": "mint_markets",
        "publisher": "Mint",
        "domain": "livemint.com",
        "url": "https://www.livemint.com/rss/markets",
        "method": "publisher_rss",
        "source_weight": 0.90,
        "required": True,
    },
    {
        "source_id": "business_standard_markets",
        "publisher": "Business Standard",
        "domain": "business-standard.com",
        "url": "https://www.business-standard.com/rss/markets-106.rss",
        "method": "publisher_rss",
        "source_weight": 0.90,
        "required": True,
    },
    {
        "source_id": "moneycontrol_search",
        "publisher": "Moneycontrol",
        "domain": "moneycontrol.com",
        "url": _google_news_feed("moneycontrol.com"),
        "method": "publisher_filtered_news_rss",
        "source_weight": 0.88,
        "required": False,
    },
    {
        "source_id": "financial_express_search",
        "publisher": "Financial Express",
        "domain": "financialexpress.com",
        "url": _google_news_feed("financialexpress.com", "markets OR stocks OR shares OR earnings"),
        "method": "publisher_filtered_news_rss",
        "source_weight": 0.88,
        "required": False,
    },
    {
        "source_id": "cnbc_tv18_search",
        "publisher": "CNBC-TV18",
        "domain": "cnbctv18.com",
        "url": _google_news_feed("cnbctv18.com", "markets OR stocks OR shares OR earnings"),
        "method": "publisher_filtered_news_rss",
        "source_weight": 0.88,
        "required": False,
    },
)


def _clean_text(value: Any, max_chars: int = 700) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def _published_utc(entry: dict[str, Any]) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    try:
        return datetime.fromtimestamp(calendar.timegm(parsed), tz=UTC)
    except (OverflowError, OSError, TypeError, ValueError):
        return None


def _article_id(url: str, title: str, publisher: str, published: datetime | None) -> str:
    canonical = str(url or "").strip().split("#", 1)[0]
    material = canonical or "|".join((
        publisher, re.sub(r"\s+", " ", title).strip().casefold(),
        published.isoformat() if published else "",
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def fetch_source(source: dict[str, Any], now_utc: datetime | None = None) -> dict[str, Any]:
    """Fetch one configured feed; source failure is recorded and does not abort peers."""
    now = (now_utc or datetime.now(UTC)).astimezone(UTC)
    output: dict[str, Any] = {
        "source_id": source["source_id"],
        "publisher": source["publisher"],
        "domain": source["domain"],
        "url": source["url"],
        "method": source["method"],
        "required": bool(source.get("required")),
        "status": "ERROR",
        "article_count": 0,
        "error": None,
        "articles": [],
    }
    try:
        response = requests.get(
            source["url"],
            headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*"},
            timeout=(4, 12),
        )
        response.raise_for_status()
        parsed_feed = feedparser.parse(response.content)
        if getattr(parsed_feed, "bozo", False) and not parsed_feed.entries:
            raise ValueError("RSS/Atom response could not be parsed")
        accepted = []
        for entry in parsed_feed.entries:
            title = _clean_text(entry.get("title"), 500)
            url = str(entry.get("link") or "").strip()
            if not title or not url or not url.startswith(("http://", "https://")):
                continue
            published = _published_utc(entry)
            if published and published > now + timedelta(hours=2):
                continue
            if published and (now - published) > timedelta(hours=MAX_ARTICLE_AGE_HOURS):
                continue
            summary = _clean_text(entry.get("summary") or entry.get("description") or "", 700)
            source_obj = entry.get("source") or {}
            publisher = str(source_obj.get("title") or source["publisher"]).strip()
            record = {
                "article_id": _article_id(url, title, publisher, published),
                "title": title,
                "summary": summary,
                "url": url,
                "publisher": publisher or source["publisher"],
                "source_id": source["source_id"],
                "source_domain": source["domain"],
                "source_method": source["method"],
                "source_weight": float(source["source_weight"]),
                "published_at_utc": published.isoformat() if published else None,
                "publication_time_quality": "FEED_TIMESTAMP" if published else "MISSING_FEED_TIMESTAMP",
            }
            accepted.append(record)
        output["articles"] = accepted
        output["article_count"] = len(accepted)
        output["status"] = "PASS" if accepted else "NO_RECENT_ARTICLES"
    except Exception as exc:
        output["error"] = f"{type(exc).__name__}: {exc}"
    return output


def collect_sources(
    sources: tuple[dict[str, Any], ...] = SOURCES,
    now_utc: datetime | None = None,
    fetcher=fetch_source,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch sources concurrently while retaining a per-feed health report."""
    now = (now_utc or datetime.now(UTC)).astimezone(UTC)
    source_status: dict[str, dict[str, Any]] = {}
    articles: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(10, max(1, len(sources)))) as pool:
        futures = {pool.submit(fetcher, source, now): source for source in sources}
        for future in as_completed(futures):
            source = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "source_id": source["source_id"],
                    "publisher": source["publisher"],
                    "domain": source["domain"],
                    "url": source["url"],
                    "method": source["method"],
                    "required": bool(source.get("required")),
                    "status": "ERROR",
                    "article_count": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "articles": [],
                }
            source_status[str(source["source_id"])] = {
                key: result.get(key) for key in (
                    "source_id", "publisher", "domain", "url", "method",
                    "required", "status", "article_count", "error",
                )
            }
            articles.extend(result.get("articles") or [])
    ordered = [source_status[s["source_id"]] for s in sources]
    # Duplicated syndicated stories are kept once, with the first source's attribution.
    by_id: dict[str, dict[str, Any]] = {}
    for article in articles:
        by_id.setdefault(str(article["article_id"]), article)
    return list(by_id.values()), ordered


def _sentiment_analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def merge_article_archive(
    previous: pd.DataFrame | None,
    fresh_articles: list[dict[str, Any]],
    now_utc: datetime,
    analyzer=None,
) -> pd.DataFrame:
    """Append newly seen article metadata without backdating first-seen availability."""
    now = now_utc.astimezone(UTC)
    old = previous.copy() if previous is not None and not previous.empty else pd.DataFrame()
    if not old.empty:
        if "article_id" not in old:
            old["article_id"] = old.apply(
                lambda r: _article_id(str(r.get("url", "")), str(r.get("title", "")),
                                      str(r.get("publisher", "")),
                                      pd.to_datetime(r.get("published_at_utc"), utc=True, errors="coerce").to_pydatetime()
                                      if pd.notna(pd.to_datetime(r.get("published_at_utc"), utc=True, errors="coerce"))
                                      else None),
                axis=1,
            )
        if "first_seen_at_utc" not in old:
            old["first_seen_at_utc"] = now.isoformat()
    existing = set(old.get("article_id", pd.Series(dtype=str)).astype(str))
    new_rows = []
    for article in fresh_articles:
        article_id = str(article.get("article_id") or "")
        if not article_id or article_id in existing:
            continue
        row = dict(article)
        row["first_seen_at_utc"] = now.isoformat()
        published = pd.to_datetime(row.get("published_at_utc"), errors="coerce", utc=True)
        row["available_at_utc"] = (
            max(pd.Timestamp(published).to_pydatetime(), now).isoformat()
            if pd.notna(published) else now.isoformat()
        )
        text = f'{row.get("title", "")}. {row.get("summary", "")}'
        row["sentiment_compound"] = (
            float(analyzer.polarity_scores(text)["compound"]) if analyzer is not None else np.nan
        )
        new_rows.append(row)
        existing.add(article_id)

    combined = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    if combined.empty:
        return pd.DataFrame(columns=[
            "article_id", "title", "summary", "url", "publisher", "source_id",
            "source_domain", "source_method", "source_weight", "published_at_utc",
            "first_seen_at_utc", "available_at_utc", "publication_time_quality",
            "sentiment_compound",
        ])
    for col in ("published_at_utc", "first_seen_at_utc", "available_at_utc"):
        if col not in combined:
            combined[col] = pd.NaT
        combined[col] = pd.to_datetime(combined[col], errors="coerce", utc=True)
    combined["article_id"] = combined["article_id"].astype(str)
    combined = combined.drop_duplicates("article_id", keep="first")
    first_seen = combined["first_seen_at_utc"].fillna(pd.Timestamp(now))
    age_from_first_seen = (pd.Timestamp(now) - first_seen).dt.total_seconds() / 3600
    published_age = (pd.Timestamp(now) - combined["published_at_utc"]).dt.total_seconds() / 3600
    keep = age_from_first_seen.le(24 * 30) & (
        combined["published_at_utc"].isna() | published_age.le(24 * 30)
    )
    combined = combined.loc[keep].sort_values(
        ["available_at_utc", "article_id"], kind="mergesort"
    ).reset_index(drop=True)
    return combined


def build_pit_layer(
    articles: pd.DataFrame,
    aliases: dict[str, dict[str, str]],
    analyzer=None,
    now_utc: datetime | None = None,
) -> pd.DataFrame:
    """Convert headline archive to a point-in-time symbol/event layer."""
    if articles is None or articles.empty:
        return pd.DataFrame(columns=[
            "symbol", "decision_timestamp", "news_available", "news_source",
            "news_sentiment", "news_article_count", "news_source_count",
            "news_source_weight", "news_title", "news_url", "news_article_id",
            "published_at_utc", "first_seen_at_utc",
        ])
    now = (now_utc or datetime.now(UTC)).astimezone(UTC)
    rows = []
    freshness_cutoff = pd.Timestamp(now) - pd.Timedelta(days=PIT_FRESHNESS_DAYS)
    for _, article in articles.iterrows():
        available = pd.to_datetime(article.get("available_at_utc"), errors="coerce", utc=True)
        if pd.isna(available) or available > pd.Timestamp(now):
            continue
        if available < freshness_cutoff:
            continue
        pub = pd.to_datetime(article.get("published_at_utc"), errors="coerce", utc=True)
        if pd.notna(pub) and pub > pd.Timestamp(now) + pd.Timedelta(hours=2):
            continue
        title = str(article.get("title") or "")
        summary = str(article.get("summary") or "")
        text = f"{title}. {summary}"
        try:
            from market_watchlist import extract_mentions
            matched = extract_mentions(text, aliases)
        except Exception:
            matched = []
        # Include the MARKET-wide stream for context but never map unmatched items to a stock.
        symbols = matched[:10] if matched else ["MARKET"]
        score = article.get("sentiment_compound")
        try:
            score = float(score)
            if not np.isfinite(score):
                score = np.nan
        except (TypeError, ValueError):
            score = np.nan
        available = pd.Timestamp(available).tz_convert("UTC")
        for symbol in symbols:
            rows.append({
                "symbol": str(symbol).strip().upper(),
                "decision_timestamp": available,
                "news_available": 1.0,
                "news_source": str(article.get("publisher") or article.get("source_id") or "UNKNOWN"),
                "news_sentiment": score,
                "news_article_count": 1,
                "news_source_count": 1,
                "news_source_weight": float(article.get("source_weight", 0.0) or 0.0),
                "news_title": title[:500],
                "news_url": str(article.get("url") or ""),
                "news_article_id": str(article.get("article_id") or ""),
                "published_at_utc": pub,
                "first_seen_at_utc": pd.to_datetime(article.get("first_seen_at_utc"), errors="coerce", utc=True),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "decision_timestamp", "news_available", "news_source",
            "news_sentiment", "news_article_count", "news_source_count",
            "news_source_weight", "news_title", "news_url", "news_article_id",
            "published_at_utc", "first_seen_at_utc",
        ])
    out = pd.DataFrame(rows)
    out["decision_timestamp"] = pd.to_datetime(out["decision_timestamp"], utc=True).astype("datetime64[ns, UTC]")
    # One row per symbol/timestamp is necessary for deterministic backward as-of joins.
    out = out.sort_values(["symbol", "decision_timestamp", "news_source_weight"], kind="mergesort")
    out = out.groupby(["symbol", "decision_timestamp"], as_index=False).agg(
        news_available=("news_available", "max"),
        news_source=("news_source", lambda s: ";".join(sorted(set(str(v) for v in s if v))[:8])),
        news_sentiment=("news_sentiment", "mean"),
        news_article_count=("news_article_id", "nunique"),
        news_source_count=("news_source", "nunique"),
        news_source_weight=("news_source_weight", "mean"),
        news_title=("news_title", "last"),
        news_url=("news_url", "last"),
        news_article_id=("news_article_id", lambda s: ";".join(sorted(set(str(v) for v in s if v))[:8])),
        published_at_utc=("published_at_utc", "max"),
        first_seen_at_utc=("first_seen_at_utc", "max"),
    )
    return out.sort_values(["symbol", "decision_timestamp"]).reset_index(drop=True)


def load_nse_universe():
    from delivery_training_universe import load_official_nse_universe
    from market_watchlist import build_alias_map

    frame = load_official_nse_universe(force_refresh=False)
    aliases = build_alias_map(frame)
    if not aliases:
        raise RuntimeError("Official NSE universe was loaded but no company/symbol aliases were resolved")
    return frame, aliases


def _drive():
    from persistent_store import GoogleDriveStore
    return GoogleDriveStore()


def _download_remote(store: Any, remote: str, local: Path) -> bool:
    from googleapiclient.http import MediaIoBaseDownload

    parts = [part for part in remote.strip("/").split("/") if part]
    if not parts:
        return False
    parent = store.root_id
    folder_mime = "application/vnd.google-apps.folder"
    for part in parts[:-1]:
        hit = store._find_child(parent, part, folder_mime)
        if not hit:
            return False
        parent = hit["id"]
    hit = store._find_child(parent, parts[-1])
    if not hit:
        return False
    local.parent.mkdir(parents=True, exist_ok=True)
    request = store.service.files().get_media(fileId=hit["id"])
    with local.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return True


def _within_collection_window(now_ist: datetime) -> tuple[bool, str]:
    # News can affect the next NSE session even on exchange holidays/weekends.
    # This is a news-collection window, deliberately independent of the cash session.
    if now_ist.time() < time(7, 30) or now_ist.time() > time(22, 0):
        return False, "OUTSIDE_0730_TO_2200_IST_COLLECTION_WINDOW"
    return True, "IN_WINDOW"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", default="/tmp/Indian_Market_Research")
    parser.add_argument("--market-window", action="store_true",
                        help="Skip scheduled runs outside NSE business days and 07:30-15:30 IST.")
    parser.add_argument("--max-article-age-hours", type=int, default=MAX_ARTICLE_AGE_HOURS)
    args = parser.parse_args()
    if args.max_article_age_hours < 1:
        parser.error("--max-article-age-hours must be positive")

    now = datetime.now(IST)
    if args.market_window:
        in_window, reason = _within_collection_window(now)
        if not in_window:
            print(json.dumps({
                "status": "SKIPPED", "reason": reason, "now_ist": now.isoformat()
            }, indent=2))
            return 0

    now_utc = now.astimezone(UTC)
    root = Path(args.drive_root)
    news_dir = root / LIVE_NEWS_DIR
    news_dir.mkdir(parents=True, exist_ok=True)
    articles_path = news_dir / "website_news_articles_2026.parquet"
    pit_path = news_dir / "website_news_pit_2026.parquet"
    manifest_path = news_dir / "website_news_manifest_2026.json"

    drive_status = None
    store = None
    try:
        from persistent_store import drive_status as check_drive
        drive_status = check_drive(require_write=True)
        if not drive_status.get("configured") or drive_status.get("reachable") is False or drive_status.get("write_ready") is False:
            raise RuntimeError("Google Drive read/write preflight failed: " + str(drive_status.get("reason", "not ready")))
        store = _drive()
        remote_found = _download_remote(store, ARTICLES_REMOTE, articles_path)
        print(f"EXISTING_NEWS_ARCHIVE: found={remote_found}", flush=True)
    except Exception as exc:
        print(f"GOOGLE_DRIVE_PREFLIGHT_ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 2

    try:
        universe, aliases = load_nse_universe()
        universe_status = {"status": "PASS", "symbols": int(len(universe)), "aliases": int(len(aliases))}
    except Exception as exc:
        aliases = {}
        universe_status = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
        print("NSE_UNIVERSE_MATCHING_UNAVAILABLE:", universe_status, flush=True)

    analyzer = _sentiment_analyzer()
    existing_articles = pd.read_parquet(articles_path) if articles_path.exists() and articles_path.stat().st_size else pd.DataFrame()
    fresh_articles, feed_status = collect_sources(now_utc=now_utc)
    article_archive = merge_article_archive(existing_articles, fresh_articles, now_utc, analyzer=analyzer)

    # Sentiment scores are archived with the article, not used as an unvalidated trade rule.
    # If sentiment library is absent, prices/news availability still flow through the pipeline.
    if analyzer is not None and "sentiment_compound" not in article_archive.columns:
        article_archive["sentiment_compound"] = np.nan
    pit = build_pit_layer(article_archive, aliases, analyzer=analyzer, now_utc=now_utc)

    articles_path.parent.mkdir(parents=True, exist_ok=True)
    article_archive.to_parquet(articles_path, index=False)
    pit.to_parquet(pit_path, index=False)

    healthy_statuses = {"PASS", "NO_RECENT_ARTICLES"}
    successful = sum(source.get("status") in healthy_statuses for source in feed_status)
    required = [source for source in feed_status if source.get("required")]
    required_success = sum(source.get("status") in healthy_statuses for source in required)
    manifest = {
        "schema_version": "website-market-news-pit-v1",
        "source": "publisher RSS / publisher-filtered Google News RSS; headline and summary only",
        "collection_status": (
            "PASS" if (
                successful >= MIN_HEALTHY_FEEDS
                and required_success >= 2
                and universe_status.get("status") == "PASS"
            ) else "PARTIAL"
        ),
        "coverage_complete": bool(
            successful >= MIN_HEALTHY_FEEDS
            and required_success >= 2
            and universe_status.get("status") == "PASS"
        ),
        "checked_at_utc": now_utc.isoformat(),
        "checked_at_ist": now.isoformat(),
        "coverage_start": str(pd.to_datetime(pit["decision_timestamp"], utc=True).min().date()) if not pit.empty else None,
        "coverage_end": str(pd.to_datetime(pit["decision_timestamp"], utc=True).max().date()) if not pit.empty else None,
        "feed_count": len(feed_status),
        "successful_feed_count": successful,
        "required_feed_count": len(required),
        "successful_required_feed_count": required_success,
        "new_article_count": len(fresh_articles),
        "retained_article_count": len(article_archive),
        "symbol_news_rows": len(pit),
        "matched_symbols": int(pit.loc[pit["symbol"].ne("MARKET"), "symbol"].nunique()) if not pit.empty else 0,
        "sentiment_model": "VADER compound; metadata only until a news-augmented predictor is separately trained and validated",
        "nse_universe": universe_status,
        "drive_preflight": {
            key: drive_status.get(key) for key in (
                "configured", "reachable", "write_ready", "auth_method", "write_auth_method"
            )
        },
        "sources": feed_status,
        "output_paths": {
            "articles": ARTICLES_REMOTE,
            "pit": PIT_REMOTE,
            "manifest": MANIFEST_REMOTE,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    assert store is not None
    for local, remote, mime_type in (
        (articles_path, ARTICLES_REMOTE, "application/octet-stream"),
        (pit_path, PIT_REMOTE, "application/octet-stream"),
        (manifest_path, MANIFEST_REMOTE, "application/json"),
    ):
        upload = store.upload_or_update(local, remote, mime_type=mime_type)
        print(json.dumps({"upload_path": remote, "result": str(upload)}, default=str), flush=True)

    print(json.dumps(manifest, indent=2, default=str), flush=True)
    # Partial source availability is useful data but must not be described as full coverage.
    return 0 if manifest["collection_status"] == "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
