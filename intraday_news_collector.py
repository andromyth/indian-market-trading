"""15-minute, auditable news collection for the autonomous NSE intraday worker.

The collector reuses the repository's multi-publisher Google News RSS adapter,
matches headlines only to symbols in the official NSE universe, derives
transparent per-symbol news context, and archives every collection to Drive.
This context is attached to forecasts; it does not alter a trained model's
feature contract or imply that the classifier used news as a numeric feature.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
HIGH_IMPACT_TERMS = (
    "results", "earnings", "profit", "loss", "guidance", "order", "contract",
    "acquisition", "merger", "demerger", "buyback", "dividend", "default",
    "fraud", "regulatory", "penalty", "downgrade", "upgrade", "credit rating",
    "plant shutdown", "fire", "strike", "court", "litigation", "insolvency",
    "fund raising", "fundraise", "pledge", "promoter", "block deal",
)


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        return value if pd.notna(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _article_id(article: dict[str, Any]) -> str:
    canonical = str(article.get("url") or "").strip()
    if not canonical:
        canonical = "|".join((
            str(article.get("publisher") or ""),
            str(article.get("title") or ""),
            str(article.get("published_utc") or ""),
        ))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sentiment(text: str) -> dict[str, Any]:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        scores = SentimentIntensityAnalyzer().polarity_scores(text)
        compound = float(scores.get("compound", 0.0))
        label = "POSITIVE" if compound >= 0.05 else ("NEGATIVE" if compound <= -0.05 else "NEUTRAL")
        return {"compound": compound, "label": label, "method": "VADER"}
    except Exception:
        return {"compound": None, "label": "UNAVAILABLE", "method": "UNAVAILABLE"}


def _aliases(universe: pd.DataFrame) -> dict[str, dict[str, str]]:
    from market_watchlist import build_alias_map
    return build_alias_map(universe)


def collect_intraday_news(
    universe: pd.DataFrame,
    *,
    now: datetime | None = None,
    persist_to_drive: bool = True,
    local_root: Path = Path("data/autonomous_intraday/news"),
) -> dict[str, Any]:
    """Fetch the existing multi-publisher news feed, enrich it and archive it."""
    from market_watchlist import fetch_watchlist_articles, extract_mentions

    checked = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    source_result = fetch_watchlist_articles(now=checked)
    aliases = _aliases(universe)
    articles = []
    by_symbol: dict[str, dict[str, Any]] = {}

    for raw in source_result.get("articles", []) or []:
        title = str(raw.get("title") or "").strip()
        body = str(raw.get("text") or title).strip()
        url = str(raw.get("url") or "").strip()
        publisher = str(raw.get("publisher") or "Unknown").strip()
        domain = str(raw.get("domain") or "").strip()
        published = str(raw.get("published_utc") or "").strip()
        combined = f"{title} {body}".strip()
        mentions = extract_mentions(combined, aliases)
        sentiment = _sentiment(combined)
        lower = combined.lower()
        event_terms = [term for term in HIGH_IMPACT_TERMS if term in lower]
        item = {
            "article_id": _article_id(raw),
            "collected_at_utc": checked.isoformat(),
            "collected_at_ist": checked.astimezone(IST).isoformat(),
            "published_at_utc": published or None,
            "publisher": publisher,
            "source_domain": domain,
            "title": title,
            "url": url,
            "matched_nse_symbols": mentions,
            "sentiment": sentiment,
            "high_impact_candidate": bool(event_terms),
            "matched_event_terms": event_terms,
        }
        articles.append(item)

        for symbol in mentions:
            entry = by_symbol.setdefault(symbol, {
                "symbol": symbol,
                "article_count": 0,
                "publishers": set(),
                "latest_published_at_utc": None,
                "sentiment_compound_sum": 0.0,
                "sentiment_count": 0,
                "positive_articles": 0,
                "negative_articles": 0,
                "high_impact_article_count": 0,
                "headlines": [],
                "source_articles": [],
            })
            entry["article_count"] += 1
            entry["publishers"].add(publisher)
            if published and (
                entry["latest_published_at_utc"] is None
                or published > entry["latest_published_at_utc"]
            ):
                entry["latest_published_at_utc"] = published
            compound = sentiment.get("compound")
            if isinstance(compound, (float, int)):
                entry["sentiment_compound_sum"] += float(compound)
                entry["sentiment_count"] += 1
                entry["positive_articles"] += int(compound >= 0.05)
                entry["negative_articles"] += int(compound <= -0.05)
            entry["high_impact_article_count"] += int(bool(event_terms))
            article_ref = {
                "article_id": item["article_id"],
                "title": title,
                "url": url,
                "publisher": publisher,
                "published_at_utc": published or None,
                "sentiment": sentiment["label"],
                "high_impact_candidate": bool(event_terms),
                "matched_event_terms": event_terms,
            }
            if not any(x["article_id"] == article_ref["article_id"] for x in entry["headlines"]):
                entry["headlines"].append(article_ref)
            entry["source_articles"].append(item["article_id"])

    symbol_context = {}
    for symbol, item in by_symbol.items():
        sentiment_count = item.pop("sentiment_count")
        sentiment_sum = item.pop("sentiment_compound_sum")
        publishers = sorted(item.pop("publishers"))
        item["publisher_count"] = len(publishers)
        item["publishers"] = publishers
        item["mean_sentiment_compound"] = (
            sentiment_sum / sentiment_count if sentiment_count else None
        )
        item["sentiment_available"] = bool(sentiment_count)
        item["news_available"] = True
        item["headlines"] = sorted(
            item["headlines"],
            key=lambda x: x.get("published_at_utc") or "",
            reverse=True,
        )[:5]
        symbol_context[symbol] = item

    source_status = source_result.get("sources", []) or []
    report = {
        "schema_version": "intraday-news-context-v1",
        "collection_id": checked.strftime("%Y%m%dT%H%M%SZ"),
        "collected_at_utc": checked.isoformat(),
        "collected_at_ist": checked.astimezone(IST).isoformat(),
        "collector": "intraday_news_collector",
        "source_count": len(source_status),
        "source_success_count": sum(
            str(x.get("status", "")).upper() == "PASS" for x in source_status
        ),
        "source_status": source_status,
        "article_count": len(articles),
        "matched_article_count": sum(bool(x["matched_nse_symbols"]) for x in articles),
        "matched_symbol_count": len(symbol_context),
        "articles": articles,
        "symbol_context": symbol_context,
    }

    local_root.mkdir(parents=True, exist_ok=True)
    local_file = local_root / f"news_{checked.astimezone(IST):%Y%m%d_%H%M}.json"
    local_file.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
    report["local_archive_path"] = str(local_file)
    report["drive_persistence_status"] = "NOT_REQUESTED"

    if persist_to_drive:
        try:
            from persistent_store import GoogleDriveStore
            store = GoogleDriveStore()
            local_date = checked.astimezone(IST)
            remote_dir = (
                f"production_data/intraday_news/"
                f"{local_date:%Y/%m/%d}"
            )
            remote_archive = f"{remote_dir}/news_{local_date:%Y%m%d_%H%M}.json"
            remote_latest = "production_data/intraday_news/news_context_latest.json"
            archive_result = store.upload_or_update(
                local_file, remote_archive, mime_type="application/json"
            )
            latest_file = local_root / "news_context_latest.json"
            latest_file.write_text(
                json.dumps(_jsonable(report), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            latest_result = store.upload_or_update(
                latest_file, remote_latest, mime_type="application/json"
            )
            report["drive_persistence_status"] = "OK"
            report["drive_archive_path"] = remote_archive
            report["drive_latest_path"] = remote_latest
            report["drive_upload_results"] = {
                "archive": _jsonable(archive_result),
                "latest": _jsonable(latest_result),
            }
            report["drive_upload_results"]["archive_path"] = remote_archive
            report["drive_upload_results"]["latest_path"] = remote_latest
        except Exception as exc:
            report["drive_persistence_status"] = "ERROR"
            report["drive_persistence_error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps({
        "event": "intraday_news_collection_completed",
        "collection_id": report["collection_id"],
        "source_count": report["source_count"],
        "source_success_count": report["source_success_count"],
        "article_count": report["article_count"],
        "matched_article_count": report["matched_article_count"],
        "matched_symbol_count": report["matched_symbol_count"],
        "drive_persistence_status": report["drive_persistence_status"],
        "drive_archive_path": report.get("drive_archive_path"),
    }, sort_keys=True), flush=True)
    report["source_result"] = source_result
    return report
