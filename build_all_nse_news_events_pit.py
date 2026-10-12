#!/usr/bin/env python3
"""Build broad NSE PIT news/events layers for the ALL-NSE model.

The pilot layer covered only three symbols. This builder intentionally removes
that universe restriction: NSE's equity corporate-announcements and corporate-
actions endpoints are requested without a symbol filter, then intersected with
the current official NSE equity universe.

Outputs are written to the same Drive paths already consumed by the ALL-NSE
Delivery builder, so no model feature-contract change is required:
  processed/pit_replay/NEWS_ENRICHED_PIT_REPLAY_2026.parquet
  processed/pit_replay/EVENTS_FUNDAMENTALS_PIT_REPLAY_2026.parquet

PIT rule: use the NSE broadcast/announcement timestamp as availability time.
No future observation is created. Corporate actions are available from the
announcement/API record date and are represented as event availability.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from delivery_training_universe import official_nse_symbols, normalize_symbol

NSE_HOME = "https://www.nseindia.com/"
ANN_API = "https://www.nseindia.com/api/corporate-announcements"
CA_API = "https://www.nseindia.com/api/corporates-corporateActions"

DEFAULT_ROOT = Path("/content/drive/MyDrive/Indian_Market_Research")
NEWS_OUT = DEFAULT_ROOT / "processed/pit_replay/NEWS_ENRICHED_PIT_REPLAY_2026.parquet"
EVENT_OUT = DEFAULT_ROOT / "processed/pit_replay/EVENTS_FUNDAMENTALS_PIT_REPLAY_2026.parquet"
CACHE_ROOT = DEFAULT_ROOT / "processed/pit_replay/nse_all_universe_raw_2026"

HIGH_IMPACT = (
    "merger", "demerger", "acquisition", "takeover", "open offer", "buyback",
    "delisting", "insolvency", "resolution plan", "default", "fraud", "forensic",
    "regulatory", "penalty", "fine", "order", "contract", "award", "major order",
    "fund raising", "fundraise", "rights", "bonus", "split", "dividend",
    "result", "financial results", "director", "management change", "resignation",
    "appointment", "credit rating", "downgrade", "upgrade", "litigation", "court",
    "approval", "commissioning", "commercial production", "shutdown", "fire",
    "strike", "plant", "capacity", "promoter", "pledge", "shareholding",
)
EVENT_WORDS = (
    "board meeting", "financial result", "results", "dividend", "bonus", "split",
    "rights", "buyback", "merger", "demerger", "acquisition", "takeover",
    "open offer", "fund raising", "fundraising", "shareholders meeting", "agm",
    "egm", "record date", "book closure", "credit rating", "insolvency",
    "director", "appointment", "resignation", "trading window", "corporate action",
    "capital reduction", "scheme of arrangement", "delisting", "court", "litigation",
)


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/134 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
        "X-Requested-With": "XMLHttpRequest",
        "Connection": "keep-alive",
    })
    last = None
    for attempt in range(4):
        try:
            r = s.get(NSE_HOME, timeout=20)
            if r.status_code in (200, 403):
                return s
        except Exception as exc:
            last = exc
        time.sleep(1.5 * (attempt + 1))
    if last:
        print(f"NSE warmup warning: {last}", flush=True)
    return s


def chunks(start: date, end: date, days: int = 31):
    cur = start
    while cur <= end:
        nxt = min(end, cur + timedelta(days=days - 1))
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def get_json(s: requests.Session, url: str, params: dict[str, Any], label: str):
    last = None
    for attempt in range(5):
        try:
            r = s.get(url, params=params, timeout=45)
            if r.status_code == 200:
                ct = r.headers.get("content-type", "")
                if "json" in ct.lower() or r.text.lstrip().startswith(("{", "[")):
                    return r.json()
                raise RuntimeError(f"non-JSON response content-type={ct}")
            if r.status_code in (401, 403, 429, 500, 502, 503, 504):
                last = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(min(12.0, 1.5 * (attempt + 1)))
                if r.status_code in (401, 403):
                    try:
                        s.get(NSE_HOME, timeout=15)
                    except Exception:
                        pass
                continue
            r.raise_for_status()
        except Exception as exc:
            last = exc
            time.sleep(min(12.0, 1.5 * (attempt + 1)))
    print(f"{label}: failed after retries: {last}", flush=True)
    return None


def records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("data", "records", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        # Some NSE responses can be a single record.
        if any(k in payload for k in ("symbol", "SYMBOL", "sym", "symbolName")):
            return [payload]
    return []


def pick(d: dict[str, Any], *names: str) -> Any:
    for n in names:
        if n in d and d[n] not in (None, "", "nan"):
            return d[n]
    return None


def parse_ts(v: Any, fallback: date | None = None) -> pd.Timestamp:
    if v is None or str(v).strip() == "":
        return pd.Timestamp(fallback, tz="Asia/Kolkata") if fallback else pd.NaT
    x = pd.to_datetime(v, errors="coerce", utc=True, dayfirst=True)
    if pd.isna(x):
        x = pd.to_datetime(v, errors="coerce", dayfirst=True)
        if pd.isna(x):
            return pd.Timestamp(fallback, tz="Asia/Kolkata") if fallback else pd.NaT
        x = x.tz_localize("Asia/Kolkata") if x.tzinfo is None else x.tz_convert("Asia/Kolkata")
        return x.tz_convert("UTC")
    return x


def text_of(r: dict[str, Any]) -> str:
    vals = [
        pick(r, "subject", "SUBJECT", "desc", "DESCRIPTION", "eventDescription", "EVENT_DESCRIPTION"),
        pick(r, "details", "DETAILS", "attchmntText", "ATTACHMENT_TEXT", "description"),
        pick(r, "companyName", "COMPANY_NAME", "company", "COMPANY"),
    ]
    return " ".join(str(x) for x in vals if x not in (None, "")).strip()


def fetch_announcements(s: requests.Session, start: date, end: date, cache_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a, b in chunks(start, end):
        key = f"ann_{a:%Y%m%d}_{b:%Y%m%d}.json"
        cp = cache_dir / key
        payload = None
        if cp.exists() and cp.stat().st_size:
            try:
                payload = json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                payload = None
        if payload is None:
            payload = get_json(s, ANN_API, {
                "index": "equities",
                "from_date": a.strftime("%d-%m-%Y"),
                "to_date": b.strftime("%d-%m-%Y"),
            }, f"announcements {a}..{b}")
            if payload is None:
                raise RuntimeError(
                    f"Refusing to publish partial PIT coverage: {label} request "
                    f"for {a}..{b} failed after retries."
                )
            cp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        rr = records(payload)
        out.extend(rr)
        print(f"ANN {a}..{b}: rows={len(rr):,}", flush=True)
        time.sleep(0.25)
    return out


def fetch_actions(s: requests.Session, start: date, end: date, cache_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a, b in chunks(start, end):
        key = f"ca_{a:%Y%m%d}_{b:%Y%m%d}.json"
        cp = cache_dir / key
        payload = None
        if cp.exists() and cp.stat().st_size:
            try:
                payload = json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                payload = None
        if payload is None:
            payload = get_json(s, CA_API, {
                "index": "equities",
                "from_date": a.strftime("%d-%m-%Y"),
                "to_date": b.strftime("%d-%m-%Y"),
            }, f"corporate actions {a}..{b}")
            if payload is None:
                continue
            cp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        rr = records(payload)
        out.extend(rr)
        print(f"CA  {a}..{b}: rows={len(rr):,}", flush=True)
        time.sleep(0.25)
    return out


def build_layers(ann_rows, action_rows, universe: set[str], start: date, end: date):
    news = []
    events = []

    for r in ann_rows:
        sym = normalize_symbol(pick(r, "symbol", "SYMBOL", "sym", "symbolName", "NSE_SYMBOL"))
        if not sym or sym not in universe:
            continue
        text = text_of(r)
        ts = parse_ts(pick(r, "broadcastDate", "BROADCAST_DATE", "broadcastDateTime", "BROADCAST_DATE_TIME", "an_dt", "date"), None)
        if pd.isna(ts):
            continue
        subject = str(pick(r, "subject", "SUBJECT", "desc", "DESCRIPTION", "eventDescription", "EVENT_DESCRIPTION") or "").strip()
        low = text.lower()
        is_event = any(k in low for k in EVENT_WORDS)
        is_high = any(k in low for k in HIGH_IMPACT)
        news.append({
            "symbol": sym,
            "decision_timestamp": ts,
            "news_available": 1,
            "news_source": "NSE_CORPORATE_ANNOUNCEMENT",
            "news_subject": subject[:500],
        })
        if is_event:
            events.append({
                "symbol": sym,
                "decision_timestamp": ts,
                "event_available": 1,
                "event_high_impact": int(is_high),
                "event_source": "NSE_CORPORATE_ANNOUNCEMENT",
                "event_text": text[:1000],
            })

    for r in action_rows:
        sym = normalize_symbol(pick(r, "SYMBOL", "symbol", "sym"))
        if not sym or sym not in universe:
            continue
        # Ex-date is the corporate-action effective date, not information-availability time.
        # Never use it as PIT timestamp; require an announcement/publication/API timestamp.
        ts = parse_ts(pick(r, "ANNOUNCEMENT_DATE", "announcementDate", "broadcastDate", "BROADCAST_DATE", "broadcastDateTime", "BROADCAST_DATE_TIME", "date"), None)
        if pd.isna(ts):
            continue
        purpose = str(pick(r, "PURPOSE", "purpose", "details", "DETAILS") or "").strip()
        low = purpose.lower()
        is_high = any(k in low for k in HIGH_IMPACT)
        events.append({
            "symbol": sym,
            "decision_timestamp": ts,
            "event_available": 1,
            "event_high_impact": int(is_high),
            "event_source": "NSE_CORPORATE_ACTION",
            "event_text": purpose[:1000],
        })

    n = pd.DataFrame(news)
    e = pd.DataFrame(events)
    for frame in (n, e):
        if not frame.empty:
            frame["decision_timestamp"] = pd.to_datetime(frame["decision_timestamp"], utc=True, errors="coerce")
            frame.dropna(subset=["symbol", "decision_timestamp"], inplace=True)
            frame.sort_values(["symbol", "decision_timestamp"], inplace=True)
            frame.drop_duplicates(["symbol", "decision_timestamp", *(["news_subject"] if "news_subject" in frame.columns else ["event_text"])], keep="last", inplace=True)
    # The merger only needs availability flags. Keep one row per symbol/time,
    # aggregating multiple announcements/actions occurring at the same time.
    if not n.empty:
        n = n.groupby(["symbol", "decision_timestamp"], as_index=False).agg(
            news_available=("news_available", "max"),
            news_source=("news_source", lambda x: ";".join(sorted(set(map(str, x)))[:4]),),
        )
    if not e.empty:
        e = e.groupby(["symbol", "decision_timestamp"], as_index=False).agg(
            event_available=("event_available", "max"),
            event_high_impact=("event_high_impact", "max"),
            event_source=("event_source", lambda x: ";".join(sorted(set(map(str, x)))[:4]),),
        )
    return n, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="", help="Coverage end date YYYY-MM-DD; defaults to today in Asia/Kolkata.")
    args = ap.parse_args()

    root = Path(args.drive_root)
    news_out = root / "processed/pit_replay/NEWS_ENRICHED_PIT_REPLAY_2026.parquet"
    event_out = root / "processed/pit_replay/EVENTS_FUNDAMENTALS_PIT_REPLAY_2026.parquet"
    cache = root / "processed/pit_replay/nse_all_universe_raw_2026"
    cache.mkdir(parents=True, exist_ok=True)
    news_out.parent.mkdir(parents=True, exist_ok=True)

    previous_manifest = {}
    previous_manifest_path = cache / "ALL_NSE_NEWS_EVENTS_MANIFEST_2026.json"
    try:
        if previous_manifest_path.exists() and previous_manifest_path.stat().st_size:
            previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        previous_manifest = {}

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = (
        datetime.strptime(args.end, "%Y-%m-%d").date()
        if args.end.strip()
        else datetime.now(ZoneInfo("Asia/Kolkata")).date()
    )
    if start > end:
        raise ValueError(f"Invalid PIT coverage range: start={start} is after end={end}")
    universe = set(official_nse_symbols(force_refresh=False))
    print(f"ALL-NSE NEWS/EVENTS: universe={len(universe):,} range={start}..{end}", flush=True)

    s = session()
    ann = fetch_announcements(s, start, end, cache)
    ca = fetch_actions(s, start, end, cache)
    news, events = build_layers(ann, ca, universe, start, end)

    # Merge with any previous file only when it contains data from the same broad
    # builder. This makes daily reruns append/update rather than erase history.
    if news_out.exists() and news_out.stat().st_size:
        try:
            old = pd.read_parquet(news_out)
            news = pd.concat([old, news], ignore_index=True)
        except Exception:
            pass
    if event_out.exists() and event_out.stat().st_size:
        try:
            old = pd.read_parquet(event_out)
            events = pd.concat([old, events], ignore_index=True)
        except Exception:
            pass

    if not news.empty:
        news["symbol"] = news["symbol"].map(normalize_symbol)
        news["decision_timestamp"] = pd.to_datetime(news["decision_timestamp"], utc=True, errors="coerce")
        news = news.dropna(subset=["symbol", "decision_timestamp"]).drop_duplicates(["symbol", "decision_timestamp"], keep="last")
        news = news.sort_values(["symbol", "decision_timestamp"]).reset_index(drop=True)
        news.to_parquet(news_out, index=False)
    else:
        pd.DataFrame(columns=["symbol", "decision_timestamp", "news_available"]).to_parquet(news_out, index=False)

    if not events.empty:
        events["symbol"] = events["symbol"].map(normalize_symbol)
        events["decision_timestamp"] = pd.to_datetime(events["decision_timestamp"], utc=True, errors="coerce")
        events = events.dropna(subset=["symbol", "decision_timestamp"]).drop_duplicates(["symbol", "decision_timestamp"], keep="last")
        events = events.sort_values(["symbol", "decision_timestamp"]).reset_index(drop=True)
        events.to_parquet(event_out, index=False)
    else:
        pd.DataFrame(columns=["symbol", "decision_timestamp", "event_available", "event_high_impact"]).to_parquet(event_out, index=False)

    coverage_start = start
    coverage_end = end
    prior_start = pd.to_datetime(previous_manifest.get("coverage_start"), errors="coerce")
    prior_end = pd.to_datetime(previous_manifest.get("coverage_end"), errors="coerce")
    prior_complete = previous_manifest.get("coverage_complete") is True
    adjacent_or_overlapping = (
        prior_complete
        and pd.notna(prior_start)
        and pd.notna(prior_end)
        and start <= (prior_end.date() + timedelta(days=1))
        and end >= (prior_start.date() - timedelta(days=1))
    )
    if adjacent_or_overlapping:
        coverage_start = min(start, prior_start.date())
        coverage_end = max(end, prior_end.date())

    manifest = {
        "source": "NSE_OFFICIAL_CORPORATE_ANNOUNCEMENTS_AND_ACTIONS",
        "universe_size": len(universe),
        "requested_range": [start.isoformat(), end.isoformat()],
        "coverage_start": coverage_start.isoformat(),
        "coverage_end": coverage_end.isoformat(),
        "prior_coverage_extended": bool(adjacent_or_overlapping),
        # Reaching this write means every requested date chunk returned a valid response.
        # Coverage is extended only when the previous complete range overlaps/is adjacent.
        "coverage_complete": True,
        "announcement_rows_fetched": len(ann),
        "corporate_action_rows_fetched": len(ca),
        "news_rows": int(len(news)),
        "news_symbols": int(news["symbol"].nunique()) if not news.empty else 0,
        "event_rows": int(len(events)),
        "event_symbols": int(events["symbol"].nunique()) if not events.empty else 0,
        "pit_rule": "NSE broadcast/announcement or corporate-action availability timestamp; backward PIT merge",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = cache / "ALL_NSE_NEWS_EVENTS_MANIFEST_2026.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Colab writes directly to the mounted Drive. In GitHub Actions, persist
    # the validated files via the configured Drive connector so the downstream
    # Delivery refresh consumes the same source files instead of ephemeral CI files.
    if not str(root).startswith("/content/drive/MyDrive/"):
        from persistent_store import GoogleDriveStore
        store = GoogleDriveStore()
        store.upload_or_update(
            news_out,
            "processed/pit_replay/NEWS_ENRICHED_PIT_REPLAY_2026.parquet",
            mime_type="application/octet-stream",
        )
        store.upload_or_update(
            event_out,
            "processed/pit_replay/EVENTS_FUNDAMENTALS_PIT_REPLAY_2026.parquet",
            mime_type="application/octet-stream",
        )
        store.upload_or_update(
            manifest_path,
            "processed/pit_replay/nse_all_universe_raw_2026/ALL_NSE_NEWS_EVENTS_MANIFEST_2026.json",
            mime_type="application/json",
        )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
