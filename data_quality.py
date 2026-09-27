"""Data provenance, integrity and NSE reconciliation for the local historical archive.

The module deliberately treats third-party archives as *research inputs*, not as
exchange-authoritative truth. It performs local bar-quality checks and can compare
locally aggregated daily OHLCV against NSE's official historical equity endpoint.

Important: NSE's public web/API endpoints can rate-limit or block automated clients.
All NSE reconciliation functions are therefore best-effort and record HTTP/error
status instead of silently treating a failed request as a successful validation.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from free_data import normalize_1m_frame

IST = "Asia/Kolkata"
NSE_HOME = "https://www.nseindia.com/"
NSE_HIST_EQUITY = "https://www.nseindia.com/api/historical/cm/equity"

SOURCE_REGISTRY = {
    "dhanhq": {
        "name": "DhanHQ v2 Data APIs",
        "authority": "broker_data_api",
        "status": "broker_primary_when_configured",
        "url": "https://dhanhq.co/docs/v2/historical-data/",
        "claimed_coverage": "Daily since instrument inception; 1/5/15/25/60-minute intraday up to 5 years for active instruments, subject to plan and API limits",
    },
    "nse_official": {
        "name": "NSE India official historical equity reports/API",
        "authority": "exchange",
        "status": "authoritative_primary",
        "url": "https://www.nseindia.com/static/resources/historical-reports-capital-market-daily-monthly-archives",
    },
    "ganesh_2018_2025": {
        "name": "ganeshbiyer/Nse_Historical_Data",
        "authority": "third_party_public_archive",
        "status": "unverified_third_party",
        "url": "https://github.com/ganeshbiyer/Nse_Historical_Data",
        "claimed_coverage": "1-minute Nifty 500 data, 2018 to 2025-12-31",
    },
    "ganesh_2026": {
        "name": "ganeshbiyer/Nse_Historical_Data_2026",
        "authority": "third_party_public_archive",
        "status": "unverified_third_party",
        "url": "https://github.com/ganeshbiyer/Nse_Historical_Data_2026",
        "claimed_coverage": "1-minute Nifty 500 data for 2026",
    },
    "voletiramu_fno": {
        "name": "voletiramu/nse-fno-1min-data",
        "authority": "third_party_public_archive",
        "status": "unverified_third_party",
        "url": "https://github.com/voletiramu/nse-fno-1min-data",
        "claimed_coverage": "214 F&O underlyings, 1-minute spot bars, 2024-04-01 to 2026-04-30",
    },
}


@dataclass
class IntegritySummary:
    rows: int = 0
    start: str | None = None
    end: str | None = None
    duplicate_timestamps: int = 0
    non_monotonic: int = 0
    weekend_rows: int = 0
    outside_market_hours: int = 0
    nonpositive_prices: int = 0
    ohlc_violations: int = 0
    missing_minutes_within_session: int = 0
    session_count: int = 0
    expected_bars: int = 0
    observed_bars: int = 0
    coverage_pct: float = 0.0
    large_move_days: int = 0
    status: str = "UNKNOWN"


@dataclass
class ReconciliationSummary:
    symbol: str
    requested_from: str
    requested_to: str
    nse_rows: int = 0
    local_days: int = 0
    matched_days: int = 0
    price_mismatches: int = 0
    volume_mismatches: int = 0
    missing_in_local: int = 0
    missing_in_nse: int = 0
    max_open_diff_pct: float = 0.0
    max_high_diff_pct: float = 0.0
    max_low_diff_pct: float = 0.0
    max_close_diff_pct: float = 0.0
    max_volume_diff_pct: float = 0.0
    status: str = "UNKNOWN"
    note: str = ""


def _nse_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "X-Requested-With": "XMLHttpRequest",
        "Connection": "keep-alive",
    }


def normalize_nse_eod(payload) -> pd.DataFrame:
    """Normalize NSE historical equity JSON or CSV-like payload to OHLCV."""
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, pd.DataFrame):
        raw = payload.copy()
    elif isinstance(payload, dict):
        data = payload.get("data", payload)
        if isinstance(data, list):
            raw = pd.DataFrame(data)
        elif isinstance(data, dict):
            raw = pd.DataFrame([data])
        else:
            return pd.DataFrame()
    elif isinstance(payload, (bytes, bytearray)):
        try:
            raw = pd.read_csv(BytesIO(payload))
        except Exception:
            return pd.DataFrame()
    else:
        return pd.DataFrame()
    if raw.empty:
        return raw
    canonical = {"Open", "High", "Low", "Close", "Volume"}
    if canonical.issubset(set(raw.columns)):
        out = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        idx = pd.to_datetime(raw.index, errors="coerce", dayfirst=True)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(IST).tz_localize(None)
        out.index = idx
        for c in canonical:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out.dropna(subset=["Open", "High", "Low", "Close"])

    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in raw.columns}

    def pick(*names):
        for name in names:
            if name in lookup:
                return lookup[name]
        return None

    date_col = pick("ch_timestamp", "date", "date_", "timestamp", "ch_timestamp_")
    mapping = {
        "Open": pick("ch_opening_price", "open", "opening_price", "ch_opening_price_"),
        "High": pick("ch_trade_high_price", "high", "trade_high_price", "ch_trade_high_price_"),
        "Low": pick("ch_trade_low_price", "low", "trade_low_price", "ch_trade_low_price_"),
        "Close": pick("ch_closing_price", "close", "closing_price", "ch_closing_price_"),
        "Volume": pick("ch_tot_traded_qty", "volume", "tot_traded_qty", "ch_tot_traded_qty_"),
    }
    if date_col is None or any(v is None for v in mapping.values()):
        return pd.DataFrame()
    out = pd.DataFrame()
    out.index = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(IST).tz_localize(None)
    for target, col in mapping.items():
        out[target] = pd.to_numeric(raw[col], errors="coerce").to_numpy()
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def _session_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(index.normalize().unique()).sort_values()
    return d[d.weekday < 5]


def integrity_check(df: pd.DataFrame, interval: str = "1m") -> IntegritySummary:
    # Inspect timestamp/index quality BEFORE normalize_1m_frame, because normalization
    # deliberately sorts and de-duplicates for downstream research use.
    raw = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    if raw.empty:
        return IntegritySummary(rows=0, status="NO_DATA")
    time_col = None
    for c in raw.columns:
        if str(c).strip().lower().replace(" ", "_") in {"timestamp", "datetime", "date", "time", "dt"}:
            time_col = c
            break
    raw_idx = pd.to_datetime(raw[time_col] if time_col is not None else raw.index, errors="coerce", utc=False)
    if getattr(raw_idx, "tz", None) is not None:
        raw_idx = raw_idx.tz_convert(IST).tz_localize(None)
    valid_idx = pd.DatetimeIndex(raw_idx[~pd.isna(raw_idx)])

    x = normalize_1m_frame(df)
    s = IntegritySummary(rows=len(raw))
    if x.empty:
        s.status = "NO_DATA"
        return s
    s.start = valid_idx.min().isoformat() if len(valid_idx) else x.index.min().isoformat()
    s.end = valid_idx.max().isoformat() if len(valid_idx) else x.index.max().isoformat()
    s.non_monotonic = int((valid_idx.to_series().diff().dropna() <= pd.Timedelta(0)).sum()) if len(valid_idx) else 0
    s.duplicate_timestamps = int(valid_idx.duplicated().sum()) if len(valid_idx) else 0
    idx = x.index
    s.weekend_rows = int((idx.dayofweek >= 5).sum())
    minutes = idx.hour * 60 + idx.minute
    s.outside_market_hours = int(((minutes < 9 * 60 + 15) | (minutes > 15 * 60 + 29)).sum())
    s.nonpositive_prices = int((x[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum())
    s.ohlc_violations = int(
        ((x.High < x[["Open", "Close"]].max(axis=1)) |
         (x.Low > x[["Open", "Close"]].min(axis=1)) |
         (x.High < x.Low)).sum()
    )

    rule = "1min" if interval in {"1m", "1min"} else "5min"
    expected_per_day = 375 if rule == "1min" else 75
    sessions = _session_dates(idx)
    s.session_count = len(sessions)
    s.observed_bars = len(x)
    s.expected_bars = s.session_count * expected_per_day
    s.coverage_pct = round(100 * s.observed_bars / s.expected_bars, 3) if s.expected_bars else 0.0

    missing = 0
    for day in sessions:
        day_slice = x.loc[(x.index.normalize() == day)]
        if day_slice.empty:
            continue
        expected = pd.date_range(day + pd.Timedelta(hours=9, minutes=15), day + pd.Timedelta(hours=15, minutes=29), freq=rule)
        observed = pd.DatetimeIndex(day_slice.index)
        # Count missing expected slots but don't call a day incomplete just because the source
        # starts/ends a few seconds differently; normalized minute timestamps handle most archives.
        missing += int(len(expected.difference(observed)))
    s.missing_minutes_within_session = missing

    daily = x.resample("1D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    if not daily.empty:
        moves = daily["Close"].pct_change().abs()
        s.large_move_days = int((moves >= 0.15).sum())

    if s.nonpositive_prices or s.ohlc_violations:
        s.status = "REJECT"
    elif s.coverage_pct >= 98 and s.duplicate_timestamps == 0 and s.weekend_rows == 0 and s.outside_market_hours == 0:
        s.status = "ACCEPT"
    elif s.coverage_pct >= 95:
        s.status = "REVIEW"
    else:
        s.status = "REJECT_REVIEW"
    return s


def read_local_file(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return normalize_1m_frame(pd.read_parquet(p))
    if p.name.endswith(".csv.gz"):
        return normalize_1m_frame(pd.read_csv(p, compression="gzip"))
    if p.suffix.lower() == ".csv":
        return normalize_1m_frame(pd.read_csv(p))
    raise ValueError(f"Unsupported historical file type: {p}")


def _request_nse_eod(session: requests.Session, symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeout: int = 30):
    params = {
        "symbol": symbol,
        "series": '["EQ"]',
        "from": start.strftime("%d-%m-%Y"),
        "to": end.strftime("%d-%m-%Y"),
    }
    # Some NSE clients have historically accepted csv=true; JSON is easier to parse when available.
    r = session.get(NSE_HIST_EQUITY, params=params, timeout=timeout)
    ctype = r.headers.get("content-type", "")
    if r.status_code != 200:
        raise RuntimeError(f"NSE HTTP {r.status_code}: {r.text[:200]}")
    if "json" in ctype:
        return normalize_nse_eod(r.json())
    return normalize_nse_eod(r.content)


def fetch_nse_eod(symbol: str, start: str | pd.Timestamp, end: str | pd.Timestamp, chunk_days: int = 90, pause: float = 0.8) -> pd.DataFrame:
    """Fetch official NSE daily equity history in conservative chunks."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        return pd.DataFrame()
    session = requests.Session()
    session.headers.update(_nse_headers())
    # Warm the NSE session for cookies / Akamai handling.
    warm = session.get(NSE_HOME, timeout=30)
    if warm.status_code not in (200, 403):
        warm.raise_for_status()

    frames = []
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days - 1), end_ts)
        last_err = None
        for attempt in range(3):
            try:
                f = _request_nse_eod(session, symbol, cursor, chunk_end)
                if not f.empty:
                    frames.append(f)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
                try:
                    session.get(NSE_HOME, timeout=15)
                except Exception:
                    pass
        if last_err is not None:
            raise last_err
        cursor = chunk_end + pd.Timedelta(days=1)
        time.sleep(max(0.0, pause))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def aggregate_local_daily(df: pd.DataFrame) -> pd.DataFrame:
    x = normalize_1m_frame(df)
    if x.empty:
        return x
    # Accept an already-daily frame (midnight timestamps) for reconciliation callers.
    if len(x) and (x.index.hour == 0).all() and (x.index.minute == 0).all():
        return x
    x = x.between_time("09:15", "15:29")
    return x.resample("1D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Open", "High", "Low", "Close"])


def _pct_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    den = b.abs().replace(0, np.nan)
    return (a - b).abs() / den * 100


def reconcile_daily(local_daily: pd.DataFrame, nse_daily: pd.DataFrame, symbol: str, price_tolerance_pct: float = 0.05, volume_tolerance_pct: float = 1.0) -> tuple[pd.DataFrame, ReconciliationSummary]:
    l = aggregate_local_daily(local_daily)
    n = normalize_nse_eod(nse_daily)
    summary = ReconciliationSummary(symbol=symbol, requested_from="", requested_to="", local_days=len(l), nse_rows=len(n))
    if l.empty or n.empty:
        summary.status = "NO_MATCH_DATA"
        summary.note = "One or both datasets are empty."
        return pd.DataFrame(), summary
    summary.requested_from = str(min(l.index.min(), n.index.min()).date())
    summary.requested_to = str(max(l.index.max(), n.index.max()).date())
    idx = l.index.intersection(n.index)
    summary.matched_days = len(idx)
    summary.missing_in_nse = int(len(l.index.difference(n.index)))
    summary.missing_in_local = int(len(n.index.difference(l.index)))
    if len(idx) == 0:
        summary.status = "NO_OVERLAP"
        return pd.DataFrame(), summary
    rows = []
    for day in idx:
        row = {"date": day.date()}
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            lv = float(l.loc[day, col]) if col in l.columns and pd.notna(l.loc[day, col]) else np.nan
            nv = float(n.loc[day, col]) if col in n.columns and pd.notna(n.loc[day, col]) else np.nan
            if np.isfinite(lv) and np.isfinite(nv) and nv != 0:
                diff = abs(lv - nv) / abs(nv) * 100
            else:
                diff = np.nan
            row[f"local_{col.lower()}"] = lv
            row[f"nse_{col.lower()}"] = nv
            row[f"diff_{col.lower()}_pct"] = diff
        row["price_match"] = all(
            pd.notna(row[f"diff_{c.lower()}_pct"]) and row[f"diff_{c.lower()}_pct"] <= price_tolerance_pct
            for c in ["Open", "High", "Low", "Close"]
        )
        row["volume_match"] = pd.isna(row["diff_volume_pct"]) or row["diff_volume_pct"] <= volume_tolerance_pct
        rows.append(row)
    comp = pd.DataFrame(rows)
    summary.price_mismatches = int((~comp["price_match"]).sum())
    summary.volume_mismatches = int((~comp["volume_match"]).sum())
    for col in ["open", "high", "low", "close", "volume"]:
        val = comp[f"diff_{col}_pct"].replace([np.inf, -np.inf], np.nan).max()
        setattr(summary, f"max_{col}_diff_pct", float(val) if pd.notna(val) else 0.0)
    overlap_target = min(summary.local_days, summary.nse_rows)
    overlap_ok = summary.matched_days >= max(1, int(0.8 * overlap_target))
    if summary.price_mismatches == 0 and summary.volume_mismatches == 0 and overlap_ok:
        summary.status = "MATCH"
    elif summary.matched_days > 0 and summary.price_mismatches <= max(1, int(0.02 * summary.matched_days)):
        summary.status = "MATCH_WITH_WARNINGS"
        summary.note = "A small number of tolerance breaches were found; inspect the reconciliation table and corporate-action flags."
    else:
        summary.status = "MISMATCH"
        summary.note = "Material differences exist. Do not use this source as ground truth until reviewed."
    return comp, summary


def choose_reconciliation_window(local_df: pd.DataFrame, sample_days: int = 30) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    x = normalize_1m_frame(local_df)
    if x.empty:
        return None
    d = _session_dates(x.index)
    if len(d) == 0:
        return None
    end = d.max()
    start = d[max(0, len(d) - sample_days)]
    return pd.Timestamp(start), pd.Timestamp(end)



def online_source_check(timeout: int = 20) -> pd.DataFrame:
    """Check that the published source URLs/repositories are reachable and report metadata.

    This is an availability/provenance check only. It does not certify bar-level accuracy.
    """
    rows = []
    checks = [
        ("ganesh_2018_2025", "https://api.github.com/repos/ganeshbiyer/Nse_Historical_Data"),
        ("ganesh_2026", "https://api.github.com/repos/ganeshbiyer/Nse_Historical_Data_2026"),
        ("voletiramu_fno", "https://api.github.com/repos/voletiramu/nse-fno-1min-data"),
        ("nse_official", "https://www.nseindia.com/static/resources/historical-reports-capital-market-daily-monthly-archives"),
    ]
    for source_id, url in checks:
        row = {"source_id": source_id, "url": url, "reachable": False, "http_status": None, "observed_name": "", "updated": "", "note": ""}
        try:
            headers = _nse_headers() if "nseindia.com" in url else {"User-Agent": "IndianStockSignalDashboard/1.0 source-provenance-check"}
            r = requests.get(url, headers=headers, timeout=timeout)
            row["http_status"] = r.status_code
            row["reachable"] = r.status_code < 400
            if "api.github.com" in url and r.ok:
                data = r.json()
                row["observed_name"] = data.get("full_name", "")
                row["updated"] = data.get("pushed_at", "") or data.get("updated_at", "")
            elif r.ok:
                row["observed_name"] = "NSE historical reports page"
            row["note"] = "Availability check only; not a bar-level accuracy certification."
        except Exception as exc:
            row["note"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)

def source_provenance_table() -> pd.DataFrame:
    rows = []
    for key, value in SOURCE_REGISTRY.items():
        row = {"source_id": key}
        row.update(value)
        rows.append(row)
    return pd.DataFrame(rows)


def audit_file(path: str | Path, source_id: str, symbol: str | None = None, reconcile: bool = False, sample_days: int = 30) -> dict:
    p = Path(path)
    sym = symbol or p.stem.replace("_1m", "")
    local = read_local_file(p)
    integrity = integrity_check(local, interval="1m")
    result = {"path": str(p), "symbol": sym, "source_id": source_id, "integrity": asdict(integrity), "reconciliation": None}
    if reconcile and not local.empty:
        window = choose_reconciliation_window(local, sample_days=sample_days)
        if window is not None:
            start, end = window
            try:
                nse = fetch_nse_eod(sym, start, end)
                _, summary = reconcile_daily(local, nse, sym)
                result["reconciliation"] = asdict(summary)
            except Exception as exc:
                result["reconciliation"] = asdict(ReconciliationSummary(
                    symbol=sym,
                    requested_from=str(start.date()),
                    requested_to=str(end.date()),
                    status="ERROR",
                    note=f"{type(exc).__name__}: {exc}",
                ))
    return result


def save_quality_report(results: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
