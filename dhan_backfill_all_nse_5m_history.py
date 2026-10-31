#!/usr/bin/env python3
"""Backfill genuine NSE five-minute OHLCV history into a separate Drive archive.

This script does not run or write to the obsolete one-minute synchronization
workflow, does not fabricate bars, and never changes a production model.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
REMOTE_ROOT = "production_data/nse_official/intraday_seed/olo_5m_archive/symbols"
MANIFEST_REMOTE = "production_data/nse_official/intraday_seed/olo_5m_archive/backfill_manifest.json"
MIN_SESSION_BARS = 50
FRESHNESS_WINDOW_DAYS = 7
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def normalize_dhan_5m(frame: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Convert Dhan's five-minute start timestamps to completed-bar close timestamps."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=IST, name="Datetime"))
    x = frame.copy()
    x.columns = [str(c).strip() for c in x.columns]
    aliases = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume", "timestamp": "Timestamp",
        "datetime": "Timestamp", "date": "Timestamp", "time": "Timestamp",
    }
    x = x.rename(columns={c: aliases.get(str(c).strip().lower(), c) for c in x.columns})
    idx = pd.DatetimeIndex(pd.to_datetime(x.pop("Timestamp"), errors="coerce")) if "Timestamp" in x else pd.DatetimeIndex(pd.to_datetime(x.index, errors="coerce"))
    idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
    x.index = idx
    for column in OHLCV:
        if column not in x:
            x[column] = np.nan
        x[column] = pd.to_numeric(x[column], errors="coerce")
    x = x.loc[~x.index.isna()].sort_index()
    x = x.loc[~x.index.duplicated(keep="last")]
    # Dhan historical chart timestamps identify the candle start; store only
    # completed bars on the same close-time convention as the active 5m worker.
    starts = pd.DatetimeIndex(x.index)
    regular = (
        (starts.time >= pd.Timestamp("09:15").time())
        & (starts.time < pd.Timestamp("15:30").time())
        & (starts.minute % 5 == 0)
    )
    x = x.loc[regular].copy()
    x.index = x.index + pd.Timedelta(minutes=5)
    x.index.name = "Datetime"
    valid_ohlc = x[["Open", "High", "Low", "Close"]].notna().all(axis=1)
    valid_ohlc &= x[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)
    x = x.loc[valid_ohlc]
    if cutoff is not None:
        boundary = pd.Timestamp(cutoff)
        boundary = boundary.tz_localize(IST) if boundary.tzinfo is None else boundary.tz_convert(IST)
        x = x.loc[x.index <= boundary]
    x = x.loc[~x.index.duplicated(keep="last")]
    return x[OHLCV].sort_index()


def usable_session_summary(frame: pd.DataFrame, minimum_bars: int = MIN_SESSION_BARS) -> dict[str, Any]:
    """Count sessions with at least 50 valid five-minute OHLC observations."""
    if frame is None or frame.empty:
        return {"raw_sessions": 0, "usable_sessions": 0, "first_timestamp_ist": None, "last_timestamp_ist": None}
    x = frame.copy()
    valid = x[["Open", "High", "Low", "Close"]].notna().all(axis=1)
    counts = valid.groupby(pd.Index(x.index.date)).sum()
    usable = counts[counts >= int(minimum_bars)].index
    return {
        "raw_sessions": int(counts.size),
        "usable_sessions": int(len(usable)),
        "first_timestamp_ist": pd.Timestamp(x.index.min()).isoformat(),
        "last_timestamp_ist": pd.Timestamp(x.index.max()).isoformat(),
        "usable_session_dates": [str(day) for day in usable],
    }


def retain_latest_sessions(frame: pd.DataFrame, maximum_sessions: int = 500, minimum_bars: int = MIN_SESSION_BARS) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=IST, name="Datetime"))
    summary = usable_session_summary(frame, minimum_bars)
    dates = summary["usable_session_dates"][-int(maximum_sessions):]
    if not dates:
        return frame.iloc[0:0].copy()
    keep = pd.Index(frame.index.date).astype(str).isin(set(dates))
    return frame.loc[keep].sort_index()


def _resolve_remote_file(store: Any, remote: str) -> dict[str, Any] | None:
    folder_mime = "application/vnd.google-apps.folder"
    parent = store.root_id
    parts = remote.strip("/").split("/")
    for component in parts[:-1]:
        hit = store._find_child(parent, component, folder_mime)
        if not hit:
            return None
        parent = hit["id"]
    return store._find_child(parent, parts[-1])


def _download_existing(store: Any, remote: str, destination: Path) -> pd.DataFrame:
    item = _resolve_remote_file(store, remote)
    if not item:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=IST, name="Datetime"))
    from googleapiclient.http import MediaIoBaseDownload
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = store.service.files().get_media(fileId=item["id"])
    with destination.open("wb") as output:
        downloader = MediaIoBaseDownload(output, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    old = pd.read_parquet(destination)
    if "Timestamp" in old.columns:
        old = old.set_index("Timestamp")
    old.index = pd.DatetimeIndex(pd.to_datetime(old.index, errors="coerce"))
    old.index = old.index.tz_localize(IST) if old.index.tz is None else old.index.tz_convert(IST)
    for column in OHLCV:
        if column not in old:
            old[column] = np.nan
        old[column] = pd.to_numeric(old[column], errors="coerce")
    old = old[OHLCV].loc[~old.index.isna()].sort_index()
    return old.loc[~old.index.duplicated(keep="last")]


def _process_symbol(
    symbol: str,
    security_id: str,
    config: Any,
    initial_start: datetime,
    end: datetime,
    incremental_calendar_days: int,
    sessions: int,
    local_root: Path,
    pause_seconds: float,
    minimum_sessions: int,
    minimum_bars: int,
) -> dict[str, Any]:
    from dhan_client import historical_intraday_range
    from persistent_store import GoogleDriveStore

    store = GoogleDriveStore()
    remote = f"{REMOTE_ROOT}/symbol={symbol}/{symbol}.parquet"
    local = local_root / f"symbol={symbol}" / f"{symbol}.parquet"
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = _download_existing(store, remote, local.with_name(f"{symbol}.existing.parquet"))
        existing_summary = usable_session_summary(existing, minimum_bars)
        latest_existing = pd.to_datetime(
            existing_summary.get("last_timestamp_ist"), errors="coerce"
        )
        if (
            existing_summary["usable_sessions"] >= minimum_sessions
            and pd.notna(latest_existing)
            and latest_existing >= pd.Timestamp(end) - pd.Timedelta(days=FRESHNESS_WINDOW_DAYS)
        ):
            start = end - timedelta(days=int(incremental_calendar_days))
        else:
            start = initial_start
        fetched = historical_intraday_range(
            config,
            security_id=security_id,
            interval=5,
            start=start,
            end=end,
            max_days=90,
            instrument="EQUITY",
            exchange_segment="NSE_EQ",
            oi=False,
            pause_seconds=pause_seconds,
        )
        fetched = normalize_dhan_5m(fetched, cutoff=pd.Timestamp.now(tz=IST).floor("5min"))
        parts = [part for part in (existing, fetched) if part is not None and not part.empty]
        merged = pd.concat(parts).sort_index() if parts else pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=IST))
        merged = merged.loc[~merged.index.duplicated(keep="last")]
        merged = retain_latest_sessions(merged, sessions, minimum_bars)
        summary = usable_session_summary(merged, minimum_bars)
        if merged.empty:
            return {"symbol": symbol, "status": "NO_DATA", **{k: v for k, v in summary.items() if k != "usable_session_dates"}}
        latest = pd.to_datetime(summary.get("last_timestamp_ist"), errors="coerce")
        fresh_latest = pd.notna(latest) and latest >= pd.Timestamp(end) - pd.Timedelta(days=FRESHNESS_WINDOW_DAYS)
        merged.to_parquet(local, index=True, compression="zstd")
        upload = store.upload_or_update(
            local,
            remote,
            mime_type="application/vnd.apache.parquet",
        )
        if str(upload.get("status") or "").upper() != "OK":
            raise RuntimeError(f"Drive upload was not confirmed: {upload}")
        session_ready = summary["usable_sessions"] >= minimum_sessions
        return {
            "symbol": symbol,
            "status": (
                "OK" if session_ready and fresh_latest
                else "STALE_LATEST_DATA" if session_ready
                else "INSUFFICIENT_HISTORY"
            ),
            "fresh_latest_data": bool(fresh_latest),
            "remote_path": remote,
            "rows": int(len(merged)),
            "fetched_rows": int(len(fetched)),
            "usable_sessions": summary["usable_sessions"],
            "raw_sessions": summary["raw_sessions"],
            "first_timestamp_ist": summary["first_timestamp_ist"],
            "last_timestamp_ist": summary["last_timestamp_ist"],
            "drive_action": upload.get("action"),
            "drive_bytes": upload.get("size"),
        }
    except Exception as exc:
        return {"symbol": symbol, "status": "ERROR", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=500)
    parser.add_argument("--calendar-days", type=int, default=1000)
    parser.add_argument("--initial-calendar-days", type=int, default=1000)
    parser.add_argument("--minimum-usable-sessions", type=int, default=200)
    parser.add_argument("--minimum-bars-per-session", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pause-seconds", type=float, default=0.55)
    parser.add_argument("--minimum-symbol-coverage", type=float, default=0.90)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--work-root", default="/tmp/nse_5m_history_backfill")
    args = parser.parse_args()
    if not 200 <= args.sessions <= 500:
        raise ValueError("--sessions must be between 200 and 500")
    if args.minimum_usable_sessions < 200 or args.minimum_usable_sessions > args.sessions:
        raise ValueError("--minimum-usable-sessions must be between 200 and --sessions")
    if args.minimum_bars_per_session < 50:
        raise ValueError("--minimum-bars-per-session must be at least 50")
    if args.workers < 1 or args.workers > 3:
        raise ValueError("--workers must be between 1 and 3")
    if not 0.5 <= args.minimum_symbol_coverage <= 1.0:
        raise ValueError("--minimum-symbol-coverage must be between 0.5 and 1.0")

    from dhan_client import DhanConfig, get_access_token, resolve_equity_symbol
    from persistent_store import GoogleDriveStore, drive_status
    config = DhanConfig.from_env()
    if not (config.has_any_credential() or config.has_totp_flow()) or not get_access_token(config):
        raise RuntimeError("Dhan TOTP credentials or a valid access token are required.")
    preflight = drive_status(require_write=True)
    if not preflight.get("configured") or preflight.get("reachable") is not True or preflight.get("write_ready") is not True:
        raise RuntimeError("Google Drive read/write preflight failed: " + str(preflight.get("reason")))

    if args.symbols.strip():
        symbols = sorted({value.strip().upper().replace(".NS", "").replace(".BO", "") for value in args.symbols.split(",") if value.strip()})
    else:
        from delivery_training_universe import official_nse_symbols
        symbols = sorted({str(value).strip().upper().replace(".NS", "").replace(".BO", "") for value in official_nse_symbols(force_refresh=True) if str(value).strip()})
    if args.max_symbols > 0:
        symbols = symbols[:args.max_symbols]
    if not symbols:
        raise RuntimeError("NSE equity universe is empty.")

    end = pd.Timestamp(args.end_date, tz=IST).to_pydatetime() if args.end_date else datetime.now(IST)
    initial_start = end - timedelta(days=int(args.initial_calendar_days))
    local_root = Path(args.work_root)
    local_root.mkdir(parents=True, exist_ok=True)

    resolved = []
    resolution_errors = []
    for symbol in symbols:
        try:
            info = resolve_equity_symbol(symbol)
            resolved.append((symbol, str(info["security_id"])))
        except Exception as exc:
            resolution_errors.append({"symbol": symbol, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _process_symbol, symbol, security_id, config, initial_start, end,
                args.calendar_days, args.sessions, local_root, args.pause_seconds,
                args.minimum_usable_sessions, args.minimum_bars_per_session,
            ): symbol
            for symbol, security_id in resolved
        }
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 100 == 0 or len(results) == len(futures):
                print(f"BACKFILL_PROGRESS completed={len(results)}/{len(futures)}", flush=True)

    results.extend({"symbol": row["symbol"], "status": "ERROR", "error": row["error"]} for row in resolution_errors)
    results.sort(key=lambda row: row["symbol"])
    session_ready = [row for row in results if int(row.get("usable_sessions") or 0) >= 200]
    ok = [row for row in results if row.get("status") == "OK"]
    preferred = [row for row in ok if int(row.get("usable_sessions") or 0) >= 500]
    report = {
        "audit": "nse_native_five_minute_historical_backfill",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "DHANHQ_HISTORICAL_NSE_EQ_INTERVAL_5",
        "bar_interval_minutes": 5,
        "timestamp_convention": "provider_bar_start_plus_5_minutes_as_bar_close",
        "minimum_bars_per_usable_session": args.minimum_bars_per_session,
        "freshness_window_days": FRESHNESS_WINDOW_DAYS,
        "target_sessions_per_symbol": args.sessions,
        "minimum_usable_sessions": args.minimum_usable_sessions,
        "calendar_days_requested": args.calendar_days,
        "range_start_ist": initial_start.isoformat(),
        "incremental_calendar_days": args.calendar_days,
        "initial_calendar_days": args.initial_calendar_days,
        "range_end_ist": end.isoformat(),
        "official_symbols_requested": len(symbols),
        "symbols_resolved": len(resolved),
        "symbols_with_200_plus_sessions": len(session_ready),
        "symbols_with_fresh_latest_data": sum(row.get("fresh_latest_data") is True for row in results),
        "symbols_with_200_plus_sessions_and_fresh_data": len(ok),
        "symbols_with_stale_latest_data": sum(row.get("status") == "STALE_LATEST_DATA" for row in results),
        "symbols_with_data_after_2026_09_24": sum(
            bool(row.get("last_timestamp_ist")) and row["last_timestamp_ist"][:10] > "2026-09-24"
            for row in results
        ),
        "latest_archive_timestamp_ist": max(
            (row.get("last_timestamp_ist") or "" for row in results), default=None
        ),
        "symbols_with_500_sessions": len(preferred),
        "symbols_with_errors": sum(row.get("status") == "ERROR" for row in results),
        "symbols_below_200_sessions": sum(row.get("status") == "INSUFFICIENT_HISTORY" for row in results),
        "symbols": results,
        "no_auto_orders": True,
        "production_model_changed": False,
    }
    manifest_path = local_root / "backfill_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    store = GoogleDriveStore()
    uploaded = store.upload_or_update(manifest_path, MANIFEST_REMOTE, mime_type="application/json")
    if str(uploaded.get("status") or "").upper() != "OK":
        raise RuntimeError(f"Backfill manifest Drive write not confirmed: {uploaded}")
    report["manifest_drive_path"] = MANIFEST_REMOTE
    print(json.dumps({key: value for key, value in report.items() if key != "symbols"}, indent=2, sort_keys=True))
    required_ok = max(1, int(np.ceil(len(symbols) * args.minimum_symbol_coverage)))
    report["overall_status"] = "PASS" if len(ok) >= required_ok else "NEEDS_ATTENTION"
    report["required_symbols_with_200_plus_sessions_and_fresh_data"] = required_ok
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    uploaded = store.upload_or_update(manifest_path, MANIFEST_REMOTE, mime_type="application/json")
    if str(uploaded.get("status") or "").upper() != "OK":
        raise RuntimeError(f"Final backfill manifest Drive write not confirmed: {uploaded}")
    return 0 if len(ok) >= required_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
