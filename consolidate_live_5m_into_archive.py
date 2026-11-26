#!/usr/bin/env python3
"""Append durable live five-minute snapshots to the native per-symbol archive.

The retraining assembler reads only olo_5m_archive/symbols/symbol=<SYM>/<SYM>.parquet.
Live captures land separately under intraday_5m_live/YYYY/MM/DD/capture_*.parquet.
This job folds recent live rows into each authoritative symbol's archive parquet so
the training frame includes the latest completed sessions without a provider refetch.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from dhan_backfill_all_nse_5m_history import (
    REMOTE_ROOT,
    _download_existing,
    _resolve_remote_file,
    retain_latest_sessions,
)
from market_session_guard import is_nse_trading_day
from persistent_store import (
    GoogleDriveStore,
    _drive_execute_with_retry,
    download_drive_file_with_retry,
)
from prediction_universe import authoritative_roster

IST = ZoneInfo("Asia/Kolkata")
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
LIVE_REMOTE_ROOT = "production_data/nse_official/intraday_5m_live"
MAX_SESSIONS = 500
MIN_BARS_PER_SESSION = 50
PARQUET_MIME = "application/vnd.apache.parquet"
FOLDER_MIME = "application/vnd.google-apps.folder"
MIN_SLOT_COVERAGE = 0.90


def expected_session_slots(first: time | None = None, last: time | None = None) -> list[time]:
    """Five-minute decision slots between ``first`` and ``last`` (inclusive)."""
    first = first or time(9, 20)
    last = last or time(15, 25)
    slots: list[time] = []
    current = datetime(2000, 1, 1, first.hour, first.minute)
    end = datetime(2000, 1, 1, last.hour, last.minute)
    while current <= end:
        slots.append(current.time())
        current += timedelta(minutes=5)
    return slots


def session_coverage_report(
    latest_session_frames: dict[str, pd.DataFrame],
    symbols: list[str],
    *,
    slots: list[time] | None = None,
    min_coverage: float = MIN_SLOT_COVERAGE,
) -> dict[str, Any]:
    """Verify that no five-minute decision slot is missing across the universe."""
    if not latest_session_frames:
        return {
            "complete": False,
            "reason": "no consolidated session rows",
            "symbols": len(symbols),
        }
    session = str(max(pd.Timestamp(frame.index.max()).date() for frame in latest_session_frames.values()))
    observed = sorted({
        stamp.time()
        for frame in latest_session_frames.values()
        for stamp in frame.index
        if str(stamp.date()) == session
    })
    if slots is None:
        slots = expected_session_slots(observed[0], observed[-1]) if observed else []
    slot_set = set(slots)
    counts = {slot: 0 for slot in slots}
    per_symbol: dict[str, int] = {}
    for symbol, frame in latest_session_frames.items():
        day_frame = frame.loc[pd.Index(frame.index.date).astype(str) == session]
        seen = {stamp.time() for stamp in day_frame.index if stamp.time() in slot_set}
        per_symbol[symbol] = len(seen)
        for slot in seen:
            counts[slot] += 1
    missing = [slot.strftime("%H:%M") for slot in slots if counts[slot] == 0]
    min_count = min(counts.values()) if counts else 0
    threshold = min_coverage * len(symbols)
    complete = not missing and min_count >= threshold
    report: dict[str, Any] = {
        "complete": bool(complete),
        "session": session,
        "expected_slots": len(slots),
        "slots_with_data": sum(count > 0 for count in counts.values()),
        "missing_slots": missing,
        "min_slot_coverage": min_count,
        "min_slot_coverage_ratio": round(min_count / len(symbols), 6) if symbols else 0.0,
        "required_slot_coverage": round(threshold, 3),
        "symbols_with_full_grid": sum(value == len(slots) for value in per_symbol.values()),
        "symbols_below_min_bars": sum(value < MIN_BARS_PER_SESSION for value in per_symbol.values()),
    }
    if not complete:
        report["reason"] = (
            f"missing slots={missing} or slot coverage {min_count}/{len(symbols)} "
            f"below required {threshold:.0f}"
        )
    return report


def _day_snapshots(
    store: GoogleDriveStore,
    day: Any,
    symbols: set[str],
    work_root: Path,
) -> dict[str, pd.DataFrame]:
    remote_dir = f"{LIVE_REMOTE_ROOT}/{day:%Y/%m/%d}"
    folder = _resolve_remote_file(store, remote_dir)
    if not folder:
        return {}
    params = {
        "q": f"'{folder['id']}' in parents and trashed = false",
        "spaces": "drive",
        "fields": "nextPageToken,files(id,name,mimeType)",
        "pageSize": 1000,
        "includeItemsFromAllDrives": True,
        "supportsAllDrives": True,
    }
    files: list[dict[str, Any]] = []
    page_token = None
    while True:
        if page_token:
            params["pageToken"] = page_token
        response = _drive_execute_with_retry(
            "live_snapshot_list",
            lambda: store.service.files().list(**params).execute(),
        )
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    params.pop("pageToken", None)
    buckets: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in symbols}
    day_root = work_root / f"{day:%Y%m%d}"
    day_root.mkdir(parents=True, exist_ok=True)
    for item in files:
        name = str(item.get("name") or "")
        if item.get("mimeType") == FOLDER_MIME:
            continue
        if not name.startswith("capture_") or not name.endswith(".parquet"):
            continue
        local = day_root / name
        try:
            if not local.exists() or local.stat().st_size == 0:
                download_drive_file_with_retry(
                    store.service, item["id"], local,
                    operation="live_snapshot_download",
                )
            frame = pd.read_parquet(
                local,
                columns=["symbol", "Timestamp", *OHLCV],
                filters=[("symbol", "in", sorted(symbols))],
            )
            if frame.empty:
                continue
            frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
            frame = frame.loc[frame["symbol"].isin(symbols)].copy()
            if frame.empty:
                continue
            index = pd.DatetimeIndex(pd.to_datetime(frame.pop("Timestamp"), errors="coerce"))
            index = index.tz_localize(IST) if index.tz is None else index.tz_convert(IST)
            frame.index = index
            for column in OHLCV:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.loc[~frame.index.isna()]
            frame = frame.loc[frame[[c for c in OHLCV if c != "Volume"]].notna().all(axis=1)]
            frame = frame.loc[frame[[c for c in OHLCV if c != "Volume"]].gt(0).all(axis=1)]
            for symbol, part in frame.groupby("symbol", sort=False):
                buckets[str(symbol)].append(part[OHLCV])
        except Exception as exc:
            print(
                "LIVE_SNAPSHOT_READ_FAILED "
                + json.dumps({"path": f"{remote_dir}/{name}", "error": type(exc).__name__}),
                flush=True,
            )
    result: dict[str, pd.DataFrame] = {}
    for symbol, parts in buckets.items():
        if not parts:
            continue
        merged = pd.concat(parts).sort_index()
        merged = merged.loc[~merged.index.duplicated(keep="last")]
        merged.index.name = "Datetime"
        result[symbol] = merged[OHLCV]
    return result


def _latest_session_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent session present in a symbol's archive frame."""
    if frame.empty:
        return frame
    latest_day = str(frame.index.max().date())
    return frame.loc[pd.Index(frame.index.date).astype(str) == latest_day, OHLCV]


def consolidate(
    *,
    store: GoogleDriveStore,
    symbols: list[str],
    days: int,
    end_date: Any,
    dry_run: bool,
    work_root: Path,
) -> dict[str, Any]:
    symbol_set = set(symbols)
    live_by_symbol: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in symbols}
    days_read = 0
    for offset in range(int(days)):
        day = end_date - timedelta(days=offset)
        if not is_nse_trading_day(day.date()):
            continue
        days_read += 1
        for symbol, frame in _day_snapshots(store, day, symbol_set, work_root).items():
            live_by_symbol.setdefault(symbol, []).append(frame)
    updated = 0
    rows_added = 0
    unchanged = 0
    missing_archive = 0
    coverage: list[dict[str, Any]] = []
    latest_session_frames: dict[str, pd.DataFrame] = {}
    for index, symbol in enumerate(symbols, start=1):
        parts = live_by_symbol.get(symbol) or []
        remote = f"{REMOTE_ROOT}/symbol={symbol}/{symbol}.parquet"
        local = work_root / "archive" / f"symbol={symbol}" / f"{symbol}.existing.parquet"
        local.parent.mkdir(parents=True, exist_ok=True)
        archive = _download_existing(store, remote, local)
        if archive.empty:
            missing_archive += 1
            continue
        if not parts:
            # No live rows for this symbol: still measure its latest archived
            # session so coverage spans the whole roster, not just updated names.
            latest_session_frames[symbol] = _latest_session_frame(archive)
            unchanged += 1
            continue
        live = pd.concat(parts).sort_index()
        live = live.loc[~live.index.duplicated(keep="last")]
        new_timestamps = live.index.difference(archive.index)
        if len(new_timestamps) == 0:
            latest_session_frames[symbol] = _latest_session_frame(archive)
            unchanged += 1
            continue
        merged = pd.concat([archive, live]).sort_index()
        merged = merged.loc[~merged.index.duplicated(keep="last")]
        merged = retain_latest_sessions(merged, MAX_SESSIONS, MIN_BARS_PER_SESSION)
        if merged.empty:
            latest_session_frames[symbol] = _latest_session_frame(archive)
            unchanged += 1
            continue
        latest_session_frames[symbol] = _latest_session_frame(merged)
        if not dry_run:
            out = work_root / "archive" / f"symbol={symbol}" / f"{symbol}.parquet"
            merged.to_parquet(out, index=True, compression="zstd")
            uploaded = store.upload_or_update(out, remote, mime_type=PARQUET_MIME)
            if str(uploaded.get("status") or "").upper() != "OK":
                raise RuntimeError(f"Archive write not confirmed: {remote}: {uploaded}")
        updated += 1
        rows_added += int(len(new_timestamps))
        coverage.append({
            "symbol": symbol,
            "rows_added": int(len(new_timestamps)),
            "last_timestamp_ist": str(merged.index.max()),
        })
        if index % 50 == 0:
            print(
                f"CONSOLIDATE_PROGRESS symbols={index}/{len(symbols)} updated={updated}",
                flush=True,
            )
    return {
        "status": "OK",
        "symbols_requested": len(symbols),
        "days_requested": int(days),
        "trading_days_read": days_read,
        "symbols_updated": updated,
        "symbols_unchanged": unchanged,
        "symbols_without_archive": missing_archive,
        "rows_added": rows_added,
        "dry_run": bool(dry_run),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "latest_session_coverage": session_coverage_report(latest_session_frames, symbols),
        "samples": coverage[:10],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-incomplete-session",
        action="store_true",
        help="Report missing five-minute slots without failing the job",
    )
    parser.add_argument("--work-root", default="")
    args = parser.parse_args(argv)
    if args.days < 1 or args.days > 30:
        raise ValueError("--days must be between 1 and 30")
    end_date = (
        pd.Timestamp(args.end_date).tz_localize(IST)
        if len(args.end_date.strip()) <= 10
        else pd.Timestamp(args.end_date).tz_convert(IST)
    ) if args.end_date.strip() else pd.Timestamp.now(tz=IST)
    symbols = authoritative_roster()
    if len(symbols) < 150:
        raise RuntimeError(f"authoritative roster too small: {len(symbols)}")
    store = GoogleDriveStore()
    with tempfile.TemporaryDirectory(prefix="consolidate_live_5m_", dir=args.work_root or None) as tmp:
        report = consolidate(
            store=store,
            symbols=symbols,
            days=args.days,
            end_date=end_date,
            dry_run=args.dry_run,
            work_root=Path(tmp),
        )
    print("CONSOLIDATE_REPORT=" + json.dumps(report, sort_keys=True), flush=True)
    coverage = report.get("latest_session_coverage") or {}
    if not coverage.get("complete") and not args.allow_incomplete_session:
        print(
            "CONSOLIDATE_INCOMPLETE_SESSION=" + json.dumps(coverage, sort_keys=True),
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
