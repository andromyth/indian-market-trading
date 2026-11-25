#!/usr/bin/env python3
"""Repair canonical five-minute snapshots after the cash session.

Two Drive-only repair paths are supported:

1. **Shard re-assembly** — a shard upload can succeed while the final assembly
   or its manifest upload fails. The live compiler only consumes canonical
   snapshots, so those slots stay missing forever even though every quote is
   still present on Drive. The original deterministic assembly is re-run and the
   canonical snapshot is published only when the persisted shards validate.
2. **Archive recovery** — when the live worker persisted no shard at all for a
   slot, the genuine provider five-minute OHLCV archive is used to reconstruct
   the canonical snapshot. The archive stores candle-close timestamps, so the
   bar stamped exactly at the decision slot is point-in-time available. These
   rows are tagged ``archive_repair`` with ``source="NSE_5M_ARCHIVE_REPAIR"`` so
   downstream readers can tell them apart from live quote captures.

This module never calls INDstocks and never invents market values: every number
comes from a stored shard or a stored exchange bar.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from all_universe_snapshot_assembly import (
    DEFAULT_WORKER_COUNT,
    SCHEMA_VERSION,
    SHARD_SUBDIR,
    assemble_from_drive,
    canonical_artifact_names,
    normalized_cycle_minute,
    read_canonical_manifest,
    write_canonical_snapshot,
)
from audit_all_universe_minute_snapshots import (
    _find_folder,
    _list_children,
    expected_minutes,
)

IST = ZoneInfo("Asia/Kolkata")
QUOTE_ROOT = "production_data/all_universe_minute_snapshots"
ARCHIVE_REPAIR_SOURCE = "NSE_5M_ARCHIVE_REPAIR"
ARCHIVE_LOOKBACK_BARS = 600
SHARD_MANIFEST_RE = re.compile(r"^manifest_(\d{8}_\d{4})_w(\d+)of(\d+)\.json$")
SHARD_SNAPSHOT_RE = re.compile(r"^snapshot_(\d{8}_\d{4})_w(\d+)of(\d+)\.parquet$")


def _default_worker_count() -> int:
    try:
        return max(1, int(os.getenv("ALL_NSE_CAPTURE_WORKERS", str(DEFAULT_WORKER_COUNT)) or DEFAULT_WORKER_COUNT))
    except (TypeError, ValueError):
        return DEFAULT_WORKER_COUNT


def _list_day_files(store: Any, session_date: date) -> list[dict[str, Any]]:
    folder_id = _find_folder(store, [
        "production_data", "all_universe_minute_snapshots",
        f"{session_date:%Y}", f"{session_date:%m}", f"{session_date:%d}",
    ])
    return _list_children(store, folder_id) if folder_id else []


def _list_shard_files(store: Any, session_date: date) -> list[dict[str, Any]]:
    folder_id = _find_folder(store, [
        "production_data", "all_universe_minute_snapshots",
        f"{session_date:%Y}", f"{session_date:%m}", f"{session_date:%d}",
        SHARD_SUBDIR,
    ])
    return _list_children(store, folder_id) if folder_id else []


def _shard_worker_count(shard_files: list[dict[str, Any]], key: str) -> int | None:
    worker_ids: set[int] = set()
    counts: set[int] = set()
    for item in shard_files:
        name = str(item.get("name") or "")
        match = SHARD_MANIFEST_RE.match(name) or SHARD_SNAPSHOT_RE.match(name)
        if not match or match.group(1) != key:
            continue
        worker_ids.add(int(match.group(2)))
        counts.add(int(match.group(3)))
    if len(counts) != 1:
        return None
    count = counts.pop()
    if set(range(count)) - worker_ids:
        return None
    return count


def repair_slot(
    store: Any,
    minute: pd.Timestamp,
    *,
    worker_count: int,
    work_dir: Path,
    expected_universe: list[str] | None = None,
) -> dict[str, Any]:
    result = assemble_from_drive(
        store,
        minute,
        worker_count,
        work_dir / f"slot_{minute:%Y%m%d_%H%M}",
        expected_universe=expected_universe,
        strict_universe=False,
        write_canonical=True,
        write_latest=False,
        allow_duplicate_repair=True,
    )
    manifest = result.get("manifest") or {}
    return {
        "status": result.get("status"),
        "cycle_minute_ist": minute.isoformat(),
        "worker_count": worker_count,
        "snapshot_persisted": bool(manifest.get("snapshot_persisted")),
        "row_count": manifest.get("row_count"),
        "quoted_symbol_count": manifest.get("quoted_symbol_count"),
        "missing_quote_count": manifest.get("missing_quote_count"),
        "missing_shards": manifest.get("missing_shards") or [],
        "repair_actions": manifest.get("repair_actions") or [],
        "fatal_issue_codes": [
            str(issue.get("code"))
            for issue in ((manifest.get("validation") or {}).get("issues") or [])
            if str(issue.get("code")) in {
                "MISSING_SHARD", "EMPTY_ASSEMBLY", "SHARD_MINUTE_MISMATCH",
                "SHARD_IDENTITY_MISMATCH", "SHARD_NOT_STORED", "SHARD_NOT_PERSISTED",
                "DUPLICATE_SYMBOL", "UNIVERSE_COUNT_MISMATCH",
            }
        ],
        "error": manifest.get("error"),
    }


def _positive_float(value: Any) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    number = float(number)
    return number if number > 0 and np.isfinite(number) else None


def _archive_history_work_dir(work_dir: Path) -> Path:
    """Reuse the compiler's Actions-cached history directory when available."""
    configured = os.getenv("NSE_PREDICTION_WORK_DIR") or "/tmp/drive_all_nse"
    candidate = Path(configured)
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError:
        return work_dir
    return candidate


def load_archive_histories(
    store: Any,
    session_date: date,
    roster: list[str],
    work_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load genuine provider five-minute bars for the roster through the last slot."""
    if not roster:
        return {}
    slots = expected_minutes(session_date)
    if not slots:
        return {}
    last_slot = pd.Timestamp(
        datetime.strptime(slots[-1], "%Y%m%d_%H%M").replace(tzinfo=IST)
    )
    from drive_all_nse_prediction_compiler import (
        MARKET_CONTEXT_SYMBOL,
        _load_drive_history,
    )

    history = _load_drive_history(
        store,
        set(roster),
        last_slot,
        _archive_history_work_dir(work_dir),
        ARCHIVE_LOOKBACK_BARS,
    )
    history.pop(MARKET_CONTEXT_SYMBOL, None)
    return history


def build_archive_snapshot_frame(
    history: dict[str, pd.DataFrame],
    minute: pd.Timestamp,
    roster: list[str],
) -> pd.DataFrame:
    """Build canonical quote rows from the candle stamped at ``minute``.

    The provider archive is close-timestamped, so the candle stamped exactly at
    the decision slot is the last fully observed bar at that point in time.
    """
    minute = normalized_cycle_minute(minute)
    rows: list[dict[str, Any]] = []
    for symbol in roster:
        frame = history.get(symbol)
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        if not isinstance(frame.index, pd.DatetimeIndex):
            continue
        exact = frame.loc[frame.index == minute]
        if exact.empty:
            continue
        bar = exact.iloc[-1]
        close = _positive_float(bar.get("Close"))
        if close is None:
            continue
        day = frame.loc[frame.index.date == minute.date()]
        if day.empty:
            continue
        day_to_slot = day.loc[day.index <= minute]

        def _numeric(column: str) -> pd.Series:
            if column not in day_to_slot.columns:
                return pd.Series(dtype="float64")
            return pd.to_numeric(day_to_slot[column], errors="coerce")

        volume = _numeric("Volume").sum(min_count=1)
        day_volume = float(volume) if pd.notna(volume) else None
        previous = frame.loc[frame.index.date < minute.date()]
        prev_close = None
        if not previous.empty and "Close" in previous.columns:
            prior = pd.to_numeric(previous["Close"], errors="coerce").dropna()
            if not prior.empty:
                prev_close = float(prior.iloc[-1])
        day_open = _positive_float(_numeric("Open").iloc[0]) if "Open" in day_to_slot.columns else None
        highs = _numeric("High").dropna()
        lows = _numeric("Low").dropna()
        day_high = float(highs.max()) if not highs.empty else None
        day_low = float(lows.min()) if not lows.empty else None
        change_pct = (
            round(100.0 * (close - prev_close) / prev_close, 4)
            if prev_close and prev_close > 0 else None
        )
        rows.append({
            "record_type": "all_universe_minute_market_observation",
            "schema_version": "all-universe-minute-snapshot-v1",
            "symbol": symbol,
            "exchange": "NSE",
            "instrument_code": f"NSE_ARCHIVE_{symbol}",
            "cycle_minute_ist": minute.isoformat(),
            "quote_observed_at_utc": minute.tz_convert("UTC").isoformat(),
            "quote_observed_at_ist": minute.isoformat(),
            "quote_timestamp_basis": "exchange_5m_bar_close_from_archive",
            "quote_freshness_status": "ARCHIVE_CRON_REPAIR",
            "quote_source": ARCHIVE_REPAIR_SOURCE,
            "quote_status": "PASS",
            "quote_error": None,
            "actual_observed_price": close,
            "day_open": day_open,
            "day_high": day_high,
            "day_low": day_low,
            "previous_close": prev_close,
            "day_volume": day_volume,
            "day_change_pct": change_pct,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("symbol", kind="stable").reset_index(drop=True)


def repair_slot_from_archive(
    store: Any,
    minute: pd.Timestamp,
    frame: pd.DataFrame,
    *,
    roster_size: int,
    work_dir: Path,
) -> dict[str, Any]:
    """Publish a canonical snapshot rebuilt from genuine archive bars."""
    minute = normalized_cycle_minute(minute)
    quoted = int(len(frame))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": ARCHIVE_REPAIR_SOURCE,
        "assembly_mode": "ARCHIVE_REPAIR",
        "archive_repair": True,
        "capture_interval_minutes": 5,
        "cycle_minute_ist": minute.isoformat(),
        "worker_count": 1,
        "expected_symbol_count": int(roster_size),
        "selected_symbol_count": int(roster_size),
        "row_count": quoted,
        "quoted_symbol_count": quoted,
        "missing_quote_count": max(0, int(roster_size) - quoted),
        "complete_quote_symbol_count": quoted,
        "coverage_is_informational": True,
        "provenance_note": (
            "Canonical snapshot reconstructed from genuine NSE five-minute archive "
            "bars because no live quote shard was persisted for this slot. Price is "
            "the archived candle close stamped at the slot; it is not a live tick."
        ),
    }
    write_result = write_canonical_snapshot(
        store, minute, frame, manifest, work_dir, write_latest=False
    )
    return {
        "status": "ARCHIVE_REPAIRED",
        "cycle_minute_ist": minute.isoformat(),
        "row_count": quoted,
        "quoted_symbol_count": quoted,
        "missing_quote_count": max(0, int(roster_size) - quoted),
        "source": ARCHIVE_REPAIR_SOURCE,
        "snapshot_persisted": True,
        **write_result,
    }


def run_repair(
    session_date: date,
    *,
    minutes: list[str] | None = None,
    expected_universe: list[str] | None = None,
    history_work_dir: Path | None = None,
) -> dict[str, Any]:
    from market_session_guard import is_nse_trading_day
    from persistent_store import GoogleDriveStore

    if not is_nse_trading_day(session_date):
        return {
            "audit_name": "missing-minute-snapshot-repair",
            "session_date": session_date.isoformat(),
            "timezone": "Asia/Kolkata",
            "expected_minutes": 0,
            "repaired_slots": [],
            "archive_repaired_slots": [],
            "skipped_slots": [],
            "repaired_count": 0,
            "archive_repaired_count": 0,
            "skipped_count": 0,
            "repair_status": "SKIPPED_NON_TRADING_DAY",
        }

    store = GoogleDriveStore()
    expected = minutes or expected_minutes(session_date)
    if not expected:
        return {
            "audit_name": "missing-minute-snapshot-repair",
            "session_date": session_date.isoformat(),
            "timezone": "Asia/Kolkata",
            "expected_minutes": 0,
            "repaired_slots": [],
            "archive_repaired_slots": [],
            "skipped_slots": [],
            "repaired_count": 0,
            "archive_repaired_count": 0,
            "skipped_count": 0,
            "repair_status": "NO_EXPECTED_SLOTS",
        }
    day_files = _list_day_files(store, session_date)
    shard_files = _list_shard_files(store, session_date) if day_files else []
    day_names = {str(item.get("name") or "") for item in day_files}
    roster = list(expected_universe or [])
    if not roster:
        from prediction_universe import authoritative_roster
        roster = authoritative_roster()
    if len(roster) == 500:
        expected_universe = sorted(roster)
    work_root = Path(tempfile.mkdtemp(prefix="repair_minute_snapshots_"))
    archive_history: dict[str, pd.DataFrame] | None = None
    archive_status = "NOT_LOADED"
    repaired: list[dict[str, Any]] = []
    archive_repaired: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def _archive_frame(minute: pd.Timestamp) -> pd.DataFrame | None:
        nonlocal archive_history, archive_status
        if archive_history is None:
            if not roster:
                archive_status = "NO_ROSTER"
                return None
            try:
                archive_history = load_archive_histories(
                    store,
                    session_date,
                    roster,
                    history_work_dir or work_root,
                )
            except Exception as exc:
                archive_status = f"LOAD_FAILED:{type(exc).__name__}"
                archive_history = {}
                return None
            archive_status = "AVAILABLE" if archive_history else "EMPTY_ARCHIVE"
        frame = build_archive_snapshot_frame(archive_history, minute, roster)
        return frame if not frame.empty else None

    for key in expected:
        minute = normalized_cycle_minute(
            datetime.strptime(key, "%Y%m%d_%H%M").replace(tzinfo=IST)
        )
        names = canonical_artifact_names(minute)
        snapshot_present = names["snapshot_name"] in day_names
        existing = read_canonical_manifest(store, minute, work_root)
        status = str((existing or {}).get("status") or "").upper()
        if snapshot_present and status in {"STORED", "OK"}:
            continue
        worker_count = _shard_worker_count(shard_files, key)
        shard_result: dict[str, Any] | None = None
        if worker_count is not None:
            shard_result = repair_slot(
                store,
                minute,
                worker_count=worker_count,
                work_dir=work_root,
                expected_universe=expected_universe,
            )
            if shard_result["status"] == "VALIDATED" and shard_result["snapshot_persisted"]:
                repaired.append(shard_result)
                continue
        frame = _archive_frame(minute)
        if frame is not None:
            try:
                archived = repair_slot_from_archive(
                    store,
                    minute,
                    frame,
                    roster_size=len(roster),
                    work_dir=work_root,
                )
            except Exception as exc:
                skipped.append({
                    "status": "ARCHIVE_REPAIR_FAILED",
                    "cycle_minute_ist": minute.isoformat(),
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "shard_repair_status": (
                        shard_result.get("status") if shard_result else None
                    ),
                })
                continue
            if shard_result is not None:
                archived["shard_repair_status"] = shard_result.get("status")
            archive_repaired.append(archived)
            repaired.append(archived)
        elif shard_result is not None:
            skipped.append(shard_result)
        else:
            skipped.append({
                "status": "NO_SHARDS_NO_ARCHIVE_BARS",
                "cycle_minute_ist": minute.isoformat(),
                "worker_count": 0,
            })

    return {
        "audit_name": "missing-minute-snapshot-repair",
        "session_date": session_date.isoformat(),
        "timezone": "Asia/Kolkata",
        "expected_minutes": len(expected),
        "repaired_slots": repaired,
        "archive_repaired_slots": archive_repaired,
        "skipped_slots": skipped,
        "repaired_count": len(repaired),
        "archive_repaired_count": len(archive_repaired),
        "skipped_count": len(skipped),
        "archive_status": archive_status,
        "repair_status": "COMPLETED",
    }


def _parse_minutes(value: str) -> list[str] | None:
    keys = [part.strip() for part in str(value or "").split(",") if part.strip()]
    return keys or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="", help="Trading date YYYY-MM-DD (default: today IST)")
    parser.add_argument("--minutes", default="", help="Optional comma-separated slot keys YYYYMMDD_HHMM")
    parser.add_argument("--expected-universe", default="", help="Optional comma-separated roster")
    args = parser.parse_args(argv)

    raw_date = str(args.date or "").strip()
    session_date = (
        datetime.strptime(raw_date, "%Y-%m-%d").date()
        if raw_date else pd.Timestamp.now(tz=IST).date()
    )
    expected_universe = [
        part.strip().upper()
        for part in str(args.expected_universe or "").split(",")
        if part.strip()
    ] or None
    report = run_repair(
        session_date,
        minutes=_parse_minutes(args.minutes),
        expected_universe=expected_universe,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
