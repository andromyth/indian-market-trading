#!/usr/bin/env python3
"""Re-assemble canonical five-minute snapshots from already-persisted shards.

A shard upload can succeed while the final assembly or its manifest upload
fails. The live compiler only consumes canonical snapshots, so those slots stay
missing forever even though every quote is still present on Drive. This module
is a Drive-only repair step: it never calls INDstocks and never fabricates a
quote. It re-runs the same deterministic assembly and publishes the canonical
snapshot only when the persisted shards validate.
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

import pandas as pd

from all_universe_snapshot_assembly import (
    DEFAULT_WORKER_COUNT,
    SHARD_SUBDIR,
    assemble_from_drive,
    canonical_artifact_names,
    normalized_cycle_minute,
    read_canonical_manifest,
)
from audit_all_universe_minute_snapshots import (
    _find_folder,
    _list_children,
    expected_minutes,
)

IST = ZoneInfo("Asia/Kolkata")
QUOTE_ROOT = "production_data/all_universe_minute_snapshots"
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


def run_repair(
    session_date: date,
    *,
    minutes: list[str] | None = None,
    expected_universe: list[str] | None = None,
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
            "skipped_slots": [],
            "repaired_count": 0,
            "skipped_count": 0,
            "repair_status": "SKIPPED_NON_TRADING_DAY",
        }

    store = GoogleDriveStore()
    day_files = _list_day_files(store, session_date)
    expected = minutes or expected_minutes(session_date)
    if not day_files:
        return {
            "audit_name": "missing-minute-snapshot-repair",
            "session_date": session_date.isoformat(),
            "timezone": "Asia/Kolkata",
            "expected_minutes": len(expected),
            "repaired_slots": [],
            "skipped_slots": [],
            "repaired_count": 0,
            "skipped_count": 0,
            "repair_status": "NO_SNAPSHOT_FOLDER",
        }
    shard_files = _list_shard_files(store, session_date)
    day_names = {str(item.get("name") or "") for item in day_files}
    work_root = Path(tempfile.mkdtemp(prefix="repair_minute_snapshots_"))
    repaired: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

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
        if worker_count is None:
            try:
                declared = int((existing or {}).get("worker_count") or 0)
            except (TypeError, ValueError):
                declared = 0
            worker_count = declared if declared > 1 else _default_worker_count()
        result = repair_slot(
            store,
            minute,
            worker_count=worker_count,
            work_dir=work_root,
            expected_universe=expected_universe,
        )
        if result["status"] == "VALIDATED" and result["snapshot_persisted"]:
            repaired.append(result)
        else:
            skipped.append(result)

    return {
        "audit_name": "missing-minute-snapshot-repair",
        "session_date": session_date.isoformat(),
        "timezone": "Asia/Kolkata",
        "expected_minutes": len(expected),
        "repaired_slots": repaired,
        "skipped_slots": skipped,
        "repaired_count": len(repaired),
        "skipped_count": len(skipped),
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
