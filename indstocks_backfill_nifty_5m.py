#!/usr/bin/env python3
"""Backfill NIFTY 50 five-minute market context through the free INDstocks API.

This fills the market-context gap after the static Kaggle CSV without Dhan or
Zerodha. It resolves the NIFTY 50 index from the provider's ``source=index``
master, downloads 5-minute candles in seven-day windows, converts candle-open
timestamps into completed bar closes, and merges the result with the existing
close-only Kaggle partition.

The output is a close-only Parquet named by a manifest so ``pretrain_pipeline.py``
can use it as ``--market-parquet`` and the nightly job can trace its provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from indstocks_backfill_all_nse_5m_history import normalize_indstocks_5m
from indstocks_client import historical, resolve_index_scrip_code

IST = ZoneInfo("Asia/Kolkata")
CANONICAL_REMOTE = (
    "production_data/nse_official/intraday_seed/olo_5m_archive/"
    "symbols/symbol=__NIFTY50__"
)
KAGGLE_NAME = "nifty50_kaggle_5m_close_only.parquet"
MERGED_NAME = "nifty50_5m_merged_close_only.parquet"
RAW_NAME = "nifty50_indstocks_5m_raw.parquet"
MANIFEST_NAME = "nifty50_5m_merged_close_only_manifest.json"
FULL_SESSION_BARS = 75


def _cloud_run_owner() -> bool:
    """Match ``indstocks_client``'s runtime guard: only Cloud Run may fetch history."""
    service = os.getenv("K_SERVICE", "").strip()
    job = os.getenv("CLOUD_RUN_JOB", "").strip()
    return service == "indian-market-intraday-worker" or job == "native-5m-backfill"


def _default_drive_root() -> Path:
    mounted = Path("/content/drive/MyDrive/Indian_Market_Research")
    if mounted.exists():
        return mounted
    import os

    return Path(os.getenv("INDIAN_MARKET_RESEARCH_ROOT", "/tmp/Indian_Market_Research"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_ist(value: Any, *, end_of_day: bool = False) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(IST)
    else:
        ts = ts.tz_convert(IST)
    if end_of_day and ts.time() == pd.Timestamp("00:00").time():
        ts = ts + pd.Timedelta(days=1)
    return ts


def _timestamp_column(frame: pd.DataFrame) -> pd.Series:
    for column in ("Timestamp", "timestamp", "Datetime", "datetime"):
        if column in frame.columns:
            return pd.to_datetime(frame[column], errors="coerce")
    return pd.to_datetime(frame.index, errors="coerce")


def _close_series(frame: pd.DataFrame | None) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype="float64", name="Close")
    close_column = "Close" if "Close" in frame.columns else "close"
    if close_column not in frame.columns:
        raise RuntimeError("NIFTY source has no Close column")
    index = pd.DatetimeIndex(_timestamp_column(frame))
    index = index.tz_localize(IST) if index.tz is None else index.tz_convert(IST)
    values = pd.to_numeric(frame[close_column], errors="coerce").to_numpy()
    series = pd.Series(values, index=index, name="Close")
    series = series.loc[~series.index.isna()].sort_index()
    series = series.loc[series > 0]
    return series.loc[~series.index.duplicated(keep="last")]


def _session_bars(index: pd.DatetimeIndex) -> dict[str, int]:
    if len(index) == 0:
        return {}
    counts = pd.Series(1, index=index.normalize()).groupby(level=0).sum()
    return {str(day.date()): int(count) for day, count in counts.items()}


def _write_manifest(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _publish(local: Path, remote_rel: str, drive_root: Path, *, no_upload: bool) -> dict[str, Any]:
    if no_upload:
        return {"mode": "local_only", "path": str(local)}
    mounted = drive_root
    if mounted.exists():
        destination = mounted / remote_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, destination)
        return {
            "mode": "mounted_drive_copy",
            "path": str(destination),
            "size": destination.stat().st_size,
        }
    from persistent_store import GoogleDriveStore

    uploaded = GoogleDriveStore().upload_or_update(
        local,
        remote_rel,
        mime_type="application/vnd.apache.parquet" if local.suffix == ".parquet" else "application/json",
    )
    if str(uploaded.get("status") or "").upper() != "OK":
        raise RuntimeError(f"Drive upload not confirmed for {remote_rel}: {uploaded}")
    return {"mode": "drive_api", **uploaded}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-name", default="NIFTY 50")
    parser.add_argument("--exchange", default="NSE")
    parser.add_argument("--scrip-code", default="", help="Provider code such as NSE_40000001; skips resolution")
    parser.add_argument("--start", default="", help="Inclusive IST date/timestamp; default: Kaggle end + 1 day")
    parser.add_argument("--end", default=date.today().isoformat(), help="Inclusive IST date/timestamp")
    parser.add_argument("--max-calls", type=int, default=0, help="Safety cap; default derived from range")
    parser.add_argument("--drive-root", type=Path, default=_default_drive_root())
    parser.add_argument("--kaggle-parquet", type=Path, default=None)
    parser.add_argument("--local-dir", type=Path, default=Path("/tmp/nifty_5m_backfill"))
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--probe", action="store_true", help="Resolve and print the provider code, then exit")
    args = parser.parse_args(argv)

    if not _cloud_run_owner():
        print(
            "INDSTOCKS_LOCAL_ACCESS_DISABLED: NIFTY 5m history is produced by the Cloud Run "
            "backfill. Read the merged artifact from Drive and trigger the Cloud Run job if "
            "it is stale; see INTEGRATION.md section 0."
        )
        return 2

    code = args.scrip_code.strip() or resolve_index_scrip_code(args.index_name, args.exchange)
    print(f"NIFTY_SCRIP_CODE={code}")
    if args.probe:
        return 0

    kaggle_path = args.kaggle_parquet or (
        args.drive_root / CANONICAL_REMOTE / KAGGLE_NAME
    )
    merged_path = args.drive_root / CANONICAL_REMOTE / MERGED_NAME
    base_path = args.kaggle_parquet or (merged_path if merged_path.exists() else kaggle_path)
    base = _close_series(pd.read_parquet(base_path)) if base_path.exists() else pd.Series(dtype="float64", name="Close")
    if base.empty:
        print(f"BASE_CONTEXT_MISSING={base_path}")

    end = _as_ist(args.end, end_of_day=True)
    if args.start.strip():
        start = _as_ist(args.start)
    elif not base.empty:
        start = base.index.max().normalize() + pd.Timedelta(days=1)
    else:
        start = _as_ist("2024-01-01")
    if start >= end:
        print("NO_FETCH_RANGE: start >= end")
        return 1

    days = max(1, math.ceil((end - start) / pd.Timedelta(days=1)))
    max_calls = args.max_calls or max(10, math.ceil(days / 7) + 2)
    print(f"FETCH_RANGE={start.isoformat()}..{end.isoformat()} max_calls={max_calls}")
    raw = historical(code, start, end, "5m", max_calls=max_calls)
    cutoff = min(pd.Timestamp.now(tz=IST).floor("5min"), end - pd.Timedelta(minutes=5))
    fresh = normalize_indstocks_5m(raw, cutoff=cutoff)
    if fresh.empty:
        raise RuntimeError("INDstocks returned no usable NIFTY 50 five-minute candles")
    fresh_close = fresh["Close"]

    merged = pd.concat([base, fresh_close]).sort_index()
    merged = merged.loc[merged.index >= (base.index.min() if not base.empty else merged.index.min())]
    merged = merged.loc[merged.index <= cutoff]
    merged = merged.loc[~merged.index.duplicated(keep="last")]

    args.local_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.local_dir / RAW_NAME
    out_path = args.local_dir / MERGED_NAME
    manifest_path = args.local_dir / MANIFEST_NAME
    fresh.to_parquet(raw_path, index=True, compression="zstd")
    output = merged.rename("Close").reset_index()
    output.columns = ["Timestamp", "Close"]
    output.to_parquet(out_path, index=False, compression="zstd")

    sessions = _session_bars(pd.DatetimeIndex(merged.index))
    incomplete = {day: bars for day, bars in sessions.items() if bars < FULL_SESSION_BARS}
    report = {
        "source": "INDSTOCKS_HISTORICAL_5M",
        "index_name": args.index_name,
        "scrip_code": code,
        "timestamp_semantics": "bar_close_ist",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start_ist": start.isoformat(),
        "requested_end_ist": end.isoformat(),
        "fetched_rows": int(len(fresh)),
        "base_parquet": str(base_path),
        "base_is_merged": bool(base_path == merged_path),
        "base_rows": int(len(base)),
        "kaggle_rows": int(len(base)) if not base_path.name == MERGED_NAME else None,
        "output_rows": int(len(merged)),
        "output_file": out_path.name,
        "output_sha256": _sha256(out_path),
        "first_timestamp_ist": merged.index.min().isoformat() if not merged.empty else None,
        "last_timestamp_ist": merged.index.max().isoformat() if not merged.empty else None,
        "sessions": len(sessions),
        "first_session": min(sessions) if sessions else None,
        "last_session": max(sessions) if sessions else None,
        "incomplete_sessions": incomplete,
        "incomplete_session_count": len(incomplete),
        "expected_full_session_bars": FULL_SESSION_BARS,
        "kaggle_parquet": str(kaggle_path) if kaggle_path.exists() else None,
        "raw_provider_file": raw_path.name,
        "raw_provider_sha256": _sha256(raw_path),
    }
    _write_manifest(manifest_path, report)

    if not args.no_upload:
        report["kaggle_publish"] = _publish(raw_path, f"{CANONICAL_REMOTE}/{RAW_NAME}", args.drive_root, no_upload=False)
        report["publish"] = _publish(out_path, f"{CANONICAL_REMOTE}/{MERGED_NAME}", args.drive_root, no_upload=False)
        _write_manifest(manifest_path, report)
        report["manifest_publish"] = _publish(manifest_path, f"{CANONICAL_REMOTE}/{MANIFEST_NAME}", args.drive_root, no_upload=False)
        _write_manifest(manifest_path, report)

    print(json.dumps({k: v for k, v in report.items() if k not in {"incomplete_sessions"}}, indent=2, sort_keys=True))
    if incomplete:
        print(f"INCOMPLETE_SESSIONS={len(incomplete)} sample={list(incomplete.items())[:10]}")
    print(f"OUTPUT={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
