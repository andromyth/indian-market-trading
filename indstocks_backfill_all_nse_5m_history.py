#!/usr/bin/env python3
"""Backfill native NSE five-minute history through the INDstocks read-only API.

The provider permits at most five scrip codes and seven calendar days per request.
This runner batches five stocks per request, walks the requested range in seven-day
windows, respects the documented 5 requests/second data limit, and checkpoints each
stock directly to the same Drive archive used by the DhanHQ backfill. Provider-omitted
instruments in successful historical responses are recorded as empty for that window,
matching the live capture parser.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from dhan_backfill_all_nse_5m_history import (
    FRESHNESS_WINDOW_DAYS,
    MANIFEST_REMOTE,
    MIN_SESSION_BARS,
    REMOTE_ROOT,
    _download_existing,
    _resolve_remote_file,
    choose_backfill_start,
    retain_latest_sessions,
    usable_session_summary,
)

BATCH_PROGRESS_REMOTE = "production_data/nse_official/intraday_seed/olo_5m_archive/backfill_batch_progress.json"


IST = ZoneInfo("Asia/Kolkata")
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
MAX_CODES_PER_REQUEST = 5
MAX_DAYS_PER_REQUEST = 7
RECENT_SESSION_LOOKBACK_DAYS = 10
FULL_REGULAR_SESSION_BARS = 75
LIVE_REMOTE_ROOT = "production_data/nse_official/intraday_5m_live"

# Set by the private Cloud Run worker so this job reuses its process-local TOTP
# token cache and yields API capacity during live-market hours.
REQUEST_GUARD = None
BACKFILL_PROGRESS_CALLBACK = None

# Authoritative owner-supplied roster for this backfill exercise.
PINNED_UNIVERSE_FILENAME = "backfill_universe.json"
# Bumping either constant invalidates previously written window checkpoints so
# bars collected before explicit date definitions existed are re-fetched once.
CHECKPOINT_SCHEMA_VERSION = "indstocks-5m-date-def-v2-2026-11-19"
HISTORY_SCHEMA = "indstocks_5m_completed_close_ist_v2"

# Defaults supplied by the Cloud Run /backfill request payload. Explicit CLI
# arguments still win; these matter because the worker calls main([]).
RUNTIME_OPTIONS: dict[str, Any] = {
    "end_date": None,
    "replace_existing_history": False,
}


def normalize_indstocks_5m(frame: pd.DataFrame | None, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Convert INDstocks candle-open timestamps into completed five-minute bar closes."""
    empty = pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=IST, name="Datetime"))
    if frame is None or frame.empty:
        return empty
    x = frame.copy()
    x.columns = [str(column).strip() for column in x.columns]
    aliases = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }
    x = x.rename(columns={column: aliases.get(str(column).strip().lower(), column) for column in x.columns})
    if "Timestamp" in x.columns:
        index = pd.DatetimeIndex(pd.to_datetime(x.pop("Timestamp"), errors="coerce"))
    else:
        index = pd.DatetimeIndex(pd.to_datetime(x.index, errors="coerce"))
    index = index.tz_localize(IST) if index.tz is None else index.tz_convert(IST)
    x.index = index
    for column in OHLCV:
        if column not in x:
            x[column] = np.nan
        x[column] = pd.to_numeric(x[column], errors="coerce")
    x = x.loc[~x.index.isna()].sort_index()
    x = x.loc[~x.index.duplicated(keep="last")]
    starts = pd.DatetimeIndex(x.index)
    regular = (
        (starts.time >= pd.Timestamp("09:15").time())
        & (starts.time < pd.Timestamp("15:30").time())
        & (starts.minute % 5 == 0)
    )
    x = x.loc[regular].copy()
    x.index = x.index + pd.Timedelta(minutes=5)
    valid = x[["Open", "High", "Low", "Close"]].notna().all(axis=1)
    valid &= x[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)
    x = x.loc[valid]
    if cutoff is not None:
        boundary = pd.Timestamp(cutoff)
        boundary = boundary.tz_localize(IST) if boundary.tzinfo is None else boundary.tz_convert(IST)
        x = x.loc[x.index <= boundary]
    x.index.name = "Datetime"
    return x[OHLCV].loc[~x.index.duplicated(keep="last")].sort_index()


def _instrument_codes() -> dict[str, str]:
    """Build the current ordinary NSE-equity symbol-to-provider-code map once."""
    from indstocks_client import _load_instruments

    frame = _load_instruments("NSE")
    required = {"EXCH", "SECURITY_ID", "TRADING_SYMBOL"}
    if not required.issubset(frame.columns):
        raise RuntimeError("INDstocks instrument master is missing NSE equity mapping columns.")
    x = frame.copy()
    x["EXCH"] = x["EXCH"].astype(str).str.strip().str.upper()
    x["SECURITY_ID"] = x["SECURITY_ID"].astype(str).str.strip()
    x["TRADING_SYMBOL"] = x["TRADING_SYMBOL"].astype(str).str.strip().str.upper()
    x = x.loc[
        x["EXCH"].eq("NSE")
        & x["SECURITY_ID"].str.fullmatch(r"\d+")
        & x["TRADING_SYMBOL"].ne("")
    ].copy()
    if "SEGMENT" in x.columns:
        equity = x["SEGMENT"].astype(str).str.strip().str.upper().isin({"E", "EQ", "EQUITY"})
        if equity.any():
            x = x.loc[equity]
    if "SERIES" in x.columns:
        x["_series_priority"] = (
            ~x["SERIES"].astype(str).str.strip().str.upper().eq("EQ")
        ).astype(int)
        x = x.sort_values(
            ["TRADING_SYMBOL", "_series_priority"],
            kind="stable",
        )
    x = x.drop_duplicates("TRADING_SYMBOL", keep="first")
    return {
        str(row.TRADING_SYMBOL): f"NSE_{row.SECURITY_ID}"
        for row in x.itertuples(index=False)
    }


def _align_to_monday(value: datetime) -> datetime:
    """Align a range start to a stable Monday midnight for reusable 7-day checkpoints."""
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize(IST) if stamp.tzinfo is None else stamp.tz_convert(IST)
    stamp = stamp.normalize() - pd.Timedelta(days=stamp.weekday())
    return stamp.to_pydatetime()


def _checkpoint_key(symbols: list[str] | set[str]) -> str:
    """Stable checkpoint folder key scoped to the current date-definition schema."""
    payload = "|".join(sorted(str(symbol) for symbol in symbols)) + "|" + CHECKPOINT_SCHEMA_VERSION
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _pinned_universe() -> list[str]:
    """Load the owner-supplied 500-symbol roster shipped with the worker image."""
    path = Path(__file__).with_name(PINNED_UNIVERSE_FILENAME)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    values = data.get("symbols") if isinstance(data, dict) else data
    if not isinstance(values, list):
        return []
    return sorted({
        str(value).strip().upper().replace(".NS", "").replace(".BO", "")
        for value in values if str(value).strip()
    })


def merge_symbol_history(
    existing: pd.DataFrame | None,
    fetched: pd.DataFrame | None,
    *,
    fetched_wins: bool,
) -> pd.DataFrame:
    """Merge archive and provider frames; keep the last duplicate for the winning source."""
    frames = [
        frame for frame in ((existing, fetched) if fetched_wins else (fetched, existing))
        if frame is not None and not frame.empty
    ]
    if not frames:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=IST, name="Datetime"))
    merged = pd.concat(frames).sort_index()
    merged = merged.loc[~merged.index.duplicated(keep="last")]
    return merged[OHLCV].sort_index()


def _download_manifest_json(store: Any, local_root: Path) -> dict[str, Any]:
    """Read the previous manifest to detect a date-definition schema change."""
    item = _resolve_remote_file(store, MANIFEST_REMOTE)
    if not item:
        return {}
    local = local_root / "backfill_manifest.prior.json"
    try:
        from persistent_store import download_drive_file_with_retry
        download_drive_file_with_retry(
            store.service, item["id"], local, operation="backfill_manifest_download"
        )
        data = json.loads(local.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _recent_incomplete_session_dates(
    frame: pd.DataFrame,
    end: datetime,
    *,
    lookback_days: int = RECENT_SESSION_LOOKBACK_DAYS,
    expected_bars: int = FULL_REGULAR_SESSION_BARS,
) -> list[str]:
    """Return recent observed regular sessions that do not contain all 75 close-time bars."""
    if frame is None or frame.empty:
        return []
    from market_session_guard import is_nse_trading_day

    x = frame.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(x.index, errors="coerce"))
    idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
    x.index = idx
    required = ["Open", "High", "Low", "Close"]
    if not set(required).issubset(x.columns):
        return []
    valid = x[required].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    valid &= x[required].apply(pd.to_numeric, errors="coerce").gt(0).all(axis=1)
    valid &= (x.index.time >= pd.Timestamp("09:20").time())
    valid &= (x.index.time <= pd.Timestamp("15:30").time())
    valid &= (x.index.minute % 5 == 0)
    observed_dates = set(x.index[valid].date)
    if not observed_dates:
        return []
    end_stamp = pd.Timestamp(end)
    end_stamp = end_stamp.tz_localize(IST) if end_stamp.tzinfo is None else end_stamp.tz_convert(IST)
    first_observed = min(observed_dates)
    start_date = max(first_observed, end_stamp.date() - timedelta(days=int(lookback_days)))
    session_dates = [
        start_date + timedelta(days=offset)
        for offset in range((end_stamp.date() - start_date).days + 1)
        if is_nse_trading_day(start_date + timedelta(days=offset))
    ]
    counts = pd.Series(1, index=x.index[valid]).groupby(
        pd.Index(x.index[valid].date)
    ).sum().to_dict()
    return [
        day.isoformat()
        for day in session_dates
        if int(counts.get(day, 0)) < int(expected_bars)
    ]


def _load_recent_live_history(
    store: Any,
    symbols: set[str],
    end: datetime,
    work_root: Path,
    *,
    lookback_days: int = RECENT_SESSION_LOOKBACK_DAYS,
    file_listing_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Load recent durable live OHLCV captures so backfill plans against all saved history."""
    from persistent_store import _drive_execute_with_retry, download_drive_file_with_retry
    from market_session_guard import is_nse_trading_day

    start_date = end.date() - timedelta(days=int(lookback_days))
    buckets: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in symbols}
    files_read = 0
    files_failed = 0
    for offset in range((end.date() - start_date).days + 1):
        day = start_date + timedelta(days=offset)
        if not is_nse_trading_day(day):
            continue
        day_key = day.isoformat()
        reuse_listing = (
            file_listing_cache is not None
            and day != end.date()
            and day_key in file_listing_cache
        )
        if reuse_listing:
            day_files = file_listing_cache[day_key]
        else:
            remote_dir = f"{LIVE_REMOTE_ROOT}/{day:%Y/%m/%d}"
            folder = _resolve_remote_file(store, remote_dir)
            if not folder:
                continue
            page_token = None
            day_files = []
            while True:
                params = {
                    "q": f"'{folder['id']}' in parents and trashed = false",
                    "spaces": "drive",
                    "fields": "nextPageToken,files(id,name,mimeType)",
                    "pageSize": 1000,
                    "includeItemsFromAllDrives": True,
                    "supportsAllDrives": True,
                }
                if page_token:
                    params["pageToken"] = page_token
                response = _drive_execute_with_retry(
                    "live_capture_archive_list",
                    lambda: store.service.files().list(**params).execute(),
                )
                day_files.extend(response.get("files", []))
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
            # Past-session capture folders are immutable during this run. Refresh
            # today's folder each batch so newly written live snapshots are visible.
            if file_listing_cache is not None and day != end.date():
                file_listing_cache[day_key] = day_files
        for item in day_files:
            name = str(item.get("name") or "")
            if item.get("mimeType") == "application/vnd.google-apps.folder":
                continue
            if not name.startswith("capture_") or not name.endswith(".parquet"):
                continue
            local = work_root / "live_history" / f"{day:%Y%m%d}" / Path(name).name
            local.parent.mkdir(parents=True, exist_ok=True)
            try:
                if not local.exists() or local.stat().st_size == 0:
                    download_drive_file_with_retry(
                        store.service, item["id"], local,
                        operation="live_capture_archive_download",
                    )
                # Push symbol filtering and column projection into Parquet/Arrow.
                # The archive files contain the full NSE universe; materializing every
                # symbol for every day multiplied peak RAM by the entire universe.
                capture = pd.read_parquet(
                    local,
                    columns=["symbol", "Timestamp", *OHLCV],
                    filters=[("symbol", "in", sorted(symbols))],
                )
                if capture.empty:
                    files_read += 1
                    continue
                if not {"symbol", "Timestamp", *OHLCV}.issubset(capture.columns):
                    files_failed += 1
                    continue
                capture["symbol"] = capture["symbol"].astype(str).str.strip().str.upper()
                capture = capture.loc[capture["symbol"].isin(symbols)].copy()
                if capture.empty:
                    files_read += 1
                    continue
                idx = pd.DatetimeIndex(pd.to_datetime(capture.pop("Timestamp"), errors="coerce"))
                idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
                capture.index = idx
                for column in OHLCV:
                    capture[column] = pd.to_numeric(capture[column], errors="coerce")
                capture = capture.loc[~capture.index.isna()]
                capture = capture.loc[capture.index <= pd.Timestamp(end)]
                capture = capture.loc[capture[["Open", "High", "Low", "Close"]].notna().all(axis=1)]
                capture = capture.loc[capture[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)]
                for symbol, part in capture.groupby("symbol", sort=False):
                    buckets[str(symbol)].append(part[OHLCV])
                files_read += 1
            except Exception:
                files_failed += 1
    result: dict[str, pd.DataFrame] = {}
    for symbol, parts in buckets.items():
        if not parts:
            result[symbol] = pd.DataFrame(
                columns=OHLCV,
                index=pd.DatetimeIndex([], tz=IST, name="Datetime"),
            )
            continue
        merged = pd.concat(parts).sort_index()
        merged = merged.loc[~merged.index.duplicated(keep="last")]
        merged.index.name = "Datetime"
        result[symbol] = merged[OHLCV]
    return result, {"live_capture_files_read": files_read, "live_capture_files_failed": files_failed}


def fetch_indstocks_5m_batch(
    symbols_and_codes: list[tuple[str, str]],
    start: datetime,
    end: datetime,
    *,
    pause_seconds: float = 1.0,
    request_guard: Any = None,
    progress_context: dict[str, Any] | None = None,
    store: Any = None,
    checkpoint_root: Path | None = None,
    force_refetch_after: datetime | None = None,
    active_symbols: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], int]:
    """Fetch a maximum of five stocks at once over consecutive seven-day windows."""
    if not symbols_and_codes:
        return {}, 0
    if len(symbols_and_codes) > MAX_CODES_PER_REQUEST:
        raise ValueError(f"INDstocks accepts at most {MAX_CODES_PER_REQUEST} scrip codes per request.")
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Historical range must use timezone-aware start/end with start < end.")

    from indstocks_client import _request

    symbols_by_code = {code: symbol for symbol, code in symbols_and_codes}
    batch_key = _checkpoint_key(list(symbols_by_code.values()))
    # Checkpoints written before FULL_BATCH_V2 were keyed by the *active* subset
    # (symbols not yet complete), so the folder changed whenever a peer finished.
    # New checkpoints are keyed by the full five-symbol group. Old folders stay
    # readable (never written) so completed windows are not downloaded again.
    full_symbols = sorted(symbols_by_code.values())
    legacy_symbols = sorted(set(active_symbols or ()) & set(full_symbols))
    legacy_key = (
        _checkpoint_key(legacy_symbols)
        if legacy_symbols and legacy_symbols != full_symbols else None
    )
    accumulated: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol, _ in symbols_and_codes}
    cursor = start
    calls = 0
    last_request_started = 0.0
    # List the batch checkpoint folder once per batch. Resolving each window
    # against Drive individually multiplied API calls by the window count.
    checkpoint_folder_id: str | None = None
    checkpoint_folder_items: dict[str, dict[str, Any]] = {}
    if store is not None and checkpoint_root is not None:
        list_folder = getattr(store, "list_folder", None)
        ensure_path = getattr(store, "ensure_path", None)
        if callable(list_folder) and callable(ensure_path):
            checkpoint_folder_remote = f"{REMOTE_ROOT}/window_checkpoints/{batch_key}"
            checkpoint_folder_id = ensure_path(
                checkpoint_folder_remote, service=store._get_write_service()
            )
            checkpoint_folder_items = {
                str(item.get("name") or ""): item
                for item in list_folder(checkpoint_folder_id)
            }
    while cursor < end:
        window_end = min(end, cursor + timedelta(days=MAX_DAYS_PER_REQUEST))
        window_key = f"{cursor:%Y%m%dT%H%M}_{window_end:%Y%m%dT%H%M}"
        checkpoint_remote = f"{REMOTE_ROOT}/window_checkpoints/{batch_key}/{window_key}.parquet"
        checkpoint_local = (checkpoint_root or Path("/tmp/nse_5m_checkpoints")) / batch_key / f"{window_key}.parquet"
        checkpoint_manifest_remote = checkpoint_remote + ".json"
        checkpoint_manifest_local = checkpoint_local.with_suffix(".parquet.json")
        lookup_enabled = store is not None and checkpoint_root is not None
        if lookup_enabled and checkpoint_folder_id is not None:
            checkpoint_item = checkpoint_folder_items.get(f"{window_key}.parquet")
            checkpoint_manifest_item = checkpoint_folder_items.get(f"{window_key}.parquet.json")
        else:
            checkpoint_item = _resolve_remote_file(store, checkpoint_remote) if lookup_enabled else None
            checkpoint_manifest_item = _resolve_remote_file(store, checkpoint_manifest_remote) if lookup_enabled else None
        expected_symbols = full_symbols
        reuse_expected = full_symbols
        read_local = checkpoint_local
        read_manifest_local = checkpoint_manifest_local
        checkpoint_source = "FULL_BATCH_V2"
        if lookup_enabled and legacy_key and not (checkpoint_item and checkpoint_manifest_item):
            legacy_remote = f"{REMOTE_ROOT}/window_checkpoints/{legacy_key}/{window_key}.parquet"
            legacy_item = _resolve_remote_file(store, legacy_remote)
            legacy_manifest_item = _resolve_remote_file(store, legacy_remote + ".json")
            if legacy_item and legacy_manifest_item:
                checkpoint_item, checkpoint_manifest_item = legacy_item, legacy_manifest_item
                reuse_expected = legacy_symbols
                read_local = (checkpoint_root or Path("/tmp/nse_5m_checkpoints")) / legacy_key / f"{window_key}.parquet"
                read_manifest_local = read_local.with_suffix(".parquet.json")
                checkpoint_source = "LEGACY_ACTIVE_SUBSET_KEY"
        reuse_checkpoint = bool(checkpoint_item and checkpoint_manifest_item)
        if reuse_checkpoint:
            if force_refetch_after is not None:
                force_after = pd.Timestamp(force_refetch_after)
                force_after = force_after.tz_localize(IST) if force_after.tzinfo is None else force_after.tz_convert(IST)
                reuse_checkpoint = pd.Timestamp(window_end) <= force_after
        if reuse_checkpoint:
            try:
                from persistent_store import download_drive_file_with_retry
                read_local.parent.mkdir(parents=True, exist_ok=True)
                download_started = time.monotonic()
                for item, destination, operation in (
                    (checkpoint_item, read_local, "checkpoint_parquet_download"),
                    (checkpoint_manifest_item, read_manifest_local, "checkpoint_manifest_download"),
                ):
                    download_drive_file_with_retry(
                        store.service, item["id"], destination, operation=operation
                    )
                if callable(BACKFILL_PROGRESS_CALLBACK):
                    BACKFILL_PROGRESS_CALLBACK({
                        "provider": "INDSTOCKS", **(progress_context or {}),
                        "stage": "CHECKPOINT_DOWNLOADED", "window_key": window_key,
                        "batch_key": batch_key, "checkpoint_source": checkpoint_source,
                        "drive_file_id": checkpoint_item["id"],
                        "bytes": read_local.stat().st_size,
                        "duration_seconds": round(time.monotonic() - download_started, 2),
                    })
                manifest = json.loads(read_manifest_local.read_text(encoding="utf-8"))
                if (
                    not isinstance(manifest, dict)
                    or manifest.get("status") != "COMPLETE"
                    or manifest.get("window_key") != window_key
                    or manifest.get("symbols") != reuse_expected
                    or not isinstance(manifest.get("row_counts"), dict)
                ):
                    raise ValueError("checkpoint completion manifest is invalid")
                saved_window = pd.read_parquet(read_local)
                if not saved_window.empty and (
                    not {"Symbol", *OHLCV}.issubset(saved_window.columns)
                    or not set(saved_window["Symbol"].dropna().astype(str)).issubset(set(reuse_expected))
                ):
                    raise ValueError("checkpoint parquet has an invalid schema or unexpected symbols")
                saved_window.index = pd.DatetimeIndex(pd.to_datetime(saved_window.index, errors="coerce"))
                if saved_window.index.isna().any():
                    raise ValueError("checkpoint parquet contains invalid timestamps")
                saved_window.index = (
                    saved_window.index.tz_localize(IST)
                    if saved_window.index.tz is None
                    else saved_window.index.tz_convert(IST)
                )
            except Exception as exc:
                reuse_checkpoint = False
                print(
                    f"BACKFILL_CHECKPOINT_REFETCH_INVALID window={window_key} batch={batch_key} "
                    f"reason={type(exc).__name__}",
                    flush=True,
                )
                if callable(BACKFILL_PROGRESS_CALLBACK):
                    BACKFILL_PROGRESS_CALLBACK({
                        "provider": "INDSTOCKS", **(progress_context or {}),
                        "stage": "CHECKPOINT_REFETCH_INVALID", "window_key": window_key,
                        "batch_key": batch_key, "reason": type(exc).__name__,
                    })
        if reuse_checkpoint:
            if not saved_window.empty:
                for symbol in expected_symbols:
                    part = saved_window.loc[saved_window["Symbol"].eq(symbol), OHLCV]
                    if not part.empty:
                        accumulated[symbol].append(part)
            cursor = window_end
            if callable(BACKFILL_PROGRESS_CALLBACK):
                BACKFILL_PROGRESS_CALLBACK({
                    "provider": "INDSTOCKS", **(progress_context or {}),
                    "stage": "RESUMED_CHECKPOINT", "window_end": window_end.isoformat(),
                    "window_key": window_key, "checkpoint_source": checkpoint_source,
                    "api_calls_in_batch": calls,
                })
            continue
        wait = pause_seconds - (time.monotonic() - last_request_started)
        if calls and wait > 0:
            time.sleep(wait)
        last_request_started = time.monotonic()
        params = {
            "scrip-codes": ",".join(symbols_by_code),
            "start_time": int(cursor.timestamp() * 1000),
            "end_time": int(window_end.timestamp() * 1000),
        }
        # A transient TLS EOF previously terminated the whole backfill because
        # SSL disconnects were not classified as retryable. Keep retries bounded,
        # but allow enough time for the provider connection to recover.
        max_request_attempts = 5
        for attempt in range(max_request_attempts):
            # Recheck immediately before every provider call; retry sleep can
            # cross the NSE live-market boundary.
            if request_guard is not None:
                request_guard()
            if callable(BACKFILL_PROGRESS_CALLBACK):
                BACKFILL_PROGRESS_CALLBACK({
                    "provider": "INDSTOCKS", **(progress_context or {}),
                    "stage": "PROVIDER_REQUEST_STARTED", "window_end": window_end.isoformat(),
                    "attempt": attempt + 1,
                })
            try:
                response = _request("GET", "/market/historical/5minute", params=params)
                break
            except Exception as exc:
                message = str(exc).upper()
                retryable = any(
                    marker in message
                    for marker in (
                        "HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504",
                        "NETWORKEXCEPTION", "REQUEST FAILED", "CONNECTIONERROR", "TIMEOUT",
                        "UNEXPECTED_EOF_WHILE_READING", "SSLEOFERROR",
                        "REMOTEENDCLOSEDERROR", "CONNECTION RESET", "CONNECTIONRESETERROR",
                        "CONNECTION ABORTED", "BROKEN PIPE",
                    )
                )
                if callable(BACKFILL_PROGRESS_CALLBACK):
                    BACKFILL_PROGRESS_CALLBACK({
                        "provider": "INDSTOCKS", **(progress_context or {}),
                        "stage": "PROVIDER_REQUEST_FAILED", "window_end": window_end.isoformat(),
                        "attempt": attempt + 1, "retryable": retryable,
                        "error_type": type(exc).__name__,
                    })
                if not retryable or attempt == max_request_attempts - 1:
                    print(json.dumps({
                        "event": "indstocks_backfill_request_failed",
                        "batch": (progress_context or {}).get("batch"),
                        "attempt": attempt + 1,
                        "max_attempts": max_request_attempts,
                        "retryable": retryable,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:180],
                    }, sort_keys=True), flush=True)
                    raise
                delay = min(2 ** (attempt + 1), 30)
                print(json.dumps({
                    "event": "indstocks_backfill_request_retry",
                    "batch": (progress_context or {}).get("batch"),
                    "attempt": attempt + 1,
                    "max_attempts": max_request_attempts,
                    "delay_seconds": delay,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:180],
                }, sort_keys=True), flush=True)
                time.sleep(delay)
        body = response.json()
        if not isinstance(body, dict) or body.get("success") is not True:
            raise RuntimeError("INDstocks five-minute historical API did not report success.")
        data = body.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"INDstocks response data is not an object for window {window_key}.")
        if callable(BACKFILL_PROGRESS_CALLBACK):
            BACKFILL_PROGRESS_CALLBACK({
                "provider": "INDSTOCKS", **(progress_context or {}),
                "stage": "PROVIDER_RESPONSE", "window_end": window_end.isoformat(),
                "api_calls_in_batch": calls + 1,
            })
        # INDstocks may omit a requested code, or its candles field, when no
        # candles exist for that instrument/window (for example before a listing).
        # The live capture path uses the same empty-history interpretation. Keep
        # this visible in the checkpoint manifest rather than failing valid peers.
        missing_codes = sorted(set(symbols_by_code) - set(data))
        window_frames: list[pd.DataFrame] = []
        row_counts: dict[str, int] = {}
        for code, symbol in symbols_by_code.items():
            instrument = data.get(code) or {}
            if not isinstance(instrument, dict):
                raise RuntimeError(f"INDstocks instrument response is invalid for {symbol} in window {window_key}.")
            candles = instrument.get("candles") or []
            if not isinstance(candles, list):
                raise RuntimeError(f"INDstocks candles field is invalid for {symbol} in window {window_key}.")
            row_counts[symbol] = len(candles)
            if not candles:
                continue
            rows = pd.DataFrame(candles)
            required = {"ts", "o", "h", "l", "c", "v"}
            if not required.issubset(rows.columns):
                raise RuntimeError(f"INDstocks candle response is incomplete for {symbol}.")
            index = pd.to_datetime(
                pd.to_numeric(rows["ts"], errors="coerce"),
                unit="s",
                utc=True,
            ).dt.tz_convert(IST)
            part = pd.DataFrame(
                {
                    "Open": pd.to_numeric(rows["o"], errors="coerce").to_numpy(),
                    "High": pd.to_numeric(rows["h"], errors="coerce").to_numpy(),
                    "Low": pd.to_numeric(rows["l"], errors="coerce").to_numpy(),
                    "Close": pd.to_numeric(rows["c"], errors="coerce").to_numpy(),
                    "Volume": pd.to_numeric(rows["v"], errors="coerce").to_numpy(),
                },
                index=pd.DatetimeIndex(index, name="Datetime"),
            )
            normalized_part = normalize_indstocks_5m(part)
            if not normalized_part.empty:
                accumulated[symbol].append(normalized_part)
                tagged = normalized_part.copy()
                tagged.insert(0, "Symbol", symbol)
                window_frames.append(tagged)
        if store is not None and checkpoint_root is not None:
            checkpoint_local.parent.mkdir(parents=True, exist_ok=True)
            window_data = pd.concat(window_frames).sort_index() if window_frames else pd.DataFrame(columns=["Symbol", *OHLCV], index=pd.DatetimeIndex([], tz=IST, name="Datetime"))
            window_data.to_parquet(checkpoint_local, index=True, compression="zstd")
            checkpoint_upload_kwargs = (
                {
                    "folder_id": checkpoint_folder_id,
                    "existing_id": (checkpoint_item or {}).get("id"),
                }
                if checkpoint_folder_id is not None
                else {}
            )
            uploaded = store.upload_or_update(
                checkpoint_local,
                checkpoint_remote,
                mime_type="application/vnd.apache.parquet",
                **checkpoint_upload_kwargs,
            )
            if str(uploaded.get("status") or "").upper() != "OK":
                raise RuntimeError(f"Drive checkpoint write failed for window {window_key}: {uploaded}")
            # The manifest is the commit marker: only a fully written parquet plus
            # a complete per-symbol API response may be skipped after a restart.
            checkpoint_manifest_local.write_text(json.dumps({
                "status": "COMPLETE",
                "window_key": window_key,
                "start": cursor.isoformat(),
                "end": window_end.isoformat(),
                "symbols": expected_symbols,
                "row_counts": row_counts,
                "provider_omitted_codes": missing_codes,
                "checkpoint_remote": checkpoint_remote,
                "batch_key": batch_key,
                "key_scheme": "FULL_BATCH_V2",
            }, sort_keys=True), encoding="utf-8")
            manifest_upload_kwargs = (
                {
                    "folder_id": checkpoint_folder_id,
                    "existing_id": (checkpoint_manifest_item or {}).get("id"),
                }
                if checkpoint_folder_id is not None
                else {}
            )
            uploaded_manifest = store.upload_or_update(
                checkpoint_manifest_local,
                checkpoint_manifest_remote,
                mime_type="application/json",
                **manifest_upload_kwargs,
            )
            if str(uploaded_manifest.get("status") or "").upper() != "OK":
                raise RuntimeError(f"Drive checkpoint manifest write failed for window {window_key}: {uploaded_manifest}")
            if callable(BACKFILL_PROGRESS_CALLBACK):
                BACKFILL_PROGRESS_CALLBACK({
                    "provider": "INDSTOCKS",
                    **(progress_context or {}),
                    "stage": "CHECKPOINT_PERSISTED",
                    "window_end": window_end.isoformat(),
                    "checkpoint_path": checkpoint_remote,
                    "api_calls_in_batch": calls + 1,
                })
        calls += 1
        cursor = window_end
        if callable(BACKFILL_PROGRESS_CALLBACK):
            BACKFILL_PROGRESS_CALLBACK({
                "provider": "INDSTOCKS",
                **(progress_context or {}),
                "stage": "FETCHING_HISTORY",
                "api_calls_in_batch": calls,
                "window_end": window_end.isoformat(),
            })

    result: dict[str, pd.DataFrame] = {}
    for symbol, parts in accumulated.items():
        if not parts:
            result[symbol] = pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=IST, name="Datetime"))
        else:
            merged = pd.concat(parts).sort_index()
            result[symbol] = merged[OHLCV].sort_index().loc[~merged.index.duplicated(keep='last')][OHLCV]
    return result, calls


def _load_batch_progress(store: Any) -> set[int]:
    """Return batch numbers already fully completed and persisted on Drive."""
    local = Path("/tmp/nse_5m_batch_progress.json")
    try:
        item = _resolve_remote_file(store, BATCH_PROGRESS_REMOTE)
    except Exception:
        return set()
    if not item:
        return set()
    try:
        download_drive_file_with_retry(store.service, item["id"], local, operation="batch_progress_download")
        data = json.loads(local.read_text(encoding="utf-8"))
        return {int(x) for x in data.get("completed_batches", []) if str(x).isdigit()}
    except Exception:
        return set()


def _save_batch_progress(store: Any, completed: set[int]) -> None:
    """Persist the set of completed batch numbers so restarts skip them."""
    local = Path("/tmp/nse_5m_batch_progress.json")
    local.write_text(
        json.dumps(
            {"completed_batches": sorted(completed),
             "updated_at_utc": datetime.now(timezone.utc).isoformat()},
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    uploaded = store.upload_or_update(local, BATCH_PROGRESS_REMOTE, mime_type="application/json")
    if str(uploaded.get("status") or "").upper() != "OK":
        print("BACKFILL_BATCH_PROGRESS_UPLOAD_FAILED " + str(uploaded), flush=True)


def _write_manifest(store: Any, root: Path, report: dict[str, Any]) -> None:
    path = root / "backfill_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    uploaded = store.upload_or_update(path, MANIFEST_REMOTE, mime_type="application/json")
    if str(uploaded.get("status") or "").upper() != "OK":
        raise RuntimeError(f"INDstocks backfill manifest Drive write was not confirmed: {uploaded}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=500)
    parser.add_argument("--calendar-days", type=int, default=1000)
    parser.add_argument("--initial-calendar-days", type=int, default=1000)
    parser.add_argument("--minimum-usable-sessions", type=int, default=200)
    parser.add_argument("--minimum-bars-per-session", type=int, default=50)
    parser.add_argument("--recent-session-lookback-days", type=int, default=RECENT_SESSION_LOOKBACK_DAYS)
    parser.add_argument("--pause-seconds", type=float, default=0.25)
    parser.add_argument("--minimum-symbol-coverage", type=float, default=0.90)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--replace-existing-history", action="store_true")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--work-root", default="/tmp/nse_5m_indstocks_backfill")
    args = parser.parse_args(argv)

    if not 200 <= args.sessions <= 500:
        raise ValueError("--sessions must be between 200 and 500")
    if not 200 <= args.minimum_usable_sessions <= args.sessions:
        raise ValueError("--minimum-usable-sessions must be 200 through --sessions")
    if args.minimum_bars_per_session < MIN_SESSION_BARS:
        raise ValueError(f"--minimum-bars-per-session must be at least {MIN_SESSION_BARS}")
    if args.recent_session_lookback_days < 1 or args.recent_session_lookback_days > 30:
        raise ValueError("--recent-session-lookback-days must be between 1 and 30")
    if not 0.5 <= args.minimum_symbol_coverage <= 1.0:
        raise ValueError("--minimum-symbol-coverage must be between 0.5 and 1.0")
    if args.pause_seconds < 0.20:
        raise ValueError("INDstocks Data API is limited to five requests/second; pause must be at least 0.20s.")

    raw_runtime_replace = RUNTIME_OPTIONS.get("replace_existing_history")
    runtime_replace = (
        raw_runtime_replace
        if isinstance(raw_runtime_replace, bool)
        else str(raw_runtime_replace).strip().lower() in {"1", "true", "yes", "on"}
    )
    replace_history = bool(args.replace_existing_history or runtime_replace)

    from indstocks_client import configured
    from delivery_training_universe import official_nse_symbols
    from persistent_store import GoogleDriveStore, drive_status

    if not configured():
        raise RuntimeError("INDstocks TOTP credentials are not configured in the Cloud Run worker.")
    preflight = drive_status(require_write=True)
    if not preflight.get("configured") or preflight.get("reachable") is not True or preflight.get("write_ready") is not True:
        raise RuntimeError("Google Drive read/write preflight failed: " + str(preflight.get("reason")))

    if args.symbols.strip():
        symbols = sorted({
            value.strip().upper().replace(".NS", "").replace(".BO", "")
            for value in args.symbols.split(",") if value.strip()
        })
        universe_source = "CLI_SYMBOLS"
    else:
        pinned_symbols = _pinned_universe()
        if pinned_symbols:
            symbols = pinned_symbols
            universe_source = "PINNED_BACKFILL_UNIVERSE"
        else:
            symbols = sorted({
                str(value).strip().upper().replace(".NS", "").replace(".BO", "")
                for value in official_nse_symbols(force_refresh=True) if str(value).strip()
            })
            universe_source = "OFFICIAL_NSE_EQUITY"
    if args.max_symbols > 0:
        symbols = symbols[:args.max_symbols]
    if not symbols:
        raise RuntimeError("Official NSE equity universe is empty.")

    codes = _instrument_codes()
    resolvable = [(symbol, codes[symbol]) for symbol in symbols if symbol in codes]
    resolution_errors = [
        {"symbol": symbol, "status": "ERROR", "error": "No NSE equity scrip-code in INDstocks instrument master."}
        for symbol in symbols if symbol not in codes
    ]
    if not resolvable:
        raise RuntimeError("INDstocks instrument master did not resolve any official NSE equity symbols.")

    raw_end_date = (args.end_date or str(RUNTIME_OPTIONS.get("end_date") or "")).strip()
    if raw_end_date:
        end_stamp = pd.Timestamp(raw_end_date)
        end_stamp = (
            end_stamp.tz_localize(IST)
            if end_stamp.tzinfo is None
            else end_stamp.tz_convert(IST)
        )
        # A bare calendar date means that session's 15:30 IST close, never midnight.
        if len(raw_end_date) <= 10 and end_stamp.hour == 0 and end_stamp.minute == 0:
            end_stamp = end_stamp.replace(hour=15, minute=30)
        end = end_stamp.to_pydatetime()
    else:
        now = datetime.now(IST)
        # The Cloud Run request may arrive during market hours and wait until
        # close. Pin the target to today's 15:30 bar boundary so today's final
        # session is included when the shared worker resumes after the close.
        end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    initial_start = _align_to_monday(end - timedelta(days=int(args.initial_calendar_days)))
    force_refetch_after = end - timedelta(days=int(args.recent_session_lookback_days))
    local_root = Path(args.work_root)
    local_root.mkdir(parents=True, exist_ok=True)
    store = GoogleDriveStore()
    if not replace_history:
        prior_manifest = _download_manifest_json(store, local_root)
        replace_history = str(prior_manifest.get("history_schema") or "") != HISTORY_SCHEMA
    results: list[dict[str, Any]] = list(resolution_errors)
    total_api_calls = 0
    live_inventory_totals = {"live_capture_files_read": 0, "live_capture_files_failed": 0}
    batches = [resolvable[pos:pos + MAX_CODES_PER_REQUEST] for pos in range(0, len(resolvable), MAX_CODES_PER_REQUEST)]
    completed_batches = _load_batch_progress(store)
    live_file_listing_cache: dict[str, list[dict[str, Any]]] = {}

    for batch_no, batch in enumerate(batches, start=1):
        if batch_no in completed_batches:
            print("BACKFILL_BATCH_SKIPPED batch=%d completed_symbols=%d" % (
                batch_no, min(batch_no * MAX_CODES_PER_REQUEST, len(resolvable))
            ), flush=True)
            if callable(BACKFILL_PROGRESS_CALLBACK):
                BACKFILL_PROGRESS_CALLBACK({
                    "provider": "INDSTOCKS", "batch": batch_no,
                    "total_batches": len(batches),
                    "completed_symbols": min(batch_no * MAX_CODES_PER_REQUEST, len(resolvable)),
                    "resolved_symbols": len(resolvable),
                    "api_calls": total_api_calls,
                    "skipped_batch": True,
                })
            continue
        if callable(REQUEST_GUARD):
            REQUEST_GUARD()
        # Load recent live captures only for the current <=5-symbol API batch.
        # Do not retain the full 2,600+ symbol live-history universe in RAM.
        live_history_started = time.monotonic()
        if callable(BACKFILL_PROGRESS_CALLBACK):
            BACKFILL_PROGRESS_CALLBACK({
                "provider": "INDSTOCKS", "batch": batch_no,
                "total_batches": len(batches), "stage": "LOADING_LIVE_HISTORY",
            })
        live_history, live_inventory = _load_recent_live_history(
            store,
            {symbol for symbol, _ in batch},
            end,
            local_root,
            lookback_days=args.recent_session_lookback_days,
            file_listing_cache=live_file_listing_cache,
        )
        if callable(BACKFILL_PROGRESS_CALLBACK):
            BACKFILL_PROGRESS_CALLBACK({
                "provider": "INDSTOCKS", "batch": batch_no,
                "total_batches": len(batches), "stage": "LIVE_HISTORY_READY",
                "elapsed_seconds": round(time.monotonic() - live_history_started, 1),
                **live_inventory,
            })
        for key in live_inventory_totals:
            live_inventory_totals[key] += int(live_inventory.get(key, 0))
        print(
            "BACKFILL_LIVE_HISTORY_INVENTORY "
            + json.dumps({"batch": batch_no, **live_inventory}, sort_keys=True),
            flush=True,
        )
        existing_by_symbol: dict[str, pd.DataFrame] = {}
        starts: dict[str, datetime] = {}
        active_batch: list[tuple[str, str]] = []
        for symbol, code in batch:
            remote = f"{REMOTE_ROOT}/symbol={symbol}/{symbol}.parquet"
            local = local_root / f"symbol={symbol}" / f"{symbol}.existing.parquet"
            live = live_history.get(symbol, pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=IST)))
            # The current batch is the only live-history slice retained in memory.
            live_rows = int(len(live))
            if replace_history:
                # Rebuild from provider history plus durable live captures. The
                # previous archive (including any legacy Kaggle rows) is not read,
                # so corrected date definitions are never shadowed by old bars.
                existing = live
            else:
                archive = _download_existing(store, remote, local)
                existing_parts = [
                    frame for frame in (archive, live)
                    if frame is not None and not frame.empty
                ]
                if existing_parts:
                    existing = pd.concat(existing_parts).sort_index()
                    existing = existing.loc[~existing.index.duplicated(keep="last")]
                else:
                    existing = archive
            # Kaggle seeds remain in their own Drive files and are excluded from
            # this INDstocks backfill by design.
            existing_by_symbol[symbol] = existing
            summary = usable_session_summary(existing, args.minimum_bars_per_session)
            latest = pd.to_datetime(summary.get("last_timestamp_ist"), errors="coerce")
            recent_gaps = _recent_incomplete_session_dates(
                existing,
                end,
                lookback_days=args.recent_session_lookback_days,
                expected_bars=args.minimum_bars_per_session,
            )
            is_complete = (not replace_history) and (
                int(summary.get("usable_sessions") or 0) >= args.minimum_usable_sessions
                and pd.notna(latest)
                and latest >= pd.Timestamp(end) - pd.Timedelta(days=FRESHNESS_WINDOW_DAYS)
                and not recent_gaps
            )
            if is_complete:
                results.append({
                    "symbol": symbol,
                    "status": "SKIPPED_EXISTING",
                    "source": "INDSTOCKS_HISTORICAL_5M",
                    "remote_path": remote,
                    "rows": int(len(existing)),
                    "live_history_rows": live_rows,
                    "recent_incomplete_session_dates": recent_gaps,
                    "usable_sessions": int(summary["usable_sessions"]),
                    "last_timestamp_ist": summary["last_timestamp_ist"],
                    "fresh_latest_data": True,
                    "reason": "Existing Drive Parquet meets history and freshness requirements.",
                })
                continue
            active_batch.append((symbol, code))
            if replace_history:
                starts[symbol] = initial_start
            elif (
                recent_gaps
                and int(summary.get("usable_sessions") or 0) >= args.minimum_usable_sessions
                and pd.notna(latest)
                and latest >= pd.Timestamp(end) - pd.Timedelta(days=FRESHNESS_WINDOW_DAYS)
            ):
                starts[symbol] = _align_to_monday(
                    end - timedelta(days=args.recent_session_lookback_days + MAX_DAYS_PER_REQUEST)
                )
            else:
                starts[symbol] = _align_to_monday(choose_backfill_start(
                    summary,
                    end,
                    initial_start,
                    args.calendar_days,
                    args.sessions,
                ))

        # Do not spend provider calls on symbols already complete in Drive.
        if not active_batch:
            progress = {
                "provider": "INDSTOCKS",
                "batch": batch_no,
                "total_batches": len(batches),
                "completed_symbols": min(batch_no * MAX_CODES_PER_REQUEST, len(resolvable)),
                "resolved_symbols": len(resolvable),
                "api_calls": total_api_calls,
                "skipped_existing_symbols": sum(row.get("status") == "SKIPPED_EXISTING" for row in results),
            }
            if callable(BACKFILL_PROGRESS_CALLBACK):
                BACKFILL_PROGRESS_CALLBACK(progress)
            del live_history
            continue

        # Batch ranges must be common per request, then deduplicate against Drive.
        batch_start = min(starts.values())
        # Fetch the whole five-symbol group (one request costs the same) so the
        # checkpoint folder key stays stable; only active symbols are consolidated.
        fetched_by_symbol, calls = fetch_indstocks_5m_batch(
            batch, batch_start, end, pause_seconds=args.pause_seconds,
            active_symbols={symbol for symbol, _ in active_batch},
            request_guard=REQUEST_GUARD,
            progress_context={
                "batch": batch_no,
                "total_batches": len(batches),
                "completed_symbols": (batch_no - 1) * MAX_CODES_PER_REQUEST,
                "resolved_symbols": len(resolvable),
                "api_calls": total_api_calls,
            },
            store=store,
            checkpoint_root=local_root / "window_checkpoints",
            force_refetch_after=force_refetch_after,
        )
        total_api_calls += calls

        for symbol, _ in active_batch:
            existing = existing_by_symbol[symbol]
            fetched = fetched_by_symbol.get(symbol, pd.DataFrame())
            if not fetched.empty:
                cutoff = pd.Timestamp.now(tz=IST).floor("5min")
                fetched = fetched.loc[fetched.index <= cutoff].sort_index()
            # Replacement runs let freshly fetched provider bars win so corrected
            # date definitions overwrite legacy rows. Normal runs preserve the
            # previous archive/live coverage on duplicate timestamps.
            merged = merge_symbol_history(existing, fetched, fetched_wins=replace_history)
            merged = retain_latest_sessions(merged, args.sessions, args.minimum_bars_per_session)
            summary = usable_session_summary(merged, args.minimum_bars_per_session)
            latest = pd.to_datetime(summary.get("last_timestamp_ist"), errors="coerce")
            fresh = pd.notna(latest) and latest >= pd.Timestamp(end) - pd.Timedelta(days=FRESHNESS_WINDOW_DAYS)
            if merged.empty:
                results.append({"symbol": symbol, "status": "NO_DATA", "usable_sessions": 0, "source": "INDSTOCKS_HISTORICAL_5M"})
                continue

            local = local_root / f"symbol={symbol}" / f"{symbol}.parquet"
            local.parent.mkdir(parents=True, exist_ok=True)
            remote = f"{REMOTE_ROOT}/symbol={symbol}/{symbol}.parquet"
            merged.to_parquet(local, index=True, compression="zstd")
            uploaded = store.upload_or_update(local, remote, mime_type="application/vnd.apache.parquet")
            if str(uploaded.get("status") or "").upper() != "OK":
                results.append({"symbol": symbol, "status": "ERROR", "error": f"Drive upload not confirmed: {uploaded}"})
                continue
            if callable(BACKFILL_PROGRESS_CALLBACK):
                BACKFILL_PROGRESS_CALLBACK({
                    "provider": "INDSTOCKS",
                    "stage": "SYMBOL_PARQUET_PERSISTED",
                    "symbol": symbol,
                    "symbol_parquet_path": remote,
                    "drive_bytes": uploaded.get("size"),
                })
            ready = summary["usable_sessions"] >= args.minimum_usable_sessions
            remaining_recent_gaps = _recent_incomplete_session_dates(
                merged,
                end,
                lookback_days=args.recent_session_lookback_days,
                expected_bars=FULL_REGULAR_SESSION_BARS,
            )
            recent_coverage_ready = not remaining_recent_gaps
            final_status = (
                "OK" if ready and fresh and recent_coverage_ready
                else "RECENT_SESSION_GAPS" if ready and fresh and not recent_coverage_ready
                else "STALE_LATEST_DATA" if ready
                else "INSUFFICIENT_HISTORY"
            )
            results.append({
                "symbol": symbol,
                "status": final_status,
                "source": "INDSTOCKS_HISTORICAL_5M",
                "remote_path": remote,
                "rows": int(len(merged)),
                "fetched_rows": int(len(fetched)),
                "live_history_rows": live_rows,
                "recent_incomplete_session_dates": recent_gaps,
                "remaining_recent_incomplete_session_dates": remaining_recent_gaps,
                "recent_coverage_ready": bool(recent_coverage_ready),
                "usable_sessions": int(summary["usable_sessions"]),
                "raw_sessions": int(summary["raw_sessions"]),
                "fresh_latest_data": bool(fresh),
                "first_timestamp_ist": summary["first_timestamp_ist"],
                "last_timestamp_ist": summary["last_timestamp_ist"],
                "drive_action": uploaded.get("action"),
                "drive_bytes": uploaded.get("size"),
            })

        # Drop batch-scoped frames before loading the next live-history slice.
        del fetched_by_symbol, live_history, existing_by_symbol
        progress = {
            "provider": "INDSTOCKS",
            "batch": batch_no,
            "total_batches": len(batches),
            "completed_symbols": min(batch_no * MAX_CODES_PER_REQUEST, len(resolvable)),
            "resolved_symbols": len(resolvable),
            "api_calls": total_api_calls,
        }
        try:
            print("BACKFILL_PROGRESS " + json.dumps(progress, sort_keys=True), flush=True)
        except BrokenPipeError:
            pass
        if callable(BACKFILL_PROGRESS_CALLBACK):
            BACKFILL_PROGRESS_CALLBACK(progress)

        batch_symbols = {symbol for symbol, _ in batch}
        batch_rows = [row for row in results if row.get("symbol") in batch_symbols]
        if batch_rows and all(row.get("status") != "ERROR" for row in batch_rows):
            completed_batches.add(batch_no)
            _save_batch_progress(store, completed_batches)

    results.sort(key=lambda row: row["symbol"])
    ready = [row for row in results if row.get("status") in {"OK", "SKIPPED_EXISTING"} and row.get("fresh_latest_data") is True]
    with_200 = [row for row in results if int(row.get("usable_sessions") or 0) >= args.minimum_usable_sessions]
    required = max(1, int(math.ceil(len(symbols) * args.minimum_symbol_coverage)))
    report = {
        "audit": "nse_native_five_minute_historical_backfill",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "INDSTOCKS_HISTORICAL_5M_BATCH_API",
        "history_schema": HISTORY_SCHEMA,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "replace_existing_history": replace_history,
        "universe_source": universe_source,
        "kaggle_excluded_from_backfill": True,
        "bar_interval_minutes": 5,
        "timestamp_convention": "provider_candle_start_plus_5_minutes_as_bar_close",
        "maximum_instruments_per_request": MAX_CODES_PER_REQUEST,
        "maximum_calendar_days_per_request": MAX_DAYS_PER_REQUEST,
        "documented_data_api_rate_limit_per_second": 5,
        "request_pause_seconds": args.pause_seconds,
        "api_calls": total_api_calls,
        "target_sessions_per_symbol": args.sessions,
        "minimum_usable_sessions": args.minimum_usable_sessions,
        "minimum_bars_per_usable_session": args.minimum_bars_per_session,
        "recent_session_full_bar_requirement": FULL_REGULAR_SESSION_BARS,
        "recent_session_lookback_days": args.recent_session_lookback_days,
        "recent_full_session_bar_requirement": FULL_REGULAR_SESSION_BARS,
        "live_capture_files_read": live_inventory_totals["live_capture_files_read"],
        "live_capture_files_failed": live_inventory_totals["live_capture_files_failed"],
        "calendar_days_requested": args.calendar_days,
        "initial_calendar_days": args.initial_calendar_days,
        "range_start_ist": initial_start.isoformat(),
        "range_end_ist": end.isoformat(),
        "official_symbols_requested": len(symbols),
        "symbols_resolved": len(resolvable),
        "symbols_with_200_plus_sessions": len(with_200),
        "symbols_with_fresh_latest_data": sum(row.get("fresh_latest_data") is True for row in results),
        "symbols_with_200_plus_sessions_and_fresh_data": len(ready),
        "symbols_with_500_sessions": sum(
            int(row.get("usable_sessions") or 0) >= args.sessions for row in results
        ),
        "symbols_with_data_after_2026_09_24": sum(
            bool(row.get("last_timestamp_ist")) and row["last_timestamp_ist"][:10] > "2026-09-24"
            for row in results
        ),
        "latest_archive_timestamp_ist": max(
            (row.get("last_timestamp_ist") or "" for row in results), default=None
        ),
        "symbols_with_errors": sum(row.get("status") == "ERROR" for row in results),
        "symbols_with_recent_session_gaps": sum(bool(row.get("remaining_recent_incomplete_session_dates")) for row in results),
        "symbols_below_200_sessions": sum(row.get("status") == "INSUFFICIENT_HISTORY" for row in results),
        "symbols_with_data_but_no_200_sessions": sum(
            row.get("status") in {"INSUFFICIENT_HISTORY", "STALE_LATEST_DATA"} for row in results
        ),
        "overall_status": "PASS" if len(ready) >= required else "NEEDS_ATTENTION",
        "required_symbols_with_200_plus_sessions_and_fresh_data": required,
        "symbols": results,
        "manifest_drive_path": MANIFEST_REMOTE,
        "no_auto_orders": True,
        "production_model_changed": False,
    }
    _write_manifest(store, local_root, report)
    try:
        print("BACKFILL_MANIFEST=" + json.dumps({
            key: report.get(key)
            for key in (
                "overall_status", "official_symbols_requested", "symbols_resolved",
                "symbols_with_200_plus_sessions_and_fresh_data",
                "required_symbols_with_200_plus_sessions_and_fresh_data",
                "symbols_with_data_after_2026_09_24", "latest_archive_timestamp_ist",
                "api_calls", "manifest_drive_path",
            )
        }, sort_keys=True))
    except BrokenPipeError:
        pass
    return 0 if report["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
