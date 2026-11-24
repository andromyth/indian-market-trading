#!/usr/bin/env python3
"""Compile the latest persisted NSE quote snapshot into predictions from Drive data.

This GitHub Actions job is a Drive consumer only. It never imports or calls
indstocks_client; INDstocks access remains inside the singleton Cloud Run worker.
"""
from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
FOLDER_MIME = "application/vnd.google-apps.folder"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
QUOTE_ROOT = "production_data/all_universe_minute_snapshots"
HISTORY_ROOT = "production_data/nse_official/intraday_seed/olo_5m_archive/symbols"
MARKET_CONTEXT_SYMBOL = "__NIFTY50__"
LIVE_HISTORY_ROOT = "production_data/nse_official/intraday_5m_live"
MAX_CAPTURE_AGE_SECONDS = 12 * 60
MAX_MINUTE_MANIFEST_CANDIDATES = 100
RECOVER_SKIP_MINUTES_ENV = "ALL_NSE_RECOVER_SKIP_MINUTES"
# Each selected symbol costs one list + one download Drive call; fetch history
# concurrently so the fixed 500-symbol roster stays inside the compiler time cap.
HISTORY_DOWNLOAD_WORKERS = max(1, int(os.getenv("ALL_NSE_COMPILER_HISTORY_WORKERS", "12") or 12))


def _selected_roster() -> set[str]:
    """Load the fixed owner-selected roster; never infer symbols from NSE."""
    from prediction_universe import authoritative_roster

    symbols = set(authoritative_roster())
    if len(symbols) != 500:
        raise RuntimeError(f"backfill_universe.json must contain exactly 500 unique symbols; found {len(symbols)}")
    return symbols


def _restrict_to_selected_roster(quotes: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Exclude any snapshot rows outside the pinned roster before inference."""
    allowed = _selected_roster()
    frame = quotes.copy()
    frame["symbol"] = (
        frame["symbol"].astype(str).str.strip().str.upper()
        .str.replace(r"\.NS$", "", regex=True)
    )
    excluded = sorted(set(frame["symbol"]) - allowed)
    return frame.loc[frame["symbol"].isin(allowed)].copy(), excluded


def _find_remote(store, remote: str):
    parent = store.root_id
    parts = [part for part in remote.strip("/").split("/") if part]
    for index, part in enumerate(parts):
        mime = FOLDER_MIME if index < len(parts) - 1 else None
        found = store._find_child(parent, part, mime)
        if not found:
            return None
        if index < len(parts) - 1:
            parent = found["id"]
    return found


def _list_children(store, parent_id: str) -> list[dict]:
    # Accepts either a store object (with a .service attribute) or a raw
    # googleapiclient Resource, so parallel download threads can hand in their
    # own per-thread service.
    client = getattr(store, "service", store)
    rows: list[dict] = []
    page = None
    while True:
        from persistent_store import _drive_execute_with_retry

        result = _drive_execute_with_retry(
            "list_children",
            lambda page=page: client.files().list(
                q=f"'{parent_id}' in parents and trashed = false",
                spaces="drive",
                pageSize=1000,
                pageToken=page,
                fields="nextPageToken,files(id,name,mimeType,size,modifiedTime)",
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            ).execute(),
        )
        rows.extend(result.get("files", []))
        page = result.get("nextPageToken")
        if not page:
            return rows


def _download(store, file_id: str, destination: Path) -> None:
    from persistent_store import download_drive_file_with_retry

    download_drive_file_with_retry(
        getattr(store, "service", store),
        file_id,
        destination,
        operation="github_drive_prediction_input",
    )


def _manifest_has_predictions(manifest: dict[str, Any]) -> bool:
    """A STORED slot with zero forecasts is a failed compile, not a completed slot."""
    if str(manifest.get("status") or "").upper() != "STORED":
        return False
    try:
        return int(manifest.get("prediction_count") or 0) > 0
    except (TypeError, ValueError):
        return False


BENIGN_EMPTY_STATES = {
    "WARMING_HISTORY", "MISSING_CURRENT_QUOTE", "NO_ELIGIBLE_SYMBOLS",
}
# WARMING_HISTORY is only expected during the opening warmup. After this slot,
# zero predictions mean the live history refresh failed and the slot must be
# recompiled from the persisted archive instead of being treated as complete.
PREDICTION_WARMUP_END = (9, 35)


def _is_prediction_warmup(minute: pd.Timestamp) -> bool:
    stamp = pd.Timestamp(minute)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(IST)
    return (int(stamp.hour), int(stamp.minute)) < PREDICTION_WARMUP_END


def _recover_skip_minutes() -> set[str]:
    """Slots the recovery workflow already found unrecoverable in this run."""
    raw = os.getenv(RECOVER_SKIP_MINUTES_ENV, "")
    return {
        part.strip()
        for part in str(raw or "").split(",")
        if part.strip()
    }


def _recover_zero_prediction_status(
    result: dict[str, Any], minute: pd.Timestamp
) -> str | None:
    """Return a non-failing recovery status when a slot cannot be rebuilt."""
    states = {
        str(key).upper()
        for key in (result.get("prediction_status_counts") or {})
        if str(key).strip()
    }
    if (
        states
        and states <= BENIGN_EMPTY_STATES
        and "WARMING_HISTORY" in states
        and not _is_prediction_warmup(minute)
    ):
        return "UNRECOVERABLE_WARMING_HISTORY"
    return None


def _has_benign_empty_prediction(
    manifest: dict[str, Any], minute: pd.Timestamp
) -> bool:
    """Return True only for empty outcomes that another compile cannot repair.

    A mid-session WARMING_HISTORY slot is a transient live-history failure: the
    post-close compiler can rebuild it from persisted OHLCV. Treating it as
    benign (the previous behavior) permanently dropped every affected slot.
    """
    states = {
        str(key).upper()
        for key in (manifest.get("prediction_status_counts") or {})
        if str(key).strip()
    }
    if not states or not states <= BENIGN_EMPTY_STATES:
        return False
    try:
        if int(manifest.get("model_failure_count") or 0) != 0:
            return False
    except (TypeError, ValueError):
        return False
    if _is_prediction_warmup(minute):
        return True
    return states <= {"MISSING_CURRENT_QUOTE", "NO_ELIGIBLE_SYMBOLS"}


def _prediction_already_stored(store, minute: pd.Timestamp) -> bool:
    remote = (
        "production_data/all_nse_minute_predictions/"
        f"{minute:%Y/%m/%d}/manifest_{minute:%Y%m%d_%H%M}.json"
    )
    item = _find_remote(store, remote)
    if not item:
        return False
    local = Path("/tmp/drive_all_nse") / f"existing_prediction_{minute:%Y%m%d_%H%M}.json"
    _download(store, item["id"], local)
    try:
        manifest = json.loads(local.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if _manifest_has_predictions(manifest):
        return True
    return _has_benign_empty_prediction(manifest, minute)


def _read_history(path: Path, symbol: str | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=IST, name="Datetime"))
    if symbol is not None and "symbol" in frame.columns:
        frame = frame.loc[
            frame["symbol"].astype(str).str.strip().str.upper().eq(symbol)
        ].copy()
    if "Timestamp" in frame.columns:
        index = pd.to_datetime(frame.pop("Timestamp"), errors="coerce")
    elif "Datetime" in frame.columns:
        index = pd.to_datetime(frame.pop("Datetime"), errors="coerce")
    else:
        index = pd.to_datetime(frame.index, errors="coerce")
    index = pd.DatetimeIndex(index)
    index = index.tz_localize(IST) if index.tz is None else index.tz_convert(IST)
    frame.index = index
    for column in OHLCV:
        if column not in frame:
            frame[column] = float("nan")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[~frame.index.isna(), OHLCV].sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    return frame


def _normalize_capture_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Return long-format (symbol, Timestamp, OHLCV) rows from one capture file."""
    if raw.empty or "symbol" not in raw.columns:
        return pd.DataFrame(columns=["symbol", "Timestamp", *OHLCV])
    frame = raw.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    if "Timestamp" in frame.columns:
        index = pd.to_datetime(frame.pop("Timestamp"), errors="coerce")
    elif "Datetime" in frame.columns:
        index = pd.to_datetime(frame.pop("Datetime"), errors="coerce")
    else:
        index = pd.to_datetime(frame.index, errors="coerce")
    index = pd.DatetimeIndex(index)
    index = index.tz_localize(IST) if index.tz is None else index.tz_convert(IST)
    valid = ~index.isna()
    frame = frame.loc[valid].copy()
    frame.index = index[valid]
    for column in OHLCV:
        if column not in frame:
            frame[column] = float("nan")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.index.name = "Timestamp"
    return frame.reset_index()[["symbol", "Timestamp", *OHLCV]]


def _load_live_history_state(
    store,
    quote_symbols: set[str],
    minute: pd.Timestamp,
    work: Path,
    lookback: int,
    history: dict[str, pd.DataFrame],
) -> None:
    """Ingest only captures not yet merged into the day's consolidated live state.

    Every compile used to re-download all of today's capture files and split each
    one into per-symbol part files (captures x symbols files per run, growing all
    session). The consolidated state keeps one parquet per trading day plus an
    ingest ledger, so a run pays only for the captures written since the previous
    run (normally one).
    """
    live_folder = _find_remote(store, f"{LIVE_HISTORY_ROOT}/{minute:%Y/%m/%d}")
    if not live_folder:
        return
    state_dir = work / "live_state" / f"{minute:%Y-%m-%d}"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.parquet"
    ingested_path = state_dir / "ingested.json"
    try:
        ingested = json.loads(ingested_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ingested = {}

    captures = [
        item
        for item in _list_children(store, live_folder["id"])
        if str(item.get("name") or "").startswith("capture_")
        and str(item.get("name") or "").endswith(".parquet")
    ]
    pending: list[dict] = []
    for item in captures:
        name = str(item.get("name") or "")
        meta = {
            "modifiedTime": item.get("modifiedTime"),
            "size": str(item.get("size") or ""),
        }
        if ingested.get(name) == meta:
            continue
        pending.append(item)

    started = time.monotonic()
    state: pd.DataFrame | None = None
    if state_path.is_file():
        try:
            state = pd.read_parquet(state_path)
        except Exception:
            state = None

    if pending:
        frames: list[pd.DataFrame] = []
        for item in pending:
            local = work / "live" / str(item.get("name") or "")
            _download(store, item["id"], local)
            frames.append(_normalize_capture_frame(pd.read_parquet(local)))
        new_rows = pd.concat(frames, ignore_index=True) if frames else None
        if new_rows is not None and not new_rows.empty:
            if state is None or state.empty:
                state = new_rows
            else:
                state = pd.concat([state, new_rows], ignore_index=True)
            state = state.drop_duplicates(subset=["symbol", "Timestamp"], keep="last")
            state = state.sort_values(["symbol", "Timestamp"])
            state.to_parquet(state_path, index=False)
        for item in pending:
            name = str(item.get("name") or "")
            ingested[name] = {
                "modifiedTime": item.get("modifiedTime"),
                "size": str(item.get("size") or ""),
            }
        ingested_path.write_text(json.dumps(ingested, sort_keys=True), encoding="utf-8")

    print(
        json.dumps({
            "event": "LIVE_HISTORY_STATE_LOADED",
            "captures_total": len(captures),
            "captures_new": len(pending),
            "state_rows": 0 if state is None else int(len(state)),
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }, sort_keys=True),
        flush=True,
    )

    if state is None or state.empty:
        return
    wanted = set(quote_symbols) | {MARKET_CONTEXT_SYMBOL}
    state = state.loc[state["symbol"].isin(wanted)]
    state = state.loc[pd.to_datetime(state["Timestamp"]) <= minute]
    if state.empty:
        return
    for symbol, group in state.groupby("symbol", sort=False):
        frame = group.set_index("Timestamp")[OHLCV].sort_index()
        frame.index = pd.DatetimeIndex(frame.index)
        frame = frame.loc[~frame.index.duplicated(keep="last")].tail(lookback)
        existing = history.get(symbol)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, frame])
            combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
            history[symbol] = combined.tail(lookback)
        else:
            history[symbol] = frame


def _latest_capture(store, now: pd.Timestamp | None = None) -> tuple[dict, pd.DataFrame, pd.Timestamp] | None:
    """Select the newest fresh STORED snapshot, even when latest_manifest is a failure pointer.

    A failed capture updates latest_manifest.json. That status pointer must not
    hide a valid same-session snapshot whose own per-minute manifest was stored.

    Returns ``None`` when every fresh slot already has a STORED prediction, which
    is the normal idle state now that the Cloud Run tick predicts in-worker.
    """
    current = pd.Timestamp.now(tz=IST) if now is None else pd.Timestamp(now)
    current = current.tz_localize(IST) if current.tzinfo is None else current.tz_convert(IST)
    day_folder = _find_remote(store, f"{QUOTE_ROOT}/{current:%Y/%m/%d}")
    latest_item = _find_remote(store, f"{QUOTE_ROOT}/latest_manifest.json")

    items: list[dict] = []
    if day_folder:
        minute_items = [
            item for item in _list_children(store, day_folder["id"])
            if re.fullmatch(r"manifest_\d{8}_\d{4}\.json", str(item.get("name") or ""))
        ]
        minute_items.sort(key=lambda item: str(item.get("name") or ""), reverse=True)
        items.extend(minute_items[:MAX_MINUTE_MANIFEST_CANDIDATES])
    if latest_item and all(item.get("id") != latest_item.get("id") for item in items):
        items.append(latest_item)
    if not items:
        raise RuntimeError("No current-session quote manifests are available in Drive")

    candidates: list[tuple[pd.Timestamp, dict, dict]] = []
    skipped: list[str] = []
    for index, item in enumerate(items):
        local = Path("/tmp/drive_all_nse") / f"capture_manifest_candidate_{index}.json"
        _download(store, item["id"], local)
        try:
            manifest = json.loads(local.read_text(encoding="utf-8"))
            minute = pd.Timestamp(manifest.get("cycle_minute_ist"))
            minute = minute.tz_localize(IST) if minute.tzinfo is None else minute.tz_convert(IST)
            candidates.append((minute, manifest, item))
        except Exception as exc:
            skipped.append(f"{item.get('name')}: invalid manifest ({type(exc).__name__})")

    # Live predictions take priority: always compile the newest fresh eligible
    # current-session slot. Historical gaps must never make the dashboard stale.
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    seen_minutes: set[pd.Timestamp] = set()
    # The Cloud Run tick owns same-slot prediction now, so most compiler runs
    # find every fresh slot already predicted. That is the healthy idle state,
    # not an error: callers get ``None`` and exit 0. A run that finds *no*
    # fresh snapshot at all still raises, because that means the worker stalled.
    fresh_slots = 0
    pending_fresh_slots = 0
    for minute, manifest, item in candidates:
        if minute in seen_minutes:
            continue
        seen_minutes.add(minute)
        status = str(manifest.get("status") or "").upper()
        age_seconds = (current - minute).total_seconds()
        if manifest.get("snapshot_persisted") is not True or status != "STORED":
            skipped.append(f"{item.get('name')}: status={status or 'MISSING'}")
            continue
        # Only the explicit assembly step may publish the canonical sharded
        # snapshot. Never compile a shard artifact or an incomplete assembly.
        worker_count = int(manifest.get("worker_count") or 1)
        if str(manifest.get("artifact_kind") or "").upper() == "SHARD":
            skipped.append(f"{item.get('name')}: shard artifact, not assembled canonical snapshot")
            continue
        if worker_count > 1:
            if str(manifest.get("assembly_status") or "").upper() != "COMPLETE":
                skipped.append(f"{item.get('name')}: sharded snapshot was not assembled")
                continue
            if int(manifest.get("shard_count_completed") or 0) < int(
                manifest.get("shard_count_expected") or worker_count
            ):
                skipped.append(f"{item.get('name')}: shard_count_incomplete")
                continue
        if minute.date() != current.date() or age_seconds < 0 or age_seconds > MAX_CAPTURE_AGE_SECONDS:
            skipped.append(f"{item.get('name')}: stale_or_future age_seconds={round(age_seconds, 1)}")
            continue
        fresh_slots += 1
        if _prediction_already_stored(store, minute):
            skipped.append(f"{item.get('name')}: prediction already STORED")
            continue
        pending_fresh_slots += 1

        snapshot_remote = str(
            manifest.get("snapshot_drive_path") or manifest.get("remote_snapshot_path") or ""
        )
        if not snapshot_remote:
            skipped.append(f"{item.get('name')}: missing snapshot path")
            continue
        snapshot_item = _find_remote(store, snapshot_remote)
        if not snapshot_item:
            skipped.append(f"{item.get('name')}: snapshot file missing")
            continue
        snapshot_path = Path("/tmp/drive_all_nse") / Path(snapshot_remote).name
        _download(store, snapshot_item["id"], snapshot_path)
        try:
            quotes = pd.read_parquet(snapshot_path)
        except Exception as exc:
            skipped.append(f"{item.get('name')}: unreadable snapshot ({type(exc).__name__})")
            continue
        if quotes.empty or "symbol" not in quotes.columns:
            skipped.append(f"{item.get('name')}: empty snapshot or missing symbol column")
            continue
        declared_rows = manifest.get("row_count")
        if declared_rows is not None and int(declared_rows) != len(quotes):
            skipped.append(f"{item.get('name')}: row_count_mismatch")
            continue

        # STORED means the parquet was persisted, not that its quotes are usable.
        # Prefer the newest same-session snapshot with at least one quote that
        # passes the exact per-symbol eligibility contract. Never turn a failed
        # provider quote into a prediction.
        from all_nse_minute_prediction_capture import _snapshot_quote_diagnostics

        quote_diagnostics = _snapshot_quote_diagnostics(
            quotes.to_dict(orient="records"), minute
        )
        eligible_count = int(
            quote_diagnostics.get("rejection_reason_counts", {}).get("ELIGIBLE", 0)
        )
        if eligible_count < 1:
            skipped.append(
                f"{item.get('name')}: no eligible quotes "
                + json.dumps(quote_diagnostics, sort_keys=True)
            )
            continue

        print(
            "DRIVE_CAPTURE_SELECTED="
            + json.dumps({
                "manifest": item.get("name"),
                "cycle_minute_ist": minute.isoformat(),
                "age_seconds": round(age_seconds, 1),
                "status": status,
                "snapshot_drive_path": snapshot_remote,
                "quote_rows": len(quotes),
                "eligible_quote_rows": eligible_count,
                "quote_diagnostics": quote_diagnostics,
                "skipped_newer_candidates": skipped,
            }, sort_keys=True),
            flush=True,
        )
        return manifest, quotes, minute

    if fresh_slots and not pending_fresh_slots:
        print(
            "DRIVE_CAPTURE_IDLE="
            + json.dumps({
                "reason": "ALL_FRESH_SLOTS_ALREADY_PREDICTED",
                "fresh_slot_count": fresh_slots,
                "skipped": skipped[-8:],
            }, sort_keys=True),
            flush=True,
        )
        return None

    raise RuntimeError(
        "No fresh, durably stored unprocessed current-session quote snapshot was found; "
        + "; ".join(skipped[-8:])
    )


def _recovery_capture(store, target: pd.Timestamp):
    """Oldest snapshot for ``target`` date whose prediction is not yet STORED.

    Live runs compile only the newest fresh slot, so a missed slot is never
    revisited. Recovery scans the day oldest-first and returns the first
    unprocessed, fully assembled snapshot.
    """
    day_folder = _find_remote(store, f"{QUOTE_ROOT}/{target:%Y/%m/%d}")
    if not day_folder:
        raise RuntimeError(f"No quote snapshot folder for {target:%Y-%m-%d}")
    items = [
        item
        for item in _list_children(store, day_folder["id"])
        if re.fullmatch(r"manifest_\d{8}_\d{4}\.json", str(item.get("name") or ""))
    ]
    items.sort(key=lambda item: str(item.get("name") or ""))
    skip_keys = _recover_skip_minutes()
    skipped: list[str] = []
    for index, item in enumerate(items):
        local = Path("/tmp/drive_all_nse") / f"recovery_manifest_{index}.json"
        _download(store, item["id"], local)
        try:
            manifest = json.loads(local.read_text(encoding="utf-8"))
            minute = pd.Timestamp(manifest.get("cycle_minute_ist"))
            minute = minute.tz_localize(IST) if minute.tzinfo is None else minute.tz_convert(IST)
        except Exception as exc:
            skipped.append(f"{item.get('name')}: invalid manifest ({type(exc).__name__})")
            continue
        if minute.strftime("%Y%m%d_%H%M") in skip_keys:
            skipped.append(f"{item.get('name')}: unrecoverable in this run")
            continue
        if minute.strftime("%Y-%m-%d") != target.strftime("%Y-%m-%d"):
            skipped.append(f"{item.get('name')}: cycle date mismatch")
            continue
        status = str(manifest.get("status") or "").upper()
        if status != "STORED" or manifest.get("snapshot_persisted") is not True:
            skipped.append(f"{item.get('name')}: status={status or 'MISSING'}")
            continue
        if str(manifest.get("artifact_kind") or "").upper() == "SHARD":
            skipped.append(f"{item.get('name')}: shard artifact")
            continue
        worker_count = int(manifest.get("worker_count") or 1)
        if worker_count > 1:
            if str(manifest.get("assembly_status") or "").upper() != "COMPLETE":
                skipped.append(f"{item.get('name')}: not assembled")
                continue
            if int(manifest.get("shard_count_completed") or 0) < int(
                manifest.get("shard_count_expected") or worker_count
            ):
                skipped.append(f"{item.get('name')}: shard_count_incomplete")
                continue
        if _prediction_already_stored(store, minute):
            continue
        snapshot_remote = str(
            manifest.get("snapshot_drive_path") or manifest.get("remote_snapshot_path") or ""
        )
        if not snapshot_remote:
            skipped.append(f"{item.get('name')}: missing snapshot path")
            continue
        snapshot_item = _find_remote(store, snapshot_remote)
        if not snapshot_item:
            skipped.append(f"{item.get('name')}: snapshot file missing")
            continue
        snapshot_path = Path("/tmp/drive_all_nse") / Path(snapshot_remote).name
        _download(store, snapshot_item["id"], snapshot_path)
        try:
            quotes = pd.read_parquet(snapshot_path)
        except Exception as exc:
            skipped.append(f"{item.get('name')}: unreadable snapshot ({type(exc).__name__})")
            continue
        if quotes.empty or "symbol" not in quotes.columns:
            skipped.append(f"{item.get('name')}: empty snapshot or missing symbol column")
            continue
        declared_rows = manifest.get("row_count")
        if declared_rows is not None and int(declared_rows) != len(quotes):
            skipped.append(f"{item.get('name')}: row_count_mismatch")
            continue
        from all_nse_minute_prediction_capture import _snapshot_quote_diagnostics

        quote_diagnostics = _snapshot_quote_diagnostics(
            quotes.to_dict(orient="records"), minute
        )
        eligible_count = int(
            quote_diagnostics.get("rejection_reason_counts", {}).get("ELIGIBLE", 0)
        )
        if eligible_count < 1:
            skipped.append(f"{item.get('name')}: no eligible quotes")
            continue
        print(
            "RECOVERY_CAPTURE_SELECTED="
            + json.dumps({
                "manifest": item.get("name"),
                "cycle_minute_ist": minute.isoformat(),
                "quote_rows": len(quotes),
                "eligible_quote_rows": eligible_count,
                "skipped_tail": skipped[-5:],
            }, sort_keys=True),
            flush=True,
        )
        return manifest, quotes, minute
    print(
        json.dumps({
            "event": "RECOVERY_SCAN_COMPLETE",
            "date": target.strftime("%Y-%m-%d"),
            "manifests": len(items),
            "skipped_tail": skipped[-5:],
        }, sort_keys=True),
        flush=True,
    )
    return None


def _thread_service(store):
    """Build one Drive service for a download worker thread.

    googleapiclient Resource objects are not thread-safe, so each history
    download thread uses its own service built from the store's shared
    (thread-safe) credentials instead of sharing ``store.service``.
    """
    from googleapiclient.discovery import build

    return build("drive", "v3", credentials=store.credentials, cache_discovery=False)


def _load_symbol_history(
    service,
    symbol: str,
    folder_id: str,
    cache_dir: Path,
    cached: dict,
    minute: pd.Timestamp,
    lookback: int,
) -> tuple[str, pd.DataFrame | None, dict | None]:
    files = _list_children(service, folder_id)
    item = next((entry for entry in files if entry.get("name") == f"{symbol}.parquet"), None)
    if not item:
        return symbol, None, None
    local = cache_dir / f"{symbol}.parquet"
    if (
        not local.is_file()
        or cached.get("modifiedTime") != item.get("modifiedTime")
        or str(cached.get("size") or "") != str(item.get("size") or "")
    ):
        _download(service, item["id"], local)
    try:
        frame = _read_history(local)
    except Exception:
        _download(service, item["id"], local)
        frame = _read_history(local)
    frame = frame.loc[frame.index <= minute].tail(lookback)
    meta = {"modifiedTime": item.get("modifiedTime"), "size": item.get("size")}
    return symbol, (frame if not frame.empty else None), meta


def _download_symbol_histories(
    store,
    tasks: list[tuple[str, str]],
    cache_dir: Path,
    minute: pd.Timestamp,
    lookback: int,
    history: dict[str, pd.DataFrame],
    cache_index: dict,
) -> None:
    """Fetch per-symbol seed parquets in parallel and merge results in-thread.

    Workers share nothing mutable: each returns ``(symbol, frame, meta)`` and
    only the calling thread touches ``history``/``cache_index``. A worker
    failure propagates through ``future.result()`` and aborts the compile,
    matching the previous fail-loud behaviour.
    """
    started = time.monotonic()
    if len(tasks) <= 1 or HISTORY_DOWNLOAD_WORKERS <= 1:
        service = getattr(store, "service", store)
        results = [
            _load_symbol_history(
                service, symbol, folder_id, cache_dir,
                cache_index.get(symbol) or {}, minute, lookback,
            )
            for symbol, folder_id in tasks
        ]
    else:
        workers = min(HISTORY_DOWNLOAD_WORKERS, len(tasks))
        # Services are built up front in the main thread so a build failure
        # raises before any pool work starts.
        services = [_thread_service(store) for _ in range(workers)]
        handoff: "queue.SimpleQueue" = queue.SimpleQueue()
        for service in services:
            handoff.put(service)
        thread_local = threading.local()

        def _init_worker() -> None:
            thread_local.service = handoff.get()

        def _task(symbol: str, folder_id: str):
            return _load_symbol_history(
                thread_local.service, symbol, folder_id, cache_dir,
                cache_index.get(symbol) or {}, minute, lookback,
            )

        with ThreadPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            futures = [pool.submit(_task, symbol, folder_id) for symbol, folder_id in tasks]
            results = [future.result() for future in futures]

    downloaded = 0
    for symbol, frame, meta in results:
        if frame is not None:
            history[symbol] = frame
        if meta:
            if (cache_index.get(symbol) or {}).get("modifiedTime") != meta["modifiedTime"]:
                downloaded += 1
            cache_index[symbol] = meta
    print(
        json.dumps({
            "event": "HISTORY_SEED_LOADED",
            "symbols": len(tasks),
            "downloaded": downloaded,
            "cached": len(tasks) - downloaded,
            "workers": 1 if len(tasks) <= 1 or HISTORY_DOWNLOAD_WORKERS <= 1 else min(HISTORY_DOWNLOAD_WORKERS, len(tasks)),
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }),
        flush=True,
    )


def _load_drive_history(store, quote_symbols: set[str], minute: pd.Timestamp, work: Path, lookback: int) -> dict[str, pd.DataFrame]:
    history: dict[str, pd.DataFrame] = {}
    cache_dir = work / "history_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"
    try:
        cache_index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache_index = {}

    archive_folder = _find_remote(store, HISTORY_ROOT)
    if archive_folder:
        wanted = set(quote_symbols) | {MARKET_CONTEXT_SYMBOL}
        tasks: list[tuple[str, str]] = []
        for folder in _list_children(store, archive_folder["id"]):
            name = str(folder.get("name") or "")
            if folder.get("mimeType") != FOLDER_MIME or not name.startswith("symbol="):
                continue
            symbol = name.split("=", 1)[1].strip().upper()
            if symbol not in wanted:
                continue
            tasks.append((symbol, folder["id"]))
        if tasks:
            _download_symbol_histories(
                store, tasks, cache_dir, minute, lookback, history, cache_index
            )
    index_path.write_text(json.dumps(cache_index, sort_keys=True), encoding="utf-8")

    _load_live_history_state(store, quote_symbols, minute, work, lookback, history)
    return history


def main() -> int:
    from market_session_guard import is_nse_trading_day
    from persistent_store import GoogleDriveStore
    from all_nse_minute_prediction_capture import (
        AllNSEMinutePredictor,
        HISTORY_LOOKBACK_BARS,
    )

    mode = (os.getenv("ALL_NSE_COMPILER_MODE") or "live").strip().lower()
    now = pd.Timestamp.now(tz=IST)
    target: pd.Timestamp | None = None
    if mode == "recover":
        raw_target = (os.getenv("ALL_NSE_RECOVER_DATE") or "").strip()
        target = pd.Timestamp(raw_target or now.strftime("%Y-%m-%d"))
        target = target.tz_localize(IST) if target.tzinfo is None else target.tz_convert(IST)
    else:
        if not is_nse_trading_day(now.date()):
            print(json.dumps({"status": "SKIPPED", "reason": "NSE_NON_TRADING_DAY", "now_ist": now.isoformat()}))
            return 0
        if (now.hour, now.minute) < (9, 15) or (now.hour, now.minute) >= (15, 30):
            print(json.dumps({"status": "SKIPPED", "reason": "OUTSIDE_NSE_CASH_SESSION", "now_ist": now.isoformat()}))
            return 0

    store = GoogleDriveStore()
    if mode == "recover":
        selection = _recovery_capture(store, target)
        if selection is None:
            print(json.dumps({
                "status": "RECOVER_COMPLETE",
                "recover_date": target.strftime("%Y-%m-%d"),
                "now_ist": now.isoformat(),
            }, sort_keys=True))
            return 0
        manifest, quotes, minute = selection
        age_minutes = (now - minute).total_seconds() / 60.0
    else:
        selection = _latest_capture(store)
        if selection is None:
            # Every fresh slot already has a STORED prediction, normally because
            # the Cloud Run tick predicted it in-worker. Idle is success.
            print(json.dumps({
                "status": "IDLE_ALL_FRESH_SLOTS_PREDICTED",
                "now_ist": now.isoformat(),
            }, sort_keys=True))
            return 0
        manifest, quotes, minute = selection
        age_minutes = (now - minute).total_seconds() / 60.0
        if age_minutes < -1 or age_minutes > MAX_CAPTURE_AGE_SECONDS / 60:
            raise RuntimeError(f"Selected quote snapshot is stale: age_minutes={age_minutes:.1f}")
    quotes, outside_roster = _restrict_to_selected_roster(quotes)
    if outside_roster:
        print(json.dumps({
            "event": "NON_SELECTED_SYMBOLS_EXCLUDED",
            "excluded_symbol_count": len(outside_roster),
            "excluded_symbol_samples": outside_roster[:20],
        }, sort_keys=True), flush=True)
    if quotes.empty:
        raise RuntimeError("Selected snapshot contains no symbols from the pinned 500-symbol roster")
    symbols = set(quotes["symbol"].astype(str).str.strip().str.upper())
    prediction_manifest_remote = (
        "production_data/all_nse_minute_predictions/"
        f"{minute:%Y/%m/%d}/manifest_{minute:%Y%m%d_%H%M}.json"
    )
    existing = _find_remote(store, prediction_manifest_remote)
    if existing:
        existing_path = Path("/tmp/drive_all_nse/existing_prediction_manifest.json")
        _download(store, existing["id"], existing_path)
        prior = json.loads(existing_path.read_text(encoding="utf-8"))
        if _manifest_has_predictions(prior):
            print(json.dumps({
                "status": "SKIPPED_ALREADY_PREDICTED",
                "cycle_minute_ist": minute.isoformat(),
                "prediction_count": prior.get("prediction_count", 0),
            }, sort_keys=True))
            return 0

    work = Path("/tmp/drive_all_nse")
    snapshot_path = work / "all_universe_minute_snapshots" / f"snapshot_{minute:%Y%m%d_%H%M}.parquet"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    quotes.to_parquet(snapshot_path, index=False)
    history = _load_drive_history(
        store, symbols, minute, work, HISTORY_LOOKBACK_BARS
    )
    # Cloud Run persists NIFTY benchmark history under the reserved symbol.
    # Pass it separately; GitHub must never fetch a replacement from an API.
    market = history.pop(
        MARKET_CONTEXT_SYMBOL,
        pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=IST, name="Datetime")),
    )
    market = market.loc[market.index <= minute].tail(HISTORY_LOOKBACK_BARS)
    print(json.dumps({
        "event": "DRIVE_MARKET_CONTEXT_LOADED",
        "rows": int(len(market)),
        "latest_candle_ist": market.index.max().isoformat() if not market.empty else None,
        "source": "GOOGLE_DRIVE_PERSISTED_HISTORY" if not market.empty else "MISSING",
    }, sort_keys=True), flush=True)

    predictor = AllNSEMinutePredictor(work)
    predictor._symbols = sorted(symbols)
    predictor._history = history
    predictor._market_history = market
    predictor._nifty_code = None
    predictor._seed_ready = True
    predictor._seed_complete = True
    predictor._seed_status = "READY" if len(history) == len(symbols) else "READY_PARTIAL"
    predictor._seed_coverage = len(history)
    predictor._seed_universe_source = "GOOGLE_DRIVE_ARCHIVE"
    result = predictor.run_for_capture(manifest, worker_root=work, drive_only=True)
    print(json.dumps({
        "mode": mode,
        "status": result.get("status"),
        "cycle_minute_ist": minute.isoformat(),
        "expected_symbol_count": manifest.get("expected_symbol_count"),
        "quote_symbol_count": len(symbols),
        "history_symbols": len(history),
        "history_missing_symbols": len(symbols) - len(history),
        "prediction_count": result.get("prediction_count"),
        "data_limited_count": result.get("data_limited_count"),
        "prediction_drive_path": result.get("remote_path"),
        "manifest_drive_path": result.get("manifest_drive_path"),
        "error": result.get("error"),
    }, sort_keys=True, default=str))
    if str(result.get("status") or "").upper() != "STORED":
        raise RuntimeError("Drive-only ALL-NSE prediction compiler did not store prediction rows")
    if int(result.get("prediction_count") or 0) < 1:
        if mode == "recover":
            unrecoverable = _recover_zero_prediction_status(result, minute)
            if unrecoverable:
                print(json.dumps({
                    "status": unrecoverable,
                    "cycle_minute_ist": minute.isoformat(),
                    "cycle_minute_key": minute.strftime("%Y%m%d_%H%M"),
                    "prediction_status_counts": result.get("prediction_status_counts"),
                    "reason": (
                        "No contiguous five-minute history is available for this "
                        "slot; the persisted live captures are missing bars."
                    ),
                }, sort_keys=True))
                return 0
        if _has_benign_empty_prediction(result, minute):
            print(json.dumps({
                "status": "NO_PREDICTIONS_EXPECTED",
                "cycle_minute_ist": minute.isoformat(),
                "cycle_minute_key": minute.strftime("%Y%m%d_%H%M"),
                "prediction_status_counts": result.get("prediction_status_counts"),
            }, sort_keys=True))
            return 0
        raise RuntimeError(
            "Drive-only ALL-NSE prediction compiler produced zero predictions for "
            f"a non-warmup slot: {result.get('prediction_status_counts')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
