#!/usr/bin/env python3
"""Compile the latest persisted NSE quote snapshot into predictions from Drive data.

This GitHub Actions job is a Drive consumer only. It never imports or calls
indstocks_client; INDstocks access remains inside the singleton Cloud Run worker.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
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
    rows: list[dict] = []
    page = None
    while True:
        result = store.service.files().list(
            q=f"'{parent_id}' in parents and trashed = false",
            spaces="drive",
            pageSize=1000,
            pageToken=page,
            fields="nextPageToken,files(id,name,mimeType,size,modifiedTime)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        rows.extend(result.get("files", []))
        page = result.get("nextPageToken")
        if not page:
            return rows


def _download(store, file_id: str, destination: Path) -> None:
    from persistent_store import download_drive_file_with_retry

    download_drive_file_with_retry(
        store.service,
        file_id,
        destination,
        operation="github_drive_prediction_input",
    )


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
    return str(manifest.get("status") or "").upper() == "STORED"


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


def _latest_capture(store, now: pd.Timestamp | None = None) -> tuple[dict, pd.DataFrame, pd.Timestamp]:
    """Select the newest fresh STORED snapshot, even when latest_manifest is a failure pointer.

    A failed capture updates latest_manifest.json. That status pointer must not
    hide a valid same-session snapshot whose own per-minute manifest was stored.
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
    for minute, manifest, item in candidates:
        if minute in seen_minutes:
            continue
        seen_minutes.add(minute)
        status = str(manifest.get("status") or "").upper()
        age_seconds = (current - minute).total_seconds()
        if manifest.get("snapshot_persisted") is not True or status != "STORED":
            skipped.append(f"{item.get('name')}: status={status or 'MISSING'}")
            continue
        if minute.date() != current.date() or age_seconds < 0 or age_seconds > MAX_CAPTURE_AGE_SECONDS:
            skipped.append(f"{item.get('name')}: stale_or_future age_seconds={round(age_seconds, 1)}")
            continue
        if _prediction_already_stored(store, minute):
            skipped.append(f"{item.get('name')}: prediction already STORED")
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

    raise RuntimeError(
        "No fresh, durably stored unprocessed current-session quote snapshot was found; "
        + "; ".join(skipped[-8:])
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
        for folder in _list_children(store, archive_folder["id"]):
            name = str(folder.get("name") or "")
            if folder.get("mimeType") != FOLDER_MIME or not name.startswith("symbol="):
                continue
            symbol = name.split("=", 1)[1].strip().upper()
            if symbol not in quote_symbols and symbol != MARKET_CONTEXT_SYMBOL:
                continue
            files = _list_children(store, folder["id"])
            item = next(
                (entry for entry in files if entry.get("name") == f"{symbol}.parquet"),
                None,
            )
            if not item:
                continue
            local = cache_dir / f"{symbol}.parquet"
            cached = cache_index.get(symbol) or {}
            if (
                not local.is_file()
                or cached.get("modifiedTime") != item.get("modifiedTime")
                or str(cached.get("size") or "") != str(item.get("size") or "")
            ):
                _download(store, item["id"], local)
            try:
                frame = _read_history(local)
            except Exception:
                _download(store, item["id"], local)
                frame = _read_history(local)
            cache_index[symbol] = {
                "modifiedTime": item.get("modifiedTime"),
                "size": item.get("size"),
            }
            frame = frame.loc[frame.index <= minute].tail(lookback)
            if not frame.empty:
                history[symbol] = frame
    index_path.write_text(json.dumps(cache_index, sort_keys=True), encoding="utf-8")

    live_folder = _find_remote(
        store,
        f"{LIVE_HISTORY_ROOT}/{minute:%Y/%m/%d}",
    )
    if live_folder:
        for item in _list_children(store, live_folder["id"]):
            name = str(item.get("name") or "")
            if not name.startswith("capture_") or not name.endswith(".parquet"):
                continue
            local = work / "live" / name
            _download(store, item["id"], local)
            raw = pd.read_parquet(local)
            if "symbol" not in raw.columns:
                continue
            raw["symbol"] = raw["symbol"].astype(str).str.strip().str.upper()
            raw = raw.loc[raw["symbol"].isin(quote_symbols | {MARKET_CONTEXT_SYMBOL})]
            for symbol, group in raw.groupby("symbol"):
                part_path = work / "live_parts" / f"{symbol}_{name}"
                part_path.parent.mkdir(parents=True, exist_ok=True)
                group.to_parquet(part_path, index=False)
                part = _read_history(part_path)
                existing = history.get(symbol)
                combined = pd.concat([existing, part]) if existing is not None else part
                combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
                history[symbol] = combined.loc[combined.index <= minute].tail(lookback)
    return history


def main() -> int:
    from market_session_guard import is_nse_trading_day
    from persistent_store import GoogleDriveStore
    from all_nse_minute_prediction_capture import (
        AllNSEMinutePredictor,
        HISTORY_LOOKBACK_BARS,
    )

    now = pd.Timestamp.now(tz=IST)
    if not is_nse_trading_day(now.date()):
        print(json.dumps({"status": "SKIPPED", "reason": "NSE_NON_TRADING_DAY", "now_ist": now.isoformat()}))
        return 0
    if (now.hour, now.minute) < (9, 15) or (now.hour, now.minute) >= (15, 30):
        print(json.dumps({"status": "SKIPPED", "reason": "OUTSIDE_NSE_CASH_SESSION", "now_ist": now.isoformat()}))
        return 0

    store = GoogleDriveStore()
    manifest, quotes, minute = _latest_capture(store)
    age_minutes = (now - minute).total_seconds() / 60.0
    if age_minutes < -1 or age_minutes > MAX_CAPTURE_AGE_SECONDS / 60:
        raise RuntimeError(f"Selected quote snapshot is stale: age_minutes={age_minutes:.1f}")
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
        if str(prior.get("status") or "").upper() == "STORED":
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
    if str(result.get("status") or "").upper() != "STORED" or int(result.get("prediction_count") or 0) < 1:
        raise RuntimeError("Drive-only ALL-NSE prediction compiler did not store prediction rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
