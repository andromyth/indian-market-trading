#!/usr/bin/env python3
"""Compile the latest persisted NSE quote snapshot into predictions from Drive data.

This GitHub Actions job is a Drive consumer only. It never imports or calls
indstocks_client; INDstocks access remains inside the singleton Cloud Run worker.
"""
from __future__ import annotations

import json
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
LIVE_HISTORY_ROOT = "production_data/nse_official/intraday_5m_live"


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
            pageSize=1000,
            pageToken=page,
            fields="nextPageToken,files(id,name,mimeType,size,modifiedTime)",
        ).execute()
        rows.extend(result.get("files", []))
        page = result.get("nextPageToken")
        if not page:
            return rows


def _download(store, file_id: str, destination: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = store.service.files().get_media(fileId=file_id)
    with destination.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


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


def _latest_capture(store) -> tuple[dict, pd.DataFrame, pd.Timestamp]:
    item = _find_remote(store, f"{QUOTE_ROOT}/latest_manifest.json")
    if not item:
        raise RuntimeError("Latest all-universe quote manifest is missing from Drive")
    local = Path("/tmp/drive_all_nse/latest_manifest.json")
    _download(store, item["id"], local)
    manifest = json.loads(local.read_text(encoding="utf-8"))
    minute = pd.Timestamp(manifest.get("cycle_minute_ist"))
    minute = minute.tz_localize(IST) if minute.tzinfo is None else minute.tz_convert(IST)
    status = str(manifest.get("status") or "").upper()
    if manifest.get("snapshot_persisted") is not True or status != "STORED":
        raise RuntimeError(
            f"Latest quote snapshot is not durably stored: status={status or 'MISSING'}"
        )
    snapshot_remote = str(manifest.get("snapshot_drive_path") or "")
    if not snapshot_remote:
        raise RuntimeError("Latest quote manifest has no snapshot_drive_path")
    snapshot_item = _find_remote(store, snapshot_remote)
    if not snapshot_item:
        raise RuntimeError("Manifest points to a missing quote snapshot in Drive")
    snapshot_path = Path("/tmp/drive_all_nse") / Path(snapshot_remote).name
    _download(store, snapshot_item["id"], snapshot_path)
    quotes = pd.read_parquet(snapshot_path)
    if quotes.empty or "symbol" not in quotes.columns:
        raise RuntimeError("Latest quote snapshot is empty or has no symbol column")
    return manifest, quotes, minute


def _load_drive_history(store, quote_symbols: set[str], minute: pd.Timestamp, work: Path, lookback: int) -> dict[str, pd.DataFrame]:
    history: dict[str, pd.DataFrame] = {}
    archive_folder = _find_remote(store, HISTORY_ROOT)
    if archive_folder:
        for folder in _list_children(store, archive_folder["id"]):
            name = str(folder.get("name") or "")
            if folder.get("mimeType") != FOLDER_MIME or not name.startswith("symbol="):
                continue
            symbol = name.split("=", 1)[1].strip().upper()
            if symbol not in quote_symbols:
                continue
            files = _list_children(store, folder["id"])
            item = next(
                (entry for entry in files if entry.get("name") == f"{symbol}.parquet"),
                None,
            )
            if not item:
                continue
            local = work / "backfill" / f"{symbol}.parquet"
            _download(store, item["id"], local)
            frame = _read_history(local)
            frame = frame.loc[frame.index <= minute].tail(lookback)
            if not frame.empty:
                history[symbol] = frame

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
            frame = _read_history(local)
            if frame.empty:
                continue
            raw = pd.read_parquet(local)
            if "symbol" not in raw.columns:
                continue
            raw["symbol"] = raw["symbol"].astype(str).str.strip().str.upper()
            raw = raw.loc[raw["symbol"].isin(quote_symbols)]
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


def _market_history(minute: pd.Timestamp) -> pd.DataFrame:
    """Yahoo is a market-context fallback only; no INDstocks request is made."""
    import yfinance as yf
    import all_nse_intraday_model as model

    raw = yf.download(
        "^NSEI", period="10d", interval="5m", auto_adjust=False,
        progress=False, threads=False, timeout=15, group_by="column",
    )
    frame = model._norm(raw)
    if frame.empty:
        return frame
    return frame.loc[frame.index <= minute].tail(500)


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
    if not (now.hour, now.minute) >= (9, 15) or (now.hour, now.minute) >= (15, 30):
        print(json.dumps({"status": "SKIPPED", "reason": "OUTSIDE_NSE_CASH_SESSION", "now_ist": now.isoformat()}))
        return 0

    store = GoogleDriveStore()
    manifest, quotes, minute = _latest_capture(store)
    age_minutes = (now - minute).total_seconds() / 60.0
    if age_minutes < -1 or age_minutes > 15:
        raise RuntimeError(f"Latest quote snapshot is stale: age_minutes={age_minutes:.1f}")
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
    market = _market_history(minute)

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
