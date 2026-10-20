#!/usr/bin/env python3
"""Backfill historical 1-minute actual candles for symbols with saved forecasts.

This is a recovery tool for validation only. It never creates or edits
predictions. Yahoo Finance candles are explicitly labelled as historical
backfill and are not represented as live or NSE-authoritative observations.
Existing partition values win over the backfill at duplicate timestamps.
"""
from __future__ import annotations

import argparse
import io
import json
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from autonomous_intraday_worker import normalize_bars
from persistent_store import GoogleDriveStore, drive_status

IST = ZoneInfo("Asia/Kolkata")
FOLDER_MIME = "application/vnd.google-apps.folder"
LEDGER_ROOT = "production_data/intraday_predictions"
BARS_ROOT = "production_data/intraday_live_worker"


def _resolve_drive_id(store: Any, remote_path: str) -> str | None:
    parts = [p for p in remote_path.replace("\\", "/").split("/") if p]
    if not parts:
        return None
    parent = store.root_id
    for part in parts[:-1]:
        hit = store._find_child(parent, part, FOLDER_MIME)
        if not hit:
            return None
        parent = hit["id"]
    hit = store._find_child(parent, parts[-1])
    return str(hit["id"]) if hit else None


def _download_drive_bytes(store: Any, remote_path: str) -> bytes | None:
    file_id = _resolve_drive_id(store, remote_path)
    if not file_id:
        return None
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, store.service.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def _upload_with_retry(store: Any, local_path: Path, remote_path: str, mime_type: str) -> dict:
    last_error = None
    for attempt in range(1, 5):
        try:
            return store.upload_or_update(local_path, remote_path, mime_type=mime_type)
        except HttpError as exc:
            last_error = exc
            status = int(getattr(exc.resp, "status", 0) or 0)
            if status < 500 or attempt == 4:
                raise
            print(f"Transient Drive HTTP {status} while uploading {remote_path}; retry {attempt}/3.", flush=True)
            time.sleep(attempt * 3)
    raise RuntimeError(f"Google Drive upload failed after retries: {type(last_error).__name__}")


def _records_for_day(payload: bytes) -> list[dict]:
    records = []
    malformed = 0
    for line in payload.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(row, dict) and row.get("symbol") and (
            row.get("forecast_key") or row.get("horizon_min") is not None
            or row.get("target_timestamp_utc")
        ):
            records.append(row)
        elif not isinstance(row, dict):
            malformed += 1
    if malformed:
        print(f"Ledger contained {malformed} malformed/non-object line(s); skipped.", flush=True)
    return records


def _fetch_yahoo_bars(symbol: str, session_day: date) -> pd.DataFrame:
    ticker = f"{symbol}.NS"
    start = session_day.isoformat()
    end = (session_day + timedelta(days=1)).isoformat()
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1m",
        prepost=False,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        timeout=20,
        group_by="column",
    )
    bars = normalize_bars(raw)
    if bars.empty:
        return bars
    idx = bars.index.tz_convert(IST)
    is_day = idx.date == session_day
    is_session = (idx.time >= datetime.strptime("09:15", "%H:%M").time()) & (
        idx.time < datetime.strptime("15:30", "%H:%M").time()
    )
    bars = bars.loc[is_day & is_session].copy()
    bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index()
    if not bars.empty:
        bars["bar_source"] = "YFINANCE_HISTORICAL_BACKFILL"
        bars["is_historical_backfill"] = True
    return bars


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-date", required=True, help="NSE session date YYYY-MM-DD")
    parser.add_argument("--segment", default="service", choices=("service", "morning", "afternoon", "once"))
    args = parser.parse_args()
    session_day = date.fromisoformat(args.session_date)

    if session_day.weekday() >= 5:
        raise SystemExit(f"{session_day} is a weekend; refusing historical backfill.")
    now_ist = datetime.now(IST)
    if session_day >= now_ist.date():
        raise SystemExit("Only completed historical sessions can be backfilled.")

    status = drive_status(require_write=True)
    safe = {k: status.get(k) for k in (
        "configured", "reachable", "write_ready", "auth_method",
        "write_auth_method", "reason",
    )}
    print("DRIVE_PREFLIGHT:", json.dumps(safe, sort_keys=True), flush=True)
    if not status.get("configured") or status.get("reachable") is False or status.get("write_ready") is False:
        raise SystemExit("Drive read/write preflight failed; no backfill was written.")

    store = GoogleDriveStore()
    day_key = f"{session_day:%Y/%m/%d}"
    ledger_path = f"{LEDGER_ROOT}/{day_key}/predictions_{args.segment}.jsonl"
    ledger_bytes = _download_drive_bytes(store, ledger_path)
    if ledger_bytes is None:
        raise SystemExit(f"No saved prediction ledger found at {ledger_path}; refusing to create predictions.")
    records = _records_for_day(ledger_bytes)
    symbols = sorted({
        str(row["symbol"]).strip().upper().replace(".NS", "").replace(".BO", "")
        for row in records if str(row.get("symbol") or "").strip()
    })
    if not symbols:
        raise SystemExit("No symbols with saved forecasts were found; no data was backfilled.")

    print(
        f"ACTUAL_CANDLE_BACKFILL_START: session={session_day} segment={args.segment} "
        f"saved_forecast_rows={len(records)} distinct_symbols={len(symbols)}",
        flush=True,
    )
    manifest = {
        "session_date": session_day.isoformat(),
        "segment": args.segment,
        "backfill_source": "YFINANCE_HISTORICAL_BACKFILL",
        "backfill_source_url": "https://finance.yahoo.com/",
        "historical_only": True,
        "predictions_created_or_modified": False,
        "existing_partition_timestamps_preferred": True,
        "source_semantics": "Historical Yahoo Finance 1-minute bars for outcome validation; not live data and not claimed NSE-authoritative.",
        "ledger_path": ledger_path,
        "saved_forecast_rows": len(records),
        "symbols": {},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    all_fetched_rows = 0
    any_write = False

    for symbol in symbols:
        report: dict[str, Any] = {
            "ticker_requested": f"{symbol}.NS",
            "status": "STARTED",
            "existing_rows": 0,
            "backfill_rows_fetched": 0,
            "new_timestamps_added": 0,
            "merged_rows": 0,
        }
        try:
            fetched = _fetch_yahoo_bars(symbol, session_day)
            report["backfill_rows_fetched"] = int(len(fetched))
            all_fetched_rows += int(len(fetched))
            if fetched.empty:
                report["status"] = "NO_HISTORICAL_1M_DATA"
                report["error"] = "Yahoo Finance returned no valid regular-session 1-minute candles."
                manifest["symbols"][symbol] = report
                print(f"BACKFILL {symbol}: NO_HISTORICAL_1M_DATA", flush=True)
                continue

            safe_symbol = "".join(c if c.isalnum() or c in {"_", "-", "^"} else "_" for c in symbol)
            remote_path = f"{BARS_ROOT}/{day_key}/{safe_symbol}.parquet"
            prior_bytes = _download_drive_bytes(store, remote_path)
            if prior_bytes:
                try:
                    existing = normalize_bars(pd.read_parquet(io.BytesIO(prior_bytes)))
                except Exception as exc:
                    raise RuntimeError(
                        f"Existing bar partition is unreadable; refusing to overwrite: {type(exc).__name__}"
                    ) from exc
            else:
                existing = pd.DataFrame()

            report["existing_rows"] = int(len(existing))
            if not existing.empty:
                if "bar_source" not in existing.columns:
                    existing["bar_source"] = "EXISTING_PARTITION_SOURCE_UNRECORDED"
                if "is_historical_backfill" not in existing.columns:
                    existing["is_historical_backfill"] = False
                old_index = set(existing.index)
            else:
                old_index = set()

            # Append only timestamps not already present. This guarantees the
            # original captured value wins at duplicate timestamps, independent
            # of pandas sort stability.
            missing_rows = fetched.loc[~fetched.index.isin(old_index)].copy()
            if existing.empty:
                combined = fetched.copy()
            else:
                combined = pd.concat([existing, missing_rows], axis=0, sort=False).sort_index()
            combined = combined.loc[~combined.index.duplicated(keep="last")]
            report["new_timestamps_added"] = int(len(missing_rows))
            report["merged_rows"] = int(len(combined))

            with tempfile.TemporaryDirectory(prefix="intraday-actual-backfill-") as tmp:
                local_path = Path(tmp) / f"{safe_symbol}.parquet"
                combined.to_parquet(local_path, index=True)
                upload = _upload_with_retry(
                    store,
                    local_path,
                    remote_path,
                    "application/vnd.apache.parquet",
                )

            report["remote_partition_path"] = remote_path
            report["drive_upload_status"] = upload.get("status", "UNKNOWN")
            report["status"] = "BACKFILLED" if report["new_timestamps_added"] > 0 else "NO_NEW_TIMESTAMPS"
            any_write = any_write or upload.get("status") == "OK"
            manifest["symbols"][symbol] = report
            print(
                f"BACKFILL {symbol}: {report['status']} fetched={report['backfill_rows_fetched']} "
                f"added={report['new_timestamps_added']} total={report['merged_rows']} "
                f"drive={report['drive_upload_status']}",
                flush=True,
            )
        except Exception as exc:
            report["status"] = "ERROR"
            report["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
            manifest["symbols"][symbol] = report
            print(f"BACKFILL {symbol}: ERROR {report['error']}", flush=True)

    manifest["total_backfill_rows_fetched"] = all_fetched_rows
    manifest["any_partition_uploaded"] = any_write
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="intraday-actual-backfill-manifest-") as tmp:
        local_manifest = Path(tmp) / f"actual_candle_backfill_{session_day:%Y%m%d}.json"
        local_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        upload = _upload_with_retry(
            store,
            local_manifest,
            f"{BARS_ROOT}/{day_key}/actual_candle_backfill_{session_day:%Y%m%d}.json",
            "application/json",
        )
        print(f"BACKFILL_MANIFEST_DRIVE_STATUS: {upload.get('status', 'UNKNOWN')}", flush=True)
    print("ACTUAL_CANDLE_BACKFILL_SUMMARY:", json.dumps(manifest, sort_keys=True), flush=True)

    if not any_write or all_fetched_rows == 0:
        return 2
    if any(row.get("status") == "ERROR" for row in manifest["symbols"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
