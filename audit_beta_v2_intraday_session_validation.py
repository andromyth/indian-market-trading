"""Read-only validation of Beta V2 intraday source Parquet contents.

Downloads one symbol file at a time from Google Drive; never writes to Drive,
trains a model, modifies a registry, or promotes a model.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from googleapiclient.http import MediaIoBaseDownload

from delivery_training_universe import official_nse_symbols
from persistent_store import GoogleDriveStore

FOLDER_MIME = "application/vnd.google-apps.folder"
SOURCE_PATH = "production_data/nse_official/intraday_seed/olo_minute_archive/symbols"
HORIZONS = (5, 10, 15, 20, 25, 30)
MIN_USABLE_SESSIONS = 200
MIN_PREFERRED_SESSIONS = 500
IST = "Asia/Kolkata"
TIME_ALIASES = ("timestamp", "datetime", "date_time", "ts", "time", "date", "__index_level_0__", "index")
CLOSE_ALIASES = ("close", "adj close", "ltp", "last", "last_price")


def list_children(store, folder_id: str):
    token = None
    while True:
        response = store.service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            spaces="drive",
            fields="nextPageToken,files(id,name,mimeType,size)",
            pageSize=1000,
            pageToken=token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        yield from response.get("files", [])
        token = response.get("nextPageToken")
        if not token:
            break


def resolve_folder(store, path: str):
    parent = store.root_id
    for part in (p for p in path.split("/") if p):
        hit = next((x for x in list_children(store, parent)
                    if x.get("name") == part and x.get("mimeType") == FOLDER_MIME), None)
        if not hit:
            return None
        parent = hit["id"]
    return parent


def download_file(store, file_id: str, destination: str):
    request = store.service.files().get_media(fileId=file_id)
    with open(destination, "wb") as output:
        downloader = MediaIoBaseDownload(output, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def normalize_source(frame: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.Series]:
    columns = {str(c).strip().lower(): c for c in frame.columns}
    ts_col = next((columns[x] for x in TIME_ALIASES if x in columns), None)
    if ts_col is not None:
        raw_ts = frame[ts_col]
    elif isinstance(frame.index, pd.DatetimeIndex):
        raw_ts = pd.Series(frame.index, index=frame.index)
    else:
        raise ValueError(f"timestamp column not found; columns={list(frame.columns)[:30]}")

    close_col = next((columns[x] for x in CLOSE_ALIASES if x in columns), None)
    if close_col is None:
        raise ValueError(f"close/price column not found; columns={list(frame.columns)[:30]}")

    timestamps = pd.to_datetime(raw_ts, errors="coerce")
    idx = pd.DatetimeIndex(timestamps)
    if idx.tz is None:
        idx = idx.tz_localize(IST, ambiguous="NaT", nonexistent="NaT")
    else:
        idx = idx.tz_convert(IST)

    close = pd.to_numeric(frame[close_col], errors="coerce")
    valid = (~idx.isna()) & np.isfinite(close.to_numpy()) & (close.to_numpy() > 0)
    idx = idx[valid]
    close = close.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    regular = (idx.time >= time(9, 15)) & (idx.time < time(15, 30))
    idx = idx[regular]
    close = close.iloc[np.flatnonzero(regular)].reset_index(drop=True)
    order = np.argsort(idx.asi8, kind="stable")
    idx = idx[order]
    close = close.iloc[order].reset_index(drop=True)
    keep = ~idx.duplicated(keep="last")
    idx = idx[keep]
    close = close.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    return idx, close


def analyze_symbol(store, symbol: str, folder: dict) -> dict:
    files = [x for x in list_children(store, folder["id"])
             if x.get("mimeType") != FOLDER_MIME and x.get("name", "").lower().endswith(".parquet")]
    if not files:
        return {"symbol": symbol, "status": "NO_PARQUET", "usable_sessions": 0}
    all_timestamps = []
    all_closes = []
    source_files = 0
    try:
        for item in files:
            with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
                download_file(store, item["id"], tmp.name)
                frame = pd.read_parquet(tmp.name)
            if frame.empty:
                continue
            idx, close = normalize_source(frame)
            if len(idx):
                all_timestamps.append(idx)
                all_closes.append(close)
            source_files += 1

        if not all_timestamps:
            return {"symbol": symbol, "status": "NO_VALID_ROWS", "usable_sessions": 0,
                    "source_files_read": source_files}

        ts = all_timestamps[0].append(all_timestamps[1:]) if len(all_timestamps) > 1 else all_timestamps[0]
        close = pd.concat(all_closes, ignore_index=True)
        order = np.argsort(ts.asi8, kind="stable")
        ts = ts[order]
        close = close.iloc[order].reset_index(drop=True)
        keep = ~ts.duplicated(keep="last")
        ts = ts[keep]
        close = close.iloc[np.flatnonzero(keep)].reset_index(drop=True)

        day = ts.normalize()
        counts = pd.Series(1, index=day).groupby(level=0).sum()
        unique_dates = pd.Index(day.unique()).sort_values()
        minute_link = pd.Series(ts, index=np.arange(len(ts))).diff().eq(pd.Timedelta(minutes=1)).to_numpy()
        same_day = np.r_[False, day[1:] == day[:-1]]
        # A decision row is feature-ready only after 50 contiguous one-minute bars
        # within the same session, matching the trainer's lookback requirement.
        feature_ready = np.zeros(len(ts), dtype=bool)
        for end in range(49, len(ts)):
            start = end - 49
            feature_ready[end] = bool(
                same_day[start + 1:end + 1].all()
                and minute_link[start + 1:end + 1].all()
            )

        horizon_sessions = {}
        horizon_rows = {}
        for horizon in HORIZONS:
            valid_by_day = set()
            valid_rows = 0
            for date in unique_dates:
                pos = np.flatnonzero(day == date)
                if len(pos) < 51:
                    continue
                day_ts = ts[pos]
                ready = feature_ready[pos]
                decision_ns = day_ts.asi8[ready]
                if not len(decision_ns):
                    continue
                target_ns = decision_ns + int(horizon) * 60_000_000_000
                target_i = np.searchsorted(day_ts.asi8, target_ns, side="left")
                in_bounds = target_i < len(day_ts)
                valid = np.zeros(len(target_i), dtype=bool)
                valid[in_bounds] = (
                    day_ts.asi8[target_i[in_bounds]] - target_ns[in_bounds]
                    <= 60_000_000_000
                )
                count = int(valid.sum())
                valid_rows += count
                if count:
                    valid_by_day.add(str(date.date()))
            horizon_sessions[str(horizon)] = len(valid_by_day)
            horizon_rows[str(horizon)] = valid_rows

        sessions_with_50_plus = int((counts >= 50).sum())
        sessions_full_375 = int((counts >= 375).sum())
        usable = min(horizon_sessions.values()) if horizon_sessions else 0
        status = "PREFERRED_500_PLUS" if usable >= MIN_PREFERRED_SESSIONS else (
            "USABLE_200_PLUS" if usable >= MIN_USABLE_SESSIONS else "INSUFFICIENT_HISTORY"
        )
        return {
            "symbol": symbol,
            "status": status,
            "source_files_read": source_files,
            "valid_rows": int(len(ts)),
            "first_timestamp_ist": ts[0].isoformat(),
            "last_timestamp_ist": ts[-1].isoformat(),
            "raw_sessions": int(len(counts)),
            "sessions_with_50_plus_bars": sessions_with_50_plus,
            "sessions_with_375_plus_bars": sessions_full_375,
            "duplicate_timestamps_removed": int(len(np.concatenate([x.asi8 for x in all_timestamps])) - len(ts)),
            "usable_sessions_by_horizon": horizon_sessions,
            "label_rows_by_horizon": horizon_rows,
            "minimum_horizon_usable_sessions": usable,
        }
    except Exception as exc:
        return {
            "symbol": symbol,
            "status": "READ_OR_SCHEMA_ERROR",
            "usable_sessions": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="beta_v2_intraday_session_validation.json")
    args = parser.parse_args()
    report = {
        "audit": "beta_v2_intraday_session_validation",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "READ_ONLY_NO_TRAINING_NO_PROMOTION",
        "source_path": SOURCE_PATH,
        "minimum_usable_sessions": MIN_USABLE_SESSIONS,
        "preferred_usable_sessions": MIN_PREFERRED_SESSIONS,
        "horizons_min": list(HORIZONS),
        "symbols": {},
        "retraining_authorized_by_this_audit": False,
        "promotion_performed": False,
    }
    store = GoogleDriveStore()
    report["drive_root_accessible"] = bool(
        store.service.files().get(fileId=store.root_id, fields="id").execute().get("id")
    )
    universe = sorted(set(official_nse_symbols(force_refresh=True)))
    report["official_nse_symbol_count"] = len(universe)
    source_folder = resolve_folder(store, SOURCE_PATH)
    if not source_folder or not universe:
        report["status"] = "BLOCKED_UNIVERSE_OR_ARCHIVE_MISSING"
        report["error"] = "Official universe or source archive folder unavailable"
    else:
        children = list(list_children(store, source_folder))
        folders = {
            str(x.get("name", "")).removeprefix("symbol=").strip().upper(): x
            for x in children
            if x.get("mimeType") == FOLDER_MIME and str(x.get("name", "")).startswith("symbol=")
        }
        for i, symbol in enumerate(universe, 1):
            folder = folders.get(symbol)
            if not folder:
                report["symbols"][symbol] = {
                    "symbol": symbol, "status": "MISSING_SYMBOL_FOLDER", "usable_sessions": 0
                }
            else:
                report["symbols"][symbol] = analyze_symbol(store, symbol, folder)
            if i % 50 == 0 or i == len(universe):
                print(f"SESSION_AUDIT_PROGRESS {i}/{len(universe)}", flush=True)

        rows = list(report["symbols"].values())
        report["summary"] = {
            "symbols_audited": len(rows),
            "preferred_500_plus": sum(x.get("status") == "PREFERRED_500_PLUS" for x in rows),
            "usable_200_plus": sum(x.get("status") in ("PREFERRED_500_PLUS", "USABLE_200_PLUS") for x in rows),
            "insufficient_history": sum(x.get("status") == "INSUFFICIENT_HISTORY" for x in rows),
            "missing_symbol_folders": sum(x.get("status") == "MISSING_SYMBOL_FOLDER" for x in rows),
            "read_or_schema_errors": sum(x.get("status") == "READ_OR_SCHEMA_ERROR" for x in rows),
            "no_parquet_or_valid_rows": sum(x.get("status") in ("NO_PARQUET", "NO_VALID_ROWS") for x in rows),
            "usable_symbols_by_horizon": {
                str(h): sum(x.get("usable_sessions_by_horizon", {}).get(str(h), 0) >= MIN_USABLE_SESSIONS for x in rows)
                for h in HORIZONS
            },
            "preferred_symbols_by_horizon": {
                str(h): sum(x.get("usable_sessions_by_horizon", {}).get(str(h), 0) >= MIN_PREFERRED_SESSIONS for x in rows)
                for h in HORIZONS
            },
        }
        report["status"] = (
            "SESSION_VALIDATION_COMPLETE"
            if report["summary"]["read_or_schema_errors"] == 0
            else "SESSION_VALIDATION_COMPLETE_WITH_ERRORS"
        )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in report.items() if k != "symbols"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
