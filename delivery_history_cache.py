#!/usr/bin/env python3
"""Persistent Drive-backed daily history cache for the certified Delivery universe."""
from __future__ import annotations

import io
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from persistent_store import GoogleDriveStore

BULK_PREFIX = "BhavCopy_NSE_CM_0_0_0_"
CACHE_ROOT = Path(__file__).resolve().parent / "data" / "nse_official" / "delivery_history"
REMOTE_ROOT = "production_data/nse_official/delivery_history"
FILENAME_RE = re.compile(r"BhavCopy_NSE_CM_0_0_0_(20\d{6})_F_0000\.parquet$", re.I)
FOLDER_MIME = "application/vnd.google-apps.folder"


def _date_from_name(name: str) -> date | None:
    m = FILENAME_RE.search(str(name))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except Exception:
        return None


def _find_folder(store: GoogleDriveStore, relative_path: str) -> str | None:
    parent = store.root_id
    for part in [p for p in relative_path.split("/") if p]:
        hit = store._find_child(parent, part, FOLDER_MIME)
        if not hit:
            return None
        parent = hit["id"]
    return parent


def _find_file_in_folder(store: GoogleDriveStore, folder_id: str, name: str) -> dict | None:
    q = f"name = '{store._q(name)}' and '{folder_id}' in parents and trashed = false"
    files = store.service.files().list(
        q=q,
        spaces="drive",
        fields="files(id,name,mimeType,size,modifiedTime)",
        pageSize=10,
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
    ).execute().get("files", [])
    return files[0] if files else None


def _download_file(store: GoogleDriveStore, file_id: str, path: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    request = store.service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


def _remote_bulk_files(store: GoogleDriveStore, start: date, end: date) -> list[dict]:
    token = None
    out: list[dict] = []
    while True:
        kwargs = {
            "q": (
                "trashed = false and "
                f"name contains '{BULK_PREFIX}'"
            ),
            "spaces": "drive",
            "fields": "nextPageToken,files(id,name,mimeType,size)",
            "pageSize": 1000,
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
            "corpora": "allDrives",
        }
        if token:
            kwargs["pageToken"] = token
        resp = store.service.files().list(**kwargs).execute()
        for item in resp.get("files", []):
            d = _date_from_name(item.get("name", ""))
            if d and start <= d <= end:
                out.append(item)
        token = resp.get("nextPageToken")
        if not token:
            break
    return sorted(out, key=lambda x: _date_from_name(x["name"]) or date.min)


def _filter_symbols(df: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    if df is None or df.empty or "symbol" not in df.columns:
        return {}
    x = df.copy()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    wanted = {s.upper() for s in symbols}
    x = x[x["symbol"].isin(wanted)].copy()
    for c in ("open", "high", "low", "close", "volume"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["symbol", "date", "open", "high", "low", "close"])
    return {
        s: g.sort_values("date").drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)
        for s, g in x.groupby("symbol", sort=False)
    }


def _save_symbol(symbol: str, frame: pd.DataFrame, store: GoogleDriveStore) -> pd.DataFrame:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = CACHE_ROOT / f"{symbol}.parquet"
    frame = frame.sort_values("date").drop_duplicates(["symbol", "date"], keep="last")
    frame.to_parquet(path, index=False)
    store.upload_or_update(path, f"{REMOTE_ROOT}/{symbol}.parquet", mime_type="application/octet-stream")
    return frame


def ensure_delivery_history(symbols: list[str], end: date, min_rows: int = 65) -> dict[str, pd.DataFrame]:
    """Load compact histories from Drive and update them with the newest official bulk EOD day.

    On first use, bootstraps the compact cache from the most recent official
    bulk EOD files already present in Drive. Subsequent runs download only the
    compact symbol file plus any newer bulk day needed for the current session.
    """
    symbols = [str(s).upper().replace(".NS", "").replace(".BO", "") for s in symbols]
    symbols = list(dict.fromkeys(symbols))
    store = GoogleDriveStore()

    folder_id = _find_folder(store, REMOTE_ROOT)
    results: dict[str, pd.DataFrame] = {}
    missing = []

    for symbol in symbols:
        local = CACHE_ROOT / f"{symbol}.parquet"
        if not local.exists() and folder_id:
            remote = _find_file_in_folder(store, folder_id, f"{symbol}.parquet")
            if remote:
                _download_file(store, remote["id"], local)
        if local.exists():
            try:
                results[symbol] = pd.read_parquet(local)
            except Exception:
                local.unlink(missing_ok=True)
        if symbol not in results or len(results[symbol]) < min_rows:
            missing.append(symbol)

    today_file = None
    recent_start = end - timedelta(days=140)
    bulk_files = _remote_bulk_files(store, recent_start, end)
    if bulk_files:
        today_file = bulk_files[-1] if _date_from_name(bulk_files[-1]["name"]) == end else None

    if missing:
        if len(bulk_files) < min_rows:
            raise RuntimeError(
                f"Drive contains only {len(bulk_files)} recent NSE bulk EOD files; "
                f"{min_rows} are required for Delivery warmup."
            )
        # Use the most recent min_rows trading-day files as the compact bootstrap.
        selected = bulk_files[-min_rows:]
        collected = {s: [] for s in symbols}
        for item in selected:
            temp = CACHE_ROOT / f".{item['name']}.tmp"
            _download_file(store, item["id"], temp)
            try:
                df = pd.read_parquet(temp)
                parts = _filter_symbols(df, symbols)
                for s, part in parts.items():
                    collected[s].append(part)
            finally:
                temp.unlink(missing_ok=True)
        for s in missing:
            frames = collected.get(s, [])
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if frame.empty or len(frame) < min_rows:
                raise RuntimeError(
                    f"Insufficient official NSE EOD history in Drive for {s}: {len(frame)} rows"
                )
            results[s] = _save_symbol(s, frame, store)

    # Add the newest official EOD bulk day if the compact cache is one session behind.
    if today_file:
        d = _date_from_name(today_file["name"])
        for s in symbols:
            current = results.get(s, pd.DataFrame())
            latest = pd.to_datetime(current["date"], errors="coerce").max().date() if not current.empty else None
            if latest == d:
                continue
            temp = CACHE_ROOT / f".{today_file['name']}.tmp"
            _download_file(store, today_file["id"], temp)
            try:
                df = pd.read_parquet(temp)
                parts = _filter_symbols(df, [s])
                if parts:
                    merged = pd.concat([current, parts[s]], ignore_index=True)
                    results[s] = _save_symbol(s, merged, store)
            finally:
                temp.unlink(missing_ok=True)

    return results
