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
    """Discover archived NSE bulk parquet files through the known Drive tree.

    The EOD ingest stores files under:
    production_data/nse_official/bulk_eod/YYYY/MM/DD/<BhavCopy...parquet>
    We traverse only the year/month/day folders needed for this window rather
    than relying on a global filename query.
    """
    base = _find_folder(store, "production_data/nse_official/bulk_eod")
    if not base:
        return []

    out: list[dict] = []

    def list_children(parent_id: str) -> list[dict]:
        token = None
        items: list[dict] = []
        while True:
            kwargs = {
                "q": f"'{parent_id}' in parents and trashed = false",
                "spaces": "drive",
                "fields": "nextPageToken,files(id,name,mimeType,size)",
                "pageSize": 500,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }
            if token:
                kwargs["pageToken"] = token
            resp = store.service.files().list(**kwargs).execute()
            items.extend(resp.get("files", []))
            token = resp.get("nextPageToken")
            if not token:
                return items

    years = sorted({d.year for d in pd.date_range(start, end, freq="D")})
    for year in years:
        yfolder = store._find_child(base, f"{year:04d}", FOLDER_MIME)
        if not yfolder:
            continue
        for month in range(1, 13):
            month_start = date(year, month, 1)
            if month_start > end or (month_start.replace(day=28) + timedelta(days=4)).replace(day=1) < start:
                continue
            mfolder = store._find_child(yfolder["id"], f"{month:02d}", FOLDER_MIME)
            if not mfolder:
                continue
            for day in range(1, 32):
                try:
                    d = date(year, month, day)
                except ValueError:
                    continue
                if d < start or d > end:
                    continue
                dfolder = store._find_child(mfolder["id"], f"{day:02d}", FOLDER_MIME)
                if not dfolder:
                    continue
                for item in list_children(dfolder["id"]):
                    dname = _date_from_name(item.get("name", ""))
                    if dname == d and str(item.get("name", "")).lower().endswith(".parquet"):
                        out.append(item)

    # Defensive fallback for archives whose date folder/file naming differs.
    if not out:
        parent_items = list_children(base)
        for year_item in parent_items:
            if year_item.get("mimeType") != FOLDER_MIME:
                continue
            try:
                year = int(year_item.get("name", ""))
            except ValueError:
                continue
            if year < start.year or year > end.year:
                continue
            for month_item in list_children(year_item["id"]):
                if month_item.get("mimeType") != FOLDER_MIME:
                    continue
                for day_item in list_children(month_item["id"]):
                    if day_item.get("mimeType") != FOLDER_MIME:
                        continue
                    for item in list_children(day_item["id"]):
                        dname = _date_from_name(item.get("name", ""))
                        if dname and start <= dname <= end and str(item.get("name", "")).lower().endswith(".parquet"):
                            out.append(item)

    unique = {str(x["id"]): x for x in out}
    return sorted(unique.values(), key=lambda x: _date_from_name(x["name"]) or date.min)


def _remote_symbol_parquets(
    store: GoogleDriveStore, symbols: list[str], start: date, end: date
) -> dict[str, list[dict]]:
    """Find already-synced per-symbol official NSE parquet chunks in Drive."""
    out = {s.upper(): [] for s in symbols}
    base = _find_folder(store, "production_data/nse_official/equity")
    if not base:
        return out

    for symbol in out:
        sfolder = store._find_child(base, symbol, FOLDER_MIME)
        if not sfolder:
            continue
        token = None
        while True:
            kwargs = {
                "q": f"'{sfolder['id']}' in parents and trashed = false",
                "spaces": "drive",
                "fields": "nextPageToken,files(id,name,mimeType,size)",
                "pageSize": 100,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }
            if token:
                kwargs["pageToken"] = token
            resp = store.service.files().list(**kwargs).execute()
            for item in resp.get("files", []):
                name = str(item.get("name", "")).lower()
                if name.endswith(".parquet") and item.get("id"):
                    out[symbol].append(item)
            token = resp.get("nextPageToken")
            if not token:
                break
    return out


def _load_remote_symbol_history(
    store: GoogleDriveStore,
    symbol: str,
    files: list[dict],
    start: date,
    end: date,
) -> pd.DataFrame:
    frames = []
    for item in files:
        temp = CACHE_ROOT / f".{symbol}.{item['id']}.tmp"
        try:
            _download_file(store, item["id"], temp)
            x = pd.read_parquet(temp)
            if "date" not in x.columns:
                continue
            x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.date
            for col in ("open", "high", "low", "close", "volume"):
                if col in x.columns:
                    x[col] = pd.to_numeric(x[col], errors="coerce")
            x = x[
                x["date"].notna()
                & (x["date"] >= start)
                & (x["date"] <= end)
            ].copy()
            if not x.empty:
                frames.append(x)
        finally:
            temp.unlink(missing_ok=True)
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    return x.sort_values("date").drop_duplicates(["symbol", "date"], keep="last")


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

    # Prefer any already-synced official NSE security-wise parquet chunks.
    # This is substantially cheaper than downloading dozens of daily bulk files.
    remote_symbol_files = _remote_symbol_parquets(store, symbols, end - timedelta(days=180), end)
    for symbol in symbols:
        if symbol in remote_symbol_files and remote_symbol_files[symbol]:
            frame = _load_remote_symbol_history(
                store, symbol, remote_symbol_files[symbol], end - timedelta(days=180), end
            )
            if len(frame) >= min_rows:
                local = CACHE_ROOT / f"{symbol}.parquet"
                results[symbol] = _save_symbol(symbol, frame, store)
        if symbol not in results:
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
