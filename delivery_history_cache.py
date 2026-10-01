#!/usr/bin/env python3
"""Persistent official NSE daily-history compiler for the Delivery engine.

Google Drive stores the compact compiled history used by Delivery prediction.
The first run bootstraps from official NSE CM-UDiFF daily EOD files. Later runs
only add the newest trading day, so daily prediction does not rediscover or
re-download the full archive.
"""
from __future__ import annotations

import io
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from nse_bulk_eod_ingest import business_days
from nse_data_layer import BULK_EOD_ROOT, fetch_nse_eod_bhavcopy
from persistent_store import GoogleDriveStore

CACHE_ROOT = Path(__file__).resolve().parent / "data" / "nse_official" / "delivery_history"
REMOTE_ROOT = "production_data/nse_official/delivery_history"
FOLDER_MIME = "application/vnd.google-apps.folder"


def _find_file(store: GoogleDriveStore, symbol: str) -> dict | None:
    parent = store.root_id
    for part in REMOTE_ROOT.split("/"):
        hit = store._find_child(parent, part, FOLDER_MIME)
        if not hit:
            return None
        parent = hit["id"]
    q = f"name = '{store._q(symbol)}.parquet' and '{parent}' in parents and trashed = false"
    files = store.service.files().list(
        q=q,
        spaces="drive",
        fields="files(id,name,mimeType,size,modifiedTime)",
        pageSize=10,
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
    ).execute().get("files", [])
    return files[0] if files else None


def _download(store: GoogleDriveStore, file_id: str, path: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    request = store.service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


def _load_compact(store: GoogleDriveStore, symbol: str, max_date: date) -> pd.DataFrame:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = CACHE_ROOT / f"{symbol}.parquet"
    if not path.exists():
        remote = _find_file(store, symbol)
        if remote:
            _download(store, remote["id"], path)
    if not path.exists():
        return pd.DataFrame()
    try:
        x = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if "date" not in x.columns:
        return pd.DataFrame()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    for c in ("open", "high", "low", "close", "volume"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["symbol", "date", "open", "high", "low", "close"])
    before = len(x)
    x = x[x["date"] <= pd.Timestamp(max_date)]
    dropped = before - len(x)
    if dropped:
        print(
            f"COMPILED_HISTORY_FUTURE_ROWS_DROPPED: {symbol} rows={dropped} "
            f"max_allowed={max_date.isoformat()}",
            flush=True,
        )
    return x.sort_values("date").drop_duplicates(["symbol", "date"], keep="last")


def _save_compact(
    store: GoogleDriveStore,
    symbol: str,
    frame: pd.DataFrame,
    max_date: date,
) -> pd.DataFrame:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = CACHE_ROOT / f"{symbol}.parquet"
    x = frame.copy()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x = x.dropna(subset=["symbol", "date", "open", "high", "low", "close"])
    x = x[x["date"] <= pd.Timestamp(max_date)]
    x = x.sort_values("date").drop_duplicates(["symbol", "date"], keep="last")
    x.to_parquet(path, index=False)
    store.upload_or_update(
        path,
        f"{REMOTE_ROOT}/{symbol}.parquet",
        mime_type="application/octet-stream",
    )
    return x


def _local_bulk_file(d: date) -> Path:
    return (
        BULK_EOD_ROOT
        / f"{d:%Y}"
        / f"{d:%m}"
        / f"{d:%d}"
        / f"BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.parquet"
    )


def _validate_day_frame(df: pd.DataFrame, expected_date: date, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "date" not in df.columns:
        raise RuntimeError(
            f"NSE EOD {source} has no date column for expected {expected_date.isoformat()}"
        )
    parsed = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    observed = sorted({ts.date() for ts in parsed.dropna()})
    if not observed:
        raise RuntimeError(
            f"NSE EOD {source} has no valid trade dates for expected {expected_date.isoformat()}"
        )
    if any(x != expected_date for x in observed):
        sample = ",".join(x.isoformat() for x in observed[:5])
        raise RuntimeError(
            f"NSE EOD date mismatch: expected {expected_date.isoformat()} "
            f"but observed {sample}"
        )
    return df


def _day_frame(d: date) -> pd.DataFrame:
    path = _local_bulk_file(d)
    if path.exists() and path.stat().st_size > 0:
        try:
            return _validate_day_frame(
                pd.read_parquet(path),
                d,
                "local-cache",
            )
        except Exception as exc:
            print(
                f"LOCAL_EOD_CACHE_INVALID: {d.isoformat()} "
                f"{type(exc).__name__}: {exc}; refreshing",
                flush=True,
            )
    fresh = fetch_nse_eod_bhavcopy(d, sync_drive=False, force_refresh=True)
    return _validate_day_frame(fresh, d, "NSE-fetch")


def _symbol_parts(df: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
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
        s: g.sort_values("date").drop_duplicates(["symbol", "date"], keep="last")
        for s, g in x.groupby("symbol", sort=False)
    }


def _bootstrap(symbols: list[str], end: date, rows_needed: int) -> dict[str, pd.DataFrame]:
    prior_end = end - timedelta(days=1)
    prior_dates = list(business_days(end - timedelta(days=370), prior_end))[-rows_needed:]
    if len(prior_dates) < rows_needed:
        raise RuntimeError(
            f"Cannot identify {rows_needed} prior NSE trading sessions for Delivery bootstrap."
        )

    collected = {s: [] for s in symbols}
    failures: list[tuple[date, str]] = []

    # Deliberately sequential: fetch_nse_eod_bhavcopy() writes the shared local
    # BULK_EOD_MANIFEST JSON, so concurrent downloads can corrupt it.
    for idx, d in enumerate(prior_dates, start=1):
        try:
            df = _day_frame(d)
            parts = _symbol_parts(df, symbols)
            for s, part in parts.items():
                collected[s].append(part)
            print(
                f"DELIVERY_BOOTSTRAP {idx}/{len(prior_dates)} {d.isoformat()} "
                f"symbols={len(parts)}",
                flush=True,
            )
        except Exception as exc:
            failures.append((d, f"{type(exc).__name__}: {exc}"))
            print(
                f"DELIVERY_BOOTSTRAP_FAILED {d.isoformat()} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        frames = collected[s]
        frame = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame()
        )
        if not frame.empty:
            frame = frame.sort_values("date").drop_duplicates(
                ["symbol", "date"], keep="last"
            )
        out[s] = frame

    if any(len(out[s]) < rows_needed for s in symbols):
        details = "; ".join(
            f"{d.isoformat()}={err}" for d, err in failures[:8]
        )
        raise RuntimeError(
            "Official NSE bootstrap completed with insufficient history: "
            + ", ".join(f"{s}={len(out[s])} rows" for s in symbols)
            + (f". Failed sessions: {details}" if details else "")
        )
    return out


def ensure_delivery_history(symbols: list[str], end: date, min_rows: int = 65) -> dict[str, pd.DataFrame]:
    symbols = [str(s).upper().replace(".NS", "").replace(".BO", "") for s in symbols]
    symbols = list(dict.fromkeys(symbols))
    store = GoogleDriveStore()
    results = {s: _load_compact(store, s, end) for s in symbols}

    need_bootstrap = [s for s, frame in results.items() if len(frame) < min_rows]
    if need_bootstrap:
        boot = _bootstrap(symbols, end, min_rows)
        for s in need_bootstrap:
            if len(boot.get(s, pd.DataFrame())) < min_rows:
                raise RuntimeError(
                    f"Official NSE bootstrap produced only {len(boot.get(s, pd.DataFrame()))} rows for {s}; "
                    f"{min_rows} are required."
                )
            results[s] = _save_compact(store, s, boot[s], end)

    # Daily compilation step: add today's official NSE EOD row to each compact history.
    # If today's archive is not published yet, preserve the existing history and let
    # the Delivery runner block rather than creating a stale prediction.
    today_parts: dict[str, pd.DataFrame] = {}
    try:
        today_parts = _symbol_parts(_day_frame(end), symbols)
    except Exception as exc:
        print(
            f"CURRENT_EOD_NOT_READY: {type(exc).__name__}: {exc}",
            flush=True,
        )

    for s in symbols:
        current = results[s]
        part = today_parts.get(s)
        if part is not None and not part.empty:
            results[s] = _save_compact(
                store,
                s,
                pd.concat([current, part], ignore_index=True),
                end,
            )
        else:
            print(f"COMPILED_HISTORY_NOT_UPDATED: {s} no current NSE EOD row", flush=True)

    return results
