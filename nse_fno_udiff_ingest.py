#!/usr/bin/env python3
"""Download and normalize the official NSE F&O UDiFF Common Bhavcopy Final."""
from __future__ import annotations

import argparse
import io
import json
import os
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from nse_data_layer import HEADERS
from persistent_store import GoogleDriveStore

BASE = Path(os.getenv("INDIAN_MARKET_RESEARCH_ROOT", "/tmp/Indian_Market_Research"))
RAW_BASE = BASE / "production_data/nse_official/fno/raw"
DERIVED = BASE / "production_data/nse_official/fno/fno_daily_features_2026.parquet"
DERIVED_META = BASE / "production_data/nse_official/fno/fno_daily_features_2026.meta.json"

PRIMARY = "https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"
FALLBACK = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"

ALIASES = {
    "symbol": ["tckrsymb", "ticker", "symbol", "underlying", "undrlyg"],
    "instrument": ["fininstrmtp", "fininstrmnm", "instrumenttype", "instrument"],
    "expiry": ["xprydt", "expirydate", "expiry"],
    "option_type": ["optn_tp", "optn_tp_cd", "optiontype", "optn type", "optn_t"],
    "close": ["clspric", "close", "closeprice"],
    "settle": ["sttlmpric", "settleprice", "settlementprice"],
    "underlying_price": ["undrlygpric", "underlyingprice", "underlying_price"],
    "volume": ["ttltradgvol", "totaltradedvolume", "tottrdqty", "volume"],
    "oi": ["ttloprnintrst", "totalopeninterest", "openinterest", "oi"],
    "oi_change": ["chnginopnintrst", "changeinopeninterest", "oichange"],
}
INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}

def _canon(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())

def _find_col(df: pd.DataFrame, names: list[str]) -> str | None:
    norm = {_canon(c): c for c in df.columns}
    for name in names:
        hit = norm.get(_canon(name))
        if hit:
            return hit
    return None

def _read_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        if not members:
            raise RuntimeError("NSE F&O UDiFF archive is empty")
        member = next((n for n in members if n.lower().endswith(".csv")), members[0])
        raw = zf.read(member)
    last = None
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Unable to parse NSE F&O UDiFF CSV: {last}")

def _download_day(d: date, force: bool = False) -> tuple[Path, pd.DataFrame]:
    key = d.strftime("%Y%m%d")
    out_dir = RAW_BASE / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"BhavCopy_NSE_FO_0_0_0_{key}_F_0000.csv.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0 and not force:
        return zip_path, _read_zip(zip_path.read_bytes())
    last = None
    for url in (PRIMARY.format(date=key), FALLBACK.format(date=key)):
        try:
            r = requests.get(
                url,
                headers={**HEADERS, "Accept": "application/zip,text/plain,*/*"},
                timeout=(5, 30),
            )
            r.raise_for_status()
            if not r.content.startswith(b"PK"):
                raise RuntimeError("NSE response is not a ZIP archive")
            zip_path.write_bytes(r.content)
            return zip_path, _read_zip(r.content)
        except Exception as exc:
            last = exc
            time.sleep(1.5)
    raise RuntimeError(f"F&O UDiFF unavailable for {key}: {last}")

def _normalize(raw: pd.DataFrame, d: date) -> pd.DataFrame:
    cols = {}
    for target, aliases in ALIASES.items():
        c = _find_col(raw, aliases)
        if c:
            cols[target] = c
    if "symbol" not in cols or "close" not in cols:
        raise RuntimeError(f"F&O UDiFF lacks required symbol/close fields: {list(raw.columns)[:80]}")
    x = raw.rename(columns={v: k for k, v in cols.items()}).copy()
    x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
    x["instrument"] = x.get("instrument", pd.Series("", index=x.index)).astype(str).str.upper()
    x["option_type"] = x.get("option_type", pd.Series("", index=x.index)).astype(str).str.upper()
    x["expiry"] = pd.to_datetime(
        x.get("expiry", pd.Series(pd.NaT, index=x.index)), errors="coerce"
    ).dt.normalize()
    for c in ("close", "settle", "underlying_price", "volume", "oi", "oi_change"):
        if c in x.columns:
            x[c] = pd.to_numeric(
                x[c].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
        else:
            x[c] = np.nan
    x["date"] = pd.Timestamp(d).normalize()
    return x

def _is_future(series: pd.Series) -> pd.Series:
    return series.str.contains("FUT", na=False)

def _is_stock_symbol(series: pd.Series) -> pd.Series:
    return (
        series.ne("")
        & ~series.isin(INDEX_SYMBOLS)
        & ~series.str.contains("NIFTY|SENSEX", regex=True, na=False)
    )

def _build_features(day: pd.DataFrame) -> pd.DataFrame:
    x = day[_is_stock_symbol(day["symbol"])].copy()
    if x.empty:
        return pd.DataFrame()
    futures = x[_is_future(x["instrument"])].copy()
    futures = futures.dropna(subset=["expiry"])
    futures = futures[futures["expiry"] >= futures["date"]]
    futures = futures.sort_values(["symbol", "expiry"]).drop_duplicates("symbol", keep="first")
    options = x[x["option_type"].isin(["CE", "PE", "CALL", "PUT"])].copy()
    rows = []
    for sym in sorted(x["symbol"].unique()):
        g = x[x["symbol"].eq(sym)]
        row = {
            "symbol": sym,
            "date": g["date"].iloc[0],
            "fno_available": 1.0,
            "fno_pcr": np.nan,
            "fno_oi_change": np.nan,
            "fno_iv_change": np.nan,
            "fno_fut_close": np.nan,
            "fno_fut_basis": np.nan,
            "fno_fut_return": np.nan,
            "fno_fut_oi": np.nan,
            "fno_fut_expiry": pd.NaT,
        }
        og = options[options["symbol"].eq(sym)]
        if not og.empty:
            calls = og[og["option_type"].isin(["CE", "CALL"])]
            puts = og[og["option_type"].isin(["PE", "PUT"])]
            call_oi = pd.to_numeric(calls["oi"], errors="coerce").sum(min_count=1)
            put_oi = pd.to_numeric(puts["oi"], errors="coerce").sum(min_count=1)
            row["fno_pcr"] = (
                float(put_oi / call_oi)
                if np.isfinite(call_oi) and call_oi != 0 and np.isfinite(put_oi)
                else np.nan
            )
            oi_change = pd.to_numeric(og["oi_change"], errors="coerce").sum(min_count=1)
            row["fno_oi_change"] = float(oi_change) if np.isfinite(oi_change) else np.nan
        fg = futures[futures["symbol"].eq(sym)]
        if not fg.empty:
            r = fg.iloc[0]
            fut_close = r.get("close", np.nan)
            if not np.isfinite(fut_close):
                fut_close = r.get("settle", np.nan)
            row["fno_fut_close"] = fut_close
            row["fno_fut_oi"] = r.get("oi", np.nan)
            row["fno_fut_expiry"] = r.get("expiry", pd.NaT)
            under = r.get("underlying_price", np.nan)
            if np.isfinite(fut_close) and np.isfinite(under) and under != 0:
                row["fno_fut_basis"] = float(fut_close / under - 1.0)
        rows.append(row)
    return pd.DataFrame(rows)

def _drive_download(remote: str, local: Path) -> bool:
    try:
        store = GoogleDriveStore()
        parts = remote.strip("/").split("/")
        parent = store.root_id
        for part in parts[:-1]:
            found = store._find_child(parent, part)
            if not found:
                return False
            parent = found["id"]
        found = store._find_child(parent, parts[-1])
        if not found:
            return False
        from googleapiclient.http import MediaIoBaseDownload
        req = store.service.files().get_media(fileId=found["id"])
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            dl = MediaIoBaseDownload(fh, req)
            done = False
            while not done:
                _, done = dl.next_chunk()
        tmp.replace(local)
        return True
    except Exception:
        return False

def _merge_and_persist(day_features: pd.DataFrame) -> None:
    DERIVED.parent.mkdir(parents=True, exist_ok=True)
    if not DERIVED.exists() or DERIVED.stat().st_size == 0:
        _drive_download(
            "production_data/nse_official/fno/fno_daily_features_2026.parquet",
            DERIVED,
        )
    old = pd.read_parquet(DERIVED) if DERIVED.exists() and DERIVED.stat().st_size > 0 else pd.DataFrame()
    all_rows = pd.concat([old, day_features], ignore_index=True) if not old.empty else day_features.copy()
    all_rows["symbol"] = all_rows["symbol"].astype(str).str.upper()
    all_rows["date"] = pd.to_datetime(all_rows["date"], errors="coerce").dt.normalize()
    all_rows = (
        all_rows.drop_duplicates(["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    all_rows["fno_fut_return"] = all_rows.groupby("symbol")["fno_fut_close"].pct_change(fill_method=None)
    all_rows.to_parquet(DERIVED, index=False)
    meta = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(all_rows)),
        "symbols": int(all_rows["symbol"].nunique()),
        "min_date": str(all_rows["date"].min().date()),
        "max_date": str(all_rows["date"].max().date()),
        "source": "NSE_OFFICIAL_FNO_UDIFF",
    }
    DERIVED_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    GoogleDriveStore().upload_or_update(DERIVED, "production_data/nse_official/fno/fno_daily_features_2026.parquet")
    GoogleDriveStore().upload_or_update(
        DERIVED_META,
        "production_data/nse_official/fno/fno_daily_features_2026.meta.json",
        mime_type="application/json",
    )

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="")
    ap.add_argument("--days-back", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.date:
        days = [pd.Timestamp(args.date).date()]
    else:
        d = datetime.now().date()
        days = []
        while len(days) < max(1, args.days_back):
            if d.weekday() < 5:
                days.append(d)
            d -= timedelta(days=1)
    for d in sorted(set(days)):
        try:
            zip_path, raw = _download_day(d, force=args.force)
            features = _build_features(_normalize(raw, d))
            if features.empty:
                print(f"{d}: no stock F&O rows in UDiFF archive", flush=True)
                continue
            GoogleDriveStore().upload_or_update(
                zip_path,
                f"production_data/nse_official/fno/raw/{d:%Y/%m/%d}/{zip_path.name}",
                mime_type="application/zip",
            )
            _merge_and_persist(features)
            print(
                f"{d}: F&O sync rows={len(features):,} symbols={features['symbol'].nunique():,}",
                flush=True,
            )
        except Exception as exc:
            print(f"{d}: F&O sync error {type(exc).__name__}: {exc}", flush=True)

if __name__ == "__main__":
    main()
