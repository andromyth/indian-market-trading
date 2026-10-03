#!/usr/bin/env python3
"""Resumeable historical NSE F&O backfill: legacy bhavcopy + UDiFF.

Official-source window handled here:
- 2016-01-01 through 2024-07-05: legacy NSE F&O bhavcopy
- 2024-07-08 onward: NSE F&O UDiFF Common Bhavcopy Final

The script:
- probes deterministic per-date archive paths
- stores raw ZIPs persistently in Google Drive
- validates ZIP/CSV structure and expected trade date
- records 404/missing sessions without silently filling them
- normalizes both historical schemas into one contract-level schema
- derives one stock-F&O row per symbol/date
- writes a cumulative research-history parquet without touching the
  production 2026 derived file
- can resume repeatedly; already-valid archives are not re-downloaded
- does not fabricate underlying prices or implied volatility

Primary research output:
  production_data/nse_official/fno/fno_daily_features_history_2016_2026.parquet
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from nse_data_layer import HEADERS

DEFAULT_BASE = (
    "/content/drive/MyDrive/Indian_Market_Research"
    if Path("/content/drive/MyDrive/Indian_Market_Research").exists()
    else "/tmp/Indian_Market_Research"
)
BASE = Path(os.getenv("INDIAN_MARKET_RESEARCH_ROOT", DEFAULT_BASE))

FNO_ROOT = BASE / "production_data/nse_official/fno"
RAW_LEGACY = FNO_ROOT / "raw_legacy"
RAW_UDIFF = FNO_ROOT / "raw_udiff"
HISTORY_DERIVED = FNO_ROOT / "fno_daily_features_history_2016_2026.parquet"
HISTORY_META = FNO_ROOT / "fno_daily_features_history_2016_2026.meta.json"
MISSING_LOG = FNO_ROOT / "historical_backfill_missing_dates_2016_2026.csv"
AUDIT_LOG = FNO_ROOT / "historical_backfill_audit_2016_2026.csv"

LEGACY_CUTOFF = date(2024, 7, 5)
UDIFF_START = date(2024, 7, 8)

LEGACY_PRIMARY = (
    "https://archives.nseindia.com/content/historical/DERIVATIVES/"
    "{year}/{month}/fo{dd}{month}{year}bhav.csv.zip"
)
LEGACY_FALLBACK = (
    "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
    "{year}/{month}/fo{dd}{month}{year}bhav.csv.zip"
)
UDIFF_PRIMARY = (
    "https://archives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)
UDIFF_FALLBACK = (
    "https://nsearchives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)

MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

INDEX_SYMBOLS = {
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
    "NIFTYNXT50", "NIFTYIT", "NIFTYPHARMA", "NIFTYAUTO",
    "NIFTYFMCG", "NIFTYMETAL", "NIFTYREALTY", "NIFTYENERGY",
    "NIFTYINFRA", "NIFTYSERVICES", "NIFTYMEDIA", "NIFTYPSUBANK",
    "NIFTYCOMMODITIES", "NIFTYPVTBANK",
}

ALIASES = {
    "symbol": [
        "tckrsymb", "ticker", "symbol", "underlying", "undrlyg"
    ],
    "instrument": [
        "fininstrmnm", "instrument", "instrumenttype", "instrument_type",
        "fininstrmtp"
    ],
    "instrument_type": [
        "fininstrmtp", "instrumenttype", "instrument_type", "instrument"
    ],
    "expiry": [
        "xprydt", "expiry_dt", "expirydate", "expiry"
    ],
    "option_type": [
        "optn_tp", "optn_tp_cd", "option_typ", "optiontype",
        "option_type", "optn type", "optn_t"
    ],
    "strike_price": [
        "strkpric", "strike_pr", "strikeprice", "strike"
    ],
    "open": ["opnpric", "open"],
    "high": ["hghpric", "high"],
    "low": ["lwpric", "low"],
    "close": ["clspric", "close", "closeprice"],
    "settle": [
        "sttlmpric", "settle_pr", "settleprice", "settlementprice"
    ],
    "underlying_price": [
        "undrlygpric", "underlyingprice", "underlying_price",
        "underlying_value"
    ],
    "volume": [
        "ttltradgvol", "totaltradedvolume", "tottrdqty",
        "contracts", "no_of_contracts", "volume"
    ],
    "turnover": [
        "ttltrfval", "valinlakh", "val_inlakh", "turnover"
    ],
    "oi": [
        "opnintrst", "open_int", "openinterest", "totalopeninterest", "oi"
    ],
    "oi_change": [
        "chnginopnintrst", "chg_in_oi", "changeinopeninterest",
        "oichange", "change_in_oi"
    ],
}

REQUIRED_BASE = {"symbol", "instrument", "expiry", "close"}


def canon(name: object) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def find_col(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    index = {canon(c): c for c in frame.columns}
    for alias in aliases:
        hit = index.get(canon(alias))
        if hit:
            return hit
    return None


def normalize_symbol(value: object) -> str:
    s = str(value or "").strip().upper()
    for suffix in (".NS", ".BO", "-EQ", ".EQ"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def archive_kind(d: date) -> str | None:
    if d <= LEGACY_CUTOFF:
        return "legacy"
    if d >= UDIFF_START:
        return "udiff"
    return None


def local_archive_paths(d: date) -> list[Path]:
    kind = archive_kind(d)
    if kind == "legacy":
        name = f"fo{d:%d}{d:%b}{d:%Y}".upper() + "bhav.csv.zip"
        return [
            RAW_LEGACY / f"{d:%Y}" / f"{d:%m}" / name,
            RAW_LEGACY / f"{d:%Y}" / d.strftime("%b").upper() / name,
            FNO_ROOT / "raw" / f"{d:%Y}" / name,
        ]
    if kind == "udiff":
        names = [
            f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip",
            f"BhavCopy_NSE_FO_{d:%Y%m%d}_F_0000.csv.zip",
        ]
        return [RAW_UDIFF / n for n in names]
    return []


def is_zip(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(4)[:2] == b"PK"
    except OSError:
        return False


def existing_archive(d: date) -> Path | None:
    for path in local_archive_paths(d):
        if is_zip(path):
            return path
    return None


def urls_for(d: date) -> list[str]:
    kind = archive_kind(d)
    if kind == "legacy":
        fmt = {
            "year": f"{d:%Y}",
            "month": d.strftime("%b").upper(),
            "dd": f"{d:%d}",
        }
        return [
            LEGACY_PRIMARY.format(**fmt),
            LEGACY_FALLBACK.format(**fmt),
        ]
    if kind == "udiff":
        key = d.strftime("%Y%m%d")
        return [
            UDIFF_PRIMARY.format(yyyymmdd=key),
            UDIFF_FALLBACK.format(yyyymmdd=key),
        ]
    return []


def read_zip(path_or_bytes: Path | bytes) -> pd.DataFrame:
    raw_bytes = (
        path_or_bytes.read_bytes()
        if isinstance(path_or_bytes, Path)
        else path_or_bytes
    )
    if not raw_bytes.startswith(b"PK"):
        raise RuntimeError("Downloaded file is not a ZIP archive")

    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        csv_members = [n for n in members if n.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError(
                f"No CSV member in archive; members={members[:10]}"
            )
        raw_csv = zf.read(csv_members[0])

    last = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(
                io.BytesIO(raw_csv),
                encoding=enc,
                low_memory=False,
            )
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Unable to parse CSV: {last}")


def validate_raw(raw: pd.DataFrame, d: date) -> None:
    if raw.empty:
        raise RuntimeError("F&O CSV is empty")

    lower = {canon(c): c for c in raw.columns}

    if not (
        any(canon(x) in lower for x in ALIASES["symbol"])
        and any(canon(x) in lower for x in ALIASES["close"])
    ):
        raise RuntimeError(
            f"Missing symbol/close fields; columns={list(raw.columns)[:60]}"
        )

    # UDiFF has TradDt/BizDt; legacy has no consistently named trade-date
    # column, so date validation is only strict when a date field exists.
    for alias in ("traddt", "bizdt", "timestamp", "trade_date", "date"):
        col = lower.get(canon(alias))
        if col:
            vals = pd.to_datetime(raw[col], errors="coerce", dayfirst=False)
            if vals.notna().any():
                observed_dates = set(vals.dropna().dt.normalize())
                target = pd.Timestamp(d).normalize()
                if target not in observed_dates:
                    raise RuntimeError(
                        f"Archive date mismatch: requested={d} "
                        f"observed_sample={sorted(observed_dates)[:5]}"
                    )
                break


def download_day(d: date, session: requests.Session, timeout=(8, 45)) -> Path | None:
    existing = existing_archive(d)
    if existing is not None:
        return existing

    urls = urls_for(d)
    if not urls:
        return None

    candidates = local_archive_paths(d)
    target = candidates[0]
    target.parent.mkdir(parents=True, exist_ok=True)

    last = None
    for url in urls:
        try:
            response = session.get(
                url,
                headers={
                    **HEADERS,
                    "Accept": "application/zip,text/plain,*/*",
                },
                timeout=timeout,
            )
            if response.status_code == 404:
                last = RuntimeError("HTTP 404")
                continue
            response.raise_for_status()
            if not response.content.startswith(b"PK"):
                raise RuntimeError(
                    f"HTTP {response.status_code} response is not a ZIP"
                )
            target.write_bytes(response.content)
            return target
        except Exception as exc:
            last = exc
            time.sleep(0.7)

    raise RuntimeError(f"archive unavailable: {last}")


def normalize_contracts(raw: pd.DataFrame, d: date) -> pd.DataFrame:
    """Normalize legacy and UDiFF schemas without alias-source collisions."""
    targets = {}
    for target, aliases in ALIASES.items():
        col = find_col(raw, aliases)
        if col:
            targets[target] = col

    # Legacy bhavcopy has one INSTRUMENT field (e.g. FUTSTK/OPTSTK).
    # UDiFF has FinInstrmNm plus FinInstrmTp. Never rename one source column
    # twice; that was the cause of the original legacy backfill failure.
    def source(target: str) -> pd.Series:
        col = targets.get(target)
        if col is None:
            return pd.Series(np.nan, index=raw.index, dtype="object")
        return raw[col]

    required_sources = []
    for target in ("symbol", "expiry", "close"):
        if targets.get(target) is None:
            required_sources.append(target)

    if targets.get("instrument") is None and targets.get("instrument_type") is None:
        required_sources.append("instrument")

    if required_sources:
        raise RuntimeError(
            "Normalized archive missing required fields: "
            f"{sorted(set(required_sources))}"
        )

    x = pd.DataFrame(index=raw.index)
    x["symbol"] = source("symbol").astype(str).str.strip().str.upper()
    x["instrument"] = source("instrument").astype(str).str.strip().str.upper()

    # Legacy: instrument_type is the same semantic value as instrument.
    # UDiFF: instrument_type is the compact code such as STF/STO.
    if targets.get("instrument_type") is not None:
        it = source("instrument_type").astype(str).str.strip().str.upper()
        x["instrument_type"] = it
    else:
        x["instrument_type"] = x["instrument"]

    x["expiry"] = pd.to_datetime(
        source("expiry"), errors="coerce", dayfirst=False
    ).dt.normalize()

    x["option_type"] = (
        source("option_type").astype(str).str.strip().str.upper()
    )

    for col in (
        "strike_price", "open", "high", "low", "close", "settle",
        "underlying_price", "volume", "turnover", "oi", "oi_change",
    ):
        s = source(col)
        x[col] = pd.to_numeric(
            s.astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

    x["date"] = pd.Timestamp(d).normalize()

    # Legacy futures use OPTION_TYP=XX; retain that semantic explicitly.
    # For UDiFF futures option_type may already be FF/FUT/blank.
    legacy_xx = x["option_type"].eq("XX")
    x.loc[legacy_xx, "option_type"] = "FUT"

    return x[
        [
            "symbol", "date", "instrument", "instrument_type", "expiry",
            "option_type", "strike_price", "open", "high", "low",
            "close", "settle", "underlying_price", "volume", "turnover",
            "oi", "oi_change",
        ]
    ].copy()

def is_future(frame: pd.DataFrame) -> pd.Series:
    instrument = frame["instrument"].astype(str).str.upper()
    type_code = frame["instrument_type"].astype(str).str.upper()
    return (
        instrument.str.contains("FUT", na=False)
        | type_code.str.contains("FUT", na=False)
    )


def stock_contracts(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x = x[x["symbol"].ne("")]
    x = x[~x["symbol"].isin(INDEX_SYMBOLS)]
    x = x[~x["symbol"].str.contains(
        "NIFTY|SENSEX|BANKEX", regex=True, na=False
    )]
    return x.copy()


def normal_cdf(x: float) -> float:
    return 0.5 * (
        1.0 + math.erf(float(x) / math.sqrt(2.0))
    )


def black76_price(
    forward: float,
    strike: float,
    t_years: float,
    sigma: float,
    is_call: bool,
) -> float:
    if not all(
        np.isfinite(v)
        for v in (forward, strike, t_years, sigma)
    ) or forward <= 0 or strike <= 0 or t_years <= 0 or sigma <= 0:
        return np.nan

    v = sigma * math.sqrt(t_years)
    d1 = (
        math.log(forward / strike)
        + 0.5 * sigma * sigma * t_years
    ) / v
    d2 = d1 - v

    if is_call:
        return forward * normal_cdf(d1) - strike * normal_cdf(d2)
    return strike * normal_cdf(-d2) - forward * normal_cdf(-d1)


def implied_vol_black76(
    price: float,
    forward: float,
    strike: float,
    t_years: float,
    is_call: bool,
) -> float:
    if not all(
        np.isfinite(v)
        for v in (price, forward, strike, t_years)
    ) or price <= 0 or forward <= 0 or strike <= 0 or t_years <= 0:
        return np.nan

    intrinsic = (
        max(forward - strike, 0.0)
        if is_call
        else max(strike - forward, 0.0)
    )
    upper = forward if is_call else strike
    if price <= intrinsic + 1e-10 or price >= upper - 1e-10:
        return np.nan

    lo, hi = 1e-4, 5.0
    hi_price = black76_price(
        forward, strike, t_years, hi, is_call
    )
    while (
        np.isfinite(hi_price)
        and hi_price < price
        and hi < 10.0
    ):
        hi *= 2.0
        hi_price = black76_price(
            forward, strike, t_years, hi, is_call
        )

    if not np.isfinite(hi_price) or hi_price < price:
        return np.nan

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = black76_price(
            forward, strike, t_years, mid, is_call
        )
        if not np.isfinite(val):
            return np.nan
        if val < price:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def atm_iv(
    options: pd.DataFrame,
    future_row: pd.Series | None,
    d: pd.Timestamp,
) -> float:
    if options.empty:
        return np.nan

    og = options.copy()
    og = og.dropna(subset=["strike_price", "expiry"])
    og = og[
        (og["expiry"] > d.normalize())
        & (og["strike_price"] > 0)
    ]
    if og.empty:
        return np.nan

    expiry = og["expiry"].min()
    og = og[og["expiry"].eq(expiry)].copy()

    forward = np.nan
    if future_row is not None:
        forward = pd.to_numeric(
            pd.Series([future_row.get("close", np.nan)]),
            errors="coerce",
        ).iloc[0]
        if not np.isfinite(forward):
            forward = pd.to_numeric(
                pd.Series([future_row.get("settle", np.nan)]),
                errors="coerce",
            ).iloc[0]

    if not np.isfinite(forward):
        uv = pd.to_numeric(
            og["underlying_price"],
            errors="coerce",
        )
        if uv.notna().any():
            forward = float(uv.dropna().median())

    if not np.isfinite(forward) or forward <= 0:
        return np.nan

    og["atm_distance"] = (
        og["strike_price"] / float(forward) - 1.0
    ).abs()
    near = og[og["atm_distance"] <= 0.05].copy()
    if near.empty:
        near = og.nsmallest(min(12, len(og)), "atm_distance")

    t_years = max((expiry - d.normalize()).days, 1) / 365.0
    ivs = []

    for row in near.itertuples(index=False):
        opt_type = str(
            getattr(row, "option_type", "")
        ).upper()
        if opt_type not in {"CE", "PE", "CALL", "PUT"}:
            continue

        price = getattr(row, "close", np.nan)
        if not np.isfinite(price) or price <= 0:
            price = getattr(row, "settle", np.nan)

        iv = implied_vol_black76(
            float(price),
            float(forward),
            float(getattr(row, "strike_price")),
            t_years,
            opt_type in {"CE", "CALL"},
        )
        if np.isfinite(iv) and 0.0001 <= iv <= 5.0:
            ivs.append(iv)

    return float(np.median(ivs)) if ivs else np.nan


def build_day_features(contracts: pd.DataFrame) -> pd.DataFrame:
    x = stock_contracts(contracts)
    if x.empty:
        return pd.DataFrame()

    futures = x[is_future(x)].copy()
    futures = futures.dropna(subset=["expiry"])
    futures = futures[futures["expiry"] >= futures["date"]]
    futures = (
        futures.sort_values(["symbol", "expiry", "volume"], ascending=[True, True, False])
        .drop_duplicates("symbol", keep="first")
    )

    options = x[
        x["option_type"].isin(["CE", "PE", "CALL", "PUT"])
    ].copy()

    rows = []
    for symbol in sorted(x["symbol"].unique()):
        g = x[x["symbol"].eq(symbol)]
        row = {
            "symbol": symbol,
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
            "fno_atm_iv": np.nan,
        }

        og = options[options["symbol"].eq(symbol)]
        if not og.empty:
            calls = og[og["option_type"].isin(["CE", "CALL"])]
            puts = og[og["option_type"].isin(["PE", "PUT"])]

            call_oi = pd.to_numeric(
                calls["oi"], errors="coerce"
            ).sum(min_count=1)
            put_oi = pd.to_numeric(
                puts["oi"], errors="coerce"
            ).sum(min_count=1)

            if np.isfinite(call_oi) and call_oi != 0 and np.isfinite(put_oi):
                row["fno_pcr"] = float(put_oi / call_oi)

            oi_change = pd.to_numeric(
                og["oi_change"], errors="coerce"
            ).sum(min_count=1)
            if np.isfinite(oi_change):
                row["fno_oi_change"] = float(oi_change)

        fg = futures[futures["symbol"].eq(symbol)]
        future_row = fg.iloc[0] if not fg.empty else None

        if future_row is not None:
            fut_close = future_row.get("close", np.nan)
            if not np.isfinite(fut_close):
                fut_close = future_row.get("settle", np.nan)

            row["fno_fut_close"] = fut_close
            row["fno_fut_oi"] = future_row.get("oi", np.nan)
            row["fno_fut_expiry"] = future_row.get("expiry", pd.NaT)

            underlying = future_row.get("underlying_price", np.nan)
            if (
                np.isfinite(fut_close)
                and np.isfinite(underlying)
                and underlying != 0
            ):
                row["fno_fut_basis"] = float(
                    fut_close / underlying - 1.0
                )

        row["fno_atm_iv"] = atm_iv(
            og,
            future_row,
            pd.Timestamp(g["date"].iloc[0]),
        )
        rows.append(row)

    return pd.DataFrame(rows)


def read_missing() -> pd.DataFrame:
    if not MISSING_LOG.exists() or MISSING_LOG.stat().st_size == 0:
        return pd.DataFrame(
            columns=["date", "kind", "reason", "attempted_at"]
        )
    return pd.read_csv(MISSING_LOG, dtype=str)


def append_missing(d: date, kind: str, reason: str) -> None:
    old = read_missing()
    row = pd.DataFrame([{
        "date": d.isoformat(),
        "kind": kind,
        "reason": reason,
        "attempted_at": datetime.now(timezone.utc).isoformat(),
    }])
    out = pd.concat([old, row], ignore_index=True)
    out = out.drop_duplicates(["date", "kind"], keep="last")
    out.to_csv(MISSING_LOG, index=False)


def persist_history(new_rows: pd.DataFrame) -> None:
    if new_rows is None or new_rows.empty:
        return

    if HISTORY_DERIVED.exists() and HISTORY_DERIVED.stat().st_size > 0:
        old = pd.read_parquet(HISTORY_DERIVED)
        combined = pd.concat(
            [old, new_rows],
            ignore_index=True,
        )
    else:
        combined = new_rows.copy()

    combined["symbol"] = combined["symbol"].map(normalize_symbol)
    combined["date"] = pd.to_datetime(
        combined["date"], errors="coerce"
    ).dt.normalize()

    combined = (
        combined.dropna(subset=["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    combined["fno_fut_close"] = pd.to_numeric(
        combined["fno_fut_close"], errors="coerce"
    )
    combined["fno_fut_oi"] = pd.to_numeric(
        combined["fno_fut_oi"], errors="coerce"
    )
    combined["fno_atm_iv"] = pd.to_numeric(
        combined["fno_atm_iv"], errors="coerce"
    )

    combined["fno_fut_return"] = combined.groupby(
        "symbol"
    )["fno_fut_close"].pct_change(fill_method=None)

    combined["fno_iv_change"] = combined.groupby(
        "symbol"
    )["fno_atm_iv"].diff()

    HISTORY_DERIVED.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(HISTORY_DERIVED, index=False)

    meta = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(combined)),
        "symbols": int(combined["symbol"].nunique()),
        "dates": int(combined["date"].nunique()),
        "min_date": str(combined["date"].min().date()),
        "max_date": str(combined["date"].max().date()),
        "source": "NSE_OFFICIAL_LEGACY_AND_UDIFF",
        "legacy_window": f"2016-01-01 through {LEGACY_CUTOFF.isoformat()}",
        "udiff_window": f"{UDIFF_START.isoformat()} onward",
        "iv_method": "ATM option IV via Black-76 using nearest non-expired option expiry; futures close/settlement preferred as forward",
        "basis_method": "NSE underlying_price where available; otherwise left NaN for PIT-safe equity-close fallback during research merge",
        "raw_legacy_root": str(RAW_LEGACY),
        "raw_udiff_root": str(RAW_UDIFF),
    }
    HISTORY_META.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def audit_row(d: date, kind: str, path: Path, contracts: pd.DataFrame, features: pd.DataFrame) -> dict:
    return {
        "date": d.isoformat(),
        "kind": kind,
        "archive": str(path),
        "contracts": int(len(contracts)),
        "stock_symbols": int(contracts.loc[
            contracts["symbol"].isin(features["symbol"]) if not features.empty else pd.Series(False, index=contracts.index),
            "symbol"
        ].nunique()) if not features.empty else 0,
        "feature_rows": int(len(features)),
        "feature_symbols": int(features["symbol"].nunique()) if not features.empty else 0,
        "atm_iv_rows": int(features["fno_atm_iv"].notna().sum()) if not features.empty else 0,
        "future_rows": int(features["fno_fut_close"].notna().sum()) if not features.empty else 0,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


def append_audit(rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if AUDIT_LOG.exists() and AUDIT_LOG.stat().st_size > 0:
        old = pd.read_csv(AUDIT_LOG)
        frame = pd.concat([old, frame], ignore_index=True)
    frame = frame.drop_duplicates(["date", "kind"], keep="last")
    frame.to_csv(AUDIT_LOG, index=False)


def date_list(start: date, end: date) -> list[date]:
    out = []
    current = start
    while current <= end:
        if current.weekday() < 5 and archive_kind(current) is not None:
            out.append(current)
        current += timedelta(days=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument(
        "--end",
        default=str(min(date.today(), date(2026, 10, 1))),
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=0,
        help="Maximum weekdays to process in this run; 0 means all requested dates.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.45,
        help="Seconds between successful date requests.",
    )
    parser.add_argument(
        "--rebuild-derived",
        action="store_true",
        help="Recompute history-derived output from all valid raw archives already on disk; does not download.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip dates already present in the history-derived parquet.",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date()

    session = requests.Session()
    session.headers.update(HEADERS)

    if args.rebuild_derived:
        if not HISTORY_DERIVED.exists():
            raise SystemExit(
                f"Cannot rebuild: {HISTORY_DERIVED} does not exist."
            )
        # Rebuild is deliberately limited to already-cached raw archives.
        rows = []
        missing = []
        for d in date_list(start, end):
            path = existing_archive(d)
            if path is None:
                missing.append(d)
                continue
            try:
                raw = read_zip(path)
                validate_raw(raw, d)
                contracts = normalize_contracts(raw, d)
                features = build_day_features(contracts)
                if not features.empty:
                    rows.append(features)
            except Exception as exc:
                print(
                    f"{d}: rebuild error {type(exc).__name__}: {exc}",
                    flush=True,
                )
        if rows:
            persist_history(pd.concat(rows, ignore_index=True))
        print(
            f"REBUILD COMPLETE: cached_dates={len(rows):,} "
            f"missing_dates={len(missing):,}",
            flush=True,
        )
        return

    existing_history_dates = set()
    if args.skip_existing and HISTORY_DERIVED.exists():
        try:
            h = pd.read_parquet(HISTORY_DERIVED, columns=["date"])
            existing_history_dates = set(
                pd.to_datetime(h["date"], errors="coerce")
                .dt.date.dropna().unique()
            )
        except Exception:
            existing_history_dates = set()

    dates = date_list(start, end)
    if args.max_days > 0:
        dates = dates[:args.max_days]

    print("=" * 100)
    print("NSE F&O HISTORICAL BACKFILL")
    print("=" * 100)
    print(f"REQUESTED WINDOW: {start} -> {end}")
    print(f"DATES IN RUN: {len(dates):,}")
    print(f"LEGACY CUTOFF: {LEGACY_CUTOFF}")
    print(f"UDIFF START: {UDIFF_START}")
    print(f"RAW LEGACY: {RAW_LEGACY}")
    print(f"RAW UDIFF: {RAW_UDIFF}")
    print(f"DERIVED HISTORY: {HISTORY_DERIVED}")
    print("=" * 100)

    feature_batches = []
    audits = []
    processed = 0
    skipped_existing = 0
    failed = 0

    for idx, d in enumerate(dates, start=1):
        kind = archive_kind(d)
        if args.skip_existing and d in existing_history_dates:
            skipped_existing += 1
            continue

        try:
            path = existing_archive(d)
            if path is None:
                path = download_day(d, session)

            if path is None:
                append_missing(d, str(kind), "unsupported_transition_date")
                failed += 1
                continue

            raw = read_zip(path)
            validate_raw(raw, d)
            contracts = normalize_contracts(raw, d)
            features = build_day_features(contracts)

            if features.empty:
                append_missing(
                    d, str(kind), "archive_valid_but_no_stock_fno_features"
                )
                failed += 1
                continue

            feature_batches.append(features)
            audits.append(
                audit_row(d, str(kind), path, contracts, features)
            )
            processed += 1

            print(
                f"[{idx:,}/{len(dates):,}] {d} {kind.upper()}: "
                f"contracts={len(contracts):,} "
                f"symbols={features['symbol'].nunique():,} "
                f"ATM_IV={features['fno_atm_iv'].notna().sum():,}",
                flush=True,
            )

            if len(feature_batches) >= 10:
                persist_history(
                    pd.concat(feature_batches, ignore_index=True)
                )
                append_audit(audits)
                feature_batches = []
                audits = []

            time.sleep(max(0.0, args.sleep))

        except Exception as exc:
            failed += 1
            append_missing(
                d,
                str(kind),
                f"{type(exc).__name__}: {exc}",
            )
            print(
                f"[{idx:,}/{len(dates):,}] {d} {kind.upper()}: "
                f"FAILED {type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(0.6)

    if feature_batches:
        persist_history(pd.concat(feature_batches, ignore_index=True))
    append_audit(audits)

    print("=" * 100)
    print("NSE F&O HISTORICAL BACKFILL COMPLETE")
    print("=" * 100)
    print(f"PROCESSED DATES: {processed:,}")
    print(f"SKIPPED EXISTING: {skipped_existing:,}")
    print(f"FAILED/MISSING: {failed:,}")
    print(f"OUTPUT: {HISTORY_DERIVED}")
    print(f"MISSING LOG: {MISSING_LOG}")
    print(f"AUDIT LOG: {AUDIT_LOG}")
    print("=" * 100)


if __name__ == "__main__":
    main()
