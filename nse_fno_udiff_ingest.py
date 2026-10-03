#!/usr/bin/env python3
"""Download and normalize the official NSE F&O UDiFF Common Bhavcopy Final."""
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
from persistent_store import GoogleDriveStore

DEFAULT_BASE = (
    "/content/drive/MyDrive/Indian_Market_Research"
    if Path("/content/drive/MyDrive/Indian_Market_Research").exists()
    else "/tmp/Indian_Market_Research"
)
BASE = Path(os.getenv("INDIAN_MARKET_RESEARCH_ROOT", DEFAULT_BASE))

# Canonical persisted location already present in the project/Drive.
UDIFF_RAW_BASE = BASE / "production_data/nse_official/fno/raw_udiff"
# Compatibility cache for files downloaded by older versions of this script.
LEGACY_RAW_BASE = BASE / "production_data/nse_official/fno/raw"
RAW_BASE = UDIFF_RAW_BASE

DERIVED = BASE / "production_data/nse_official/fno/fno_daily_features_2026.parquet"
DERIVED_META = BASE / "production_data/nse_official/fno/fno_daily_features_2026.meta.json"

PRIMARY = "https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"
FALLBACK = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"

ALIASES = {
    "symbol": ["tckrsymb", "ticker", "symbol", "underlying", "undrlyg"],
    "instrument": ["fininstrmtp", "fininstrmnm", "instrumenttype", "instrument"],
    "expiry": ["xprydt", "expirydate", "expiry"],
    "option_type": ["optn_tp", "optn_tp_cd", "optiontype", "optn type", "optn_t"],
    "strike_price": ["strkpric", "strikeprice", "strike"],
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


def _udiff_local_path(d: date) -> Path:
    UDIFF_RAW_BASE.mkdir(parents=True, exist_ok=True)
    key = d.strftime("%Y%m%d")
    return UDIFF_RAW_BASE / f"BhavCopy_NSE_FO_0_0_0_{key}_F_0000.csv.zip"


def _legacy_partition_path(d: date) -> Path:
    out_dir = LEGACY_RAW_BASE / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}"
    return out_dir / f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"


def _existing_local_archive(d: date) -> Path | None:
    canonical = _udiff_local_path(d)
    if canonical.exists() and canonical.stat().st_size > 0:
        return canonical
    legacy = _legacy_partition_path(d)
    if legacy.exists() and legacy.stat().st_size > 0:
        return legacy
    return None


def _download_day(d: date, force: bool = False) -> tuple[Path, pd.DataFrame]:
    key = d.strftime("%Y%m%d")
    zip_path = _udiff_local_path(d)
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
        raise RuntimeError(
            "F&O UDiFF lacks required symbol/close fields: "
            f"{list(raw.columns)[:80]}"
        )
    x = raw.rename(columns={v: k for k, v in cols.items()}).copy()
    x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
    x["instrument"] = x.get("instrument", pd.Series("", index=x.index)).astype(str).str.upper()
    x["option_type"] = x.get("option_type", pd.Series("", index=x.index)).astype(str).str.upper()
    x["expiry"] = pd.to_datetime(
        x.get("expiry", pd.Series(pd.NaT, index=x.index)), errors="coerce"
    ).dt.normalize()
    for c in (
        "close",
        "settle",
        "underlying_price",
        "strike_price",
        "volume",
        "oi",
        "oi_change",
    ):
        if c in x.columns:
            x[c] = pd.to_numeric(
                x[c].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
        else:
            x[c] = np.nan
    x["date"] = pd.Timestamp(d).normalize()
    return x


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _black76_price(forward: float, strike: float, t_years: float, sigma: float, is_call: bool) -> float:
    if not all(
        np.isfinite(v)
        for v in (forward, strike, t_years, sigma)
    ) or forward <= 0 or strike <= 0 or t_years <= 0 or sigma <= 0:
        return np.nan
    vol_sqrt_t = sigma * math.sqrt(t_years)
    d1 = (
        math.log(forward / strike)
        + 0.5 * sigma * sigma * t_years
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if is_call:
        return forward * _normal_cdf(d1) - strike * _normal_cdf(d2)
    return strike * _normal_cdf(-d2) - forward * _normal_cdf(-d1)


def _implied_vol_black76(
    price: float,
    forward: float,
    strike: float,
    t_years: float,
    is_call: bool,
) -> float:
    """Return Black-76 IV; invalid/illiquid prices remain NaN."""
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
    hi_price = _black76_price(
        forward, strike, t_years, hi, is_call
    )
    while (
        np.isfinite(hi_price)
        and hi_price < price
        and hi < 10.0
    ):
        hi *= 2.0
        hi_price = _black76_price(
            forward, strike, t_years, hi, is_call
        )
    if not np.isfinite(hi_price) or hi_price < price:
        return np.nan

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = _black76_price(
            forward, strike, t_years, mid, is_call
        )
        if not np.isfinite(val):
            return np.nan
        if val < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _atm_iv(
    options: pd.DataFrame,
    futures_row: pd.Series | None,
    day_date: pd.Timestamp,
) -> float:
    if options.empty or "strike_price" not in options.columns:
        return np.nan

    og = options.copy()
    og["strike_price"] = pd.to_numeric(
        og["strike_price"], errors="coerce"
    )
    og["expiry"] = pd.to_datetime(
        og["expiry"], errors="coerce"
    ).dt.normalize()
    og["close"] = pd.to_numeric(
        og["close"], errors="coerce"
    )
    if "settle" in og.columns:
        og["settle"] = pd.to_numeric(
            og["settle"], errors="coerce"
        )
    og = og.dropna(subset=["strike_price", "expiry"])
    og = og[
        (og["expiry"] > day_date.normalize())
        & (og["strike_price"] > 0)
    ]
    if og.empty:
        return np.nan

    near_expiry = og["expiry"].min()
    og = og[og["expiry"].eq(near_expiry)].copy()

    forward = np.nan
    if futures_row is not None and not futures_row.empty:
        forward = pd.to_numeric(
            pd.Series([futures_row.get("close", np.nan)]),
            errors="coerce",
        ).iloc[0]
        if not np.isfinite(forward):
            forward = pd.to_numeric(
                pd.Series([futures_row.get("settle", np.nan)]),
                errors="coerce",
            ).iloc[0]

    if not np.isfinite(forward):
        uv = pd.to_numeric(
            og.get(
                "underlying_price",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
        if len(uv):
            forward = (
                float(uv.dropna().median())
                if uv.notna().any()
                else np.nan
            )

    if not np.isfinite(forward) or forward <= 0:
        return np.nan

    og["atm_distance"] = (
        og["strike_price"] / float(forward) - 1.0
    ).abs()
    near = og[og["atm_distance"] <= 0.05].copy()
    if near.empty:
        near = og.nsmallest(
            min(12, len(og)),
            "atm_distance",
        ).copy()

    ivs = []
    t_years = max(
        (near_expiry - day_date.normalize()).days,
        1,
    ) / 365.0

    for rr in near.itertuples(index=False):
        option_type = str(
            getattr(rr, "option_type", "")
        ).upper()
        is_call = option_type in {"CE", "CALL"}
        if option_type not in {"CE", "PE", "CALL", "PUT"}:
            continue

        px = getattr(rr, "close", np.nan)
        if not np.isfinite(px) or px <= 0:
            px = getattr(rr, "settle", np.nan)

        iv = _implied_vol_black76(
            float(px),
            float(forward),
            float(getattr(rr, "strike_price")),
            t_years,
            is_call,
        )
        if np.isfinite(iv) and 0.0001 <= iv <= 5.0:
            ivs.append(iv)

    return float(np.median(ivs)) if ivs else np.nan


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
    futures = (
        futures.sort_values(["symbol", "expiry"])
        .drop_duplicates("symbol", keep="first")
    )

    options = x[
        x["option_type"].isin(["CE", "PE", "CALL", "PUT"])
    ].copy()

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
            "fno_atm_iv": np.nan,
        }

        og = options[options["symbol"].eq(sym)]
        if not og.empty:
            calls = og[
                og["option_type"].isin(["CE", "CALL"])
            ]
            puts = og[
                og["option_type"].isin(["PE", "PUT"])
            ]
            call_oi = pd.to_numeric(
                calls["oi"], errors="coerce"
            ).sum(min_count=1)
            put_oi = pd.to_numeric(
                puts["oi"], errors="coerce"
            ).sum(min_count=1)

            row["fno_pcr"] = (
                float(put_oi / call_oi)
                if np.isfinite(call_oi)
                and call_oi != 0
                and np.isfinite(put_oi)
                else np.nan
            )

            oi_change = pd.to_numeric(
                og["oi_change"], errors="coerce"
            ).sum(min_count=1)
            row["fno_oi_change"] = (
                float(oi_change)
                if np.isfinite(oi_change)
                else np.nan
            )

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
            if (
                np.isfinite(fut_close)
                and np.isfinite(under)
                and under != 0
            ):
                row["fno_fut_basis"] = (
                    float(fut_close / under - 1.0)
                )

        future_row = (
            fg.iloc[0]
            if not fg.empty
            else None
        )
        row["fno_atm_iv"] = _atm_iv(
            og,
            future_row,
            pd.Timestamp(g["date"].iloc[0]),
        )
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

        req = store.service.files().get_media(
            fileId=found["id"]
        )
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


def _persist_derived_rows(rows: pd.DataFrame) -> None:
    DERIVED.parent.mkdir(parents=True, exist_ok=True)

    all_rows = rows.copy()
    all_rows["symbol"] = all_rows["symbol"].astype(str).str.upper()
    all_rows["date"] = pd.to_datetime(
        all_rows["date"], errors="coerce"
    ).dt.normalize()

    all_rows = (
        all_rows.dropna(subset=["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    all_rows["fno_fut_return"] = all_rows.groupby(
        "symbol"
    )["fno_fut_close"].pct_change(fill_method=None)

    if "fno_atm_iv" not in all_rows.columns:
        all_rows["fno_atm_iv"] = np.nan

    all_rows["fno_atm_iv"] = pd.to_numeric(
        all_rows["fno_atm_iv"],
        errors="coerce",
    )
    all_rows["fno_iv_change"] = all_rows.groupby(
        "symbol"
    )["fno_atm_iv"].diff()

    all_rows.to_parquet(DERIVED, index=False)

    meta = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(all_rows)),
        "symbols": int(all_rows["symbol"].nunique()),
        "min_date": (
            str(all_rows["date"].min().date())
            if not all_rows.empty
            else None
        ),
        "max_date": (
            str(all_rows["date"].max().date())
            if not all_rows.empty
            else None
        ),
        "source": "NSE_OFFICIAL_FNO_UDIFF",
        "raw_archive_root": str(UDIFF_RAW_BASE),
        "fno_iv_method": (
            "ATM option IV from closing/settlement prices via "
            "Black-76, unit discount factor, near-expiry contracts"
        ),
        "fno_basis_method": (
            "NSE underlying_price when present; PIT-safe equity "
            "close fallback in candidate panel"
        ),
    }
    DERIVED_META.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    g = GoogleDriveStore()
    g.upload_or_update(
        DERIVED,
        "production_data/nse_official/fno/fno_daily_features_2026.parquet",
    )
    g.upload_or_update(
        DERIVED_META,
        "production_data/nse_official/fno/fno_daily_features_2026.meta.json",
        mime_type="application/json",
    )


def _merge_and_persist(day_features: pd.DataFrame) -> None:
    old = (
        pd.read_parquet(DERIVED)
        if DERIVED.exists() and DERIVED.stat().st_size > 0
        else pd.DataFrame()
    )
    all_rows = (
        pd.concat([old, day_features], ignore_index=True)
        if not old.empty
        else day_features.copy()
    )
    _persist_derived_rows(all_rows)


def _rebuild_derived_2026() -> None:
    start = date(2026, 1, 1)
    end = min(date.today(), date(2026, 12, 31))
    frames = []
    current = start
    scanned = 0
    loaded = 0

    while current <= end:
        if current.weekday() < 5:
            scanned += 1

            try:
                zip_path = _existing_local_archive(current)

                if zip_path is None:
                    canonical = _udiff_local_path(current)
                    ok = _drive_download(
                        (
                            "production_data/nse_official/fno/"
                            f"raw_udiff/{canonical.name}"
                        ),
                        canonical,
                    )
                    if ok:
                        zip_path = canonical

                if zip_path is None:
                    zip_path, raw = _download_day(
                        current,
                        force=False,
                    )
                else:
                    raw = _read_zip(
                        zip_path.read_bytes()
                    )

                features = _build_features(
                    _normalize(raw, current)
                )

                if not features.empty:
                    frames.append(features)
                    loaded += 1
                    print(
                        f"{current}: rebuilt F&O features "
                        f"rows={len(features):,} "
                        f"symbols={features['symbol'].nunique():,}",
                        flush=True,
                    )
            except Exception as exc:
                print(
                    f"{current}: rebuild skipped "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        current += timedelta(days=1)

    if not frames:
        raise RuntimeError(
            "No 2026 F&O archives could be rebuilt"
        )

    combined = pd.concat(frames, ignore_index=True)
    _persist_derived_rows(combined)

    print(
        "F&O derived rebuild complete: "
        f"weekday_dates_scanned={scanned:,} "
        f"dates_loaded={loaded:,} "
        f"rows={len(combined):,} "
        f"symbols={combined['symbol'].nunique():,}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="")
    ap.add_argument("--days-back", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--rebuild-derived-2026",
        action="store_true",
        help="Rebuild the 2026 derived F&O feature file from cached raw archives.",
    )
    args = ap.parse_args()

    if args.rebuild_derived_2026:
        _rebuild_derived_2026()
        return

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
            zip_path = _existing_local_archive(d)
            if zip_path is None or args.force:
                zip_path, raw = _download_day(
                    d,
                    force=args.force,
                )
            else:
                raw = _read_zip(
                    zip_path.read_bytes()
                )

            features = _build_features(
                _normalize(raw, d)
            )
            if features.empty:
                print(
                    f"{d}: no stock F&O rows in UDiFF archive",
                    flush=True,
                )
                continue

            # Canonical remote archive location matches the persisted
            # raw_udiff tree already used by this project.
            if zip_path.parent == UDIFF_RAW_BASE:
                remote_archive = (
                    "production_data/nse_official/fno/"
                    f"raw_udiff/{zip_path.name}"
                )
            else:
                # Legacy local cache is uploaded into the canonical UDiFF tree.
                remote_archive = (
                    "production_data/nse_official/fno/"
                    f"raw_udiff/{zip_path.name}"
                )

            GoogleDriveStore().upload_or_update(
                zip_path,
                remote_archive,
                mime_type="application/zip",
            )
            _merge_and_persist(features)

            print(
                f"{d}: F&O sync rows={len(features):,} "
                f"symbols={features['symbol'].nunique():,}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"{d}: F&O sync error "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )


if __name__ == "__main__":
    main()
