#!/usr/bin/env python3
"""Audit whether persisted NSE F&O raw archives contain enough inputs for ATM IV."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nse_fno_udiff_ingest import (
    _atm_iv,
    _drive_download,
    _normalize,
    _read_zip,
    RAW_BASE,
    BASE,
)


def _archive_path(day: pd.Timestamp) -> Path:
    day = pd.Timestamp(day).normalize()
    out_dir = RAW_BASE / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"
    return out_dir / f"BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"


def _materialize_drive_archive(path: Path, day: pd.Timestamp) -> bool:
    """Materialize one known Drive archive into the local RAW_BASE cache."""
    if path.exists() and path.stat().st_size > 0:
        return True
    rel = (
        f"production_data/nse_official/fno/raw/"
        f"{day:%Y/%m/%d}/{path.name}"
    )
    return _drive_download(rel, path)


def audit_file(path: Path) -> dict:
    try:
        raw = _read_zip(path.read_bytes())
        day = pd.Timestamp(
            path.name.split("_0_0_0_")[1][:8], format="%Y%m%d"
        ).date()
        x = _normalize(raw, day)
        option = x[
            x["option_type"].isin(["CE", "PE", "CALL", "PUT"])
        ].copy()
        futures = x[
            x["instrument"].str.contains("FUT", na=False)
        ].copy()
        futures = futures.dropna(subset=["expiry"])
        futures = futures[futures["expiry"] >= x["date"].iloc[0]]
        futures = (
            futures.sort_values(["symbol", "expiry"])
            .drop_duplicates("symbol", keep="first")
        )

        with_strike = (
            int(option["strike_price"].notna().sum())
            if "strike_price" in option.columns
            else 0
        )
        with_price = int(
            pd.to_numeric(option["close"], errors="coerce").gt(0).sum()
        )
        with_future = int(
            futures["symbol"].isin(option["symbol"]).sum()
        )

        iv_symbols = 0
        iv_values = []
        for sym in sorted(option["symbol"].dropna().unique()):
            og = option[option["symbol"].eq(sym)]
            fg = futures[futures["symbol"].eq(sym)]
            fr = fg.iloc[0] if not fg.empty else None
            iv = _atm_iv(
                og,
                fr,
                pd.Timestamp(x["date"].iloc[0]),
            )
            if np.isfinite(iv):
                iv_symbols += 1
                iv_values.append(float(iv))

        return {
            "file": str(path),
            "date": str(x["date"].iloc[0].date()),
            "rows": int(len(x)),
            "symbols": int(x["symbol"].nunique()),
            "option_rows": int(len(option)),
            "option_rows_with_strike": with_strike,
            "option_rows_with_positive_close": with_price,
            "symbols_with_future": with_future,
            "symbols_with_recoverable_atm_iv": iv_symbols,
            "median_atm_iv": (
                float(np.median(iv_values))
                if iv_values
                else np.nan
            ),
            "status": "PASS",
        }
    except Exception as exc:
        return {
            "file": str(path),
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _candidate_days(sample_days: int) -> list[pd.Timestamp]:
    """Return recent 2026 weekdays, newest first, for Drive-backed sampling."""
    end = min(pd.Timestamp.today().normalize(), pd.Timestamp("2026-12-31"))
    start = pd.Timestamp("2026-01-01")
    dates = pd.date_range(start, end, freq="B")
    if sample_days > 0:
        dates = dates[-sample_days:]
    return [pd.Timestamp(d) for d in dates[::-1]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=str(RAW_BASE))
    ap.add_argument("--sample-days", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.raw_root)
    files = sorted(
        root.rglob("BhavCopy_NSE_FO_0_0_0_*_F_0000.csv.zip")
    )

    materialized = 0
    if not files:
        # The raw archive is persisted on Drive, but may not yet be present in
        # the local Colab filesystem. Materialize a recent sample using the
        # exact same Drive path contract as the production rebuild.
        for day in _candidate_days(max(args.sample_days, 1)):
            path = _archive_path(day)
            if _materialize_drive_archive(path, day):
                materialized += 1
        files = sorted(
            root.rglob("BhavCopy_NSE_FO_0_0_0_*_F_0000.csv.zip")
        )
    else:
        # Materialize enough recent Drive-backed files when the local cache is
        # sparse, without downloading the full historical archive.
        have_days = {
            pd.Timestamp(
                p.name.split("_0_0_0_")[1][:8],
                format="%Y%m%d",
            ).normalize()
            for p in files
        }
        for day in _candidate_days(max(args.sample_days, 1)):
            if day in have_days:
                continue
            path = _archive_path(day)
            if _materialize_drive_archive(path, day):
                materialized += 1
        files = sorted(
            root.rglob("BhavCopy_NSE_FO_0_0_0_*_F_0000.csv.zip")
        )

    if not files:
        raise SystemExit(
            "No NSE F&O raw archives found locally or in the configured "
            f"Google Drive path under {BASE}"
        )

    if args.sample_days > 0:
        # Prefer the newest materialized/sample files for a current input audit.
        files = sorted(files, reverse=True)[: args.sample_days]
        files = sorted(files)

    rows = [audit_file(p) for p in files]
    df = pd.DataFrame(rows)

    print("=" * 90)
    print("NSE F&O IV INPUT AUDIT")
    print("=" * 90)
    print("files_scanned:", len(df))
    print("raw_root:", root)
    print("drive_archives_materialized:", materialized)

    good = df[df["status"].eq("PASS")].copy()
    if not good.empty:
        print("option_rows:", int(good["option_rows"].sum()))
        print(
            "option_rows_with_strike:",
            int(good["option_rows_with_strike"].sum()),
        )
        print(
            "option_rows_with_positive_close:",
            int(good["option_rows_with_positive_close"].sum()),
        )
        print(
            "symbols_with_future:",
            int(good["symbols_with_future"].sum()),
        )
        print(
            "symbols_with_recoverable_atm_iv:",
            int(
                good["symbols_with_recoverable_atm_iv"].sum()
            ),
        )
        display_cols = [
            "date",
            "symbols",
            "option_rows",
            "option_rows_with_strike",
            "option_rows_with_positive_close",
            "symbols_with_future",
            "symbols_with_recoverable_atm_iv",
            "median_atm_iv",
        ]
        print("\nPer-file summary:")
        print(
            good[display_cols].to_string(index=False)
        )

    bad = df[df["status"].ne("PASS")]
    if not bad.empty:
        print("\nFailures:")
        print(bad.to_string(index=False))

    out = Path(
        "/content/drive/MyDrive/Indian_Market_Research/"
        "diagnostics/fno_iv_input_audit_20261003.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\nOUTPUT:", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
