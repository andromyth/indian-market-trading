#!/usr/bin/env python3
"""Audit whether persisted NSE F&O UDiFF archives contain enough inputs for ATM IV."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from nse_fno_udiff_ingest import _atm_iv, _normalize, _read_zip, UDIFF_RAW_BASE


ARCHIVE_PREFIX = "BhavCopy_NSE_FO_0_0_0_"
ARCHIVE_SUFFIX = "_F_0000.csv.zip"
OUTPUT = Path(
    "/content/drive/MyDrive/Indian_Market_Research/"
    "diagnostics/fno_iv_input_audit_20261003.csv"
)


def audit_file(path: Path) -> dict:
    try:
        raw = _read_zip(path.read_bytes())
        token = path.name.split("_0_0_0_")[1][:8]
        day = pd.Timestamp(token, format="%Y%m%d").date()
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

        with_strike = int(option["strike_price"].notna().sum())
        with_close = int(
            pd.to_numeric(
                option["close"], errors="coerce"
            ).gt(0).sum()
        )
        with_settle = int(
            pd.to_numeric(
                option["settle"], errors="coerce"
            ).gt(0).sum()
        )
        with_underlying = int(
            pd.to_numeric(
                option["underlying_price"],
                errors="coerce",
            ).gt(0).sum()
        )
        symbols_with_future = int(
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
            "date": str(day),
            "rows": int(len(x)),
            "symbols": int(x["symbol"].nunique()),
            "option_rows": int(len(option)),
            "option_rows_with_strike": with_strike,
            "option_rows_with_positive_close": with_close,
            "option_rows_with_positive_settle": with_settle,
            "option_rows_with_positive_underlying": with_underlying,
            "symbols_with_future": symbols_with_future,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-root",
        default=os.getenv(
            "FNO_UDIFF_RAW_ROOT",
            str(UDIFF_RAW_BASE),
        ),
    )
    ap.add_argument("--sample-days", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.raw_root)
    print("=" * 90)
    print("NSE F&O IV INPUT AUDIT")
    print("=" * 90)
    print("RAW_UDIFF_ROOT:", root)
    print("INDIAN_MARKET_RESEARCH_ROOT:",
          os.environ.get("INDIAN_MARKET_RESEARCH_ROOT", ""))

    files = sorted(
        root.rglob(
            f"{ARCHIVE_PREFIX}*{ARCHIVE_SUFFIX}"
        )
    )

    print("RAW_UDIFF_ARCHIVES_FOUND:", len(files))

    if not files:
        raise SystemExit(
            "No NSE F&O UDiFF archives found under "
            f"{root}"
        )

    if args.sample_days > 0:
        files = sorted(files, reverse=True)[:args.sample_days]
        files = sorted(files)

    rows = [audit_file(path) for path in files]
    df = pd.DataFrame(rows)

    good = df[df["status"].eq("PASS")].copy()
    bad = df[df["status"].ne("PASS")].copy()

    print("FILES_SCANNED:", len(df))

    if not good.empty:
        print("OPTION_ROWS:",
              int(good["option_rows"].sum()))
        print(
            "OPTION_ROWS_WITH_STRIKE:",
            int(good["option_rows_with_strike"].sum()),
        )
        print(
            "OPTION_ROWS_WITH_POSITIVE_CLOSE:",
            int(good["option_rows_with_positive_close"].sum()),
        )
        print(
            "OPTION_ROWS_WITH_POSITIVE_SETTLE:",
            int(good["option_rows_with_positive_settle"].sum()),
        )
        print(
            "OPTION_ROWS_WITH_POSITIVE_UNDERLYING:",
            int(
                good[
                    "option_rows_with_positive_underlying"
                ].sum()
            ),
        )
        print(
            "SYMBOLS_WITH_FUTURE:",
            int(good["symbols_with_future"].sum()),
        )
        print(
            "SYMBOLS_WITH_RECOVERABLE_ATM_IV:",
            int(
                good[
                    "symbols_with_recoverable_atm_iv"
                ].sum()
            ),
        )

        print("\nPer-file summary:")
        print(
            good[
                [
                    "date",
                    "symbols",
                    "option_rows",
                    "option_rows_with_strike",
                    "option_rows_with_positive_close",
                    "option_rows_with_positive_settle",
                    "option_rows_with_positive_underlying",
                    "symbols_with_future",
                    "symbols_with_recoverable_atm_iv",
                    "median_atm_iv",
                ]
            ].to_string(index=False)
        )

    if not bad.empty:
        print("\nFAILURES:")
        print(bad.to_string(index=False))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print("\nOUTPUT:", OUTPUT)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
