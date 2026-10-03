#!/usr/bin/env python3
"""Audit whether persisted NSE F&O raw archives contain enough inputs for ATM IV."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nse_fno_udiff_ingest import _atm_iv, _normalize, _read_zip, RAW_BASE


def audit_file(path: Path) -> dict:
    try:
        raw = _read_zip(path.read_bytes())
        day = pd.Timestamp(
            path.name.split("_0_0_0_")[1][:8], format="%Y%m%d"
        ).date()
        x = _normalize(raw, day)
        option = x[x["option_type"].isin(["CE", "PE", "CALL", "PUT"])].copy()
        futures = x[x["instrument"].str.contains("FUT", na=False)].copy()
        futures = futures.dropna(subset=["expiry"])
        futures = futures[futures["expiry"] >= x["date"].iloc[0]]
        futures = futures.sort_values(["symbol", "expiry"]).drop_duplicates("symbol", keep="first")

        with_strike = int(option["strike_price"].notna().sum()) if "strike_price" in option.columns else 0
        with_price = int(pd.to_numeric(option["close"], errors="coerce").gt(0).sum())
        with_future = int(futures["symbol"].isin(option["symbol"]).sum())

        iv_symbols = 0
        iv_values = []
        for sym in sorted(option["symbol"].dropna().unique()):
            og = option[option["symbol"].eq(sym)]
            fg = futures[futures["symbol"].eq(sym)]
            fr = fg.iloc[0] if not fg.empty else None
            iv = _atm_iv(og, fr, pd.Timestamp(x["date"].iloc[0]))
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
            "median_atm_iv": float(np.median(iv_values)) if iv_values else np.nan,
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
    ap.add_argument("--raw-root", default=str(RAW_BASE))
    ap.add_argument("--sample-days", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.raw_root)
    files = sorted(root.glob("*/*/*/*.zip"))
    if not files:
        # Support the actual persisted archive filename regardless of partition depth.
        files = sorted(root.rglob("BhavCopy_NSE_FO_0_0_0_*_F_0000.csv.zip"))
    if not files:
        raise SystemExit(f"No NSE F&O raw archives found under {root}")

    if args.sample_days > 0:
        files = files[: args.sample_days]

    rows = [audit_file(p) for p in files]
    df = pd.DataFrame(rows)
    print("=" * 90)
    print("NSE F&O IV INPUT AUDIT")
    print("=" * 90)
    print("files_scanned:", len(df))
    print("raw_root:", root)

    good = df[df["status"].eq("PASS")].copy()
    if not good.empty:
        print("option_rows:", int(good["option_rows"].sum()))
        print("option_rows_with_strike:", int(good["option_rows_with_strike"].sum()))
        print("option_rows_with_positive_close:", int(good["option_rows_with_positive_close"].sum()))
        print("symbols_with_future:", int(good["symbols_with_future"].sum()))
        print("symbols_with_recoverable_atm_iv:", int(good["symbols_with_recoverable_atm_iv"].sum()))
        display_cols = [
            "date", "symbols", "option_rows", "option_rows_with_strike",
            "option_rows_with_positive_close", "symbols_with_future",
            "symbols_with_recoverable_atm_iv", "median_atm_iv"
        ]
        print("\nPer-file summary:")
        print(good[display_cols].to_string(index=False))

    bad = df[df["status"].ne("PASS")]
    if not bad.empty:
        print("\nFailures:")
        print(bad.to_string(index=False))

    out = Path("/content/drive/MyDrive/Indian_Market_Research/diagnostics/fno_iv_input_audit_20261003.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\nOUTPUT:", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
