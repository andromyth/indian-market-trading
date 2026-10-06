"""One-symbol equivalence audit for the fast Beta V2 intraday preparer."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from colab_build_all_nse_intraday_5m_multifactor import load_market, load_symbol
from selfdriving.sd_intraday_fast_prepare import build_compact_symbol
from selfdriving.sd_intraday_v2 import (
    _session_rows,
)

HORIZONS = (5, 10, 15, 20, 25, 30)


def compact_old(g, market):
    generated = _session_rows(
        g,
        market,
        HORIZONS,
        max_decisions=None,
    )
    if not generated:
        return pd.DataFrame()

    long = pd.DataFrame(generated)
    long["decision_timestamp"] = pd.to_datetime(
        long["decision_timestamp"], utc=True, errors="coerce"
    )
    long["date"] = pd.to_datetime(long["date"], errors="coerce").dt.normalize()
    base_cols = [
        "symbol",
        "decision_timestamp",
        "date",
        "reference_close",
    ] + [
        c for c in long.columns
        if c in __import__("selfdriving.sd_intraday_v2", fromlist=["feature_columns"]).feature_columns()
    ]
    base = (
        long[base_cols]
        .drop_duplicates("decision_timestamp", keep="first")
        .sort_values("decision_timestamp")
        .reset_index(drop=True)
    )
    for h in HORIZONS:
        part = (
            long[long["horizon_min"].eq(h)][
                ["decision_timestamp", "future_close", "fwd_ret"]
            ]
            .drop_duplicates("decision_timestamp", keep="last")
            .rename(
                columns={
                    "future_close": f"future_close_{h}m",
                    "fwd_ret": f"fwd_ret_{h}m",
                }
            )
        )
        base = base.merge(part, on="decision_timestamp", how="left")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/Indian_Market_Research")
    ap.add_argument("--symbol", default="20MICRONS")
    args = ap.parse_args()

    root = __import__("pathlib").Path(args.drive_root)
    market = load_market(root)
    raw = load_symbol(root, args.symbol)
    if raw is None or raw.empty:
        raise RuntimeError(f"No data for {args.symbol}")

    old = compact_old(raw, market)
    fast = build_compact_symbol(raw, market, HORIZONS, max_decisions=None)

    if len(old) != len(fast):
        raise AssertionError(f"ROW_COUNT_MISMATCH old={len(old)} fast={len(fast)}")

    keys = ["decision_timestamp"]
    merged = old.merge(fast, on=keys, how="outer", suffixes=("_old", "_fast"), indicator=True)
    if not (merged["_merge"] == "both").all():
        raise AssertionError("TIMESTAMP_SET_MISMATCH")

    compare_cols = [
        c for c in fast.columns
        if c != "decision_timestamp" and c in old.columns
    ]

    worst = []
    for c in compare_cols:
        a = pd.to_numeric(merged[f"{c}_old"], errors="coerce").to_numpy()
        b = pd.to_numeric(merged[f"{c}_fast"], errors="coerce").to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        finite = np.isfinite(a) & np.isfinite(b)
        if np.any(np.isfinite(a) ^ np.isfinite(b)):
            raise AssertionError(f"NULL_MASK_MISMATCH:{c}")
        err = np.nanmax(np.abs(a[finite] - b[finite])) if finite.any() else 0.0
        worst.append((c, float(err)))

    worst.sort(key=lambda x: x[1], reverse=True)
    print(f"FAST_EQUIVALENCE_PASS symbol={args.symbol} rows={len(fast):,}")
    print("largest absolute differences:")
    for c, err in worst[:10]:
        print(f"  {c}: {err:.3e}")


if __name__ == "__main__":
    main()
