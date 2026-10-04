#!/usr/bin/env python3
"""Audit the persisted 5-minute multi-horizon prediction dataset.

This script is read-only: it never changes a model or prediction.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

HORIZONS = (5, 10, 15, 20, 25, 30)
KEY = ["symbol", "decision_timestamp", "horizon_minutes"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        default="/content/drive/MyDrive/Indian_Market_Research/models/"
                "all_nse_intraday_5m_multifactor_replacement_2026/"
                "holdout_predictions_all_horizons_2026.parquet",
    )
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    required = {
        "symbol", "decision_timestamp", "target_timestamp",
        "actual_future_timestamp", "horizon_minutes",
        "reference_price", "predicted_price", "actual_future_price",
        "target_alignment_seconds", "realized_horizon_seconds",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError("Missing columns: " + ", ".join(missing))

    dec = pd.to_datetime(df["decision_timestamp"], utc=True, errors="coerce")
    target = pd.to_datetime(df["target_timestamp"], utc=True, errors="coerce")
    actual = pd.to_datetime(df["actual_future_timestamp"], utc=True, errors="coerce")
    horizon = pd.to_numeric(df["horizon_minutes"], errors="coerce")

    target_delta = (target - dec).dt.total_seconds()
    alignment = (actual - target).dt.total_seconds()
    duplicate_keys = int(df.duplicated(KEY).sum())
    bad_horizon = int((~horizon.isin(HORIZONS)).sum())

    groups = df.groupby(["symbol", "decision_timestamp"])["horizon_minutes"].apply(
        lambda x: set(int(v) for v in x.dropna())
    )
    incomplete_groups = int((groups.apply(lambda x: x != set(HORIZONS))).sum())

    checks = {
        "DUPLICATE_KEYS": duplicate_keys,
        "BAD_HORIZON_ROWS": bad_horizon,
        "BAD_TARGET_INTERVAL_ROWS": int((target_delta != horizon * 60).sum()),
        "ACTUAL_BEFORE_TARGET_ROWS": int((alignment < 0).sum()),
        "ACTUAL_OVER_60S_ROWS": int((alignment > 60).sum()),
        "ACTUAL_NA_ROWS": int(actual.isna().sum()),
        "PREDICTED_PRICE_NA_ROWS": int(pd.to_numeric(df["predicted_price"], errors="coerce").isna().sum()),
        "ACTUAL_PRICE_NA_ROWS": int(pd.to_numeric(df["actual_future_price"], errors="coerce").isna().sum()),
        "DECISION_GROUPS_MISSING_HORIZONS": incomplete_groups,
    }

    print("ROWS:", len(df))
    print("SYMBOLS:", int(df["symbol"].nunique()))
    print("HORIZONS:", sorted(horizon.dropna().unique().tolist()))
    for k, v in checks.items():
        print(f"{k}:", v)

    failed = any(v != 0 for v in checks.values())
    print("AUDIT_STATUS:", "PASS" if not failed else "FAIL")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
