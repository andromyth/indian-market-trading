#!/usr/bin/env python3
"""Smoke-test one historical NSE F&O archive before full backfill."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

import COLAB_BACKFILL_FNO_HISTORY_2016_20261003 as hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2016-01-04")
    args = ap.parse_args()

    d = pd.Timestamp(args.date).date()
    if hist.archive_kind(d) != "legacy":
        raise SystemExit(f"{d} is not in the legacy NSE window")

    session = __import__("requests").Session()
    session.headers.update(hist.HEADERS if hasattr(hist, "HEADERS") else {})

    path = hist.existing_archive(d)
    if path is None:
        path = hist.download_day(d, session)

    if path is None:
        raise SystemExit(f"No archive available for {d}")

    raw = hist.read_zip(path)
    hist.validate_raw(raw, d)
    normalized = hist.normalize_contracts(raw, d)
    features = hist.build_day_features(normalized)

    print("=" * 100)
    print(f"NSE F&O LEGACY SMOKE TEST: {d}")
    print("=" * 100)
    print(f"ARCHIVE: {path}")
    print(f"RAW_ROWS: {len(raw):,}")
    print(f"RAW_COLUMNS: {list(raw.columns)}")
    print(f"NORMALIZED_ROWS: {len(normalized):,}")
    print(f"NORMALIZED_INSTRUMENTS: {sorted(normalized['instrument'].dropna().astype(str).unique())}")
    print(f"FEATURE_ROWS: {len(features):,}")
    print(f"FEATURE_SYMBOLS: {features['symbol'].nunique():,}")
    print(f"ATM_IV_ROWS: {int(features['fno_atm_iv'].notna().sum()):,}")
    print(f"FUTURE_ROWS: {int(features['fno_fut_close'].notna().sum()):,}")
    print("SMOKE_TEST: PASS")


if __name__ == "__main__":
    main()
