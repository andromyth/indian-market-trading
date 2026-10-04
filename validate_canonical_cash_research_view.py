#!/usr/bin/env python3
"""Read-only validation of the canonical NSE cash research projection.

This gate exercises the same projection the ALL-NSE Delivery trainer will use.
It never writes the cash master and never trains a model.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from delivery_training_universe import load_official_nse_universe
from nse_data_layer import (
    CANONICAL_CASH_MASTER,
    CANONICAL_RESEARCH_SERIES,
    load_canonical_cash_research_history,
)

BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
OUT = BASE / "diagnostics" / "canonical_cash_master_validation"
REPORT = OUT / "canonical_cash_research_view_validation_latest.json"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    if not CANONICAL_CASH_MASTER.exists() or CANONICAL_CASH_MASTER.stat().st_size == 0:
        raise FileNotFoundError(f"Canonical cash master missing: {CANONICAL_CASH_MASTER}")

    universe = load_official_nse_universe(force_refresh=False)
    universe_symbols = sorted(set(universe["_symbol"].astype(str)))
    view = load_canonical_cash_research_history(
        CANONICAL_CASH_MASTER,
        symbols=universe_symbols,
    )

    if view.empty:
        raise RuntimeError("Canonical cash research view returned zero rows.")

    duplicate_symbol_date = int(
        view.duplicated(["symbol", "date"], keep=False).sum()
    )
    t0_rows = int(view["series"].eq("T0").sum()) if "series" in view else 0
    invalid_ohlc = int(
        (
            (view["high"] < view["low"])
            | (view["close"] < view["low"])
            | (view["close"] > view["high"])
            | (view["open"] < view["low"])
            | (view["open"] > view["high"])
        ).sum()
    )

    sessions = view.groupby("symbol")["date"].nunique()
    trainable = sorted(sessions[sessions >= 60].index.astype(str).tolist())
    symbols_with_history = sorted(sessions.index.astype(str).tolist())
    excluded = sorted(set(universe_symbols) - set(trainable))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_master": str(CANONICAL_CASH_MASTER),
        "research_series_policy": list(CANONICAL_RESEARCH_SERIES),
        "t0_excluded": True,
        "universe_symbol_count": len(universe_symbols),
        "symbols_with_research_history": len(symbols_with_history),
        "trainable_symbol_count": len(trainable),
        "excluded_symbol_count": len(excluded),
        "rows": int(len(view)),
        "min_date": str(view["date"].min().date()),
        "max_date": str(view["date"].max().date()),
        "distinct_trading_dates": int(view["date"].nunique()),
        "duplicate_symbol_date_rows": duplicate_symbol_date,
        "t0_rows": t0_rows,
        "invalid_ohlc_rows": invalid_ohlc,
        "series_counts": {
            str(k): int(v) for k, v in view["series"].value_counts().items()
        },
        "status": "PASS" if duplicate_symbol_date == 0 and t0_rows == 0 and invalid_ohlc == 0 else "FAIL",
    }

    REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("CANONICAL NSE CASH — RESEARCH VIEW VALIDATION")
    print("=" * 100)
    print(f"Universe symbols             : {len(universe_symbols):,}")
    print(f"Research-view rows           : {len(view):,}")
    print(f"Symbols with history         : {len(symbols_with_history):,}")
    print(f"Trainable (>=60 sessions)    : {len(trainable):,}")
    print(f"Excluded                     : {len(excluded):,}")
    print(f"Date range                   : {view['date'].min()} -> {view['date'].max()}")
    print(f"Trading dates                : {view['date'].nunique():,}")
    print(f"Duplicate symbol/date rows   : {duplicate_symbol_date:,}")
    print(f"T0 rows in research view     : {t0_rows:,}")
    print(f"Invalid OHLC rows            : {invalid_ohlc:,}")
    print(f"Series distribution          : {report['series_counts']}")
    print(f"Report                       : {REPORT}")
    print(f"STATUS                       : {report['status']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
