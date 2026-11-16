#!/usr/bin/env python3
"""Publish the day's affordable, ranked tradable picks.

Reads the newest prediction frame and quote snapshot from Drive, applies the
capital plan (price cap, position count, minimum move) and writes
``production_data/daily_picks/YYYY/MM/DD/picks.json`` plus ``latest.json``.

The selection math lives in :mod:`prediction_universe`; this module only adds
Drive IO so the decision stays testable offline.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from prediction_universe import (
    PREDICTIONS_ROOT,
    SNAPSHOT_ROOT,
    _download_bytes,
    _latest_file,
    load_capital_plan,
    tradable_picks,
)

IST = ZoneInfo("Asia/Kolkata")
PICKS_ROOT = "production_data/daily_picks"
REALISED_PNL_PATH = f"{PICKS_ROOT}/realised_pnl.json"


def _read_parquet(store, remote: str) -> pd.DataFrame:
    payload = _download_bytes(store, remote) if store else None
    if not payload:
        return pd.DataFrame()
    try:
        return pd.read_parquet(io.BytesIO(payload))
    except Exception:
        return pd.DataFrame()


def _snapshot_prices(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or "symbol" not in frame.columns:
        return {}
    price_col = next((c for c in ("actual_observed_price", "live_price", "ltp") if c in frame.columns), None)
    if price_col is None:
        return {}
    prices = pd.to_numeric(frame[price_col], errors="coerce")
    rows = pd.DataFrame({"symbol": frame["symbol"].astype(str).str.upper(), "price": prices})
    rows = rows.dropna().groupby("symbol")["price"].max()
    return {str(s): float(p) for s, p in rows.items() if p > 0}


def _realised_pnl(store) -> float:
    payload = _download_bytes(store, REALISED_PNL_PATH) if store else None
    if not payload:
        return 0.0
    try:
        return float(json.loads(payload).get("realised_pnl", 0.0) or 0.0)
    except Exception:
        return 0.0


def build_picks(store=None, day: datetime | None = None, realised_pnl: float | None = None,
                plan: dict[str, Any] | None = None, sync: bool = True) -> dict[str, Any]:
    plan = plan or load_capital_plan()
    day = day or datetime.now(IST)
    if store is None and sync:
        from persistent_store import GoogleDriveStore
        store = GoogleDriveStore()

    folder = f"{PREDICTIONS_ROOT}/{day:%Y/%m/%d}"
    prediction_path = _latest_file(store, folder, ".parquet") if store else None
    predictions = _read_parquet(store, prediction_path) if prediction_path else pd.DataFrame()
    snapshot_path = _latest_file(store, f"{SNAPSHOT_ROOT}/{day:%Y/%m/%d}", ".parquet") if store else None
    prices = _snapshot_prices(_read_parquet(store, snapshot_path)) if snapshot_path else {}
    if realised_pnl is None:
        realised_pnl = _realised_pnl(store)

    result = tradable_picks(predictions, plan, realised_pnl=realised_pnl, prices=prices)
    document = {
        "status": "OK" if result["picks"] else "NO_PICKS",
        "generated_at": datetime.now(IST).isoformat(),
        "trading_day": f"{day:%Y-%m-%d}",
        "prediction_file": prediction_path,
        "snapshot_file": snapshot_path,
        "realised_pnl": realised_pnl,
        "prediction_rows": int(len(predictions)),
        **result,
    }
    if sync and store is not None:
        for remote in (f"{PICKS_ROOT}/{day:%Y/%m/%d}/picks.json", f"{PICKS_ROOT}/latest.json"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "picks.json"
                path.write_text(json.dumps(document, indent=2), encoding="utf-8")
                store.upload_or_update(path, remote, "application/json")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish daily tradable picks")
    parser.add_argument("--day", default="", help="trading day YYYY-MM-DD (default: today IST)")
    parser.add_argument("--realised-pnl", type=float, default=None)
    parser.add_argument("--no-sync", action="store_true")
    args = parser.parse_args(argv)
    day = datetime.strptime(args.day, "%Y-%m-%d").replace(tzinfo=IST) if args.day else None
    document = build_picks(day=day, realised_pnl=args.realised_pnl, sync=not args.no_sync)
    print(json.dumps({k: v for k, v in document.items() if k != "picks"}, indent=2))
    for pick in document["picks"]:
        print(f"{pick['symbol']:>12} x{pick['quantity']:<3} @ {pick['price']:>8.2f}  "
              f"ret={pick['expected_return']:.2%} p={pick['probability_up']:.2f}")
    return 0 if document["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
