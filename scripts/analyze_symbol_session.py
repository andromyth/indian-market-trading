#!/usr/bin/env python3
"""Read-only replay of one symbol through one session.

Downloads the stored prediction frame for an entry slot and the stored quote
snapshots for the entry and exit slots, then reports the model's call and the
rupee P&L for a position sized from ``capital_plan.json``.

It never writes to Drive, never retrains and never places orders.
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime
from math import floor
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from prediction_universe import (  # noqa: E402
    PREDICTIONS_ROOT,
    SNAPSHOT_ROOT,
    _download_bytes,
    load_capital_plan,
    position_budget,
)
from session_prediction_health import _list_children, _resolve_folder  # noqa: E402

LIVE_HISTORY_ROOT = "production_data/nse_official/intraday_5m_live"
PRICE_COLUMNS = ("actual_observed_price", "live_price", "ltp", "close", "Close")


def out(key: str, value: Any) -> None:
    print(f"{key}={value}", flush=True)


def folder_files(store, service, root: str, day: datetime) -> list[dict[str, Any]]:
    parts = root.split("/") + [day.strftime("%Y"), day.strftime("%m"), day.strftime("%d")]
    node = _resolve_folder(store, parts)
    if not node:
        return []
    return _list_children(service, node["id"])


def slot_file(files: list[dict[str, Any]], stamp: str, hhmm: str) -> dict[str, Any] | None:
    suffix = f"_{stamp}_{hhmm}.parquet"
    matches = [f for f in files if str(f.get("name") or "").lower().endswith(suffix)]
    if not matches:
        return None
    return sorted(matches, key=lambda f: str(f.get("name")))[0]


def read_frame(store, root: str, day: datetime, name: str) -> pd.DataFrame:
    remote = "/".join([root, day.strftime("%Y"), day.strftime("%m"), day.strftime("%d"), name])
    payload = _download_bytes(store, remote)
    if not payload:
        return pd.DataFrame()
    try:
        return pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:  # pragma: no cover - diagnostic only
        out("PARQUET_READ_ERROR", f"{type(exc).__name__}: {exc}")
        return pd.DataFrame()


def symbol_price(frame: pd.DataFrame, symbol: str) -> float | None:
    if frame.empty or "symbol" not in frame.columns:
        return None
    column = next((c for c in PRICE_COLUMNS if c in frame.columns), None)
    if column is None:
        return None
    rows = frame[frame["symbol"].astype(str).str.upper().str.replace(".NS", "", regex=False).str.strip() == symbol]
    if rows.empty:
        return None
    value = pd.to_numeric(rows[column], errors="coerce").dropna()
    return float(value.iloc[0]) if not value.empty else None


def trading_costs(buy_value: float, sell_value: float, brokerage_per_leg: float = 20.0) -> dict[str, float]:
    """Statutory NSE intraday equity charges plus flat brokerage."""
    brokerage = brokerage_per_leg * 2
    stt = 0.00025 * sell_value
    exchange = 0.0000297 * (buy_value + sell_value)
    sebi = 0.000001 * (buy_value + sell_value)
    stamp = 0.00003 * buy_value
    gst = 0.18 * (brokerage + exchange + sebi)
    total = brokerage + stt + exchange + sebi + stamp + gst
    return {
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange": round(exchange, 2),
        "sebi": round(sebi, 2),
        "stamp": round(stamp, 2),
        "gst": round(gst, 2),
        "total": round(total, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay one symbol through one session")
    parser.add_argument("--symbol", default="SBIN")
    parser.add_argument("--date", default="", help="YYYY-MM-DD (default: today IST)")
    parser.add_argument("--entry", default="11:00", help="entry slot HH:MM IST")
    parser.add_argument("--exit", default="13:45", help="requested exit slot HH:MM IST")
    parser.add_argument("--horizon", type=int, default=30, help="model horizon minutes")
    args = parser.parse_args(argv)

    day = (
        datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=IST)
        if args.date
        else datetime.now(IST)
    )
    stamp = day.strftime("%Y%m%d")
    symbol = args.symbol.strip().upper().replace(".NS", "")
    entry_hhmm = args.entry.replace(":", "")
    exit_hhmm = args.exit.replace(":", "")

    def shift(hhmm: str, minutes: int) -> str:
        parsed = datetime.strptime(hhmm, "%H%M")
        return (parsed + pd.Timedelta(minutes=minutes)).strftime("%H%M")

    horizon_hhmm = shift(entry_hhmm, args.horizon)

    out("SYMBOL", symbol)
    out("SESSION_DATE", day.strftime("%Y-%m-%d"))
    out("ENTRY_SLOT", entry_hhmm)
    out("REQUESTED_EXIT_SLOT", exit_hhmm)
    out("HORIZON_SLOT", horizon_hhmm)

    from persistent_store import GoogleDriveStore

    store = GoogleDriveStore()
    service = store.service

    # --- the model's call at the entry slot -------------------------------
    pred_files = folder_files(store, service, PREDICTIONS_ROOT, day)
    out("PREDICTION_FILES_PRESENT", len(pred_files))
    pred_meta = slot_file(pred_files, stamp, entry_hhmm)
    if pred_meta is None:
        out("PREDICTION_FILE", "MISSING")
        return 2
    out("PREDICTION_FILE", pred_meta["name"])
    predictions = read_frame(store, PREDICTIONS_ROOT, day, str(pred_meta["name"]))
    out("PREDICTION_ROWS", len(predictions))

    row_frame = predictions[
        predictions["symbol"].astype(str).str.upper().str.replace(".NS", "", regex=False).str.strip() == symbol
    ] if "symbol" in predictions.columns and not predictions.empty else pd.DataFrame()

    if row_frame.empty:
        out("SYMBOL_ROW", "MISSING")
        return 3

    row = row_frame.iloc[0]
    out("SYMBOL_ROW", "PRESENT")
    for column in (
        "prediction_status",
        "signal",
        "signal_raw",
        "probability_up_30m",
        "expected_return_30m",
        "data_readiness_score",
        "data_limited",
        "readiness_segment",
        "model_family",
        "horizon_min",
        "actual_market_value",
        "actual_observed_at_ist",
    ):
        if column in row_frame.columns:
            out(f"MODEL_{column.upper()}", row.get(column))

    action = str(row.get("signal", "")).strip().upper() or "HOLD"
    try:
        probability = float(row.get("probability_up_30m"))
    except (TypeError, ValueError):
        probability = None
    try:
        expected_return = float(row.get("expected_return_30m"))
    except (TypeError, ValueError):
        expected_return = None

    # --- prices -----------------------------------------------------------
    snap_files = folder_files(store, service, SNAPSHOT_ROOT, day)
    out("SNAPSHOT_FILES_PRESENT", len(snap_files))
    live_files = folder_files(store, service, LIVE_HISTORY_ROOT, day)
    out("LIVE_HISTORY_FILES_PRESENT", len(live_files))

    def price_at(hhmm: str) -> tuple[float | None, str]:
        for root, files in ((SNAPSHOT_ROOT, snap_files), (LIVE_HISTORY_ROOT, live_files)):
            meta = slot_file(files, stamp, hhmm)
            if meta is None:
                continue
            price = symbol_price(read_frame(store, root, day, str(meta["name"])), symbol)
            if price is not None:
                return price, f"{root.split('/')[-1]}/{meta['name']}"
        return None, "MISSING"

    entry_price, entry_source = price_at(entry_hhmm)
    out("ENTRY_PRICE", entry_price)
    out("ENTRY_PRICE_SOURCE", entry_source)
    if entry_price is None:
        try:
            entry_price = float(row.get("actual_market_value"))
            out("ENTRY_PRICE", entry_price)
            out("ENTRY_PRICE_SOURCE", "prediction_row.actual_market_value")
        except (TypeError, ValueError):
            out("ENTRY_PRICE_SOURCE", "UNAVAILABLE")
            return 4

    exit_price, exit_source = price_at(exit_hhmm)
    out("EXIT_PRICE", exit_price)
    out("EXIT_PRICE_SOURCE", exit_source)

    horizon_price, horizon_source = price_at(horizon_hhmm)
    out("HORIZON_PRICE", horizon_price)
    out("HORIZON_PRICE_SOURCE", horizon_source)

    # --- position sizing from the live capital plan -----------------------
    plan = load_capital_plan()
    budget = position_budget(plan)
    out("CAPITAL", plan.get("monthly_capital"))
    out("INVESTABLE", budget.get("investable"))
    out("POSITIONS", budget.get("positions"))
    out("PER_TRADE", round(float(budget.get("per_trade", 0.0)), 2))
    out("PRICE_CAP", round(float(budget.get("price_cap", 0.0)), 2))

    per_trade = float(budget.get("per_trade", 0.0))
    price_cap = float(budget.get("price_cap", 0.0))
    tradable = action in {"BUY", "SELL"} and entry_price <= price_cap
    out("MODEL_ACTION", action)
    out("GATE_PASSED", "YES" if tradable else "NO")
    if not tradable:
        out("REASON", "model did not emit BUY/SELL or price is above the plan price cap")
        out("NET_PROFIT_RS", 0.0)
        return 0

    quantity = int(floor(per_trade // entry_price))
    out("QUANTITY", quantity)
    out("POSITION_VALUE", round(quantity * entry_price, 2))
    if quantity < 1:
        out("REASON", "per-trade budget cannot buy one share")
        out("NET_PROFIT_RS", 0.0)
        return 0

    def pnl(exit_value: float | None, label: str) -> None:
        if exit_value is None:
            out(f"PNL_{label}", "UNAVAILABLE")
            return
        if action == "BUY":
            gross = (exit_value - entry_price) * quantity
            buy_value, sell_value = quantity * entry_price, quantity * exit_value
        else:
            gross = (entry_price - exit_value) * quantity
            sell_value, buy_value = quantity * entry_price, quantity * exit_value
        costs = trading_costs(buy_value, sell_value)
        net = gross - costs["total"]
        out(f"EXIT_PRICE_{label}", round(exit_value, 2))
        out(f"GROSS_{label}", round(gross, 2))
        out(f"COSTS_{label}", costs["total"])
        out(f"COST_DETAIL_{label}", " ".join(f"{k}:{v}" for k, v in costs.items() if k != "total"))
        out(f"NET_PROFIT_{label}", round(net, 2))
        out(f"RETURN_PCT_{label}", round(100.0 * net / (quantity * entry_price), 3))

    pnl(horizon_price, f"HORIZON_{args.horizon}M")
    pnl(exit_price, f"EXIT_{exit_hhmm}")

    out("MODEL_EXPECTED_RETURN_PCT", None if expected_return is None else round(expected_return * 100.0, 3))
    out("MODEL_PROBABILITY_UP", probability)
    return 0


if __name__ == "__main__":
    sys.exit(main())
