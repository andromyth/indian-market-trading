#!/usr/bin/env python3
"""Rolling simulation of one symbol's stored 5-minute signals.

Every prediction has a 30-minute horizon, so a signal issued at 12:30 is scored
against the price at 13:00. This replays the whole session: long on BUY, short
on SELL, held for the model horizon, with the capital plan's position limit and
real NSE intraday charges.

Read-only. It never writes to Drive, never retrains and never places orders.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from math import floor
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_symbol_session import (  # noqa: E402
    PREDICTIONS_ROOT,
    SNAPSHOT_ROOT,
    folder_files,
    out,
    read_frame,
    slot_file,
    symbol_price,
    trading_costs,
)
from prediction_universe import load_capital_plan, position_budget  # noqa: E402


def parquet_slots(files: list[dict[str, Any]], stamp: str) -> list[str]:
    slots = set()
    suffix = ".parquet"
    for item in files:
        name = str(item.get("name") or "")
        if not name.endswith(suffix):
            continue
        # .../predictions_YYYYMMDD_HHMM.parquet or snapshot_YYYYMMDD_HHMM.parquet
        parts = name[: -len(suffix)].split("_")
        if len(parts) < 2:
            continue
        day, slot = parts[-2], parts[-1]
        if day == stamp and len(slot) == 4 and slot.isdigit():
            slots.add(slot)
    return sorted(slots)


def shift(slot: str, minutes: int) -> str:
    parsed = datetime.strptime(slot, "%H%M")
    return (parsed + timedelta(minutes=minutes)).strftime("%H%M")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate one symbol's session")
    parser.add_argument("--symbol", default="SBIN")
    parser.add_argument("--date", default="")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--max-positions", type=int, default=0, help="0 = use capital plan")
    parser.add_argument("--gate", type=float, default=None,
                        help="only trade signals whose expected 30m move clears this magnitude (e.g. 0.015); "
                             "applied to BUY as expected>=gate and to SELL as expected<=-gate. "
                             "Default: take every BUY/SELL")
    parser.add_argument("--target-slot", default="",
                        help="fixed exit HH:MM; every signal issued from target-30m through target is held to it")
    parser.add_argument("--window-minutes", type=int, default=0,
                        help="how far back from the fixed exit a signal stays usable (0 = horizon)")
    args = parser.parse_args(argv)

    day = (
        datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=IST)
        if args.date
        else datetime.now(IST)
    )
    stamp = day.strftime("%Y%m%d")
    symbol = args.symbol.strip().upper().replace(".NS", "")

    from persistent_store import GoogleDriveStore

    store = GoogleDriveStore()
    service = store.service

    plan = load_capital_plan()
    budget = position_budget(plan)
    per_trade = float(budget["per_trade"])
    price_cap = float(budget["price_cap"])
    max_positions = args.max_positions or int(budget["positions"])

    pred_files = folder_files(store, service, PREDICTIONS_ROOT, day)
    snap_files = folder_files(store, service, SNAPSHOT_ROOT, day)
    pred_slots = parquet_slots(pred_files, stamp)
    snap_slots = set(parquet_slots(snap_files, stamp))

    out("SYMBOL", symbol)
    out("SESSION_DATE", day.strftime("%Y-%m-%d"))
    out("PREDICTION_SLOTS", len(pred_slots))
    out("SNAPSHOT_SLOTS", len(snap_slots))
    out("HORIZON_MINUTES", args.horizon)
    out("PER_TRADE", round(per_trade, 2))
    out("PRICE_CAP", round(price_cap, 2))
    out("MAX_POSITIONS", max_positions)
    out("GATE", "ALL_SIGNALS" if args.gate is None else args.gate)

    fixed_exit = ""
    if args.target_slot:
        # A 30-minute model refreshed every 5 minutes gives exactly one strict
        # horizon match per exit slot. But a signal issued at 13:35 is still
        # usable for a 13:45 exit -- you just hold it for 10 minutes instead of
        # 30. So the window is every slot from exit-window through exit.
        fixed_exit = args.target_slot.replace(":", "")
        window = args.window_minutes or args.horizon
        earliest = shift(fixed_exit, -window)
        pred_slots = [s for s in pred_slots if earliest <= s <= fixed_exit]
        out("TARGET_WINDOW_EXIT", fixed_exit)
        out("TARGET_WINDOW_START", earliest)
        out("TARGET_WINDOW_SLOTS", ",".join(pred_slots) if pred_slots else "NONE")

    def slot_price(slot: str) -> float | None:
        meta = slot_file(snap_files, stamp, slot)
        if meta is None:
            return None
        return symbol_price(read_frame(store, SNAPSHOT_ROOT, day, str(meta["name"])), symbol)

    open_until: list[str] = []
    tradeable = 0
    skipped_capacity = 0
    pending = 0
    holds = 0
    buys = sells = 0
    wins = losses = 0
    gross_total = cost_total = net_total = 0.0
    row_number = 0

    for slot in pred_slots:
        meta = slot_file(pred_files, stamp, slot)
        if meta is None:
            continue
        frame = read_frame(store, PREDICTIONS_ROOT, day, str(meta["name"]))
        if frame.empty or "symbol" not in frame.columns:
            continue
        rows = frame[
            frame["symbol"].astype(str).str.upper().str.replace(".NS", "", regex=False).str.strip() == symbol
        ]
        if rows.empty:
            continue
        row = rows.iloc[0]
        action = str(row.get("signal", "")).strip().upper() or "HOLD"
        try:
            probability = float(row.get("probability_up_30m"))
        except (TypeError, ValueError):
            probability = float("nan")
        try:
            expected = float(row.get("expected_return_30m"))
        except (TypeError, ValueError):
            expected = float("nan")
        try:
            readiness = float(row.get("data_readiness_score"))
        except (TypeError, ValueError):
            readiness = float("nan")

        target_slot = fixed_exit or shift(slot, args.horizon)
        entry = slot_price(slot)

        if action == "HOLD":
            holds += 1
            print(
                f"SLOT={slot} TARGET={target_slot} SIGNAL=HOLD P={probability:.4f} "
                f"EXP={expected:.4f} READY={readiness:.4f} ENTRY={entry} ACTION=NO_TRADE",
                flush=True,
            )
            continue

        if action == "BUY":
            buys += 1
        else:
            sells += 1

        if args.gate is not None:
            # A SELL is a bet on a fall, so its expected return is negative.
            # Gate on the magnitude of the predicted move, signed by direction.
            gated = expected >= args.gate if action == "BUY" else expected <= -args.gate
            if not gated:
                print(
                    f"SLOT={slot} TARGET={target_slot} SIGNAL={action} P={probability:.4f} "
                    f"EXP={expected:.4f} ENTRY={entry} ACTION=SKIP_GATE",
                    flush=True,
                )
                continue

        if entry is None or entry <= 0:
            print(f"SLOT={slot} TARGET={target_slot} SIGNAL={action} ACTION=SKIP_NO_ENTRY_PRICE", flush=True)
            continue
        if entry > price_cap:
            print(f"SLOT={slot} TARGET={target_slot} SIGNAL={action} ENTRY={entry} ACTION=SKIP_PRICE_CAP", flush=True)
            continue

        exit_price = slot_price(target_slot)
        if exit_price is None:
            pending += 1
            print(
                f"SLOT={slot} TARGET={target_slot} SIGNAL={action} P={probability:.4f} "
                f"EXP={expected:.4f} ENTRY={entry} ACTION=PENDING_TARGET_MISSING",
                flush=True,
            )
            continue

        open_until = [value for value in open_until if value > slot]
        if len(open_until) >= max_positions:
            skipped_capacity += 1
            print(
                f"SLOT={slot} TARGET={target_slot} SIGNAL={action} ENTRY={entry} "
                f"ACTION=SKIP_CAPACITY({len(open_until)}/{max_positions})",
                flush=True,
            )
            continue

        quantity = int(floor(per_trade // entry))
        if quantity < 1:
            print(f"SLOT={slot} TARGET={target_slot} SIGNAL={action} ACTION=SKIP_LOT_SIZE", flush=True)
            continue

        if action == "BUY":
            gross = (exit_price - entry) * quantity
            buy_value, sell_value = quantity * entry, quantity * exit_price
        else:
            gross = (entry - exit_price) * quantity
            sell_value, buy_value = quantity * entry, quantity * exit_price
        costs = trading_costs(buy_value, sell_value)
        net = gross - costs["total"]

        open_until.append(target_slot)
        tradeable += 1
        row_number += 1
        gross_total += gross
        cost_total += costs["total"]
        net_total += net
        wins += 1 if net > 0 else 0
        losses += 1 if net <= 0 else 0

        hold_minutes = int(
            (datetime.strptime(target_slot, "%H%M") - datetime.strptime(slot, "%H%M")).total_seconds() // 60
        )
        print(
            f"TRADE={row_number} SLOT={slot} ENTRY={entry} TARGET={target_slot} EXIT={round(exit_price, 2)} "
            f"SIGNAL={action} QTY={quantity} HOLD_MIN={hold_minutes} GROSS={round(gross, 2)} "
            f"COSTS={costs['total']} NET={round(net, 2)}",
            flush=True,
        )

    out("SIGNALS_TOTAL", buys + sells + holds)
    out("SIGNALS_BUY", buys)
    out("SIGNALS_SELL", sells)
    out("SIGNALS_HOLD", holds)
    out("TRADES_TAKEN", tradeable)
    out("SKIPPED_CAPACITY", skipped_capacity)
    out("PENDING_TARGET_MISSING", pending)
    out("WINS", wins)
    out("LOSSES", losses)
    out("TOTAL_GROSS_RS", round(gross_total, 2))
    out("TOTAL_COSTS_RS", round(cost_total, 2))
    out("TOTAL_NET_RS", round(net_total, 2))
    if tradeable:
        out("AVG_NET_PER_TRADE_RS", round(net_total / tradeable, 2))
        out("WIN_RATE_PCT", round(100.0 * wins / tradeable, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
