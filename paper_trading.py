"""Long/short NSE cash-equity paper trading, isolated from broker execution.

The ledger records simulated fills in SQLite. It never submits broker orders.
"""
from __future__ import annotations

import math
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_CAPITAL = 100_000.0
DEFAULT_FEE_BPS = 10.0
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_INTRADAY_ALLOCATION = 5_000.0
# NSE cash-market rates currently in force (2026).
NSE_TRANSACTION_CHARGE_RATE = 0.000030699
NSE_IPFT_CHARGE_RATE = 0.000000001
SEBI_TURNOVER_FEE_RATE = 0.000001
GST_RATE = 0.18
# Illustrative simulator assumptions; these are not exchange or broker margin rates.
INITIAL_SHORT_MARGIN_RATE = 1.50
MAINTENANCE_SHORT_MARGIN_RATE = 0.30
TRADING_SEGMENTS = ("INTRADAY", "DELIVERY", "LEGACY")


def _database_path() -> Path:
    configured = os.getenv("PAPER_TRADING_DB_PATH", "").strip()
    path = Path(configured) if configured else Path(__file__).resolve().parent / "data" / "paper_trading.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(str(_database_path()), timeout=20)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def _connection():
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _now_ist() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


def _money(value: float) -> float:
    """Round a rupee amount to paise using financial half-up rounding."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def calculate_trade_charges(
    segment: str,
    side: str,
    turnover: float,
    brokerage_bps: float,
) -> dict[str, float | str]:
    """Estimate Indian NSE cash-equity charges for a single executed side.

    Per-fill paise rounding is a simulator estimate; broker/exchange contract
    note aggregation may differ by a few paise. Brokerage is configurable.
    """
    selected_segment = str(segment).strip().upper()
    direction = str(side).strip().upper()
    amount = float(turnover)
    brokerage_rate = float(brokerage_bps)
    if selected_segment not in {"INTRADAY", "DELIVERY"}:
        raise ValueError("Charges can only be estimated for INTRADAY or DELIVERY.")
    if direction not in {"BUY", "SELL"}:
        raise ValueError("Side must be BUY or SELL.")
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("Turnover must be a finite non-negative amount.")
    if not math.isfinite(brokerage_rate) or not 0 <= brokerage_rate <= 500:
        raise ValueError("Brokerage rate must be between 0 and 500 bps.")

    brokerage = _money(amount * brokerage_rate / 10_000.0)
    if selected_segment == "INTRADAY":
        stt = _money(amount * 0.00025) if direction == "SELL" else 0.0
        stamp_duty = _money(amount * 0.00003) if direction == "BUY" else 0.0
    else:
        stt = _money(amount * 0.001)
        stamp_duty = _money(amount * 0.00015) if direction == "BUY" else 0.0

    exchange_charges = _money(amount * NSE_TRANSACTION_CHARGE_RATE)
    sebi_charges = _money(amount * SEBI_TURNOVER_FEE_RATE)
    ipft_charges = _money(amount * NSE_IPFT_CHARGE_RATE)
    gst = _money((brokerage + exchange_charges + sebi_charges + ipft_charges) * GST_RATE)
    total = _money(
        brokerage + stt + stamp_duty + exchange_charges
        + sebi_charges + ipft_charges + gst
    )
    return {
        "brokerage": brokerage,
        "stt": stt,
        "stamp_duty": stamp_duty,
        "exchange_charges": exchange_charges,
        "sebi_charges": sebi_charges,
        "ipft_charges": ipft_charges,
        "gst": gst,
        "total_charges": total,
        "charge_model_version": "NSE_CASH_EQUITY_2026",
    }


def normalize_symbol(value: str) -> str:
    """Return a bare NSE symbol; the simulator intentionally excludes indices."""
    symbol = str(value or "").strip().upper()
    for suffix in (".NS", ".BO"):
        if symbol.endswith(suffix):
            symbol = symbol[:-len(suffix)]
    if not symbol or len(symbol) > 30 or not re.fullmatch(r"[A-Z0-9&._-]+", symbol):
        raise ValueError("Enter a valid NSE cash-equity symbol, for example SBIN or RELIANCE.")
    if symbol.startswith("^"):
        raise ValueError("Index instruments cannot be traded in this cash-equity simulator.")
    return symbol


def init_db() -> None:
    with _connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_account (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                initial_cash REAL NOT NULL,
                fee_bps REAL NOT NULL,
                slippage_bps REAL NOT NULL,
                created_at TEXT NOT NULL,
                intraday_initial_cash REAL NOT NULL DEFAULT 0,
                delivery_initial_cash REAL NOT NULL DEFAULT 0,
                legacy_initial_cash REAL NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ist TEXT NOT NULL,
                symbol TEXT NOT NULL,
                segment TEXT NOT NULL DEFAULT 'LEGACY',
                side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                reference_price REAL NOT NULL,
                fill_price REAL NOT NULL,
                gross REAL NOT NULL,
                charges REAL NOT NULL,
                brokerage REAL,
                stt REAL,
                stamp_duty REAL,
                exchange_charges REAL,
                sebi_charges REAL,
                ipft_charges REAL,
                gst REAL,
                charge_model_version TEXT,
                source TEXT NOT NULL DEFAULT '',
                quote_timestamp TEXT NOT NULL DEFAULT '',
                prediction_signal TEXT NOT NULL DEFAULT '',
                prediction_probability_up REAL,
                prediction_expected_return REAL,
                prediction_model_version TEXT NOT NULL DEFAULT '',
                prediction_decision_timestamp TEXT NOT NULL DEFAULT '',
                prediction_horizon_min INTEGER
            )
            """
        )
        # Existing fills have no strategy classification. Preserve them in an
        # explicit LEGACY bucket instead of guessing Intraday or Delivery.
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(paper_fills)").fetchall()
        }
        if "segment" not in columns:
            connection.execute(
                "ALTER TABLE paper_fills ADD COLUMN segment TEXT NOT NULL DEFAULT 'LEGACY'"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_fills_symbol_id ON paper_fills(symbol, id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_fills_segment_id ON paper_fills(segment, id)"
        )

        account_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(paper_account)").fetchall()
        }
        added_allocations = False
        for column in ("intraday_initial_cash", "delivery_initial_cash", "legacy_initial_cash"):
            if column not in account_columns:
                connection.execute(
                    f"ALTER TABLE paper_account ADD COLUMN {column} REAL NOT NULL DEFAULT 0"
                )
                added_allocations = True
        if added_allocations:
            account = connection.execute(
                "SELECT initial_cash FROM paper_account WHERE id = 1"
            ).fetchone()
            if account:
                total_capital = float(account["initial_cash"])
                has_legacy_fills = connection.execute(
                    "SELECT 1 FROM paper_fills WHERE segment = 'LEGACY' LIMIT 1"
                ).fetchone() is not None
                intraday_capital = min(DEFAULT_INTRADAY_ALLOCATION, total_capital)
                residual_capital = total_capital - intraday_capital
                if has_legacy_fills:
                    delivery_capital = 0.0
                    legacy_capital = residual_capital
                else:
                    delivery_capital = residual_capital
                    legacy_capital = 0.0
                connection.execute(
                    """
                    UPDATE paper_account
                    SET intraday_initial_cash = ?,
                        delivery_initial_cash = ?,
                        legacy_initial_cash = ?
                    WHERE id = 1
                    """,
                    (intraday_capital, delivery_capital, legacy_capital),
                )

        fill_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(paper_fills)").fetchall()
        }
        charge_columns = {
            "brokerage": "REAL",
            "stt": "REAL",
            "stamp_duty": "REAL",
            "exchange_charges": "REAL",
            "sebi_charges": "REAL",
            "ipft_charges": "REAL",
            "gst": "REAL",
            "charge_model_version": "TEXT",
        }
        for column, definition in charge_columns.items():
            if column not in fill_columns:
                connection.execute(
                    f"ALTER TABLE paper_fills ADD COLUMN {column} {definition}"
                )

        old_charge_rows = connection.execute(
            """
            SELECT id, segment, side, gross, charges
            FROM paper_fills
            WHERE charge_model_version IS NULL
            ORDER BY id
            """
        ).fetchall()
        for row in old_charge_rows:
            segment = str(row["segment"] or "LEGACY").upper()
            if segment in {"INTRADAY", "DELIVERY"}:
                # Preserve historic estimate as brokerage, then add statutory
                # levies that the previous one-rate model omitted.
                brokerage = _money(max(0.0, float(row["charges"])) )
                statutory = calculate_trade_charges(
                    segment, str(row["side"]), float(row["gross"]), 0.0
                )
                exchange = float(statutory["exchange_charges"])
                sebi = float(statutory["sebi_charges"])
                ipft = float(statutory["ipft_charges"])
                gst = _money((brokerage + exchange + sebi + ipft) * GST_RATE)
                stt = float(statutory["stt"])
                stamp_duty = float(statutory["stamp_duty"])
                total = _money(
                    brokerage + stt + stamp_duty + exchange + sebi + ipft + gst
                )
                connection.execute(
                    """
                    UPDATE paper_fills
                    SET brokerage = ?, stt = ?, stamp_duty = ?,
                        exchange_charges = ?, sebi_charges = ?,
                        ipft_charges = ?, gst = ?, charges = ?,
                        charge_model_version = 'NSE_CASH_EQUITY_2026_MIGRATED'
                    WHERE id = ?
                    """,
                    (
                        brokerage, stt, stamp_duty, exchange, sebi,
                        ipft, gst, total, int(row["id"]),
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE paper_fills
                    SET charge_model_version = 'LEGACY_UNITEMIZED'
                    WHERE id = ?
                    """,
                    (int(row["id"]),),
                )


def get_account() -> dict | None:
    init_db()
    with _connection() as connection:
        row = connection.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
    return dict(row) if row else None


def _segment_capitals_from_account(account: dict) -> dict[str, float]:
    total = float(account["initial_cash"])
    values = {
        "INTRADAY": float(account.get("intraday_initial_cash", min(DEFAULT_INTRADAY_ALLOCATION, total))),
        "DELIVERY": float(account.get("delivery_initial_cash", max(0.0, total - DEFAULT_INTRADAY_ALLOCATION))),
        "LEGACY": float(account.get("legacy_initial_cash", 0.0)),
    }
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("Paper account contains an invalid strategy cash allocation.")
    if abs(sum(values.values()) - total) > 0.01:
        raise ValueError("Strategy cash allocations must equal total starting capital.")
    return values


def create_account(initial_cash: float, fee_bps: float = DEFAULT_FEE_BPS,
                   slippage_bps: float = DEFAULT_SLIPPAGE_BPS) -> None:
    values = (float(initial_cash), float(fee_bps), float(slippage_bps))
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Account settings must be finite numbers.")
    if values[0] <= 0 or values[1] < 0 or values[2] < 0 or values[1] > 500 or values[2] > 500:
        raise ValueError("Capital must be positive; fee and slippage must be between 0 and 500 bps.")
    intraday_capital = min(DEFAULT_INTRADAY_ALLOCATION, values[0])
    delivery_capital = values[0] - intraday_capital
    init_db()
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO paper_account(
                id, initial_cash, fee_bps, slippage_bps, created_at,
                intraday_initial_cash, delivery_initial_cash, legacy_initial_cash
            )
            VALUES (1, ?, ?, ?, ?, ?, ?, 0)
            """,
            (*values, _now_ist(), intraday_capital, delivery_capital),
        )


def update_segment_capital_allocations(
    intraday_initial_cash: float,
    delivery_initial_cash: float,
    legacy_initial_cash: float = 0.0,
) -> None:
    values = tuple(float(value) for value in (
        intraday_initial_cash, delivery_initial_cash, legacy_initial_cash
    ))
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Strategy allocations must be finite, non-negative amounts.")
    with _connection() as connection:
        account = connection.execute(
            "SELECT initial_cash FROM paper_account WHERE id = 1"
        ).fetchone()
        if not account:
            raise RuntimeError("Paper account is not initialized.")
        total = float(account["initial_cash"])
        if abs(sum(values) - total) > 0.01:
            raise ValueError(
                f"Allocations must total ₹{total:,.2f}; supplied total is ₹{sum(values):,.2f}."
            )
        connection.execute(
            """
            UPDATE paper_account
            SET intraday_initial_cash = ?, delivery_initial_cash = ?,
                legacy_initial_cash = ?
            WHERE id = 1
            """,
            values,
        )


def update_cost_settings(fee_bps: float, slippage_bps: float) -> None:
    fee, slippage = float(fee_bps), float(slippage_bps)
    if not math.isfinite(fee) or not math.isfinite(slippage) or not (0 <= fee <= 500) or not (0 <= slippage <= 500):
        raise ValueError("Fee and slippage must be between 0 and 500 bps.")
    with _connection() as connection:
        cursor = connection.execute(
            "UPDATE paper_account SET fee_bps = ?, slippage_bps = ? WHERE id = 1",
            (fee, slippage),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("Paper account is not initialized.")


def reset_account(initial_cash: float) -> None:
    capital = float(initial_cash)
    if not math.isfinite(capital) or capital <= 0:
        raise ValueError("Starting capital must be a positive finite amount.")
    with _connection() as connection:
        account = connection.execute("SELECT fee_bps, slippage_bps FROM paper_account WHERE id = 1").fetchone()
        if not account:
            raise RuntimeError("Paper account is not initialized.")
        connection.execute("DELETE FROM paper_fills")
        intraday_capital = min(DEFAULT_INTRADAY_ALLOCATION, capital)
        delivery_capital = capital - intraday_capital
        connection.execute(
            """
            UPDATE paper_account
            SET initial_cash = ?, created_at = ?,
                intraday_initial_cash = ?, delivery_initial_cash = ?,
                legacy_initial_cash = 0
            WHERE id = 1
            """,
            (capital, _now_ist(), intraday_capital, delivery_capital),
        )


def list_fills(limit: int | None = None, segment: str | None = None) -> list[dict]:
    init_db()
    query = "SELECT * FROM paper_fills"
    params: list = []
    if segment is not None:
        normalized_segment = str(segment).strip().upper()
        if normalized_segment not in TRADING_SEGMENTS:
            raise ValueError("Segment must be INTRADAY, DELIVERY or LEGACY.")
        query += " WHERE segment = ?"
        params.append(normalized_segment)
    query += " ORDER BY id DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(max(1, int(limit)))
    with _connection() as connection:
        rows = connection.execute(query, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def calculate_portfolio(
    initial_cash: float,
    fills: list[dict],
    market_prices: dict[str, float] | None = None,
) -> dict:
    """Rebuild a netted long/short portfolio from immutable BUY/SELL fills.

    BUY covers an existing short before opening a long. SELL closes an existing
    long before opening a short. Entry charges are capitalised into position
    basis; exit charges reduce realised P&L. Short-sale proceeds remain in cash
    but are encumbered by the initial-margin reserve.
    """
    cash = float(initial_cash)
    positions: dict[str, dict[str, float]] = {}
    realised = 0.0

    for fill in sorted(fills, key=lambda item: int(item.get("id", 0))):
        symbol = normalize_symbol(fill["symbol"])
        side = str(fill["side"]).strip().upper()
        quantity = int(fill["quantity"])
        fill_price = float(fill["fill_price"])
        gross = float(fill.get("gross", quantity * fill_price))
        charges = float(fill.get("charges", 0.0))
        if (
            quantity <= 0
            or not all(math.isfinite(x) for x in (fill_price, gross, charges))
            or fill_price <= 0
            or gross < 0
            or charges < 0
            or side not in {"BUY", "SELL"}
        ):
            raise ValueError("Paper fill ledger contains invalid trade or numeric values.")

        position = positions.setdefault(symbol, {"quantity": 0, "cost_basis": 0.0})
        signed_qty = int(position["quantity"])
        basis = float(position["cost_basis"])
        unit_gross = gross / quantity
        fee_per_share = charges / quantity
        remaining = quantity

        if side == "BUY":
            cash -= gross + charges
            # First cover any short shares. Short basis is net sale proceeds
            # after the entry fee, so closing P&L includes both sides' charges.
            if signed_qty < 0:
                covered = min(remaining, -signed_qty)
                short_basis_per_share = basis / (-signed_qty)
                realised += (
                    short_basis_per_share - unit_gross - fee_per_share
                ) * covered
                basis -= short_basis_per_share * covered
                signed_qty += covered
                remaining -= covered
            # Any excess BUY opens or adds to a long position.
            if remaining:
                signed_qty += remaining
                basis += (unit_gross + fee_per_share) * remaining

        else:  # SELL
            cash += gross - charges
            # First close any existing long shares.
            if signed_qty > 0:
                closed = min(remaining, signed_qty)
                long_basis_per_share = basis / signed_qty
                realised += (
                    unit_gross - fee_per_share - long_basis_per_share
                ) * closed
                basis -= long_basis_per_share * closed
                signed_qty -= closed
                remaining -= closed
            # Any excess SELL opens or adds to a short position. Proceeds are
            # net of the entry fee and form the short position's cover basis.
            if remaining:
                signed_qty -= remaining
                basis += (unit_gross - fee_per_share) * remaining

        if signed_qty == 0:
            basis = 0.0
        position["quantity"] = signed_qty
        position["cost_basis"] = max(0.0, basis)

    open_positions: dict[str, dict[str, float | int | str]] = {}
    long_market_value = 0.0
    short_liability = 0.0
    unrealised = 0.0
    market_prices = market_prices or {}

    for symbol, values in positions.items():
        quantity = int(values["quantity"])
        if quantity == 0:
            continue
        basis = float(values["cost_basis"])
        units = abs(quantity)
        average_cost = basis / units
        mark = _finite_or_none(market_prices.get(symbol))
        if mark is None or mark <= 0:
            mark = average_cost
        position_value = units * mark
        position_pnl = (
            (mark - average_cost) * units
            if quantity > 0
            else (average_cost - mark) * units
        )
        unrealised += position_pnl
        if quantity > 0:
            long_market_value += position_value
            side_name = "LONG"
        else:
            short_liability += position_value
            side_name = "SHORT"
        open_positions[symbol] = {
            "side": side_name,
            "quantity": quantity,
            "cost_basis": _money(basis),
            "average_cost": average_cost,
            "mark_price": mark,
            "market_value": position_value,
            "unrealised_pnl": position_pnl,
        }

    equity = cash + long_market_value - short_liability
    initial_margin = short_liability * INITIAL_SHORT_MARGIN_RATE
    maintenance_margin = short_liability * MAINTENANCE_SHORT_MARGIN_RATE
    available_cash = cash - initial_margin
    margin_excess = equity - maintenance_margin
    return {
        "starting_capital": float(initial_cash),
        "cash": cash,
        "positions": open_positions,
        "realised_pnl": realised,
        "unrealised_pnl": unrealised,
        "long_market_value": long_market_value,
        "short_liability": short_liability,
        "equity": equity,
        "initial_margin_required": initial_margin,
        "maintenance_margin_required": maintenance_margin,
        "available_cash": available_cash,
        "margin_excess": margin_excess,
        "margin_status": "CALL" if short_liability > 0 and margin_excess < -0.005 else "OK",
    }


def calculate_segmented_portfolio(
    initial_cash: float,
    fills: list[dict],
    market_prices: dict[str, float] | None = None,
    segment_capitals: dict[str, float] | None = None,
) -> dict:
    """Calculate isolated books and aggregate their cash and equity.

    If segment_capitals is supplied, each book starts with its own allocation.
    Omitting it retains compatibility with previous callers.
    """
    grouped: dict[str, list[dict]] = {name: [] for name in TRADING_SEGMENTS}
    for fill in fills:
        segment = str(fill.get("segment") or "LEGACY").strip().upper()
        if segment not in TRADING_SEGMENTS:
            segment = "LEGACY"
        grouped[segment].append(fill)

    capital_map: dict[str, float] | None = None
    if segment_capitals is not None:
        capital_map = {
            name: float(segment_capitals.get(name, 0.0))
            for name in TRADING_SEGMENTS
        }
        if any(not math.isfinite(value) or value < 0 for value in capital_map.values()):
            raise ValueError("Strategy allocations must be finite, non-negative amounts.")
        if abs(sum(capital_map.values()) - float(initial_cash)) > 0.01:
            raise ValueError("Strategy allocations must equal total starting capital.")

    segment_states = {
        name: calculate_portfolio(
            capital_map[name] if capital_map is not None else 0.0,
            rows,
            market_prices=market_prices,
        )
        for name, rows in grouped.items()
    }
    starting_capital = (
        sum(capital_map.values()) if capital_map is not None else float(initial_cash)
    )
    cash = (
        sum(state["cash"] for state in segment_states.values())
        if capital_map is not None
        else starting_capital + sum(state["cash"] for state in segment_states.values())
    )
    long_market_value = sum(state["long_market_value"] for state in segment_states.values())
    short_liability = sum(state["short_liability"] for state in segment_states.values())
    realised = sum(state["realised_pnl"] for state in segment_states.values())
    unrealised = sum(state["unrealised_pnl"] for state in segment_states.values())
    initial_margin = sum(state["initial_margin_required"] for state in segment_states.values())
    maintenance_margin = sum(state["maintenance_margin_required"] for state in segment_states.values())
    equity = cash + long_market_value - short_liability
    available_cash = cash - initial_margin
    margin_excess = equity - maintenance_margin
    return {
        "starting_capital": round(starting_capital, 2),
        "cash": _money(cash),
        "segments": segment_states,
        "realised_pnl": _money(realised),
        "unrealised_pnl": _money(unrealised),
        "long_market_value": _money(long_market_value),
        "short_liability": _money(short_liability),
        "equity": _money(equity),
        "initial_margin_required": _money(initial_margin),
        "maintenance_margin_required": _money(maintenance_margin),
        "available_cash": _money(available_cash),
        "margin_excess": _money(margin_excess),
        "margin_status": "CALL" if short_liability > 0 and margin_excess < -0.005 else "OK",
    }


def place_order(
    symbol: str,
    side: str,
    quantity: int,
    reference_price: float,
    account: dict,
    *,
    source: str = "",
    quote_timestamp: str = "",
    prediction: dict | None = None,
    market_prices: dict[str, float] | None = None,
    segment: str = "DELIVERY",
) -> dict:
    """Create a simulated fill in the selected strategy book."""
    root = normalize_symbol(symbol)
    selected_segment = str(segment).strip().upper()
    if selected_segment not in {"INTRADAY", "DELIVERY"}:
        raise ValueError("New fills must belong to INTRADAY or DELIVERY.")
    direction = str(side).strip().upper()
    if direction not in {"BUY", "SELL"}:
        raise ValueError("Side must be BUY or SELL.")
    if isinstance(quantity, bool) or int(quantity) != quantity or int(quantity) <= 0 or int(quantity) > 1_000_000:
        raise ValueError("Quantity must be a whole number between 1 and 1,000,000.")
    qty = int(quantity)
    reference = float(reference_price)
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("Reference price must be a positive finite number.")

    fee_bps = float(account["fee_bps"])
    slippage_bps = float(account["slippage_bps"])
    if not all(math.isfinite(x) and 0 <= x <= 500 for x in (fee_bps, slippage_bps)):
        raise ValueError("Invalid account fee or slippage settings.")
    impact = slippage_bps / 10_000.0
    fill_price = _money(reference * (1.0 + impact if direction == "BUY" else 1.0 - impact))
    gross = _money(qty * fill_price)
    charge_breakdown = calculate_trade_charges(
        selected_segment, direction, gross, fee_bps
    )
    charges = float(charge_breakdown["total_charges"])

    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        current_account = connection.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
        if not current_account:
            raise RuntimeError("Paper account is not initialized.")
        fills = [
            dict(row) for row in connection.execute("SELECT * FROM paper_fills ORDER BY id").fetchall()
        ]
        initial_cash = float(current_account["initial_cash"])
        segment_capitals = _segment_capitals_from_account(dict(current_account))
        mark_prices = dict(market_prices or {})
        mark_prices[root] = fill_price
        before = calculate_segmented_portfolio(
            initial_cash, fills, market_prices=mark_prices,
            segment_capitals=segment_capitals,
        )
        before_book = before["segments"][selected_segment]
        before_qty = int(before_book["positions"].get(root, {}).get("quantity", 0))

        model = prediction or {}
        candidate = {
            "id": max((int(item.get("id", 0)) for item in fills), default=0) + 1,
            "symbol": root,
            "segment": selected_segment,
            "side": direction,
            "quantity": qty,
            "fill_price": fill_price,
            "gross": gross,
            "charges": charges,
        }
        after = calculate_segmented_portfolio(
            initial_cash,
            fills + [candidate],
            market_prices=mark_prices,
            segment_capitals=segment_capitals,
        )
        after_book = after["segments"][selected_segment]
        after_qty = int(after_book["positions"].get(root, {}).get("quantity", 0))
        increased_long = max(after_qty, 0) > max(before_qty, 0)
        increased_short = max(-after_qty, 0) > max(-before_qty, 0)

        # Existing positions can always be reduced or closed, including during
        # a margin call. New long or short exposure must fit available cash.
        if (increased_long or increased_short) and after_book["available_cash"] < -0.005:
            raise ValueError(
                f"Insufficient free cash in the {selected_segment.title()} book after "
                "reserving initial short margin. "
                f"Book cash: ₹{after_book['cash']:,.2f}; "
                f"required reserve: ₹{after_book['initial_margin_required']:,.2f}; "
                f"free cash: ₹{after_book['available_cash']:,.2f}."
            )
        if increased_short and after_book["margin_excess"] < -0.005:
            raise ValueError(
                f"Short order rejected: projected {selected_segment.title()} equity "
                "would fall below its maintenance-margin requirement by "
                f"₹{abs(after_book['margin_excess']):,.2f}."
            )

        cursor = connection.execute(
            """
            INSERT INTO paper_fills(
                timestamp_ist, symbol, segment, side, quantity, reference_price, fill_price,
                gross, charges, brokerage, stt, stamp_duty, exchange_charges,
                sebi_charges, ipft_charges, gst, charge_model_version,
                source, quote_timestamp, prediction_signal,
                prediction_probability_up, prediction_expected_return,
                prediction_model_version, prediction_decision_timestamp,
                prediction_horizon_min
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now_ist(), root, selected_segment, direction, qty, reference,
                fill_price, gross, charges,
                float(charge_breakdown["brokerage"]),
                float(charge_breakdown["stt"]),
                float(charge_breakdown["stamp_duty"]),
                float(charge_breakdown["exchange_charges"]),
                float(charge_breakdown["sebi_charges"]),
                float(charge_breakdown["ipft_charges"]),
                float(charge_breakdown["gst"]),
                str(charge_breakdown["charge_model_version"]),
                str(source or ""), str(quote_timestamp or ""),
                str(model.get("signal") or ""),
                _finite_or_none(model.get("probability_up")),
                _finite_or_none(model.get("expected_return")),
                str(model.get("model_version") or ""),
                str(model.get("decision_timestamp") or ""),
                _int_or_none(model.get("horizon_min")),
            ),
        )
        row = connection.execute("SELECT * FROM paper_fills WHERE id = ?", (cursor.lastrowid,)).fetchone()
        connection.commit()
        return dict(row)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _finite_or_none(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _select_prediction_for_horizon(
    predictions: list[dict],
    horizon_min: int,
    preferred: dict | None = None,
) -> dict:
    """Select only a prediction that explicitly matches the requested minute horizon."""
    candidates = ([preferred] if isinstance(preferred, dict) and preferred else []) + list(predictions or [])
    for item in candidates:
        if _int_or_none(item.get("horizon_min")) == int(horizon_min):
            return dict(item)
    return {}


def _intraday_horizon_table_rows(
    predictions: list[dict] | None,
    unavailable_reason: str = "",
) -> list[dict]:
    """Build a stable six-row table; missing horizons remain explicitly unavailable."""
    by_horizon = {}
    for item in predictions or []:
        horizon = _int_or_none(item.get("horizon_min"))
        if horizon in (5, 10, 15, 20, 25, 30):
            by_horizon[horizon] = dict(item)

    rows = []
    for horizon in (5, 10, 15, 20, 25, 30):
        item = by_horizon.get(horizon)
        if item is None:
            rows.append({
                "Horizon": f"+{horizon} min",
                "Signal": "UNAVAILABLE",
                "P(up)": "—",
                "Expected return": "—",
                "Forecast price": "—",
                "Feature coverage": "—",
                "Model status": "NOT RUN",
                "Reason": unavailable_reason or "No output returned for this horizon.",
            })
            continue

        probability = _finite_or_none(item.get("probability_up"))
        expected_return = _finite_or_none(item.get("expected_return"))
        predicted_price = _finite_or_none(item.get("predicted_price"))
        coverage = _finite_or_none(item.get("feature_coverage"))
        signal = str(item.get("signal") or "ABSTAIN").upper()
        reason = str(item.get("reason") or item.get("signal_reason") or "")
        if signal == "ABSTAIN":
            model_status = "ABSTAIN"
        elif predicted_price is None:
            model_status = str(item.get("price_forecast_status") or "SIGNAL ONLY")
        else:
            model_status = "OUTPUT"
        rows.append({
            "Horizon": f"+{horizon} min",
            "Signal": signal,
            "P(up)": f"{probability:.1%}" if probability is not None else "—",
            "Expected return": f"{expected_return:+.2%}" if expected_return is not None else "—",
            "Forecast price": f"₹{predicted_price:,.2f}" if predicted_price is not None else "—",
            "Feature coverage": f"{coverage:.1%}" if coverage is not None else "—",
            "Model status": model_status,
            "Reason": reason,
        })
    return rows


def _format_timestamp(value: str):
    if not value:
        return "—"
    try:
        return pd.Timestamp(value).strftime("%d %b %Y, %H:%M:%S")
    except Exception:
        return str(value)


def _fetch_live_market_marks(
    fills: list[dict],
    selected_snapshot: dict,
    snapshot_loader,
    refresh_revision: int,
) -> dict[str, float]:
    """Fetch one usable mark per symbol held across all strategy books."""
    symbols: list[str] = []
    grouped: dict[str, list[dict]] = {name: [] for name in TRADING_SEGMENTS}
    for fill in fills:
        segment = str(fill.get("segment") or "LEGACY").upper()
        if segment not in TRADING_SEGMENTS:
            segment = "LEGACY"
        grouped[segment].append(fill)
    for rows in grouped.values():
        book_state = calculate_portfolio(0.0, rows)
        for symbol in book_state["positions"]:
            if symbol not in symbols:
                symbols.append(symbol)

    selected_symbol = str((selected_snapshot or {}).get("symbol") or "").upper()
    prices: dict[str, float] = {}
    for symbol in symbols:
        quote = None
        if symbol == selected_symbol:
            quote = _finite_or_none((selected_snapshot or {}).get("price"))
        if quote is None or quote <= 0:
            try:
                other = snapshot_loader(f"{symbol}.NS", refresh_revision, False) or {}
                quote = _finite_or_none(other.get("price"))
            except Exception:
                quote = None
        if quote is not None and quote > 0:
            prices[symbol] = float(quote)
    return prices


def _open_positions_with_marks(
    state: dict,
    snapshot: dict | None,
    snapshot_loader,
    refresh_revision: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fetch best available marks; unpriced positions fall back to entry basis."""
    rows = []
    market_prices: dict[str, float] = {}
    selected_symbol = str((snapshot or {}).get("symbol") or "").upper()
    for symbol, position in state["positions"].items():
        mark = None
        quote_time = ""
        if symbol == selected_symbol:
            mark = _finite_or_none((snapshot or {}).get("price"))
            quote_time = str((snapshot or {}).get("quote_timestamp") or "")
        if mark is None or mark <= 0:
            try:
                other = snapshot_loader(f"{symbol}.NS", refresh_revision, False) or {}
                mark = _finite_or_none(other.get("price"))
                quote_time = str(other.get("quote_timestamp") or "")
            except Exception:
                mark = None
        if mark is None or mark <= 0:
            mark = float(position["average_cost"])
            quote_time = "Entry basis (no usable market mark)"
            mark_source = "ENTRY_BASIS_FALLBACK"
        else:
            mark_source = "LIVE_QUOTE"
        market_prices[symbol] = float(mark)

        signed_qty = int(position["quantity"])
        quantity = abs(signed_qty)
        side = "LONG" if signed_qty > 0 else "SHORT"
        value = quantity * mark
        average_cost = float(position["average_cost"])
        pnl = (
            (mark - average_cost) * quantity
            if signed_qty > 0
            else (average_cost - mark) * quantity
        )
        rows.append({
            "Price source": mark_source,
            "Symbol": symbol,
            "Position": side,
            "Quantity": quantity,
            "Average entry": round(average_cost, 2),
            "Mark price": round(mark, 2),
            "Long market value": round(value, 2) if signed_qty > 0 else 0.0,
            "Short liability": round(value, 2) if signed_qty < 0 else 0.0,
            "Unrealised P&L": round(pnl, 2),
            "Unrealised return %": round(
                pnl / float(position["cost_basis"]) * 100
                if position["cost_basis"] else 0,
                2,
            ),
            "Price timestamp": quote_time,
        })
    return pd.DataFrame(rows), market_prices


def _render_live_market_overview(
    fills: list[dict],
    selected_snapshot: dict,
    snapshot_loader,
    revision: int,
) -> None:
    """Show live/last available marks for held shares and a compact NSE watchlist."""
    symbols = list(dict.fromkeys(
        [str(row.get("symbol") or "").upper() for row in fills]
        + ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "IDEA"]
    ))
    selected_symbol = str((selected_snapshot or {}).get("symbol") or "").upper()
    if selected_symbol:
        symbols = [selected_symbol] + [value for value in symbols if value != selected_symbol]

    with st.expander("Live NSE market watch", expanded=True):
        refresh = st.button("Refresh market watch", key="paper_refresh_market_watch")
        if refresh:
            try:
                snapshot_loader.clear()
            except Exception:
                pass
            st.session_state["paper_refresh_revision"] = int(
                st.session_state.get("paper_refresh_revision", revision)
            ) + 1
            st.rerun()

        rows = []
        for root in symbols:
            try:
                snapshot = (
                    selected_snapshot if root == selected_symbol
                    else snapshot_loader(f"{root}.NS", revision, False) or {}
                )
                quote = _finite_or_none(snapshot.get("price"))
                if quote is None or quote <= 0:
                    rows.append({
                        "Symbol": root,
                        "Last price": None,
                        "Change %": None,
                        "Source": snapshot.get("price_source") or snapshot.get("source") or "UNAVAILABLE",
                        "Quote time": "Unavailable",
                        "Status": snapshot.get("price_freshness_status") or "NO_DATA",
                    })
                    continue
                change = _finite_or_none(snapshot.get("change_percent"))
                rows.append({
                    "Symbol": root,
                    "Last price": round(float(quote), 2),
                    "Change %": round(float(change), 2) if change is not None else None,
                    "Source": snapshot.get("price_source") or snapshot.get("source") or "UNKNOWN",
                    "Quote time": _format_timestamp(str(snapshot.get("quote_timestamp") or "")),
                    "Status": (
                        "FRESH" if snapshot.get("price_fresh")
                        else snapshot.get("price_freshness_status") or "UNVERIFIED"
                    ),
                })
            except Exception as exc:
                rows.append({
                    "Symbol": root, "Last price": None, "Change %": None,
                    "Source": "ERROR", "Quote time": "—",
                    "Status": f"{type(exc).__name__}",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "Quotes are fetched through the project's configured NSE/broker/Yahoo adapter. "
            "Freshness applies to the completed candle; standalone quote fallbacks are marked unverified."
        )


def render_paper_trading(snapshot_loader, delivery_prediction_loader=None) -> None:
    """Render separate Intraday and Delivery books using existing project models."""
    init_db()
    st.title("Paper Trading")
    st.caption(
        "Simulated NSE cash-equity trading. Every fill is manual, simulated and stored separately from production data. "
        "Live quotes use the project's NSE → authenticated broker → Dhan → Yahoo fallback chain."
    )

    account = get_account()
    if account is None:
        st.info("Create a virtual account. No broker account or real money is involved.")
        with st.form("paper_create_account"):
            capital = st.number_input("Starting virtual capital (₹)", min_value=1_000.0, value=DEFAULT_CAPITAL, step=10_000.0)
            fee = st.number_input("Brokerage estimate (bps of turnover)", min_value=0.0, max_value=500.0, value=DEFAULT_FEE_BPS, step=1.0)
            slippage = st.number_input("Simulated slippage (bps)", min_value=0.0, max_value=500.0, value=DEFAULT_SLIPPAGE_BPS, step=1.0)
            submitted = st.form_submit_button("Create paper account", type="primary")
        if submitted:
            try:
                create_account(capital, fee, slippage)
                st.success("Paper account created.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not create paper account: {type(exc).__name__}: {exc}")
        return

    with st.expander("Account settings and reset", expanded=False):
        left, right = st.columns(2)
        with left:
            with st.form("paper_cost_settings"):
                fee_value = st.number_input(
                    "Brokerage estimate (bps of turnover)", min_value=0.0, max_value=500.0,
                    value=float(account["fee_bps"]), step=1.0,
                )
                slippage_value = st.number_input(
                    "Simulated slippage (bps)", min_value=0.0, max_value=500.0,
                    value=float(account["slippage_bps"]), step=1.0,
                )
                save_costs = st.form_submit_button("Save cost settings")
            if save_costs:
                try:
                    update_cost_settings(fee_value, slippage_value)
                    st.success("Cost settings saved.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save settings: {type(exc).__name__}: {exc}")
        with right:
            with st.form("paper_reset_account"):
                reset_capital = st.number_input(
                    "New starting capital (₹)", min_value=1_000.0,
                    value=float(account["initial_cash"]), step=10_000.0,
                )
                confirm_reset = st.checkbox("I understand that reset permanently clears Intraday, Delivery and Legacy paper fills.")
                reset = st.form_submit_button("Reset paper account")
            if reset:
                if not confirm_reset:
                    st.error("Tick the confirmation box before resetting.")
                else:
                    try:
                        reset_account(reset_capital)
                        st.success("All paper-trading tracks reset.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not reset account: {type(exc).__name__}: {exc}")

    segment_capitals = _segment_capitals_from_account(account)
    with st.expander("Strategy cash allocations", expanded=False):
        st.caption(
            f"Allocate the total starting capital of ₹{float(account['initial_cash']):,.2f} "
            "across Intraday, Delivery and Legacy. The three values must add up to the "
            "account starting capital. This changes book allocations only; it does not "
            "delete or rewrite previous fills."
        )
        with st.form("paper_capital_allocations"):
            alloc1, alloc2, alloc3 = st.columns(3)
            intraday_allocation = alloc1.number_input(
                "Intraday starting cash (₹)", min_value=0.0,
                value=float(segment_capitals["INTRADAY"]), step=500.0,
            )
            delivery_allocation = alloc2.number_input(
                "Delivery starting cash (₹)", min_value=0.0,
                value=float(segment_capitals["DELIVERY"]), step=1_000.0,
            )
            legacy_allocation = alloc3.number_input(
                "Legacy starting cash (₹)", min_value=0.0,
                value=float(segment_capitals["LEGACY"]), step=500.0,
            )
            save_allocations = st.form_submit_button("Save cash allocations")
        if save_allocations:
            try:
                update_segment_capital_allocations(
                    intraday_allocation, delivery_allocation, legacy_allocation
                )
                st.success("Strategy cash allocations saved.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save cash allocations: {type(exc).__name__}: {exc}")

    segment = st.radio(
        "Trading track",
        ["INTRADAY", "DELIVERY"],
        horizontal=True,
        key="paper_trading_segment",
        format_func=lambda value: "Intraday · 30-minute forecast" if value == "INTRADAY" else "Delivery · 7-session OHLC forecast",
    )
    st.caption(
        "Intraday, Delivery and Legacy use separately allocated cash and margin. "
        "The combined account adds the three books without duplicating starting capital."
    )

    symbol_text = st.text_input("NSE cash symbol", value="SBIN", key="paper_symbol_input")
    try:
        symbol = normalize_symbol(symbol_text)
        symbol_error = ""
    except ValueError as exc:
        symbol, symbol_error = "", str(exc)
    refresh_col, status_col = st.columns([1, 3])
    with refresh_col:
        refresh = st.button("Refresh quote and model", type="primary")
    if refresh:
        st.session_state["paper_refresh_revision"] = int(st.session_state.get("paper_refresh_revision", 0)) + 1
    revision = int(st.session_state.get("paper_refresh_revision", 0))

    snapshot = {}
    delivery_forecast = {}
    if symbol and not symbol_error:
        try:
            snapshot = snapshot_loader(
                f"{symbol}.NS", revision, segment == "INTRADAY"
            ) or {}
            if segment == "DELIVERY" and delivery_prediction_loader is not None:
                delivery_forecast = delivery_prediction_loader(symbol, revision) or {}
        except Exception as exc:
            snapshot = {"error": f"{type(exc).__name__}: {exc}"}
    with status_col:
        if symbol_error:
            st.error(symbol_error)
        elif snapshot.get("error"):
            st.warning(f"Market-data snapshot unavailable: {snapshot['error']}")
        elif snapshot:
            st.caption(
                f"Source: {snapshot.get('source') or 'unknown'} · "
                f"Last bar: {_format_timestamp(str(snapshot.get('quote_timestamp') or ''))} · "
                f"Price freshness: {snapshot.get('price_freshness_status') or 'UNKNOWN'}"
            )

    price = _finite_or_none(snapshot.get("price"))
    predictions = snapshot.get("predictions") or []
    intraday_horizons_returned = {
        int(item.get("horizon_min", 0) or 0)
        for item in predictions
        if int(item.get("horizon_min", 0) or 0) in (5, 10, 15, 20, 25, 30)
    }
    if segment == "INTRADAY":
        selected_prediction = _select_prediction_for_horizon(
            predictions, 30, snapshot.get("prediction")
        )
        prediction_status = "READY" if selected_prediction and snapshot.get("prediction_ready") else "NOT READY"
        prediction_unavailable_reason = snapshot.get("prediction_freshness_reason") or (
            "A fresh completed stock bar and fresh benchmark context are required for the 30-minute model."
        )
    else:
        delivery_rows = delivery_forecast.get("forecasts", []) or []
        selected_prediction = {
            "signal": "7-SESSION OHLC" if delivery_forecast.get("status") == "CALCULATED" and delivery_rows else "UNAVAILABLE",
            "probability_up": None,
            "expected_return": None,
            "predicted_price": _finite_or_none(delivery_rows[-1].get("close")) if delivery_rows else None,
            "model_version": delivery_forecast.get("model") or "7-session daily OHLC model",
            "decision_timestamp": delivery_forecast.get("as_of_date") or "",
            "reason": delivery_forecast.get("reason") or "",
        }
        prediction_status = str(delivery_forecast.get("status") or "UNAVAILABLE").upper()
        prediction_unavailable_reason = (
            delivery_forecast.get("reason")
            or f"Seven-session OHLC forecast status: {prediction_status}."
        )
    fresh = bool(snapshot.get("price_fresh"))
    price_source = str(snapshot.get("price_source") or snapshot.get("source") or "UNKNOWN")
    quote_timestamp = str(snapshot.get("quote_timestamp") or "")
    if price is not None and price > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Live market price", f"₹{price:,.2f}")
        c2.metric("Quote status", "FRESH BAR" if fresh else "QUOTE ONLY / STALE")
        c3.metric(
            "Intraday horizons" if segment == "INTRADAY" else "7-session forecast",
            (
                f"{len(intraday_horizons_returned)}/6 returned"
                if segment == "INTRADAY" else
                str(selected_prediction.get("signal") or "UNAVAILABLE")
            ),
        )
        st.caption(
            f"Quote source: {price_source} · Last update: {_format_timestamp(quote_timestamp)}"
        )
        if not fresh:
            if segment == "INTRADAY":
                st.warning(
                    f"Quote is available, but it is not verified as a fresh completed current-session OHLC bar. "
                    f"{snapshot.get('price_freshness_reason') or 'The 30-minute intraday model requires fresh stock and benchmark inputs.'}"
                )
            else:
                st.caption(
                    "The intraday quote is stale or unverified. The Delivery forecast uses completed daily OHLC "
                    "history and is checked separately for source-session freshness."
                )
    elif snapshot:
        st.warning("No market quote is currently available. Enter a manual reference price to test the ledger.")
    else:
        st.info(
            "Enter an NSE symbol and refresh to load its current market value and "
            + ("30-minute Intraday prediction." if segment == "INTRADAY" else "seven-session Delivery OHLC forecast.")
        )

    if segment == "INTRADAY":
        st.subheader("Intraday predictions · 5, 10, 15, 20, 25 and 30 minutes")
        intraday_rows = _intraday_horizon_table_rows(
            predictions,
            unavailable_reason=prediction_unavailable_reason,
        )
        returned_horizons = intraday_horizons_returned
        st.caption(
            f"Model outputs received: {len(returned_horizons)}/6 · "
            "Each horizon is evaluated independently from the same latest eligible decision anchor."
        )
        st.dataframe(
            pd.DataFrame(intraday_rows),
            use_container_width=True,
            hide_index=True,
        )
        if not returned_horizons:
            st.warning(
                "No intraday horizon could be calculated. All six horizons are shown as unavailable; "
                + (prediction_unavailable_reason or "fresh stock and benchmark OHLC bars are required.")
            )
        elif len(returned_horizons) < 6:
            missing = sorted(set((5, 10, 15, 20, 25, 30)) - returned_horizons)
            st.warning(f"Missing model outputs for: {', '.join(f'{value} min' for value in missing)}.")
        if selected_prediction:
            prediction_caption = (
                f"Order annotation default: +30 min · "
                f"Model: {selected_prediction.get('model_version') or 'unknown'} · "
                f"Decision time: {_format_timestamp(str(selected_prediction.get('decision_timestamp') or ''))}"
            )
            st.caption(prediction_caption)
    else:
        st.subheader("Delivery forecast · next 7 NSE sessions")
        forecasts = delivery_forecast.get("forecasts", []) or []
        target_dates = delivery_forecast.get("target_sessions", []) or []
        if prediction_status == "CALCULATED" and forecasts:
            rows = []
            for index, item in enumerate(forecasts):
                skill = _finite_or_none(item.get("mean_mae_skill_vs_median"))
                rows.append({
                    "Horizon": f"+{int(item.get('horizon', index + 1))} session(s)",
                    "Target date": target_dates[index] if index < len(target_dates) else "—",
                    "Open (₹)": _finite_or_none(item.get("open")),
                    "High (₹)": _finite_or_none(item.get("high")),
                    "Low (₹)": _finite_or_none(item.get("low")),
                    "Close (₹)": _finite_or_none(item.get("close")),
                    "Validation skill vs median": f"{skill:.1%}" if skill is not None else "—",
                    "Validation": item.get("validation_status") or "NOT_EVALUATED",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            skill = _finite_or_none(delivery_forecast.get("overall_mean_mae_skill_vs_median"))
            st.caption(
                f"History as of: {delivery_forecast.get('as_of_date') or '—'} · "
                f"Holdout skill vs median: {skill:.1%}" if skill is not None
                else f"History as of: {delivery_forecast.get('as_of_date') or '—'}"
            )
            st.caption(
                "Forecasts are for future NSE trading sessions, not calendar dates. "
                "The +1 session is omitted once that session is already underway."
            )
            if delivery_forecast.get("active_session_target_omitted"):
                st.info(
                    f"The {delivery_forecast['active_session_target_omitted']} session has already started; "
                    "its +1 forecast is withheld. The table shows the remaining future modelled sessions."
                )
            st.caption(
                f"Validation: {delivery_forecast.get('validation_status') or 'NOT_EVALUATED'} · "
                f"Trading eligibility: {delivery_forecast.get('trade_eligibility') or 'NOT_ELIGIBLE'}"
            )
            if delivery_forecast.get("reason"):
                st.warning(str(delivery_forecast["reason"]))
        else:
            st.warning(f"Seven-session Delivery forecast withheld: {prediction_unavailable_reason}")

    st.subheader("Place a simulated order")
    current_state = calculate_portfolio(
        0.0, list_fills(segment=segment)
    )
    available_shares = int(current_state["positions"].get(symbol, {}).get("quantity", 0)) if symbol else 0
    current_side = "LONG" if available_shares > 0 else "SHORT" if available_shares < 0 else "FLAT"
    st.caption(
        f"Current {symbol or 'symbol'} position: {current_side} · {abs(available_shares):,} shares. "
        "SELL closes longs then opens shorts; BUY covers shorts then opens longs."
    )
    with st.form("paper_place_order"):
        side = st.radio("Side", ["BUY", "SELL"], horizontal=True)
        quantity = st.number_input("Quantity (shares)", min_value=1, max_value=1_000_000, value=1, step=1)
        manual_price = st.number_input(
            "Manual reference price (₹); use 0 for the latest market bar",
            min_value=0.0, value=0.0, step=0.05,
        )
        execution_price = manual_price if manual_price > 0 else price
        estimated_fill = (
            round(
                execution_price * (1 + float(account["slippage_bps"]) / 10_000)
                if side == "BUY"
                else execution_price * (1 - float(account["slippage_bps"]) / 10_000),
                2,
            )
            if execution_price else 0.0
        )
        estimated_turnover = round(float(quantity) * estimated_fill, 2)
        estimated_costs = calculate_trade_charges(
            segment, side, estimated_turnover, float(account["fee_bps"])
        )
        estimated_charges = float(estimated_costs["total_charges"])
        estimated_cash_flow = (
            -(estimated_turnover + estimated_charges)
            if side == "BUY" else estimated_turnover - estimated_charges
        )
        st.caption(
            f"Estimated fill: ₹{estimated_fill:,.2f} · Turnover: ₹{estimated_turnover:,.2f} · "
            f"Brokerage ₹{estimated_costs['brokerage']:,.2f} · STT ₹{estimated_costs['stt']:,.2f} · "
            f"Stamp ₹{estimated_costs['stamp_duty']:,.2f} · "
            f"NSE/SEBI/IPFT ₹{estimated_costs['exchange_charges'] + estimated_costs['sebi_charges'] + estimated_costs['ipft_charges']:,.2f} · "
            f"GST ₹{estimated_costs['gst']:,.2f} · Total ₹{estimated_charges:,.2f} · "
            f"{'Cash debit' if side == 'BUY' else 'Cash credit'} ₹{abs(estimated_cash_flow):,.2f}"
        )
        allow_stale = st.checkbox("Allow a stale / last-session price for this paper fill")
        confirm_order = st.checkbox("I confirm this is a simulated cash-equity order")
        submit_order = st.form_submit_button("Execute paper order")
    if submit_order:
        if not symbol or symbol_error:
            st.error(symbol_error or "Enter a valid NSE symbol.")
        elif not confirm_order:
            st.error("Tick the simulation confirmation before submitting.")
        else:
            reference_price = manual_price if manual_price > 0 else price
            if reference_price is None or reference_price <= 0:
                st.error("No usable reference price. Provide a manual reference price.")
            elif manual_price <= 0 and not fresh and not allow_stale:
                st.error("Fresh current-session data is unavailable. Tick the stale-price override or enter a manual price.")
            else:
                try:
                    attached_prediction = selected_prediction if selected_prediction else {}
                    existing_fills = list_fills()
                    current_market_prices = _fetch_live_market_marks(
                        existing_fills, snapshot, snapshot_loader, revision
                    )
                    fill = place_order(
                        symbol,
                        side,
                        int(quantity),
                        reference_price,
                        account,
                        source="MANUAL_REFERENCE" if manual_price > 0 else str(snapshot.get("source") or ""),
                        quote_timestamp="USER_ENTERED" if manual_price > 0 else str(snapshot.get("quote_timestamp") or ""),
                        prediction=attached_prediction,
                        market_prices=current_market_prices,
                        segment=segment,
                    )
                    st.success(
                        f"{fill['side']} {fill['quantity']} {fill['symbol']} filled at "
                        f"₹{fill['fill_price']:,.2f} (simulated). Estimated charges: ₹{fill['charges']:,.2f}."
                    )
                except Exception as exc:
                    st.error(f"Paper order rejected: {type(exc).__name__}: {exc}")

    fills = list_fills()
    with st.expander("Live quote watchlist (optional)", expanded=False):
        _render_live_market_overview(fills, snapshot, snapshot_loader, revision)
    market_prices = _fetch_live_market_marks(fills, snapshot, snapshot_loader, revision)
    segment_capitals = _segment_capitals_from_account(account)
    state = calculate_segmented_portfolio(
        float(account["initial_cash"]), fills, market_prices=market_prices,
        segment_capitals=segment_capitals,
    )
    selected_book_state = state["segments"][segment]
    selected_book_capital = float(segment_capitals[segment])
    selected_book_pnl = _money(
        float(selected_book_state["realised_pnl"])
        + float(selected_book_state["unrealised_pnl"])
    )
    selected_book_return = (
        selected_book_pnl / selected_book_capital * 100
        if selected_book_capital > 0 else 0.0
    )
    selected_book_fills = [
        fill for fill in fills
        if str(fill.get("segment") or "LEGACY").upper() == segment
    ]

    st.divider()
    st.subheader(f"{segment.title()} paper account")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Cash balance", f"₹{selected_book_state['cash']:,.2f}")
    k2.metric("Free cash after margin", f"₹{selected_book_state['available_cash']:,.2f}")
    k3.metric("Long market value", f"₹{selected_book_state['long_market_value']:,.2f}")
    k4.metric("Short liability", f"₹{selected_book_state['short_liability']:,.2f}")
    k5.metric("Net liquidation value", f"₹{selected_book_state['equity']:,.2f}")
    k6.metric("Total P&L", f"₹{selected_book_pnl:,.2f}", f"{selected_book_return:.2f}%")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Starting allocation", f"₹{selected_book_capital:,.2f}")
    m2.metric("Initial short margin", f"₹{selected_book_state['initial_margin_required']:,.2f}")
    m3.metric("Maintenance margin", f"₹{selected_book_state['maintenance_margin_required']:,.2f}")
    m4.metric("Margin excess", f"₹{selected_book_state['margin_excess']:,.2f}")
    m5.metric("Realised P&L", f"₹{selected_book_state['realised_pnl']:,.2f}")
    if selected_book_state["margin_status"] == "CALL":
        st.error(
            f"{segment.title()} MARGIN CALL: net liquidation value is below the "
            "maintenance-margin requirement. Reduce or cover short exposure."
        )
    elif selected_book_state["available_cash"] < -0.005:
        if selected_book_state["short_liability"] > 0:
            st.warning("This book's initial short-margin reserve exceeds its cash balance.")
        else:
            st.warning("This book has a cash funding deficit. New risk-increasing orders are blocked.")
    st.caption(
        f"Unrealised P&L: ₹{selected_book_state['unrealised_pnl']:,.2f} · "
        f"Brokerage estimate: {float(account['fee_bps']):g} bps · "
        f"Slippage: {float(account['slippage_bps']):g} bps · "
        f"Recorded charges in this book: ₹{sum(float(fill.get('charges') or 0.0) for fill in selected_book_fills):,.2f} · "
        f"Short margin assumptions: {INITIAL_SHORT_MARGIN_RATE:.0%} initial / "
        f"{MAINTENANCE_SHORT_MARGIN_RATE:.0%} maintenance."
    )
    st.caption(
        f"Price source: {snapshot.get('source') or 'unavailable'} · "
        f"Quote status: {snapshot.get('price_freshness_status') or 'UNKNOWN'}"
    )

    combined_pnl = _money(float(state["realised_pnl"]) + float(state["unrealised_pnl"]))
    with st.expander("Combined account · all strategy books", expanded=False):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Combined cash", f"₹{state['cash']:,.2f}")
        c2.metric("Combined free cash", f"₹{state['available_cash']:,.2f}")
        c3.metric("Long market value", f"₹{state['long_market_value']:,.2f}")
        c4.metric("Short liability", f"₹{state['short_liability']:,.2f}")
        c5.metric("Net liquidation value", f"₹{state['equity']:,.2f}")
        c6.metric(
            "Total P&L", f"₹{combined_pnl:,.2f}",
            f"{combined_pnl / float(account['initial_cash']) * 100:.2f}%",
        )
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Starting capital", f"₹{state['starting_capital']:,.2f}")
        d2.metric("Initial short margin", f"₹{state['initial_margin_required']:,.2f}")
        d3.metric("Maintenance margin", f"₹{state['maintenance_margin_required']:,.2f}")
        d4.metric("Margin excess", f"₹{state['margin_excess']:,.2f}")
        d5.metric("Realised P&L", f"₹{state['realised_pnl']:,.2f}")
    st.subheader("Separate strategy books")
    book_tabs = st.tabs(["Intraday", "Delivery", "Legacy / Unassigned"])
    for book_name, book_tab in zip(TRADING_SEGMENTS, book_tabs):
        with book_tab:
            book_state = state["segments"][book_name]
            book_fills = [row for row in fills if str(row.get("segment") or "LEGACY").upper() == book_name]
            b1,b2,b3,b4,b5,b6 = st.columns(6)
            b1.metric("Starting allocation", f"₹{segment_capitals[book_name]:,.2f}")
            b2.metric("Cash balance", f"₹{book_state['cash']:,.2f}")
            b3.metric("Free cash after margin", f"₹{book_state['available_cash']:,.2f}")
            b4.metric("Realised P&L", f"₹{book_state['realised_pnl']:,.2f}")
            b5.metric("Unrealised P&L", f"₹{book_state['unrealised_pnl']:,.2f}")
            b6.metric("Margin excess", f"₹{book_state['margin_excess']:,.2f}")
            v1, v2 = st.columns(2)
            v1.metric("Long market value", f"₹{book_state['long_market_value']:,.2f}")
            v2.metric("Short liability", f"₹{book_state['short_liability']:,.2f}")

            book_positions, _ = _open_positions_with_marks(
                book_state, snapshot, snapshot_loader, revision
            )
            if not book_positions.empty:
                book_positions.insert(0, "Track", book_name)
                st.dataframe(book_positions, use_container_width=True, hide_index=True)
            else:
                st.caption("No open positions in this track.")

            st.markdown("**Trade ledger**")
            if book_fills:
                ledger = pd.DataFrame(book_fills)
                visible = {
                    "timestamp_ist": "Timestamp (IST)", "symbol": "Symbol", "side": "Side",
                    "quantity": "Qty", "reference_price": "Reference price", "fill_price": "Fill price",
                    "gross": "Turnover", "brokerage": "Brokerage", "stt": "STT",
                    "stamp_duty": "Stamp duty", "exchange_charges": "NSE charges",
                    "sebi_charges": "SEBI fee", "ipft_charges": "IPFT",
                    "gst": "GST", "charges": "Total charges", "charge_model_version": "Charge model",
                    "segment": "Track",
                    "prediction_signal": "Model signal", "prediction_probability_up": "P(up)",
                    "prediction_expected_return": "Expected return", "prediction_model_version": "Model version",
                    "prediction_decision_timestamp": "Model decision time",
                    "prediction_horizon_min": "Model horizon (min)", "source": "Price source",
                    "quote_timestamp": "Quote timestamp",
                }
                ledger = ledger[[column for column in visible if column in ledger.columns]].rename(columns=visible)
                st.dataframe(ledger, use_container_width=True, hide_index=True)
                st.download_button(
                    f"Download {book_name.lower()} ledger CSV",
                    data=ledger.to_csv(index=False).encode("utf-8"),
                    file_name=f"paper_trade_{book_name.lower()}.csv",
                    mime="text/csv",
                    key=f"paper_download_{book_name.lower()}",
                )
            else:
                st.caption("No fills recorded in this track.")

    st.caption(
        "Simulation only: NSE cash-equity long/short accounting; no F&O or broker integration. "
        "Cash and margin are allocated independently to Intraday, Delivery and Legacy. "
        "Each fill records estimated brokerage, STT, stamp duty, NSE transaction/IPFT, SEBI fee and GST; "
        "entry charges are included in position cost basis and exit charges reduce realised P&L. "
        "Broker-specific per-order caps, DP charges, penalties and other contract-note items are not modelled. "
        "Legacy fills retain historic unitemized fees. These amounts are estimates, not a contract note. "
        "Use a persistent mounted path via PAPER_TRADING_DB_PATH if the hosting filesystem is ephemeral."
    )
