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
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_CAPITAL = 100_000.0
DEFAULT_FEE_BPS = 10.0
DEFAULT_SLIPPAGE_BPS = 5.0
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
                created_at TEXT NOT NULL
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


def get_account() -> dict | None:
    init_db()
    with _connection() as connection:
        row = connection.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
    return dict(row) if row else None


def create_account(initial_cash: float, fee_bps: float = DEFAULT_FEE_BPS,
                   slippage_bps: float = DEFAULT_SLIPPAGE_BPS) -> None:
    values = (float(initial_cash), float(fee_bps), float(slippage_bps))
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Account settings must be finite numbers.")
    if values[0] <= 0 or values[1] < 0 or values[2] < 0 or values[1] > 500 or values[2] > 500:
        raise ValueError("Capital must be positive; fee and slippage must be between 0 and 500 bps.")
    init_db()
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO paper_account(id, initial_cash, fee_bps, slippage_bps, created_at)
            VALUES (1, ?, ?, ?, ?)
            """,
            (*values, _now_ist()),
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
        connection.execute(
            "UPDATE paper_account SET initial_cash = ?, created_at = ? WHERE id = 1",
            (capital, _now_ist()),
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
                    short_basis_per_share - fill_price - fee_per_share
                ) * covered
                basis -= short_basis_per_share * covered
                signed_qty += covered
                remaining -= covered
            # Any excess BUY opens or adds to a long position.
            if remaining:
                signed_qty += remaining
                basis += (fill_price + fee_per_share) * remaining

        else:  # SELL
            cash += gross - charges
            # First close any existing long shares.
            if signed_qty > 0:
                closed = min(remaining, signed_qty)
                long_basis_per_share = basis / signed_qty
                realised += (
                    fill_price - fee_per_share - long_basis_per_share
                ) * closed
                basis -= long_basis_per_share * closed
                signed_qty -= closed
                remaining -= closed
            # Any excess SELL opens or adds to a short position. Proceeds are
            # net of the entry fee and form the short position's cover basis.
            if remaining:
                signed_qty -= remaining
                basis += (fill_price - fee_per_share) * remaining

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
            "cost_basis": round(basis, 2),
            "average_cost": average_cost,
            "mark_price": mark,
            "market_value": position_value,
            "unrealised_pnl": position_pnl,
        }

    cash = round(cash, 2)
    long_market_value = round(long_market_value, 2)
    short_liability = round(short_liability, 2)
    equity = round(cash + long_market_value - short_liability, 2)
    initial_margin = round(short_liability * INITIAL_SHORT_MARGIN_RATE, 2)
    maintenance_margin = round(short_liability * MAINTENANCE_SHORT_MARGIN_RATE, 2)
    available_cash = round(cash - initial_margin, 2)
    margin_excess = round(equity - maintenance_margin, 2)
    return {
        "cash": cash,
        "positions": open_positions,
        "realised_pnl": round(realised, 2),
        "unrealised_pnl": round(unrealised, 2),
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
) -> dict:
    """Calculate separate strategy books against one shared cash/margin pool.

    Position netting is performed inside each segment only. Cash flows and
    short-margin reserves are aggregated across all segments, preventing
    duplicated buying power while keeping Intraday and Delivery P&L separate.
    Unclassified historical fills remain in LEGACY.
    """
    grouped: dict[str, list[dict]] = {name: [] for name in TRADING_SEGMENTS}
    for fill in fills:
        segment = str(fill.get("segment") or "LEGACY").strip().upper()
        if segment not in TRADING_SEGMENTS:
            segment = "LEGACY"
        grouped[segment].append(fill)

    segment_states = {
        name: calculate_portfolio(0.0, rows, market_prices=market_prices)
        for name, rows in grouped.items()
    }
    cash = round(float(initial_cash) + sum(state["cash"] for state in segment_states.values()), 2)
    long_market_value = round(sum(state["long_market_value"] for state in segment_states.values()), 2)
    short_liability = round(sum(state["short_liability"] for state in segment_states.values()), 2)
    realised = round(sum(state["realised_pnl"] for state in segment_states.values()), 2)
    unrealised = round(sum(state["unrealised_pnl"] for state in segment_states.values()), 2)
    initial_margin = round(sum(state["initial_margin_required"] for state in segment_states.values()), 2)
    maintenance_margin = round(sum(state["maintenance_margin_required"] for state in segment_states.values()), 2)
    equity = round(cash + long_market_value - short_liability, 2)
    available_cash = round(cash - initial_margin, 2)
    margin_excess = round(equity - maintenance_margin, 2)
    return {
        "cash": cash,
        "segments": segment_states,
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
    fill_price = reference * (1.0 + impact if direction == "BUY" else 1.0 - impact)
    gross = round(qty * fill_price, 2)
    charges = round(gross * fee_bps / 10_000.0, 2)

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
        mark_prices = dict(market_prices or {})
        mark_prices[root] = fill_price
        before = calculate_segmented_portfolio(initial_cash, fills, market_prices=mark_prices)
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
        )
        after_book = after["segments"][selected_segment]
        after_qty = int(after_book["positions"].get(root, {}).get("quantity", 0))
        increased_long = max(after_qty, 0) > max(before_qty, 0)
        increased_short = max(-after_qty, 0) > max(-before_qty, 0)

        # Existing positions can always be reduced or closed, including during
        # a margin call. New long or short exposure must fit available cash.
        if (increased_long or increased_short) and after["available_cash"] < -0.005:
            raise ValueError(
                "Insufficient free cash after reserving initial short margin. "
                f"Required reserve: ₹{after['initial_margin_required']:,.2f}; "
                f"cash balance after order: ₹{after['cash']:,.2f}; "
                f"free cash: ₹{after['available_cash']:,.2f}."
            )
        if increased_short and after["margin_excess"] < -0.005:
            raise ValueError(
                "Short order rejected: projected equity would fall below the "
                f"maintenance-margin requirement by ₹{abs(after['margin_excess']):,.2f}."
            )

        cursor = connection.execute(
            """
            INSERT INTO paper_fills(
                timestamp_ist, symbol, segment, side, quantity, reference_price, fill_price,
                gross, charges, source, quote_timestamp, prediction_signal,
                prediction_probability_up, prediction_expected_return,
                prediction_model_version, prediction_decision_timestamp,
                prediction_horizon_min
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now_ist(), root, selected_segment, direction, qty, reference, fill_price, gross, charges,
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


def _format_timestamp(value: str) -> str:
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
            fee = st.number_input("Estimated charges (bps of turnover)", min_value=0.0, max_value=500.0, value=DEFAULT_FEE_BPS, step=1.0)
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
                    "Estimated charges (bps of turnover)", min_value=0.0, max_value=500.0,
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

    segment = st.radio(
        "Trading track",
        ["INTRADAY", "DELIVERY"],
        horizontal=True,
        key="paper_trading_segment",
        format_func=lambda value: "Intraday · 5-minute model" if value == "INTRADAY" else "Delivery · Beta V2 Delivery model",
    )
    st.caption(
        "Intraday and Delivery fills, positions and P&L are reported separately. "
        "They share one account cash balance and one account-wide margin pool; no capital is duplicated. "
        "Pre-existing unclassified fills remain in Legacy / Unassigned."
    )

    symbol_text = st.text_input("NSE cash symbol", value="SBIN", key="paper_symbol_input")
    try:
        symbol = normalize_symbol(symbol_text)
        symbol_error = ""
    except ValueError as exc:
        symbol, symbol_error = "", str(exc)
    horizon = 5
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
    if segment == "INTRADAY":
        selected_prediction = snapshot.get("prediction") or next(
            (item for item in predictions if int(item.get("horizon_min", 0) or 0) == 5),
            {},
        )
        prediction_status = "READY" if snapshot.get("prediction_ready") else "NOT READY"
        prediction_unavailable_reason = snapshot.get("prediction_freshness_reason") or (
            "Fresh stock and benchmark OHLC inputs are required."
        )
    else:
        delivery_signal = (
            delivery_forecast.get("prior_close_forecast_signal")
            or delivery_forecast.get("signal")
            or "UNAVAILABLE"
        )
        delivery_probability = (
            delivery_forecast.get("prior_close_probability_up")
            if delivery_forecast.get("prior_close_forecast_signal") is not None
            else delivery_forecast.get("p")
        )
        selected_prediction = {
            "signal": delivery_signal,
            "probability_up": _finite_or_none(delivery_probability),
            "expected_return": _finite_or_none(
                delivery_forecast.get("expected_return")
                or delivery_forecast.get("prediction_expected_return")
            ),
            "predicted_price": _finite_or_none(delivery_forecast.get("predicted_price")),
            "model_version": (
                delivery_forecast.get("model_version")
                or delivery_forecast.get("champion_model_version")
                or delivery_forecast.get("model_bundle")
                or "Beta V2 Delivery"
            ),
            "decision_timestamp": (
                delivery_forecast.get("forecast_as_of_utc")
                or delivery_forecast.get("feature_cutoff_timestamp")
                or ""
            ),
            "reason": delivery_forecast.get("signal_reason") or delivery_forecast.get("reason") or "",
            "prediction_status": delivery_forecast.get("prediction_status") or "UNKNOWN",
            "target_session_date": delivery_forecast.get("target_session_date"),
        }
        prediction_status = str(delivery_forecast.get("prediction_status") or "UNAVAILABLE")
        prediction_unavailable_reason = (
            delivery_forecast.get("signal_reason")
            or delivery_forecast.get("reason")
            or "The existing Beta V2 Delivery forecast is unavailable."
        )
    fresh = bool(snapshot.get("price_fresh"))
    model_fresh = bool(snapshot.get("prediction_ready")) if segment == "INTRADAY" else prediction_status == "PASS"
    price_source = str(snapshot.get("price_source") or snapshot.get("source") or "UNKNOWN")
    quote_timestamp = str(snapshot.get("quote_timestamp") or "")
    if price is not None and price > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Current market value", f"₹{price:,.2f}")
        c2.metric("Quote status", "FRESH BAR" if fresh else "QUOTE ONLY / STALE")
        c3.metric(
            "5-minute signal" if segment == "INTRADAY" else "Delivery signal",
            str(selected_prediction.get("signal") or "UNAVAILABLE"),
        )
        st.caption(
            f"Quote source: {price_source} · Last update: {_format_timestamp(quote_timestamp)}"
        )
        if not fresh:
            st.warning(
                f"Quote is available, but it is not verified as a fresh completed current-session OHLC bar. "
                f"{snapshot.get('price_freshness_reason') or 'The price is display-only and cannot unlock model inference.'}"
            )
    elif snapshot:
        st.warning("No market quote is currently available. Enter a manual reference price to test the ledger.")
    else:
        st.info(
            "Enter an NSE symbol and refresh to load its current market value and "
            + ("five-minute Intraday prediction." if segment == "INTRADAY" else "Delivery forecast.")
        )

    st.subheader("5-minute Intraday prediction" if segment == "INTRADAY" else "Delivery forecast")
    if selected_prediction and (
        segment == "INTRADAY" or prediction_status not in {"ERROR", "BLOCKED", "ABSTAIN", "UNAVAILABLE"}
    ):
        prediction_row = {
            "Track": segment,
            "Horizon": "5 minutes" if segment == "INTRADAY" else "Next NSE session",
            "Status": prediction_status,
            "Signal": selected_prediction.get("signal", "—"),
            "Probability up": _finite_or_none(selected_prediction.get("probability_up")),
            "Expected return": _finite_or_none(selected_prediction.get("expected_return")),
            "Predicted price": _finite_or_none(selected_prediction.get("predicted_price")),
            "Target session": selected_prediction.get("target_session_date") or "—",
            "Model version": selected_prediction.get("model_version", "—"),
            "Decision time": selected_prediction.get("decision_timestamp", "—"),
            "Reason": selected_prediction.get("signal_reason") or selected_prediction.get("reason") or "",
        }
        st.dataframe(pd.DataFrame([prediction_row]), use_container_width=True, hide_index=True)
    else:
        st.caption(f"{segment.title()} prediction unavailable: {prediction_unavailable_reason}")

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
            execution_price * (1 + float(account["slippage_bps"]) / 10_000)
            if execution_price and side == "BUY"
            else execution_price * (1 - float(account["slippage_bps"]) / 10_000)
            if execution_price else 0.0
        )
        estimated_turnover = float(quantity) * estimated_fill
        estimated_charges = estimated_turnover * float(account["fee_bps"]) / 10_000
        estimated_cash_flow = (
            -(estimated_turnover + estimated_charges)
            if side == "BUY" else estimated_turnover - estimated_charges
        )
        st.caption(
            f"Estimated fill: ₹{estimated_fill:,.2f} · Turnover: ₹{estimated_turnover:,.2f} · "
            f"Charges: ₹{estimated_charges:,.2f} · "
            f"{'Cash debit' if side == 'BUY' else 'Cash credit'}: ₹{abs(estimated_cash_flow):,.2f}"
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
    _render_live_market_overview(fills, snapshot, snapshot_loader, revision)
    market_prices = _fetch_live_market_marks(fills, snapshot, snapshot_loader, revision)
    state = calculate_segmented_portfolio(
        float(account["initial_cash"]), fills, market_prices=market_prices
    )
    cash = float(state["cash"])
    equity = float(state["equity"])
    total_pnl = equity - float(account["initial_cash"])
    unrealised = float(state["unrealised_pnl"])

    st.divider()
    st.subheader("Combined paper account · shared cash and margin")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Cash balance", f"₹{cash:,.2f}")
    k2.metric("Free cash after margin", f"₹{state['available_cash']:,.2f}")
    k3.metric("Long market value", f"₹{state['long_market_value']:,.2f}")
    k4.metric("Short liability", f"₹{state['short_liability']:,.2f}")
    k5.metric("Net liquidation value", f"₹{equity:,.2f}")
    k6.metric("Total P&L", f"₹{total_pnl:,.2f}", f"{(total_pnl / float(account['initial_cash']) * 100):.2f}%")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Initial short margin", f"₹{state['initial_margin_required']:,.2f}")
    m2.metric("Maintenance margin", f"₹{state['maintenance_margin_required']:,.2f}")
    m3.metric("Margin excess", f"₹{state['margin_excess']:,.2f}")
    m4.metric("Realised P&L", f"₹{state['realised_pnl']:,.2f}")
    if state["margin_status"] == "CALL":
        st.error(
            "MARGIN CALL: net liquidation value is below the maintenance-margin "
            "requirement. Reduce or cover short exposure; new risk-increasing orders are blocked."
        )
    elif state["available_cash"] < -0.005:
        if state["short_liability"] > 0:
            st.warning("Initial short-margin reserve exceeds the cash balance. New risk-increasing orders are blocked.")
        else:
            st.warning("The paper account has a cash funding deficit. New risk-increasing orders are blocked.")
    st.caption(
        f"Unrealised P&L: ₹{unrealised:,.2f} · "
        f"Estimated charges: {float(account['fee_bps']):g} bps · "
        f"Slippage: {float(account['slippage_bps']):g} bps · "
        f"Simulator assumptions: {INITIAL_SHORT_MARGIN_RATE:.0%} initial short margin and "
        f"{MAINTENANCE_SHORT_MARGIN_RATE:.0%} maintenance margin against short liability."
    )
    st.caption(
        f"Pricing source: {snapshot.get('source') or 'unavailable'} · "
        f"Selected-symbol quote: {snapshot.get('price_freshness_status') or 'UNKNOWN'} · "
        f"{segment.title()} model: {prediction_status}"
    )
    probability = _finite_or_none(selected_prediction.get("probability_up"))
    expected_return = _finite_or_none(selected_prediction.get("expected_return"))
    predicted_price = _finite_or_none(selected_prediction.get("predicted_price"))
    p1,p2,p3,p4 = st.columns(4)
    p1.metric(f"{segment.title()} signal", str(selected_prediction.get("signal") or "UNAVAILABLE"))
    p2.metric("P(up)", f"{probability:.1%}" if probability is not None else "—")
    p3.metric("Expected return", f"{expected_return:+.2%}" if expected_return is not None else "—")
    p4.metric("Forecast price", f"₹{predicted_price:,.2f}" if predicted_price is not None else "—")
    st.caption(
        f"Model: {selected_prediction.get('model_version') or 'unknown'} · "
        f"Decision: {_format_timestamp(str(selected_prediction.get('decision_timestamp') or ''))} · "
        + ("Horizon: 5 minutes" if segment == "INTRADAY" else
           f"Target NSE session: {selected_prediction.get('target_session_date') or '—'}")
    )
    st.caption(str(selected_prediction.get("reason") or ""))

    st.subheader("Separate strategy books")
    book_tabs = st.tabs(["Intraday", "Delivery", "Legacy / Unassigned"])
    for book_name, book_tab in zip(TRADING_SEGMENTS, book_tabs):
        with book_tab:
            book_state = state["segments"][book_name]
            book_fills = [row for row in fills if str(row.get("segment") or "LEGACY").upper() == book_name]
            b1,b2,b3,b4 = st.columns(4)
            b1.metric("Realised P&L", f"₹{book_state['realised_pnl']:,.2f}")
            b2.metric("Unrealised P&L", f"₹{book_state['unrealised_pnl']:,.2f}")
            b3.metric("Long market value", f"₹{book_state['long_market_value']:,.2f}")
            b4.metric("Short liability", f"₹{book_state['short_liability']:,.2f}")

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
                    "gross": "Turnover", "charges": "Estimated charges", "segment": "Track",
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
        "Short-sale proceeds are included in cash but reserved through the initial-margin calculation. "
        "Initial/maintenance margin rates are illustrative simulator assumptions, not live exchange or broker requirements. "
        "Charges and slippage are estimates, not a contract note. Use a persistent mounted path via "
        "PAPER_TRADING_DB_PATH if the hosting filesystem is ephemeral."
    )
