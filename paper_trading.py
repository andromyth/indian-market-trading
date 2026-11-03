"""Long-only NSE cash-equity paper trading, isolated from broker execution.

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
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_fills_symbol_id ON paper_fills(symbol, id)"
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


def list_fills(limit: int | None = None) -> list[dict]:
    init_db()
    query = "SELECT * FROM paper_fills ORDER BY id DESC"
    params: tuple = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (max(1, int(limit)),)
    with _connection() as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def calculate_portfolio(initial_cash: float, fills: list[dict]) -> dict:
    """Rebuild cash, open positions and realised P&L solely from immutable fills."""
    cash = float(initial_cash)
    positions: dict[str, dict[str, float]] = {}
    realised = 0.0
    for fill in sorted(fills, key=lambda item: int(item.get("id", 0))):
        symbol = normalize_symbol(fill["symbol"])
        side = str(fill["side"]).upper()
        quantity = int(fill["quantity"])
        fill_price = float(fill["fill_price"])
        gross = float(fill.get("gross", quantity * fill_price))
        charges = float(fill.get("charges", 0.0))
        if quantity <= 0 or not all(math.isfinite(x) for x in (fill_price, gross, charges)) or fill_price <= 0 or gross < 0 or charges < 0:
            raise ValueError("Paper fill ledger contains invalid numeric values.")
        position = positions.setdefault(symbol, {"quantity": 0.0, "cost_basis": 0.0})
        if side == "BUY":
            cash -= gross + charges
            position["quantity"] += quantity
            position["cost_basis"] += gross + charges
        elif side == "SELL":
            held = int(position["quantity"])
            if quantity > held:
                raise ValueError(f"Paper fill ledger would create a short position in {symbol}.")
            average_cost = position["cost_basis"] / held if held else 0.0
            cash += gross - charges
            realised += gross - charges - average_cost * quantity
            position["quantity"] -= quantity
            position["cost_basis"] -= average_cost * quantity
        else:
            raise ValueError(f"Unsupported paper fill side: {side}")
    positions = {
        symbol: {
            "quantity": int(values["quantity"]),
            "cost_basis": round(values["cost_basis"], 2),
            "average_cost": (values["cost_basis"] / values["quantity"]) if values["quantity"] else 0.0,
        }
        for symbol, values in positions.items()
        if values["quantity"] > 0
    }
    return {
        "cash": round(cash, 2),
        "positions": positions,
        "realised_pnl": round(realised, 2),
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
) -> dict:
    """Create an immediate simulated fill; never calls a broker or order API."""
    root = normalize_symbol(symbol)
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
        state = calculate_portfolio(float(current_account["initial_cash"]), fills)
        held = int(state["positions"].get(root, {}).get("quantity", 0))
        if direction == "BUY" and gross + charges > state["cash"] + 0.000001:
            raise ValueError(
                f"Insufficient paper cash. Required ₹{gross + charges:,.2f}; available ₹{state['cash']:,.2f}."
            )
        if direction == "SELL" and qty > held:
            raise ValueError(f"Cannot sell {qty} {root}; only {held} shares are held. Short selling is disabled.")

        model = prediction or {}
        cursor = connection.execute(
            """
            INSERT INTO paper_fills(
                timestamp_ist, symbol, side, quantity, reference_price, fill_price,
                gross, charges, source, quote_timestamp, prediction_signal,
                prediction_probability_up, prediction_expected_return,
                prediction_model_version, prediction_decision_timestamp,
                prediction_horizon_min
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now_ist(), root, direction, qty, reference, fill_price, gross, charges,
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


def _open_positions_with_marks(state: dict, snapshot: dict | None, snapshot_loader, refresh_revision: int) -> tuple[pd.DataFrame, float]:
    rows = []
    market_value = 0.0
    selected_symbol = str((snapshot or {}).get("symbol") or "")
    for symbol, position in state["positions"].items():
        mark = None
        quote_time = ""
        if symbol == selected_symbol:
            mark = _finite_or_none((snapshot or {}).get("price"))
            quote_time = str((snapshot or {}).get("quote_timestamp") or "")
        if mark is None:
            try:
                other = snapshot_loader(f"{symbol}.NS", refresh_revision, False) or {}
                mark = _finite_or_none(other.get("price"))
                quote_time = str(other.get("quote_timestamp") or "")
            except Exception:
                mark = None
        if mark is None or mark <= 0:
            mark = float(position["average_cost"])
            quote_time = "Last available cost basis (no market mark)"
        qty = int(position["quantity"])
        value = qty * mark
        pnl = value - float(position["cost_basis"])
        market_value += value
        rows.append({
            "Symbol": symbol,
            "Quantity": qty,
            "Average cost": round(float(position["average_cost"]), 2),
            "Mark price": round(mark, 2),
            "Market value": round(value, 2),
            "Unrealised P&L": round(pnl, 2),
            "Unrealised return %": round((pnl / float(position["cost_basis"]) * 100) if position["cost_basis"] else 0, 2),
            "Price timestamp": quote_time,
        })
    return pd.DataFrame(rows), market_value


def render_paper_trading(snapshot_loader) -> None:
    """Render the paper terminal. snapshot_loader(symbol, revision, include_predictions)."""
    init_db()
    st.title("Paper Trading")
    st.caption("Simulated NSE cash-equity trading. Every fill is manual, simulated and stored separately from production data.")

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
                confirm_reset = st.checkbox("I understand that reset permanently clears this paper ledger.")
                reset = st.form_submit_button("Reset paper account")
            if reset:
                if not confirm_reset:
                    st.error("Tick the confirmation box before resetting.")
                else:
                    try:
                        reset_account(reset_capital)
                        st.success("Paper ledger reset.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not reset account: {type(exc).__name__}: {exc}")

    symbol_text = st.text_input("NSE cash symbol", value="SBIN", key="paper_symbol_input")
    try:
        symbol = normalize_symbol(symbol_text)
        symbol_error = ""
    except ValueError as exc:
        symbol, symbol_error = "", str(exc)
    horizon = st.selectbox("Model horizon to associate with a trade", [5, 10, 15, 20, 25, 30], index=5, format_func=lambda value: f"{value} minutes")
    refresh_col, status_col = st.columns([1, 3])
    with refresh_col:
        refresh = st.button("Refresh quote and model", type="primary")
    if refresh:
        st.session_state["paper_refresh_revision"] = int(st.session_state.get("paper_refresh_revision", 0)) + 1
    revision = int(st.session_state.get("paper_refresh_revision", 0))

    snapshot = {}
    if symbol and not symbol_error:
        try:
            snapshot = snapshot_loader(f"{symbol}.NS", revision, True) or {}
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
    selected_prediction = next(
        (item for item in predictions if int(item.get("horizon_min", 0) or 0) == int(horizon)),
        {},
    )
    fresh = bool(snapshot.get("price_fresh"))
    model_fresh = bool(snapshot.get("prediction_ready"))
    if price is not None and price > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest reference price", f"₹{price:,.2f}")
        c2.metric("Market input", "FRESH" if fresh else "STALE / CLOSED")
        c3.metric("Selected model signal", str(selected_prediction.get("signal") or "UNAVAILABLE"))
        if not fresh:
            st.warning(
                f"Latest price is not a fresh current-session quote. {snapshot.get('price_freshness_reason') or 'Use only with explicit stale-price override.'}"
            )
    elif snapshot:
        st.warning("No usable market price is available. Enter a manual reference price in the order form to test the ledger.")
    else:
        st.info("Enter a symbol and refresh the market snapshot to load its quote and prediction.")

    if predictions:
        prediction_rows = []
        for item in predictions:
            prediction_rows.append({
                "Horizon": f"{item.get('horizon_min')} min",
                "Signal": item.get("signal", "—"),
                "P(up)": _finite_or_none(item.get("probability_up")),
                "Expected return": _finite_or_none(item.get("expected_return")),
                "Model version": item.get("model_version", "—"),
                "Reason": item.get("signal_reason") or item.get("reason") or "",
            })
        st.dataframe(pd.DataFrame(prediction_rows), use_container_width=True, hide_index=True)
    elif snapshot and not model_fresh:
        st.caption(f"Model not run: {snapshot.get('prediction_freshness_reason') or 'fresh stock and benchmark inputs are required.'}")

    st.subheader("Place a simulated order")
    with st.form("paper_place_order"):
        side = st.radio("Side", ["BUY", "SELL"], horizontal=True)
        quantity = st.number_input("Quantity (shares)", min_value=1, max_value=1_000_000, value=1, step=1)
        manual_price = st.number_input(
            "Manual reference price (₹); use 0 for the latest market bar",
            min_value=0.0, value=0.0, step=0.05,
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
                    fill = place_order(
                        symbol,
                        side,
                        int(quantity),
                        reference_price,
                        account,
                        source="MANUAL_REFERENCE" if manual_price > 0 else str(snapshot.get("source") or ""),
                        quote_timestamp="USER_ENTERED" if manual_price > 0 else str(snapshot.get("quote_timestamp") or ""),
                        prediction=attached_prediction,
                    )
                    st.success(
                        f"{fill['side']} {fill['quantity']} {fill['symbol']} filled at "
                        f"₹{fill['fill_price']:,.2f} (simulated). Estimated charges: ₹{fill['charges']:,.2f}."
                    )
                except Exception as exc:
                    st.error(f"Paper order rejected: {type(exc).__name__}: {exc}")

    fills = list_fills()
    state = calculate_portfolio(float(account["initial_cash"]), fills)
    positions_frame, market_value = _open_positions_with_marks(
        state, snapshot, snapshot_loader, revision
    )
    cash = float(state["cash"])
    equity = cash + market_value
    total_pnl = equity - float(account["initial_cash"])
    unrealised = float(positions_frame["Unrealised P&L"].sum()) if not positions_frame.empty else 0.0

    st.divider()
    st.subheader("Paper account")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Cash available", f"₹{cash:,.2f}")
    k2.metric("Open market value", f"₹{market_value:,.2f}")
    k3.metric("Net liquidation value", f"₹{equity:,.2f}")
    k4.metric("Realised P&L", f"₹{state['realised_pnl']:,.2f}")
    k5.metric("Total P&L", f"₹{total_pnl:,.2f}", f"{(total_pnl / float(account['initial_cash']) * 100):.2f}%")
    st.caption(
        f"Unrealised P&L: ₹{unrealised:,.2f} · "
        f"Estimated charges: {float(account['fee_bps']):g} bps · "
        f"Slippage: {float(account['slippage_bps']):g} bps"
    )

    st.subheader("Open positions")
    if positions_frame.empty:
        st.caption("No open positions.")
    else:
        st.dataframe(positions_frame, use_container_width=True, hide_index=True)

    st.subheader("Trade ledger")
    if fills:
        ledger = pd.DataFrame(fills)
        visible = {
            "timestamp_ist": "Timestamp (IST)", "symbol": "Symbol", "side": "Side",
            "quantity": "Qty", "reference_price": "Reference price", "fill_price": "Fill price",
            "gross": "Turnover", "charges": "Estimated charges", "prediction_signal": "Model signal",
            "prediction_horizon_min": "Model horizon (min)", "prediction_probability_up": "P(up)",
            "prediction_model_version": "Model version", "source": "Price source",
            "quote_timestamp": "Quote timestamp",
        }
        ledger = ledger[[column for column in visible if column in ledger.columns]].rename(columns=visible)
        st.dataframe(ledger, use_container_width=True, hide_index=True)
        st.download_button(
            "Download paper-trade ledger CSV",
            data=ledger.to_csv(index=False).encode("utf-8"),
            file_name="paper_trade_ledger.csv",
            mime="text/csv",
        )
    else:
        st.caption("No simulated fills have been recorded yet.")

    st.caption(
        "Simulation only. Cash-equity, long-only; no F&O, leverage, short selling, or broker integration. "
        "Charges and slippage are configurable estimates, not an exchange or broker contract note. "
        "Use a persistent mounted path via PAPER_TRADING_DB_PATH if the hosting filesystem is ephemeral."
    )
