"""Streamlit UI for delivery prediction history and price-path presentation.

The active ALL-NSE Delivery candidate directly predicts next-session direction
probability and return. OHLC/entry/exit figures shown here are explicitly
DERIVED estimates from the reference OHLC, the model expected return, and a
14-session ATR; they are not separately trained OHLC targets.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from nse_data_layer import fetch_equity_history
from prediction_ledger import load_records


def _root(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace(".NS", "").replace(".BO", "")


def _ohlc_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    if "date" not in x.columns:
        return pd.DataFrame()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    for c in ("open", "high", "low", "close"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    required = [c for c in ("open", "high", "low", "close") if c in x.columns]
    if "close" not in required:
        return pd.DataFrame()
    return x.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _atr14(df: pd.DataFrame) -> float:
    x = _ohlc_frame(df)
    if len(x) < 2 or not all(c in x.columns for c in ("high", "low", "close")):
        return np.nan
    prev = x["close"].shift(1)
    tr = pd.concat([(x["high"] - x["low"]), (x["high"] - prev).abs(), (x["low"] - prev).abs()], axis=1).max(axis=1)
    v = tr.ewm(alpha=1 / 14, adjust=False, min_periods=5).mean().iloc[-1]
    return float(v) if np.isfinite(v) else np.nan


def _derived_price_path(reference: pd.Series, expected_return: float, atr: float) -> dict[str, float]:
    close = float(reference.get("close", np.nan))
    if not np.isfinite(close):
        return {k: np.nan for k in ("predicted_open", "predicted_high", "predicted_low", "predicted_close", "purchase_price", "sale_price", "stop_price")}
    er = float(expected_return) if np.isfinite(expected_return) else 0.0
    predicted_close = close * (1.0 + er)
    # These are deliberately transparent derived estimates, not separate ML targets.
    predicted_open = close * (1.0 + 0.25 * er)
    vol = float(atr) if np.isfinite(atr) and atr > 0 else close * 0.01
    predicted_high = max(predicted_open, predicted_close) + 0.50 * vol
    predicted_low = min(predicted_open, predicted_close) - 0.50 * vol
    purchase_price = close
    sale_price = predicted_close
    stop_price = close - vol if er >= 0 else close + vol
    return {
        "predicted_open": predicted_open,
        "predicted_high": predicted_high,
        "predicted_low": max(0.0, predicted_low),
        "predicted_close": predicted_close,
        "purchase_price": purchase_price,
        "sale_price": sale_price,
        "stop_price": max(0.0, stop_price),
    }


def _history(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        return _ohlc_frame(fetch_equity_history(_root(symbol), start.date(), end.date(), series="EQ", sync_drive=True))
    except Exception:
        return pd.DataFrame()


def _reference_row(symbol: str, reference_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series | None, float]:
    df = _history(symbol, reference_date - pd.Timedelta(days=40), reference_date + pd.Timedelta(days=2))
    if df.empty:
        return df, None, np.nan
    rows = df[df["date"] <= reference_date.normalize()]
    if rows.empty:
        return df, None, np.nan
    ref = rows.iloc[-1]
    atr = _atr14(df[df["date"] <= ref["date"]])
    return df, ref, atr


def render_delivery_forecast(results: dict[str, dict[str, Any]]) -> None:
    """Render detailed next-session price-path fields for live Delivery results."""
    passed = [r for r in (results or {}).values() if r.get("prediction_status") == "PASS"]
    if not passed:
        return
    st.markdown("### Next-session price forecast")
    st.caption(
        "The classifier directly predicts P(up) and the regressor directly predicts expected return. "
        "Projected Open/High/Low/Close and purchase/sale levels below are derived estimates using the last completed NSE OHLC and 14-session ATR; they are not separately trained OHLC targets."
    )
    rows = []
    for dlv in passed:
        ref_date = pd.Timestamp(dlv.get("reference_session"))
        _, ref, atr = _reference_row(dlv.get("symbol", ""), ref_date)
        if ref is None:
            rows.append({"symbol": dlv.get("symbol"), "status": "Reference OHLC unavailable"})
            continue
        path = _derived_price_path(ref, float(dlv.get("predicted_return_next_session", np.nan)), atr)
        rows.append({
            "symbol": dlv.get("symbol"),
            "reference_date": ref_date.date(),
            "reference_close": float(ref["close"]),
            "predicted_open": path["predicted_open"],
            "predicted_high": path["predicted_high"],
            "predicted_low": path["predicted_low"],
            "predicted_close": path["predicted_close"],
            "purchase_price": path["purchase_price"],
            "sale_price": path["sale_price"],
            "stop_price": path["stop_price"],
            "signal": dlv.get("signal"),
            "p_up": dlv.get("probability_up_next_session"),
            "expected_return": dlv.get("predicted_return_next_session"),
        })
    if rows:
        out = pd.DataFrame(rows)
        for c in ["reference_close", "predicted_open", "predicted_high", "predicted_low", "predicted_close", "purchase_price", "sale_price", "stop_price"]:
            if c in out.columns:
                out[c] = out[c].map(lambda x: f"₹{x:,.2f}" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else x)
        for c in ["p_up", "expected_return"]:
            if c in out.columns:
                out[c] = out[c].map(lambda x: f"{x:.1%}" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else x)
        st.dataframe(out, width="stretch", hide_index=True)


def render_prediction_history() -> None:
    """Render stored Delivery predictions against the next actual NSE session."""
    st.markdown("## Previous Delivery Prediction vs Actual")
    st.caption(
        "Select a date for which the live Delivery model stored a prediction. The table compares that stored prediction with the next completed NSE session. "
        "It does not rerun the model using future information."
    )
    try:
        records = [r for r in load_records() if str(r.get("mode", "")).lower() == "delivery"]
    except Exception as exc:
        st.error(f"Prediction ledger could not be loaded: {type(exc).__name__}: {exc}")
        return
    if not records:
        st.info("No stored Delivery predictions are available yet. Run Analyze on the Live Dashboard first.")
        return

    dated = []
    for r in records:
        raw = r.get("reference_session") or str(r.get("decision_timestamp", ""))[:10]
        try:
            d = pd.Timestamp(raw).normalize()
            if pd.notna(d):
                dated.append(d.date())
        except Exception:
            continue
    dates = sorted(set(dated), reverse=True)
    if not dates:
        st.info(
            "The prediction ledger contains Delivery records but no usable reference dates."
        )
        return

    earliest_prediction_date = min(dates)
    latest_prediction_date = max(dates)
    today = pd.Timestamp.now(
        tz="Asia/Kolkata"
    ).date()

    # Use a real calendar instead of a selectbox limited to dates that already
    # exist in the ledger. A user can inspect any valid historical date; when
    # no stored prediction exists for that date, we report that explicitly.
    default_date = min(
        max(
            latest_prediction_date,
            earliest_prediction_date,
        ),
        today,
    )

    selected = st.date_input(
        "Prediction date",
        value=default_date,
        min_value=earliest_prediction_date,
        max_value=today,
        format="DD-MM-YYYY",
        help=(
            "Choose the date on which an ALL-NSE Delivery prediction was made. "
            "Dates with no stored prediction are allowed and will be reported "
            "as unavailable rather than reconstructed or fabricated."
        ),
    )

    if selected not in dates:
        st.info(
            f"No stored ALL-NSE Delivery prediction exists for "
            f"{selected.strftime('%d-%m-%Y')}. "
            f"Currently stored prediction dates run through "
            f"{latest_prediction_date.strftime('%d-%m-%Y')}."
        )
        return

    selected_records = []
    for r in records:
        raw = r.get("reference_session") or str(r.get("decision_timestamp", ""))[:10]
        try:
            if pd.Timestamp(raw).date() == selected:
                selected_records.append(r)
        except Exception:
            pass

    symbol_options = ["All"] + sorted({_root(r.get("symbol", "")) for r in selected_records if r.get("symbol")})
    selected_symbol = st.selectbox("Stock", symbol_options)
    if selected_symbol != "All":
        selected_records = [r for r in selected_records if _root(r.get("symbol")) == selected_symbol]

    rows = []
    for r in selected_records:
        symbol = _root(r.get("symbol"))
        ref_date = pd.Timestamp(selected)
        df, ref, atr = _reference_row(symbol, ref_date)
        if ref is None or df.empty:
            rows.append({"symbol": symbol, "prediction": r.get("signal"), "status": "Reference/actual NSE history unavailable"})
            continue
        future = df[df["date"] > ref["date"]]
        if future.empty:
            rows.append({"symbol": symbol, "prediction": r.get("signal"), "status": "Next NSE session actual not available yet"})
            continue
        actual = future.iloc[0]
        actual_return = float(actual["close"] / ref["close"] - 1.0) if ref["close"] else np.nan
        p_up = float(r.get("probability_up_next_session", np.nan))
        expected = float(r.get("predicted_return_next_session", np.nan))
        path = _derived_price_path(ref, expected, atr)
        predicted_direction = "UP" if p_up >= 0.5 else "DOWN" if np.isfinite(p_up) else "N/A"
        actual_direction = "UP" if actual_return > 0 else "DOWN" if actual_return < 0 else "FLAT"
        rows.append({
            "symbol": symbol,
            "prediction_date": ref["date"].date(),
            "actual_date": actual["date"].date(),
            "signal": r.get("signal"),
            "p_up": p_up,
            "expected_return": expected,
            "predicted_open": path["predicted_open"],
            "predicted_high": path["predicted_high"],
            "predicted_low": path["predicted_low"],
            "predicted_close": path["predicted_close"],
            "purchase_price": path["purchase_price"],
            "sale_price": path["sale_price"],
            "actual_open": float(actual.get("open", np.nan)),
            "actual_high": float(actual.get("high", np.nan)),
            "actual_low": float(actual.get("low", np.nan)),
            "actual_close": float(actual["close"]),
            "actual_return": actual_return,
            "predicted_vs_actual_close_pct": (path["predicted_close"] / float(actual["close"]) - 1.0) if actual["close"] else np.nan,
            "direction_match": (predicted_direction == actual_direction) if actual_direction != "FLAT" else None,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        st.info("No matching prediction records for the selected date/stock.")
        return

    valid = out[out.get("actual_return", pd.Series(dtype=float)).notna()] if "actual_return" in out.columns else pd.DataFrame()
    if not valid.empty:
        m1, m2, m3 = st.columns(3)
        m1.metric("Predictions", len(valid))
        m2.metric("Actuals available", len(valid))
        matches = valid["direction_match"].dropna()
        m3.metric("Direction matches", f"{int(matches.sum())}/{len(matches)}" if len(matches) else "N/A")

    money_cols = [c for c in out.columns if c.endswith("_open") or c.endswith("_high") or c.endswith("_low") or c.endswith("_close") or c in {"purchase_price", "sale_price"}]
    for c in money_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"₹{x:,.2f}" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else x)
    for c in ["p_up", "expected_return", "actual_return", "predicted_vs_actual_close_pct"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:.2%}" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else x)
    st.dataframe(out, width="stretch", hide_index=True)
    st.caption("Direction match compares the stored P(up) direction with the next session's actual close direction. Price-path fields are derived estimates, not separate trained targets.")
