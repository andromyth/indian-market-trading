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
from all_nse_delivery_model import predict_symbol as all_nse_delivery_predict


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


def _historical_ml_vs_actual(
    symbol: str,
    prediction_date: pd.Timestamp,
) -> dict[str, Any]:

    root = _root(symbol)

    # Reconstruct only from information available through the selected
    # historical NSE session. Existing PIT snapshots are used when available;
    # otherwise the ALL-NSE model builds its fixed 59-feature row from the
    # official historical archive.
    decision_ts = pd.Timestamp(
        prediction_date.date()
    ).tz_localize(
        "Asia/Kolkata"
    ) + pd.Timedelta(
        hours=15,
        minutes=30,
    )

    prediction = all_nse_delivery_predict(
        root,
        decision_ts,
    )

    if prediction.get("prediction_status") != "PASS":
        return {
            "symbol": root,
            "prediction_source": "HISTORICAL_ML_REPLAY",
            "status": "PREDICTION_BLOCKED",
            "prediction_reason": prediction.get(
                "reason",
                "Historical model replay did not produce a prediction.",
            ),
        }

    ref_date = pd.Timestamp(
        prediction.get(
            "reference_session",
            prediction_date.date(),
        )
    ).normalize()

    df, ref, atr = _reference_row(
        root,
        ref_date,
    )

    if ref is None or df.empty:
        return {
            "symbol": root,
            "prediction_source": "HISTORICAL_ML_REPLAY",
            "prediction_date": ref_date.date(),
            "status": "REFERENCE_DATA_UNAVAILABLE",
        }

    future = df[
        df["date"] > ref_date
    ].sort_values("date")

    p_up = float(
        prediction.get(
            "probability_up_next_session",
            np.nan,
        )
    )

    expected = float(
        prediction.get(
            "predicted_return_next_session",
            np.nan,
        )
    )

    path = _derived_price_path(
        ref,
        expected,
        atr,
    )

    row = {
        "symbol": root,
        "prediction_date": ref_date.date(),
        "prediction_source": "HISTORICAL_ML_REPLAY",
        "model_version": prediction.get(
            "model_version"
        ),
        "signal": prediction.get(
            "signal"
        ),
        "p_up": p_up,
        "expected_return": expected,
        "reference_close": float(
            ref["close"]
        ),
        "predicted_open": path[
            "predicted_open"
        ],
        "predicted_high": path[
            "predicted_high"
        ],
        "predicted_low": path[
            "predicted_low"
        ],
        "predicted_close": path[
            "predicted_close"
        ],
        "purchase_price": path[
            "purchase_price"
        ],
        "sale_price": path[
            "sale_price"
        ],
        "stop_price": path[
            "stop_price"
        ],
    }

    if future.empty:
        row["status"] = (
            "Prediction available; next NSE session actual not available yet"
        )
        return row

    actual = future.iloc[0]

    actual_return = (
        float(
            actual["close"] / ref["close"] - 1.0
        )
        if float(ref["close"]) != 0
        else np.nan
    )

    predicted_direction = (
        "UP"
        if p_up >= 0.5
        else "DOWN"
        if np.isfinite(p_up)
        else "N/A"
    )

    actual_direction = (
        "UP"
        if actual_return > 0
        else "DOWN"
        if actual_return < 0
        else "FLAT"
    )

    row.update({
        "actual_date": actual["date"].date(),
        "actual_open": float(
            actual.get("open", np.nan)
        ),
        "actual_high": float(
            actual.get("high", np.nan)
        ),
        "actual_low": float(
            actual.get("low", np.nan)
        ),
        "actual_close": float(
            actual["close"]
        ),
        "actual_return": actual_return,
        "predicted_vs_actual_close_pct": (
            path["predicted_close"]
            / float(actual["close"])
            - 1.0
            if float(actual["close"]) != 0
            else np.nan
        ),
        "predicted_direction": predicted_direction,
        "actual_direction": actual_direction,
        "direction_match": (
            predicted_direction == actual_direction
            if actual_direction != "FLAT"
            else None
        ),
        "status": "OK",
    })

    return row


def _stored_prediction_vs_actual(
    stored: dict[str, Any],
    symbol: str,
    selected: pd.Timestamp,
) -> dict[str, Any]:

    ref_date = pd.Timestamp(
        stored.get(
            "reference_session",
            selected.date(),
        )
    ).normalize()

    df, ref, atr = _reference_row(
        symbol,
        ref_date,
    )

    if ref is None or df.empty:
        return {
            "symbol": symbol,
            "prediction_source": "STORED_LIVE",
            "status": "REFERENCE_DATA_UNAVAILABLE",
        }

    future = df[
        df["date"] > ref["date"]
    ].sort_values("date")

    p_up = float(
        stored.get(
            "probability_up_next_session",
            np.nan,
        )
    )

    expected = float(
        stored.get(
            "predicted_return_next_session",
            np.nan,
        )
    )

    path = _derived_price_path(
        ref,
        expected,
        atr,
    )

    row = {
        "symbol": symbol,
        "prediction_date": ref["date"].date(),
        "prediction_source": "STORED_LIVE",
        "model_version": stored.get(
            "model_version"
        ),
        "signal": stored.get("signal"),
        "p_up": p_up,
        "expected_return": expected,
        "reference_close": float(ref["close"]),
        "predicted_open": path["predicted_open"],
        "predicted_high": path["predicted_high"],
        "predicted_low": path["predicted_low"],
        "predicted_close": path["predicted_close"],
        "purchase_price": path["purchase_price"],
        "sale_price": path["sale_price"],
        "stop_price": path["stop_price"],
    }

    if future.empty:
        row["status"] = (
            "Prediction available; next NSE session actual not available yet"
        )
        return row

    actual = future.iloc[0]

    actual_return = (
        float(
            actual["close"] / ref["close"] - 1.0
        )
        if float(ref["close"]) != 0
        else np.nan
    )

    predicted_direction = (
        "UP"
        if p_up >= 0.5
        else "DOWN"
        if np.isfinite(p_up)
        else "N/A"
    )

    actual_direction = (
        "UP"
        if actual_return > 0
        else "DOWN"
        if actual_return < 0
        else "FLAT"
    )

    row.update({
        "actual_date": actual["date"].date(),
        "actual_open": float(
            actual.get("open", np.nan)
        ),
        "actual_high": float(
            actual.get("high", np.nan)
        ),
        "actual_low": float(
            actual.get("low", np.nan)
        ),
        "actual_close": float(actual["close"]),
        "actual_return": actual_return,
        "predicted_vs_actual_close_pct": (
            path["predicted_close"]
            / float(actual["close"])
            - 1.0
            if float(actual["close"]) != 0
            else np.nan
        ),
        "predicted_direction": predicted_direction,
        "actual_direction": actual_direction,
        "direction_match": (
            predicted_direction == actual_direction
            if actual_direction != "FLAT"
            else None
        ),
        "status": "OK",
    })

    return row


def render_prediction_history() -> None:
    """Compare stored Delivery predictions or replay the ALL-NSE model historically."""

    st.markdown("## Previous Delivery Prediction vs Actual")

    st.caption(
        "Choose any historical date and stock. Stored live predictions are compared "
        "directly with the next actual NSE session. For dates without a stored live "
        "prediction, the same 59-feature ALL-NSE model is reconstructed from the "
        "official historical archive using only information available by that date. "
        "Historical replay is retrospective analysis, not prospective OOS validation."
    )

    try:
        records = [
            r
            for r in load_records()
            if str(r.get("mode", "")).lower() == "delivery"
        ]
    except Exception:
        records = []

    stored_by_date_symbol: dict[tuple, dict[str, Any]] = {}

    for r in records:
        raw = (
            r.get("reference_session")
            or str(
                r.get(
                    "decision_timestamp",
                    "",
                )
            )[:10]
        )

        try:
            d = pd.Timestamp(raw).date()
        except Exception:
            continue

        symbol = _root(
            r.get(
                "symbol",
                "",
            )
        )

        if symbol:
            stored_by_date_symbol[
                (d, symbol)
            ] = r

    today = pd.Timestamp.now(
        tz="Asia/Kolkata"
    ).date()

    active = str(
        st.session_state.get(
            "active_watchlist",
            "RELIANCE.NS",
        )
    )

    default_symbol = _root(
        active.split(",")[0]
    ) or "RELIANCE"

    c1, c2 = st.columns(2)

    selected = c1.date_input(
        "Prediction date",
        value=min(
            today,
            pd.Timestamp("2018-01-01").date()
            if today < pd.Timestamp("2018-01-01").date()
            else today,
        ),
        min_value=pd.Timestamp(
            "2018-01-01"
        ).date(),
        max_value=today,
        format="DD-MM-YYYY",
    )

    stock_text = c2.text_input(
        "Stock / stocks",
        value=f"{default_symbol}.NS",
        help=(
            "Example: RELIANCE.NS, SBIN.NS, TCS.NS. "
            "Limit interactive historical replay to 20 stocks."
        ),
    )

    requested_symbols = [
        _root(x)
        for x in stock_text.split(",")
        if str(x).strip()
    ]

    requested_symbols = [
        x
        for x in dict.fromkeys(requested_symbols)
        if x
    ]

    run = st.button(
        "Check Prediction vs Actual",
        type="primary",
    )

    if not run:
        return

    if not requested_symbols:
        st.error("Enter at least one NSE symbol.")
        return

    if len(requested_symbols) > 20:
        st.error(
            "Limit interactive historical replay to 20 stocks at a time."
        )
        return

    rows = []

    for symbol in requested_symbols:

        stored = stored_by_date_symbol.get(
            (
                selected,
                symbol,
            )
        )

        try:

            if stored is not None:

                rows.append(
                    _stored_prediction_vs_actual(
                        stored,
                        symbol,
                        pd.Timestamp(selected),
                    )
                )

            else:

                with st.spinner(
                    f"Reconstructing {symbol} for "
                    f"{selected.strftime('%d-%m-%Y')}..."
                ):

                    rows.append(
                        _historical_ml_vs_actual(
                            symbol,
                            pd.Timestamp(selected),
                        )
                    )

        except Exception as exc:

            rows.append({
                "symbol": symbol,
                "prediction_source": (
                    "STORED_LIVE"
                    if stored is not None
                    else "HISTORICAL_ML_REPLAY"
                ),
                "status": (
                    f"Replay failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            })

    out = pd.DataFrame(rows)

    if out.empty:
        st.info(
            "No prediction/actual comparison could be produced."
        )
        return

    st.markdown(
        "### Prediction vs actual"
    )

    valid = (
        out[
            out["actual_return"].notna()
        ]
        if "actual_return" in out.columns
        else pd.DataFrame()
    )

    if not valid.empty:

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Predictions",
            len(out),
        )

        m2.metric(
            "Actuals available",
            len(valid),
        )

        matches = (
            valid["direction_match"]
            .dropna()
        )

        m3.metric(
            "Direction matches",
            (
                f"{int(matches.sum())}/{len(matches)}"
                if len(matches)
                else "N/A"
            ),
        )

    money_cols = [
        col
        for col in out.columns
        if col.endswith("_open")
        or col.endswith("_high")
        or col.endswith("_low")
        or col.endswith("_close")
        or col in {
            "reference_close",
            "purchase_price",
            "sale_price",
            "stop_price",
        }
    ]

    for col in money_cols:

        if col in out.columns:

            out[col] = out[col].map(
                lambda x: (
                    f"₹{x:,.2f}"
                    if isinstance(
                        x,
                        (
                            int,
                            float,
                            np.floating,
                        ),
                    )
                    and np.isfinite(x)
                    else x
                )
            )

    for col in [
        "p_up",
        "expected_return",
        "actual_return",
        "predicted_vs_actual_close_pct",
    ]:

        if col in out.columns:

            out[col] = out[col].map(
                lambda x: (
                    f"{x:.2%}"
                    if isinstance(
                        x,
                        (
                            int,
                            float,
                            np.floating,
                        ),
                    )
                    and np.isfinite(x)
                    else x
                )
            )

    preferred_columns = [
        "symbol",
        "prediction_date",
        "actual_date",
        "prediction_source",
        "model_version",
        "signal",
        "p_up",
        "expected_return",
        "reference_close",
        "predicted_open",
        "predicted_high",
        "predicted_low",
        "predicted_close",
        "purchase_price",
        "sale_price",
        "stop_price",
        "actual_open",
        "actual_high",
        "actual_low",
        "actual_close",
        "actual_return",
        "predicted_vs_actual_close_pct",
        "predicted_direction",
        "actual_direction",
        "direction_match",
        "status",
        "prediction_reason",
    ]

    display_columns = [
        col
        for col in preferred_columns
        if col in out.columns
    ]

    st.dataframe(
        out[display_columns],
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "STORED_LIVE rows use the prediction actually recorded by the live system. "
        "HISTORICAL_ML_REPLAY rows reconstruct the same current 59-feature ALL-NSE "
        "model from point-in-time official NSE data. Retrospective replay is not "
        "prospective OOS validation. Projected OHLC, purchase and sale levels are "
        "derived estimates, not separately trained OHLC targets."
    )
