"""Streamlit UI for live Delivery price forecasts and historical comparison.

The active ALL-NSE Delivery model directly predicts next-session direction
probability and expected return. OHLC/entry/exit figures are derived estimates,
not separately trained OHLC targets.

Historical replay is retrospective analysis only. It does not alter the live
model, prediction ledger, or prospective OOS baseline.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def _root(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    for suffix in (".NS", ".BO"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def _clean_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    x = frame.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]

    if x.columns.duplicated().any():
        x = x.loc[:, ~x.columns.duplicated(keep="first")].copy()

    if "date" not in x.columns:
        return pd.DataFrame()

    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce",
    ).dt.normalize()

    for col in ("open", "high", "low", "close", "volume", "delivery_qty", "delivery_pct"):
        if col in x.columns:
            x[col] = pd.to_numeric(
                x[col],
                errors="coerce",
            )

    if "close" not in x.columns:
        return pd.DataFrame()

    return (
        x.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def _atr14(frame: pd.DataFrame) -> float:
    x = _clean_ohlc(frame)

    if len(x) < 2 or not all(
        col in x.columns
        for col in ("high", "low", "close")
    ):
        return np.nan

    prev_close = x["close"].shift(1)

    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    value = (
        tr.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=5,
        )
        .mean()
        .iloc[-1]
    )

    return float(value) if np.isfinite(value) else np.nan


def _price_path(
    reference: pd.Series,
    expected_return: float,
    atr: float,
) -> dict[str, float]:

    close = float(
        reference.get(
            "close",
            np.nan,
        )
    )

    if not np.isfinite(close):
        return {
            key: np.nan
            for key in (
                "predicted_open",
                "predicted_high",
                "predicted_low",
                "predicted_close",
                "purchase_price",
                "sale_price",
                "stop_price",
            )
        }

    er = (
        float(expected_return)
        if np.isfinite(expected_return)
        else 0.0
    )

    predicted_close = close * (1.0 + er)
    predicted_open = close * (1.0 + 0.25 * er)

    volatility = (
        float(atr)
        if np.isfinite(atr) and atr > 0
        else close * 0.01
    )

    predicted_high = (
        max(
            predicted_open,
            predicted_close,
        )
        + 0.50 * volatility
    )

    predicted_low = (
        min(
            predicted_open,
            predicted_close,
        )
        - 0.50 * volatility
    )

    return {
        "predicted_open": predicted_open,
        "predicted_high": predicted_high,
        "predicted_low": max(
            0.0,
            predicted_low,
        ),
        "predicted_close": predicted_close,
        "purchase_price": close,
        "sale_price": predicted_close,
        "stop_price": max(
            0.0,
            close - volatility
            if er >= 0
            else close + volatility,
        ),
    }


def _stock_history_from_drive(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load official NSE bulk-EOD rows directly from the persisted Drive archive.

    This historical-replay loader intentionally does not call fetch_equity_history
    or the live NSE transport layer.
    """
    from pathlib import Path
    from persistent_store import GoogleDriveStore
    from googleapiclient.http import MediaIoBaseDownload

    root = _root(symbol)
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()

    store = GoogleDriveStore()

    # Find the persisted bulk-EOD root.
    parent = store.root_id

    for part in (
        "production_data",
        "nse_official",
        "bulk_eod",
    ):
        folder = store._find_child(
            parent,
            part,
            "application/vnd.google-apps.folder",
        )

        if not folder:
            raise RuntimeError(
                f"Google Drive archive folder not found: {part}"
            )

        parent = folder["id"]

    pieces = []

    # The ALL-NSE archive is daily and organized as YYYY/MM/DD folders.
    for year in range(
        start_day.year,
        end_day.year + 1,
    ):

        year_folder = store._find_child(
            parent,
            f"{year:04d}",
            "application/vnd.google-apps.folder",
        )

        if not year_folder:
            continue

        month_first = (
            start_day.month
            if year == start_day.year
            else 1
        )

        month_last = (
            end_day.month
            if year == end_day.year
            else 12
        )

        for month in range(
            month_first,
            month_last + 1,
        ):

            month_folder = store._find_child(
                year_folder["id"],
                f"{month:02d}",
                "application/vnd.google-apps.folder",
            )

            if not month_folder:
                continue

            days = pd.date_range(
                max(
                    start_day,
                    pd.Timestamp(
                        year,
                        month,
                        1,
                    ),
                ),
                min(
                    end_day,
                    (
                        pd.Timestamp(
                            year,
                            month,
                            1,
                        )
                        + pd.offsets.MonthEnd(1)
                    ),
                ),
                freq="D",
            )

            for day in days:

                day_folder = store._find_child(
                    month_folder["id"],
                    f"{day.day:02d}",
                    "application/vnd.google-apps.folder",
                )

                if not day_folder:
                    continue

                filename = (
                    f"BhavCopy_NSE_CM_0_0_0_"
                    f"{day:%Y%m%d}_F_0000.parquet"
                )

                item = store._find_child(
                    day_folder["id"],
                    filename,
                    None,
                )

                if not item:
                    continue

                cache_dir = (
                    Path(".historical_bulk_cache")
                    / root
                )

                cache_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                cache_file = (
                    cache_dir
                    / filename
                )

                try:

                    if (
                        not cache_file.exists()
                        or cache_file.stat().st_size == 0
                    ):

                        request = (
                            store.service
                            .files()
                            .get_media(
                                fileId=item["id"]
                            )
                        )

                        temp = cache_file.with_suffix(
                            ".tmp"
                        )

                        with temp.open("wb") as fh:

                            downloader = MediaIoBaseDownload(
                                fh,
                                request,
                            )

                            done = False

                            while not done:
                                _, done = downloader.next_chunk()

                        temp.replace(
                            cache_file
                        )

                    frame = pd.read_parquet(
                        cache_file
                    )

                except Exception:
                    continue

                if frame is None or frame.empty:
                    continue

                frame.columns = [
                    str(col).strip().lower()
                    for col in frame.columns
                ]

                if frame.columns.duplicated().any():
                    frame = frame.loc[
                        :,
                        ~frame.columns.duplicated(
                            keep="first"
                        ),
                    ].copy()

                if "symbol" not in frame.columns:
                    continue

                symbols = (
                    frame["symbol"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

                hit = frame[
                    symbols == root
                ].copy()

                if hit.empty:
                    continue

                if "date" not in hit.columns:
                    hit["date"] = day

                hit["date"] = pd.to_datetime(
                    hit["date"],
                    errors="coerce",
                ).dt.normalize()

                # Normalize the official UDiFF field names used by the archive.
                aliases = {
                    "open": (
                        "open",
                        "opnpric",
                        "ch_opening_price",
                    ),
                    "high": (
                        "high",
                        "hghpric",
                        "ch_trade_high_price",
                    ),
                    "low": (
                        "low",
                        "lwpric",
                        "ch_trade_low_price",
                    ),
                    "close": (
                        "close",
                        "clspric",
                        "ch_closing_price",
                    ),
                    "volume": (
                        "volume",
                        "ttltradgvol",
                        "ch_tot_traded_qty",
                    ),
                    "delivery_qty": (
                        "delivery_qty",
                        "cop_deliv_qty",
                    ),
                    "delivery_pct": (
                        "delivery_pct",
                        "cop_deliv_perc",
                    ),
                }

                for target, names in aliases.items():
                    source = next(
                        (
                            name
                            for name in names
                            if name in hit.columns
                        ),
                        None,
                    )

                    if source is None:
                        hit[target] = np.nan

                    elif target != source:
                        hit[target] = hit[source]

                    hit[target] = pd.to_numeric(
                        hit[target],
                        errors="coerce",
                    )

                hit = hit.dropna(
                    subset=[
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                    ]
                )

                if not hit.empty:
                    pieces.append(
                        hit[
                            [
                                "symbol",
                                "date",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                                "delivery_qty",
                                "delivery_pct",
                            ]
                        ]
                    )

    if not pieces:
        return pd.DataFrame()

    out = (
        pd.concat(
            pieces,
            ignore_index=True,
        )
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    return out


def _benchmark_history(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:

    """Load retrospective benchmark data without calling NSE's blocked index API."""

    bank_symbols = {
        "HDFCBANK",
        "ICICIBANK",
        "SBIN",
        "AXISBANK",
        "KOTAKBANK",
        "INDUSINDBK",
        "BANKBARODA",
        "PNB",
    }

    root = _root(symbol)

    benchmark = (
        "^NSEBANK"
        if root in bank_symbols
        else "^NSEI"
    )

    # Use Yahoo only for retrospective replay. It is explicitly labelled in
    # the result and is never used by the live PIT prediction path.
    data = yf.download(
        benchmark,
        start=start.date(),
        end=(end + pd.Timedelta(days=1)).date(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        raise RuntimeError(
            f"No retrospective benchmark data available for {benchmark}."
        )

    x = data.copy()

    if isinstance(
        x.columns,
        pd.MultiIndex,
    ):
        flattened = []

        for col in x.columns:
            parts = [
                str(part).strip().lower()
                for part in col
                if str(part).strip()
            ]

            flattened.append(
                parts[0]
                if parts
                else ""
            )

        x.columns = flattened

    else:
        x.columns = [
            str(col).strip().lower()
            for col in x.columns
        ]

    if x.columns.duplicated().any():
        x = x.loc[
            :,
            ~x.columns.duplicated(
                keep="first"
            ),
        ].copy()

    x = x.reset_index()
    x.columns = [
        str(col).strip().lower()
        for col in x.columns
    ]

    if x.columns.duplicated().any():
        x = x.loc[
            :,
            ~x.columns.duplicated(
                keep="first"
            ),
        ].copy()

    date_col = next(
        (
            col
            for col in ("date", "datetime")
            if col in x.columns
        ),
        None,
    )

    if date_col is None:
        raise RuntimeError(
            f"Retrospective benchmark has no date column for {benchmark}."
        )

    x["date"] = pd.to_datetime(
        x[date_col],
        errors="coerce",
    ).dt.normalize()

    for col in (
        "open",
        "high",
        "low",
        "close",
    ):
        if col not in x.columns:
            raise RuntimeError(
                f"Retrospective benchmark missing {col} for {benchmark}."
            )

        x[col] = pd.to_numeric(
            x[col],
            errors="coerce",
        )

    x = (
        x.dropna(
            subset=[
                "date",
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    if x.empty:
        raise RuntimeError(
            f"Retrospective benchmark has no usable rows for {benchmark}."
        )

    x["symbol"] = benchmark

    return (
        x[
            [
                "symbol",
                "date",
                "open",
                "high",
                "low",
                "close",
            ]
        ],
        "YAHOO_RETROSPECTIVE_REFERENCE",
    )


def _historical_prediction(
    symbol: str,
    selected_date: pd.Timestamp,
) -> dict[str, Any]:
    """Replay a historical prediction using data strictly before target date.

    The selected date is the target NSE session. No stock, benchmark, or other
    feature information from the target session is allowed into the model
    inputs. The actual target-session OHLC is used only after prediction.
    """

    from pathlib import Path
    from all_nse_delivery_model import (
        BUY_P,
        MIN_EXPECTED_RETURN,
        SELL_P,
        _build_latest_features,
        _load_models,
    )

    root = _root(symbol)

    target_date = pd.Timestamp(
        selected_date
    ).normalize()

    # Strict causal boundary: model inputs stop on the previous calendar day.
    # The NSE archive naturally resolves this to the previous trading session.
    input_end = (
        target_date
        - pd.Timedelta(days=1)
    ).normalize()

    history_start = (
        input_end
        - pd.Timedelta(days=365)
    )

    # Load enough history through the day before the target session.
    stock_input = _stock_history_from_drive(
        root,
        history_start,
        input_end,
    )

    if stock_input.empty:
        raise RuntimeError(
            f"Official NSE bulk-EOD history unavailable for {root} "
            f"through {input_end.date()}."
        )

    input_rows = stock_input[
        stock_input["date"] <= input_end
    ].sort_values("date")

    if input_rows.empty:
        raise RuntimeError(
            f"No model input session exists before target date "
            f"{target_date.date()} for {root}."
        )

    reference_date = pd.Timestamp(
        input_rows["date"].iloc[-1]
    ).normalize()

    # Benchmark also stops at the reference session. The target session is
    # deliberately excluded to prevent look-ahead leakage.
    benchmark_start = (
        reference_date
        - pd.Timedelta(days=365)
    )

    benchmark, benchmark_source = _benchmark_history(
        root,
        benchmark_start,
        reference_date,
    )

    features = _build_latest_features(
        input_rows,
        benchmark,
    )

    clf, reg, manifest = _load_models()

    expected_features = list(
        manifest["features"]
    )

    if list(features.columns) != expected_features:
        raise RuntimeError(
            "Historical replay feature order does not match "
            "the locked 59-feature model."
        )

    p_up = float(
        np.clip(
            clf.predict_proba(features)[0, 1],
            0.0,
            1.0,
        )
    )

    expected_return = float(
        reg.predict(features)[0]
    )

    if (
        p_up >= BUY_P
        and expected_return > MIN_EXPECTED_RETURN
    ):
        signal = "BUY"
    elif (
        p_up <= SELL_P
        and expected_return < -MIN_EXPECTED_RETURN
    ):
        signal = "SELL"
    else:
        signal = "HOLD"

    # Actual target-session data is deliberately loaded separately, after the
    # prediction has been generated.
    future = _stock_history_from_drive(
        root,
        target_date,
        target_date,
    )

    actual_rows = future[
        future["date"] == target_date
    ].sort_values("date")

    reference = input_rows[
        input_rows["date"] == reference_date
    ].iloc[-1]

    atr = _atr14(input_rows)

    path = _price_path(
        reference,
        expected_return,
        atr,
    )

    result = {
        "symbol": root,
        "prediction_date": target_date.date(),
        "input_through_date": reference_date.date(),
        "prediction_source": "HISTORICAL_ML_REPLAY",
        "model_version": manifest.get(
            "engine_version",
        ),
        "benchmark_source": benchmark_source,
        "signal": signal,
        "p_up": p_up,
        "expected_return": expected_return,
        "data_readiness": float(
            features.notna().sum(axis=1).iloc[0]
            / len(expected_features)
        ),
        "feature_count": len(
            expected_features
        ),
        "reference_close": float(
            reference["close"]
        ),
        "predicted_open": path["predicted_open"],
        "predicted_high": path["predicted_high"],
        "predicted_low": path["predicted_low"],
        "predicted_close": path["predicted_close"],
        "purchase_price": path["purchase_price"],
        "sale_price": path["sale_price"],
        "stop_price": path["stop_price"],
    }

    if actual_rows.empty:
        result["status"] = (
            "Prediction generated from data through the previous session, "
            "but target-session actual is unavailable."
        )
        return result

    actual = actual_rows.iloc[0]

    actual_return = (
        float(
            actual["close"]
            / reference["close"]
            - 1.0
        )
        if float(reference["close"]) != 0
        else np.nan
    )

    predicted_direction = (
        "UP"
        if p_up >= 0.5
        else "DOWN"
    )

    actual_direction = (
        "UP"
        if actual_return > 0
        else "DOWN"
        if actual_return < 0
        else "FLAT"
    )

    result.update({
        "actual_date": actual["date"].date(),
        "actual_open": float(
            actual["open"]
        ),
        "actual_high": float(
            actual["high"]
        ),
        "actual_low": float(
            actual["low"]
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
        ),
        "status": "OK",
    })

    return result


def render_delivery_forecast(
    results: dict[str, dict[str, Any]]
) -> None:
    """Show derived next-session price levels for live Delivery predictions."""

    passed = [
        result
        for result in (results or {}).values()
        if result.get("prediction_status") == "PASS"
    ]

    if not passed:
        return

    st.markdown(
        "### Next-session price forecast"
    )

    st.caption(
        "P(up) and expected return are the model's direct predictions. "
        "Open/High/Low/Close, purchase and sale levels are transparent "
        "derived estimates based on the reference close, expected return and ATR."
    )

    rows = []

    for prediction in passed:

        symbol = prediction.get(
            "symbol",
            "",
        )

        reference_date = pd.Timestamp(
            prediction.get(
                "reference_session"
            )
        )

        history = _stock_history_from_drive(
            symbol,
            reference_date - pd.Timedelta(days=60),
            reference_date,
        )

        if history.empty:
            rows.append({
                "symbol": symbol,
                "status": "Reference OHLC unavailable",
            })
            continue

        reference = history.iloc[-1]
        atr = _atr14(history)

        path = _price_path(
            reference,
            float(
                prediction.get(
                    "predicted_return_next_session",
                    np.nan,
                )
            ),
            atr,
        )

        rows.append({
            "symbol": symbol,
            "reference_date": reference_date.date(),
            "reference_close": float(
                reference["close"]
            ),
            "predicted_open": path["predicted_open"],
            "predicted_high": path["predicted_high"],
            "predicted_low": path["predicted_low"],
            "predicted_close": path["predicted_close"],
            "purchase_price": path["purchase_price"],
            "sale_price": path["sale_price"],
            "stop_price": path["stop_price"],
            "signal": prediction.get("signal"),
            "p_up": prediction.get(
                "probability_up_next_session"
            ),
            "expected_return": prediction.get(
                "predicted_return_next_session"
            ),
        })

    if not rows:
        return

    out = pd.DataFrame(rows)

    for col in (
        "reference_close",
        "predicted_open",
        "predicted_high",
        "predicted_low",
        "predicted_close",
        "purchase_price",
        "sale_price",
        "stop_price",
    ):
        if col in out.columns:
            out[col] = out[col].map(
                lambda value: (
                    f"₹{value:,.2f}"
                    if isinstance(
                        value,
                        (
                            int,
                            float,
                            np.floating,
                        ),
                    )
                    and np.isfinite(value)
                    else value
                )
            )

    for col in (
        "p_up",
        "expected_return",
    ):
        if col in out.columns:
            out[col] = out[col].map(
                lambda value: (
                    f"{value:.1%}"
                    if isinstance(
                        value,
                        (
                            int,
                            float,
                            np.floating,
                        ),
                    )
                    and np.isfinite(value)
                    else value
                )
            )

    st.dataframe(
        out,
        width="stretch",
        hide_index=True,
    )


def render_prediction_history() -> None:
    """Historical Prediction → Actual comparison."""

    st.markdown(
        "## Previous Delivery Prediction vs Actual"
    )

    st.caption(
        "Select any historical date and NSE stock. "
        "For dates without a stored live prediction, the app reconstructs "
        "the current 59-feature ALL-NSE model using data only through the "
        "previous NSE session, then compares that prediction with the selected "
        "target session. Historical replay is retrospective analysis, not "
        "prospective OOS validation."
    )

    today = pd.Timestamp.now(
        tz="Asia/Kolkata"
    ).date()

    c1, c2 = st.columns(2)

    selected = c1.date_input(
        "Prediction target date",
        value=today,
        min_value=pd.Timestamp(
            "2026-01-01"
        ).date(),
        max_value=today,
        format="DD-MM-YYYY",
    )

    active_watchlist = str(
        st.session_state.get(
            "active_watchlist",
            "RELIANCE.NS",
        )
    )

    default_stock = (
        active_watchlist.split(",")[0].strip()
        or "RELIANCE.NS"
    )

    stock_text = c2.text_input(
        "Stock / stocks",
        value=default_stock,
        help=(
            "Example: SBIN.NS, RELIANCE.NS, TCS.NS. "
            "Limit historical replay to 10 stocks per request."
        ),
    )

    requested = [
        _root(item)
        for item in stock_text.split(",")
        if str(item).strip()
    ]

    requested = list(
        dict.fromkeys(
            item
            for item in requested
            if item
        )
    )

    run = st.button(
        "Check Prediction vs Actual",
        type="primary",
    )

    if not run:
        return

    if not requested:
        st.error(
            "Enter at least one NSE symbol."
        )
        return

    if len(requested) > 10:
        st.error(
            "Limit historical replay to 10 stocks at a time."
        )
        return

    st.info(
        f"Historical replay is causal: the model uses data only through "
        f"{(pd.Timestamp(selected) - pd.Timedelta(days=1)).strftime('%d-%m-%Y')} "
        f"(previous calendar day; the actual previous NSE trading session is "
        f"resolved from the archive). The selected date "
        f"{pd.Timestamp(selected).strftime('%d-%m-%Y')} is the prediction target."
    )

    rows = []

    for symbol in requested:

        with st.spinner(
            f"Reconstructing {symbol} "
            f"for {selected.strftime('%d-%m-%Y')}..."
        ):

            try:
                rows.append(
                    _historical_prediction(
                        symbol,
                        pd.Timestamp(
                            selected
                        ),
                    )
                )

            except Exception as exc:

                rows.append({
                    "symbol": symbol,
                    "prediction_source": "HISTORICAL_ML_REPLAY",
                    "status": (
                        f"Replay failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                })

    out = pd.DataFrame(rows)

    st.markdown(
        "### Prediction vs actual"
    )

    if "actual_return" in out.columns:

        valid = out[
            out["actual_return"].notna()
        ]

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

            matches = valid[
                "direction_match"
            ].dropna()

            m3.metric(
                "Direction matches",
                (
                    f"{int(matches.sum())}/{len(matches)}"
                    if len(matches)
                    else "N/A"
                ),
            )

    display = out.copy()

    for col in (
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
    ):
        if col in display.columns:
            display[col] = display[col].map(
                lambda value: (
                    f"₹{value:,.2f}"
                    if isinstance(
                        value,
                        (
                            int,
                            float,
                            np.floating,
                        ),
                    )
                    and np.isfinite(value)
                    else value
                )
            )

    for col in (
        "p_up",
        "expected_return",
        "actual_return",
        "predicted_vs_actual_close_pct",
        "data_readiness",
    ):
        if col in display.columns:
            display[col] = display[col].map(
                lambda value: (
                    f"{value:.2%}"
                    if isinstance(
                        value,
                        (
                            int,
                            float,
                            np.floating,
                        ),
                    )
                    and np.isfinite(value)
                    else value
                )
            )

    preferred = [
        "symbol",
        "prediction_date",
        "actual_date",
        "prediction_source",
        "model_version",
        "benchmark_source",
        "signal",
        "p_up",
        "expected_return",
        "data_readiness",
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
    ]

    columns = [
        col
        for col in preferred
        if col in display.columns
    ]

    st.dataframe(
        display[columns],
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "STORED_LIVE and HISTORICAL_ML_REPLAY are kept distinct. "
        "Historical replay uses only data available on/before the selected "
        "reference session. Benchmark source is explicitly reported. "
        "Projected OHLC, purchase and sale levels are derived estimates."
    )
