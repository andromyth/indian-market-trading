"""Streamlit UI for live Delivery price forecasts and historical comparison.

The active ALL-NSE Delivery model directly predicts next-session direction
probability and expected return. OHLC/entry/exit figures are derived estimates,
not separately trained OHLC targets.

Historical replay is retrospective analysis only. It does not alter the live
model, prediction ledger, or prospective OOS baseline.
"""

from __future__ import annotations

from pathlib import Path


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

    # Canonicalize official NSE field aliases used by persisted compact
    # histories. The 59-feature model requires these exact names.
    alias_map = {
        "cop_deliv_qty": "delivery_qty",
        "delivery_qty": "delivery_qty",
        "delivery quantity": "delivery_qty",
        "delivery_quantity": "delivery_qty",
        "cop_deliv_perc": "delivery_pct",
        "delivery %": "delivery_pct",
        "delivery_percentage": "delivery_pct",
        "delivery_percent": "delivery_pct",
        "ttltradgvol": "volume",
        "ch_tot_traded_qty": "volume",
        "opnpric": "open",
        "hghpric": "high",
        "lwpric": "low",
        "clspric": "close",
        "ch_opening_price": "open",
        "ch_trade_high_price": "high",
        "ch_trade_low_price": "low",
        "ch_closing_price": "close",
    }

    rename = {
        col: alias_map[col]
        for col in x.columns
        if col in alias_map
        and alias_map[col] != col
    }

    if rename:
        x = x.rename(columns=rename)

    if x.columns.duplicated().any():
        # Merge duplicate canonical/alias columns deterministically by taking
        # the first non-null value in each row, then keep one column.
        deduped = pd.DataFrame(
            index=x.index
        )

        for column in dict.fromkeys(x.columns):
            block = x.loc[
                :,
                x.columns == column,
            ]

            if isinstance(
                block,
                pd.Series,
            ):
                deduped[column] = block
            else:
                deduped[column] = (
                    block
                    .bfill(axis=1)
                    .iloc[:, 0]
                )

        x = deduped

    if "date" not in x.columns:
        return pd.DataFrame()

    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce",
    ).dt.normalize()

    for col in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "delivery_qty",
        "delivery_pct",
    ):
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


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
@st.cache_resource(ttl=3600)
def _drive_store():
    from persistent_store import GoogleDriveStore
    return GoogleDriveStore()


@st.cache_data(ttl=3600, show_spinner=False)
def _compact_symbol_history(symbol: str) -> pd.DataFrame:
    from delivery_history_cache import _load_compact

    root = _root(symbol)

    frame = _load_compact(
        _drive_store(),
        root,
        pd.Timestamp("2099-12-31").date(),
    )

    if frame is None or frame.empty:
        return pd.DataFrame()

    return _clean_ohlc(frame)


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def _bulk_eod_drive_index() -> dict[str, str]:
    """Index official NSE BhavCopy parquet files in Drive by filename.

    The production archive is the dated:
      production_data/nse_official/bulk_eod
    collection. We build the filename→file-id index once per Streamlit
    runtime instead of traversing year/month/day folders for every request.
    """
    from persistent_store import GoogleDriveStore

    store = _drive_store()

    candidates = []

    # First use the known production archive path.
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
            parent = None
            break

        parent = folder["id"]

    if parent:
        token = None
        while True:
            kwargs = {
                "q": (
                    f"'{parent}' in parents "
                    "and trashed = false "
                    "and name contains 'BhavCopy_NSE_CM_0_0_0_'"
                ),
                "spaces": "drive",
                "fields": "nextPageToken,files(id,name,mimeType)",
                "pageSize": 1000,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }

            if token:
                kwargs["pageToken"] = token

            response = store.service.files().list(
                **kwargs
            ).execute()

            candidates.extend(
                response.get(
                    "files",
                    [],
                )
            )

            token = response.get(
                "nextPageToken"
            )

            if not token:
                break

    # Fallback: global filename search if the configured folder layout differs.
    if not candidates:
        token = None

        while True:
            kwargs = {
                "q": (
                    "trashed = false "
                    "and name contains 'BhavCopy_NSE_CM_0_0_0_'"
                ),
                "spaces": "drive",
                "fields": "nextPageToken,files(id,name,mimeType)",
                "pageSize": 1000,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
                "corpora": "allDrives",
            }

            if token:
                kwargs["pageToken"] = token

            response = store.service.files().list(
                **kwargs
            ).execute()

            candidates.extend(
                response.get(
                    "files",
                    [],
                )
            )

            token = response.get(
                "nextPageToken"
            )

            if not token:
                break

    index = {}

    for item in candidates:

        if item.get(
            "mimeType"
        ) == "application/vnd.google-apps.folder":
            continue

        name = str(
            item.get(
                "name",
                "",
            )
        ).strip()

        file_id = str(
            item.get(
                "id",
                "",
            )
        ).strip()

        if (
            name
            and file_id
            and name.endswith(".parquet")
        ):
            index[name] = file_id

    if not index:
        raise RuntimeError(
            "No official NSE BhavCopy parquet files were found in the "
            "configured Google Drive archive."
        )

    return index


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def _historical_symbol_bulk_history(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Read only the requested symbol from official NSE BhavCopy files.

    The Drive file index is created once. Only business-day BhavCopy files
    whose dates fall inside the requested range are downloaded/read.
    """
    from googleapiclient.http import MediaIoBaseDownload

    root = _root(symbol)

    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()

    index = _bulk_eod_drive_index()

    # Select relevant dated BhavCopy files without walking Drive folders.
    selected = []

    for name, file_id in index.items():

        match = re.search(
            r"BhavCopy_NSE_CM_0_0_0_(20\d{6})",
            name,
        )

        if not match:
            continue

        try:
            day = pd.Timestamp(
                match.group(1)
            ).normalize()
        except Exception:
            continue

        if (
            start_day
            <= day
            <= end_day
        ):
            selected.append(
                (
                    day,
                    name,
                    file_id,
                )
            )

    selected.sort(
        key=lambda x: x[0]
    )

    if not selected:
        return pd.DataFrame()

    store = _drive_store()

    cache_dir = (
        Path(".historical_bhavcopy_cache")
        / root
    )
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pieces = []

    for day, name, file_id in selected:

        local = (
            cache_dir
            / name
        )

        try:

            if (
                not local.exists()
                or local.stat().st_size == 0
            ):

                request = (
                    store.service
                    .files()
                    .get_media(
                        fileId=file_id
                    )
                )

                temp = local.with_suffix(
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

                temp.replace(local)

            raw = pd.read_parquet(
                local
            )

            if raw is None or raw.empty:
                continue

            raw.columns = [
                str(col).strip().lower()
                for col in raw.columns
            ]

            if raw.columns.duplicated().any():
                raw = raw.loc[
                    :,
                    ~raw.columns.duplicated(
                        keep="first"
                    ),
                ].copy()

            if "symbol" not in raw.columns:
                continue

            symbols = (
                raw["symbol"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            hit = raw[
                symbols == root
            ].copy()

            if hit.empty:
                continue

            aliases = {
                "date": (
                    "date",
                    "date1",
                    "traddt",
                    "trading_date",
                ),
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
                        col
                        for col in names
                        if col in hit.columns
                    ),
                    None,
                )

                if source is None:
                    hit[target] = np.nan

                elif source != target:
                    hit[target] = hit[source]

            hit["date"] = pd.to_datetime(
                hit["date"],
                errors="coerce",
            ).dt.normalize()

            for col in (
                "open",
                "high",
                "low",
                "close",
                "volume",
                "delivery_qty",
                "delivery_pct",
            ):
                hit[col] = pd.to_numeric(
                    hit[col],
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

        except Exception:
            continue

    if not pieces:
        return pd.DataFrame()

    return (
        pd.concat(
            pieces,
            ignore_index=True,
        )
        .sort_values("date")
        .drop_duplicates(
            ["symbol", "date"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def _stock_history_from_drive(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return _historical_symbol_bulk_history(
        symbol,
        start,
        end,
    )


def _stock_history_from_drive(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Return a filtered view of cached compact official NSE history."""
    frame = _compact_symbol_history(
        _root(symbol)
    )

    if frame.empty:
        return pd.DataFrame()

    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()

    return frame[
        (frame["date"] >= start_day)
        & (frame["date"] <= end_day)
    ].copy()

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

    # The model's longest rolling windows are far shorter than one year.
    # ~130 calendar days provides ample prior NSE sessions while keeping replay fast.
    history_start = (
        input_end
        - pd.Timedelta(days=130)
    )
    # Load enough history through the day before the target session.
    stock_input = _historical_stock_history_from_drive(
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
    # Market context only needs enough sessions for 12-session returns
    # and the 20-session stress statistic.
    benchmark_start = (
        reference_date
        - pd.Timedelta(days=130)
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

    # Read the actual target row from the same cached compact history.
    # The target row is not passed into the feature builder.
    all_history = _historical_symbol_bulk_history(
        root,
        target_date,
        target_date,
    )

    actual_rows = all_history[
        all_history["date"] == target_date
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

    st.markdown("### Result")

    def _money(value):
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
            return f"₹{float(value):,.2f}"
        return "—"

    def _pct(value):
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
            return f"{float(value):+.2%}"
        return "—"

    for _, row in out.iterrows():

        st.markdown(
            f"## {row.get('symbol', '')}"
        )

        if row.get("status") != "OK":
            st.error(
                str(
                    row.get(
                        "status",
                        "Prediction failed.",
                    )
                )
            )
            continue

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Target session",
            str(
                row.get(
                    "prediction_date",
                    "—",
                )
            ),
        )

        c2.metric(
            "Model data through",
            str(
                row.get(
                    "input_through_date",
                    "—",
                )
            ),
        )

        c3.metric(
            "Signal",
            str(
                row.get(
                    "signal",
                    "—",
                )
            ),
        )

        c4.metric(
            "P(up)",
            (
                f"{float(row['p_up']):.1%}"
                if "p_up" in row
                and np.isfinite(
                    float(row["p_up"])
                )
                else "—"
            ),
        )

        st.markdown(
            "### Model prediction"
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Expected move",
            _pct(
                row.get(
                    "expected_return"
                )
            ),
        )

        m2.metric(
            "Data readiness",
            (
                f"{float(row['data_readiness']):.0%}"
                if "data_readiness" in row
                and np.isfinite(
                    float(row["data_readiness"])
                )
                else "—"
            ),
        )

        m3.metric(
            "Model",
            "ALL-NSE 59-feature",
        )

        st.markdown(
            "### Indicative price levels"
        )

        st.caption(
            "Projected prices are derived from expected return and ATR; "
            "they are not separate OHLC model targets."
        )

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "Reference close",
            _money(
                row.get(
                    "reference_close"
                )
            ),
        )

        p2.metric(
            "Projected open",
            _money(
                row.get(
                    "predicted_open"
                )
            ),
        )

        p3.metric(
            "Projected high",
            _money(
                row.get(
                    "predicted_high"
                )
            ),
        )

        p4.metric(
            "Projected low",
            _money(
                row.get(
                    "predicted_low"
                )
            ),
        )

        p5, p6, p7, p8 = st.columns(4)

        p5.metric(
            "Projected close",
            _money(
                row.get(
                    "predicted_close"
                )
            ),
        )

        p6.metric(
            "Indicative purchase",
            _money(
                row.get(
                    "purchase_price"
                )
            ),
        )

        p7.metric(
            "Indicative sale target",
            _money(
                row.get(
                    "sale_price"
                )
            ),
        )

        p8.metric(
            "Risk stop reference",
            _money(
                row.get(
                    "stop_price"
                )
            ),
        )

        st.markdown(
            "### Actual target-session result"
        )

        a1, a2, a3, a4 = st.columns(4)

        a1.metric(
            "Actual open",
            _money(
                row.get(
                    "actual_open"
                )
            ),
        )

        a2.metric(
            "Actual high",
            _money(
                row.get(
                    "actual_high"
                )
            ),
        )

        a3.metric(
            "Actual low",
            _money(
                row.get(
                    "actual_low"
                )
            ),
        )

        a4.metric(
            "Actual close",
            _money(
                row.get(
                    "actual_close"
                )
            ),
        )

        e1, e2, e3 = st.columns(3)

        e1.metric(
            "Actual return",
            _pct(
                row.get(
                    "actual_return"
                )
            ),
        )

        e2.metric(
            "Predicted vs actual close",
            _pct(
                row.get(
                    "predicted_vs_actual_close_pct"
                )
            ),
        )

        e3.metric(
            "Direction match",
            "YES"
            if bool(
                row.get(
                    "direction_match"
                )
            )
            else "NO",
        )

        st.caption(
            f"Direction: "
            f"{row.get('predicted_direction', '—')} → "
            f"{row.get('actual_direction', '—')} | "
            f"Benchmark: "
            f"{row.get('benchmark_source', '—')}"
        )

        with st.expander(
            "Raw diagnostic data"
        ):

            raw = row.to_dict()

            for key in (
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

                if key in raw:
                    raw[key] = _money(
                        raw[key]
                    )

            for key in (
                "p_up",
                "expected_return",
                "actual_return",
                "predicted_vs_actual_close_pct",
                "data_readiness",
            ):

                if key in raw:
                    raw[key] = _pct(
                        raw[key]
                    )

            st.json(raw)

    st.caption(
        "Historical replay is retrospective analysis. The selected target "
        "session is excluded from model inputs. Price-path figures are "
        "derived estimates rather than independently trained OHLC predictions."
    )

