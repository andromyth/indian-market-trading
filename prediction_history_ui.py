"""Streamlit UI for delivery prediction history and price-path presentation.

The active ALL-NSE Delivery candidate directly predicts next-session direction
probability and return. OHLC/entry/exit figures shown here are explicitly
DERIVED estimates from the reference OHLC, the model expected return, and a
14-session ATR; they are not separately trained OHLC targets.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
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


def _load_persisted_market_context(
    benchmark: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    """Load previously persisted official NSE market context from Drive.

    The research build already stores NIFTY context artifacts. Historical
    replay should consume those artifacts rather than call a hosted market
    endpoint that may return HTML/403 or otherwise fail.
    """
    benchmark = str(benchmark).strip().upper()

    try:
        from persistent_store import GoogleDriveStore
        from googleapiclient.http import MediaIoBaseDownload

        store = GoogleDriveStore()

        # Candidate persisted artifacts created by the research pipeline.
        artifact_names = [
            (
                "EXPANSION_NIFTY_CONTEXT_2026.parquet",
                "processed/universe_expansion_2026_v4",
            ),
            (
                "STEP45_NIFTY_MARKET_CONTEXT_2026.parquet",
                "processed/universe_expansion_2026_v4",
            ),
            (
                "NIFTY50-INDEX.parquet",
                "raw/2026",
            ),
        ]

        for filename, folder_path in artifact_names:

            parent_id = store.root_id

            for part in [
                p
                for p in folder_path.split("/")
                if p
            ]:

                folder = store._find_child(
                    parent_id,
                    part,
                    "application/vnd.google-apps.folder",
                )

                if not folder:
                    parent_id = None
                    break

                parent_id = folder["id"]

            if not parent_id:
                continue

            file_item = store._find_child(
                parent_id,
                filename,
                None,
            )

            if not file_item:
                continue

            local_dir = Path(
                ".market_context_cache"
            )
            local_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            local_path = (
                local_dir / filename
            )

            if (
                not local_path.exists()
                or local_path.stat().st_size == 0
            ):

                request = (
                    store.service
                    .files()
                    .get_media(
                        fileId=file_item["id"]
                    )
                )

                temp = local_path.with_suffix(
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
                    local_path
                )

            frame = pd.read_parquet(
                local_path
            )

            if frame.empty:
                continue

            x = frame.copy()

            x.columns = [
                str(col).strip().lower()
                for col in x.columns
            ]

            # Handle the two research-artifact schemas:
            # 1) timestamp + market_* fields
            # 2) date + OHLC index values.
            if "timestamp" in x.columns:
                x["date"] = pd.to_datetime(
                    x["timestamp"],
                    errors="coerce",
                ).dt.normalize()

            elif "date" in x.columns:
                x["date"] = pd.to_datetime(
                    x["date"],
                    errors="coerce",
                ).dt.normalize()

            else:
                continue

            if "market_ret_1" not in x.columns:
                close_col = (
                    "nifty_close"
                    if "nifty_close" in x.columns
                    else "close"
                    if "close" in x.columns
                    else None
                )

                if close_col is None:
                    continue

                cclose = pd.to_numeric(
                    x[close_col],
                    errors="coerce",
                )

                x["market_ret_1"] = (
                    cclose.pct_change(1)
                )
                x["market_ret_3"] = (
                    cclose.pct_change(3)
                )
                x["market_ret_6"] = (
                    cclose.pct_change(6)
                )
                x["market_ret_12"] = (
                    cclose.pct_change(12)
                )

                delta = cclose.diff()

                gain = (
                    delta.clip(lower=0)
                    .rolling(
                        14,
                        min_periods=14,
                    )
                    .mean()
                )

                loss = (
                    (-delta.clip(upper=0))
                    .rolling(
                        14,
                        min_periods=14,
                    )
                    .mean()
                )

                rs = gain / loss

                x["market_rsi14"] = (
                    100
                    - 100 / (1 + rs)
                )

                x["market_rsi14"] = x[
                    "market_rsi14"
                ].where(
                    loss != 0,
                    100.0,
                )

            for col in (
                "market_ret_1",
                "market_ret_3",
                "market_ret_6",
                "market_ret_12",
                "market_rsi14",
            ):
                if col not in x.columns:
                    x[col] = np.nan
                x[col] = pd.to_numeric(
                    x[col],
                    errors="coerce",
                )

            x["market_mood_proxy"] = (
                x["market_ret_1"]
            )
            x["market_stress_proxy"] = (
                x["market_ret_1"]
                .rolling(
                    20,
                    min_periods=5,
                )
                .std()
            )

            x = x[
                (x["date"] >= start.normalize())
                & (x["date"] <= end.normalize())
            ].copy()

            if x.empty:
                continue

            return (
                x[
                    [
                        "date",
                        "market_ret_1",
                        "market_ret_3",
                        "market_ret_6",
                        "market_ret_12",
                        "market_rsi14",
                        "market_mood_proxy",
                        "market_stress_proxy",
                    ]
                ]
                .sort_values("date")
                .drop_duplicates(
                    "date",
                    keep="last",
                )
                .reset_index(drop=True),
                "PERSISTED_OFFICIAL_NSE_CONTEXT",
            )

    except Exception:
        pass

    return pd.DataFrame(), ""


def _replay_benchmark_history(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:

    """Load benchmark context for retrospective replay.

    Persisted official NSE research context is preferred. No fabricated index
    values are generated. If that artifact is unavailable, the replay is
    explicitly blocked rather than silently substituting an online provider.
    """
    benchmark = (
        "^NSEBANK"
        if _root(symbol)
        in {
            "HDFCBANK",
            "ICICIBANK",
            "SBIN",
            "AXISBANK",
            "KOTAKBANK",
            "INDUSINDBK",
            "BANKBARODA",
            "PNB",
        }
        else "^NSEI"
    )

    frame, source = _load_persisted_market_context(
        benchmark,
        start,
        end,
    )

    if frame.empty:
        raise RuntimeError(
            "Persisted official NSE market-context history is unavailable "
            f"for retrospective replay of {benchmark}."
        )

    return frame, source


def _drive_nse_bulk_history(
    symbol: str,
    end_date: pd.Timestamp,
    min_sessions: int = 70,
) -> pd.DataFrame:
    """Read point-in-time stock history directly from the official NSE bulk-EOD archive.

    This deliberately bypasses the live security-wise NSE endpoint. The archive
    is the persistent exchange-data layer already used to build the ALL-NSE
    candidate.
    """
    root = _root(symbol)
    end_day = pd.Timestamp(end_date).normalize()

    try:
        from persistent_store import GoogleDriveStore
        from googleapiclient.http import MediaIoBaseDownload

        store = GoogleDriveStore()

        production = store._find_child(
            store.root_id,
            "production_data",
            "application/vnd.google-apps.folder",
        )
        if not production:
            raise RuntimeError(
                "Google Drive production_data folder not found."
            )

        nse = store._find_child(
            production["id"],
            "nse_official",
            "application/vnd.google-apps.folder",
        )
        if not nse:
            raise RuntimeError(
                "Google Drive nse_official folder not found."
            )

        bulk = store._find_child(
            nse["id"],
            "bulk_eod",
            "application/vnd.google-apps.folder",
        )
        if not bulk:
            raise RuntimeError(
                "Google Drive bulk_eod folder not found."
            )

        # The current official archive spans 2026. Discover month/day folders
        # once per month, then read only the dated parquet files needed.
        years = {}
        page_token = None

        while True:

            kwargs = {
                "q": (
                    f"'{bulk['id']}' in parents "
                    "and mimeType = "
                    "'application/vnd.google-apps.folder' "
                    "and trashed = false"
                ),
                "spaces": "drive",
                "fields": "nextPageToken,files(id,name)",
                "pageSize": 100,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }

            if page_token:
                kwargs["pageToken"] = page_token

            response = store.service.files().list(
                **kwargs
            ).execute()

            for item in response.get("files", []):
                years[str(item["name"])] = item["id"]

            page_token = response.get(
                "nextPageToken"
            )

            if not page_token:
                break

        available_days = []

        local_dir = Path(
            ".historical_bulk_cache"
        ) / root
        local_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Scan all archive dates that can precede the requested date.
        for year_name, year_id in years.items():

            try:
                year_num = int(year_name)
            except Exception:
                continue

            if year_num > end_day.year:
                continue

            month_page = None
            while True:

                kwargs = {
                    "q": (
                        f"'{year_id}' in parents "
                        "and mimeType = "
                        "'application/vnd.google-apps.folder' "
                        "and trashed = false"
                    ),
                    "spaces": "drive",
                    "fields": "nextPageToken,files(id,name)",
                    "pageSize": 100,
                    "includeItemsFromAllDrives": True,
                    "supportsAllDrives": True,
                }

                if month_page:
                    kwargs["pageToken"] = month_page

                response = store.service.files().list(
                    **kwargs
                ).execute()

                for month_item in response.get("files", []):

                    try:
                        month_num = int(
                            str(
                                month_item["name"]
                            )
                        )
                    except Exception:
                        continue

                    if year_num == end_day.year and month_num > end_day.month:
                        continue

                    day_page = None

                    while True:

                        day_kwargs = {
                            "q": (
                                f"'{month_item['id']}' in parents "
                                "and mimeType = "
                                "'application/vnd.google-apps.folder' "
                                "and trashed = false"
                            ),
                            "spaces": "drive",
                            "fields": "nextPageToken,files(id,name)",
                            "pageSize": 100,
                            "includeItemsFromAllDrives": True,
                            "supportsAllDrives": True,
                        }

                        if day_page:
                            day_kwargs[
                                "pageToken"
                            ] = day_page

                        day_response = store.service.files().list(
                            **day_kwargs
                        ).execute()

                        for day_item in day_response.get("files", []):

                            try:
                                day_num = int(
                                    str(
                                        day_item["name"]
                                    )
                                )
                            except Exception:
                                continue

                            try:
                                day = pd.Timestamp(
                                    year_num,
                                    month_num,
                                    day_num,
                                ).normalize()
                            except Exception:
                                continue

                            if day > end_day:
                                continue

                            available_days.append(
                                (
                                    day,
                                    day_item["id"],
                                )
                            )

                        day_page = day_response.get(
                            "nextPageToken"
                        )

                        if not day_page:
                            break

                month_page = response.get(
                    "nextPageToken"
                )

                if not month_page:
                    break

        available_days = sorted(
            set(available_days),
            key=lambda x: x[0],
        )

        if not available_days:
            raise RuntimeError(
                f"No official NSE bulk-EOD sessions found through {end_day.date()}."
            )

        # Start from the latest date and accumulate until we have sufficient
        # clean sessions for the 48/50-session features.
        selected_days = list(
            reversed(available_days)
        )[: max(min_sessions + 20, 90)]

        parts = []

        for day, folder_id in selected_days:

            if day > end_day:
                continue

            files_page = None
            files = []

            while True:

                kwargs = {
                    "q": (
                        f"'{folder_id}' in parents "
                        "and name contains '.parquet' "
                        "and trashed = false"
                    ),
                    "spaces": "drive",
                    "fields": "nextPageToken,files(id,name)",
                    "pageSize": 100,
                    "includeItemsFromAllDrives": True,
                    "supportsAllDrives": True,
                }

                if files_page:
                    kwargs["pageToken"] = files_page

                response = store.service.files().list(
                    **kwargs
                ).execute()

                files.extend(
                    response.get(
                        "files",
                        [],
                    )
                )

                files_page = response.get(
                    "nextPageToken"
                )

                if not files_page:
                    break

            target = None

            for item in files:

                name = str(
                    item.get("name", "")
                )

                if (
                    "BhavCopy_NSE_CM_0_0_0_"
                    in name
                    and name.endswith(".parquet")
                ):
                    target = item
                    break

            if target is None:
                continue

            cache_file = (
                local_dir
                / f"{day:%Y%m%d}.parquet"
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
                            fileId=target["id"]
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

            symbol_col = next(
                (
                    col
                    for col in (
                        "symbol",
                        "tckrsymb",
                        "ch_symbol",
                    )
                    if col in frame.columns
                ),
                None,
            )

            if symbol_col is None:
                continue

            frame[symbol_col] = (
                frame[symbol_col]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            hit = frame[
                frame[symbol_col].map(
                    _root
                ) == root
            ].copy()

            if hit.empty:
                continue

            if symbol_col != "symbol":
                hit["symbol"] = hit[symbol_col]

            date_col = next(
                (
                    col
                    for col in (
                        "date",
                        "traddt",
                        "trading_date",
                    )
                    if col in hit.columns
                ),
                None,
            )

            if date_col is None:
                hit["date"] = day
            else:
                hit["date"] = pd.to_datetime(
                    hit[date_col],
                    errors="coerce",
                ).dt.normalize()

            def first_col(*names):
                return next(
                    (
                        col
                        for col in names
                        if col in hit.columns
                    ),
                    None,
                )

            aliases = {
                "open": first_col(
                    "open",
                    "opnpric",
                    "ch_opening_price",
                ),
                "high": first_col(
                    "high",
                    "hghpric",
                    "ch_trade_high_price",
                ),
                "low": first_col(
                    "low",
                    "lwpric",
                    "ch_trade_low_price",
                ),
                "close": first_col(
                    "close",
                    "clspric",
                    "ch_closing_price",
                ),
                "volume": first_col(
                    "volume",
                    "tt ltradgvol".replace(" ", ""),
                    "ttltradgvol",
                    "ch_tot_traded_qty",
                ),
                "delivery_qty": first_col(
                    "delivery_qty",
                    "cop_deliv_qty",
                ),
                "delivery_pct": first_col(
                    "delivery_pct",
                    "cop_deliv_perc",
                ),
            }

            for target_col, source_col in aliases.items():
                if target_col not in hit.columns:

                    if source_col is not None:
                        hit[target_col] = hit[source_col]

                    else:
                        hit[target_col] = np.nan

                hit[target_col] = pd.to_numeric(
                    hit[target_col].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                )

            required = [
                "date",
                "open",
                "high",
                "low",
                "close",
            ]

            hit = hit.dropna(
                subset=required
            )

            if not hit.empty:
                parts.append(
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

            if sum(len(x) for x in parts) >= min_sessions:
                # Continue a little further only if the latest rows do not
                # contain enough pre-reference history.
                if len(
                    pd.concat(
                        parts,
                        ignore_index=True,
                    )["date"].unique()
                ) >= min_sessions:
                    break

        if not parts:
            raise RuntimeError(
                f"No official NSE bulk-EOD rows found for {root} through {end_day.date()}."
            )

        out = (
            pd.concat(
                parts,
                ignore_index=True,
            )
            .sort_values("date")
            .drop_duplicates(
                "date",
                keep="last",
            )
            .reset_index(drop=True)
        )

        out = out[
            out["date"] <= end_day
        ].copy()

        return out


def _persisted_benchmark_daily(
    symbol: str,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    """Load a persisted official benchmark artifact for historical replay."""
    root = _root(symbol)

    benchmark = (
        "^NSEBANK"
        if root
        in {
            "HDFCBANK",
            "ICICIBANK",
            "SBIN",
            "AXISBANK",
            "KOTAKBANK",
            "INDUSINDBK",
            "BANKBARODA",
            "PNB",
        }
        else "^NSEI"
    )

    try:
        from persistent_store import GoogleDriveStore
        from googleapiclient.http import MediaIoBaseDownload

        store = GoogleDriveStore()

        candidate_names = (
            [
                "NIFTYBANK-INDEX.parquet",
                "NIFTY_BANK_CONTEXT_2026.parquet",
                "EXPANSION_NIFTY_BANK_CONTEXT_2026.parquet",
            ]
            if benchmark == "^NSEBANK"
            else [
                "NIFTY50-INDEX.parquet",
                "EXPANSION_NIFTY_CONTEXT_2026.parquet",
                "STEP45_NIFTY_MARKET_CONTEXT_2026.parquet",
            ]
        )

        # Exact-name Drive search; this searches the connected Drive and avoids
        # hard-coding a folder layout for the benchmark artifact.
        for name in candidate_names:

            response = store.service.files().list(
                q=(
                    f"name = '{name}' "
                    "and trashed = false"
                ),
                spaces="drive",
                fields="files(id,name,mimeType,parents)",
                pageSize=20,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            ).execute()

            items = response.get(
                "files",
                []
            )

            for item in items:

                if item.get(
                    "mimeType"
                ) == "application/vnd.google-apps.folder":
                    continue

                local_dir = Path(
                    ".benchmark_cache"
                )
                local_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                local_path = (
                    local_dir
                    / name
                )

                if (
                    not local_path.exists()
                    or local_path.stat().st_size == 0
                ):

                    request = (
                        store.service
                        .files()
                        .get_media(
                            fileId=item["id"]
                        )
                    )

                    temp = local_path.with_suffix(
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
                        local_path
                    )

                frame = pd.read_parquet(
                    local_path
                )

                if frame.empty:
                    continue

                frame.columns = [
                    str(col).strip().lower()
                    for col in frame.columns
                ]

                if "timestamp" in frame.columns:
                    frame["date"] = pd.to_datetime(
                        frame["timestamp"],
                        errors="coerce",
                    ).dt.normalize()
                elif "date" in frame.columns:
                    frame["date"] = pd.to_datetime(
                        frame["date"],
                        errors="coerce",
                    ).dt.normalize()
                else:
                    continue

                # Research context artifacts already contain the market-return
                # fields. Preserve them when present.
                if "market_ret_1" in frame.columns:
                    for col in (
                        "market_ret_1",
                        "market_ret_3",
                        "market_ret_6",
                        "market_ret_12",
                        "market_rsi14",
                    ):
                        if col not in frame.columns:
                            frame[col] = np.nan

                    frame["market_mood_proxy"] = frame[
                        "market_ret_1"
                    ]

                    if "market_stress_proxy" not in frame.columns:
                        frame["market_stress_proxy"] = (
                            frame["market_ret_1"]
                            .rolling(
                                20,
                                min_periods=5,
                            )
                            .std()
                        )

                else:

                    close_col = (
                        "nifty_close"
                        if "nifty_close" in frame.columns
                        else "close"
                        if "close" in frame.columns
                        else None
                    )

                    if close_col is None:
                        continue

                    frame[close_col] = pd.to_numeric(
                        frame[close_col],
                        errors="coerce",
                    )

                    close = frame[close_col]

                    frame["market_ret_1"] = close.pct_change(1)
                    frame["market_ret_3"] = close.pct_change(3)
                    frame["market_ret_6"] = close.pct_change(6)
                    frame["market_ret_12"] = close.pct_change(12)

                    delta = close.diff()

                    gain = (
                        delta.clip(lower=0)
                        .rolling(
                            14,
                            min_periods=14,
                        )
                        .mean()
                    )

                    loss = (
                        (-delta.clip(upper=0))
                        .rolling(
                            14,
                            min_periods=14,
                        )
                        .mean()
                    )

                    rs = gain / loss

                    frame["market_rsi14"] = (
                        100
                        - 100 / (1 + rs)
                    )

                    frame["market_rsi14"] = frame[
                        "market_rsi14"
                    ].where(
                        loss != 0,
                        100.0,
                    )

                    frame["market_mood_proxy"] = frame[
                        "market_ret_1"
                    ]

                    frame["market_stress_proxy"] = frame[
                        "market_ret_1"
                    ].rolling(
                        20,
                        min_periods=5,
                    ).std()

                frame = (
                    frame[
                        frame["date"]
                        <= end_date.normalize()
                    ]
                    .sort_values("date")
                    .groupby(
                        "date",
                        as_index=False,
                    )
                    .tail(1)
                    .reset_index(drop=True)
                )

                if len(frame) >= 30:
                    return (
                        frame[
                            [
                                "date",
                                "market_ret_1",
                                "market_ret_3",
                                "market_ret_6",
                                "market_ret_12",
                                "market_rsi14",
                                "market_mood_proxy",
                                "market_stress_proxy",
                            ]
                        ],
                        benchmark + "_PERSISTED_OFFICIAL",
                    )

    except Exception:
        pass

    raise RuntimeError(
        f"No persisted official benchmark artifact available for {benchmark} "
        f"through {end_date.date()}."
    )


def _historical_ml_vs_actual(
    symbol: str,
    prediction_date: pd.Timestamp,
) -> dict[str, Any]:

    root = _root(symbol)

    history = _drive_nse_bulk_history(
        root,
        prediction_date,
        min_sessions=70,
    )

    if history.empty:
        return {
            "symbol": root,
            "prediction_source": "HISTORICAL_ML_REPLAY",
            "status": "HISTORICAL_NSE_DATA_UNAVAILABLE",
        }

    # The selected date is a user-facing calendar date. Use the last official
    # NSE session on/before it as the actual model reference session.
    ref_date = pd.Timestamp(
        history["date"].iloc[-1]
    ).normalize()

    benchmark_history, benchmark_source = _persisted_benchmark_daily(
        root,
        ref_date,
    )

    from all_nse_delivery_model import (
        _build_latest_features,
        _load_models,
    )

    feature_frame = _build_latest_features(
        history,
        benchmark_history,
    )

    clf, reg, manifest = _load_models()

    expected_features = list(
        manifest["features"]
    )

    if list(feature_frame.columns) != expected_features:
        raise RuntimeError(
            "Historical replay feature order does not match the locked 59-feature model."
        )

    p_up = float(
        np.clip(
            clf.predict_proba(
                feature_frame
            )[0, 1],
            0.0,
            1.0,
        )
    )

    expected = float(
        reg.predict(
            feature_frame
        )[0]
    )

    if (
        p_up >= 0.55
        and expected > 0
    ):
        signal = "BUY"
    elif (
        p_up <= 0.45
        and expected < 0
    ):
        signal = "SELL"
    else:
        signal = "HOLD"

    df = history.copy()
    future = df[
        df["date"] > ref_date
    ].sort_values("date")

    # If the first history load ended exactly at ref_date, retrieve only the
    # subsequent archive window needed for the actual comparison.
    if future.empty:
        try:
            future_history = _drive_nse_bulk_history(
                root,
                ref_date + pd.Timedelta(days=15),
                min_sessions=5,
            )
            future = future_history[
                future_history["date"] > ref_date
            ].sort_values("date")
        except Exception:
            future = pd.DataFrame()

    ref = df[
        df["date"] == ref_date
    ].iloc[-1]

    atr = _atr14(
        df[
            df["date"] <= ref_date
        ]
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
        "model_version": manifest.get(
            "engine_version"
        ),
        "benchmark_source": benchmark_source,
        "signal": signal,
        "p_up": p_up,
        "expected_return": expected,
        "reference_close": float(
            ref["close"]
        ),
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
            "Prediction available; next official NSE actual is unavailable"
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
            predicted_direction
            == actual_direction
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
        value=today,
        min_value=pd.Timestamp(
            "2026-01-01"
        ).date(),
        max_value=today,
        format="DD-MM-YYYY",
        help=(
            "Historical replay uses the official NSE bulk-EOD archive "
            "persisted in Google Drive and the matching persisted market-context "
            "artifact. No live NSE API is used for the replay."
        ),
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
        "benchmark_source",
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
