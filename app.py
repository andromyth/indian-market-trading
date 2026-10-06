"""Indian-market stock signal dashboard: delivery, intraday and F&O context.

The dashboard produces decision-support signals only. It does not place orders.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import subprocess
import sys

from data_fetcher import (
    INDEX_TICKER_MAP,
    benchmark_for_symbol,
    get_corporate_announcements,
    get_historical,
    get_news,
    get_option_chain,
    market_is_open,
    news_for_symbol,
    stock_news_keywords,
    source_diagnostics,
    get_latest_price,
    fetch_yahoo_intraday_1m,
    fetch_live_intraday_1m,
)
from signals import (
    add_indicators,
    combined_signal,
    market_regime_score,
    news_metrics,
    technical_components,
    technical_score,
)
from backtest import backtest, holdout_backtest, random_baseline
from audit import AuditConfig, DEFAULT_AUDIT_UNIVERSE, run_universe_audit
from fo_analysis import summarize as fo_summarize
from triggers import announcement_to_news, event_risk_score, tag_news_items
from data_quality import online_source_check, source_provenance_table
from research_readiness import (
    STEP52_AUDIT, STEP52_JOINED_ROWS, STEP52_LAST_TEST_DATE, STEP52_PREDICTORS,
    STEP52_SYMBOLS, news_perception,
)
from dhan_client import DhanConfig, status as dhan_status
from dhan_download import download_symbol as dhan_download_symbol
from free_data import (
    CORE_SYMBOLS, discover_repo_symbols, download_equity_universe, download_fno_archive,
    scan_coverage, scan_quality, storage_summary, local_history_provider, ensure_dirs as ensure_data_dirs,
)
from selfdriving.sd_v2_runtime import (
    beta_delivery_predict,
    beta_intraday_predict,
    beta_status,
)

from prediction_validation import validate_ledger, rolling_7day_scorecard
from model_health import assess_delivery_health
from historical_audit import run_batch_intraday_audit
from historical_replay import (
    historical_price, historical_benchmark, historical_news, historical_announcements,
    actual_outcome, replay_rule_delivery, _entry_exit_metrics,
)
from prediction_history_ui import render_delivery_forecast, render_prediction_history
from nse_data_layer import (
    fetch_equity_history as nse_fetch_equity_history,
    fetch_intraday_1m_current as nse_fetch_intraday_1m_current,
    fetch_corporate_actions as nse_fetch_corporate_actions,
    fetch_nse_drive_bulk_symbol_history,
    fetch_nse_equity_universe,
    nse_equity_symbols,
    nse_last_diagnostic,
)
from persistent_store import (
    drive_status as persistent_drive_status,
    persist_1m_frame,
    scan_coverage as persistent_scan_coverage,
    storage_summary as persistent_storage_summary,
    sync_pending as persistent_sync_pending,
    load_persistent_1m,
)

st.set_page_config(page_title="Indian Market Trading", layout="wide")

interface_mode = st.sidebar.radio(
    "Interface",
    ["Live Test", "Research / Diagnostics"],
    index=0,
    help="Live Test is the production-facing screen. Research / Diagnostics contains replay, audits and legacy investigation tools.",
)

if interface_mode == "Research / Diagnostics":
    st.title("Indian Market Trading — Research & Diagnostics")
    st.caption(
        "Development and research tools. The active production Delivery model is the ALL-NSE 59-feature path; "
        "the active Intraday model is the certified 137-feature target_30m path. No orders are placed."
    )

INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}

default_watchlist = "BHARTIARTL.NS, RELIANCE.NS, SBIN.NS"
st.sidebar.header("Watchlist")

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _searchable_nse_universe() -> pd.DataFrame:
    """Load the official NSE securities list for symbol/company search."""
    frame = fetch_nse_equity_universe(force_refresh=False)
    if frame is None or frame.empty:
        return pd.DataFrame()
    x = frame.copy()
    x.columns = [str(col).strip() for col in x.columns]
    symbol_col = next(
        (col for col in x.columns if str(col).strip().upper() in {"SYMBOL", "CH_SYMBOL"}),
        None,
    )
    if symbol_col is None:
        return pd.DataFrame()
    company_col = next(
        (
            col for col in x.columns
            if str(col).strip().upper() in {
                "NAME OF COMPANY", "COMPANY NAME", "NAME", "COMPANYNAME"
            }
        ),
        None,
    )
    series_col = next(
        (col for col in x.columns if str(col).strip().upper() in {"SERIES", "CH_SERIES"}),
        None,
    )
    out = pd.DataFrame({
        "symbol": x[symbol_col].astype(str).str.strip().str.upper(),
        "company": (
            x[company_col].astype(str).str.strip()
            if company_col is not None
            else ""
        ),
    })
    if series_col is not None:
        series = x[series_col].astype(str).str.strip().str.upper()
        # Prefer ordinary equity listings in search results while retaining
        # other current NSE securities if the source does not expose SERIES.
        out["series"] = series
        eq = out[out["series"].eq("EQ")].copy()
        if not eq.empty:
            out = eq
    out = out[
        out["symbol"].ne("")
        & ~out["symbol"].isin({"NAN", "NONE", "SYMBOL"})
    ].drop_duplicates("symbol", keep="first")
    return out.sort_values(["company", "symbol"], na_position="last").reset_index(drop=True)


if "watchlist_revision" not in st.session_state:
    st.session_state["watchlist_revision"] = 0

# Keep text entry separate from the active watchlist so typing does not trigger
# a full analysis on every keystroke. The explicit Analyze button is the
# authoritative submit action; it also fixes the previous Ctrl+Enter confusion
# where the text area was not inside a Streamlit form.
if "active_watchlist" not in st.session_state:
    st.session_state["active_watchlist"] = default_watchlist

st.sidebar.caption("Find a stock by NSE symbol or company name")
try:
    _search_df = _searchable_nse_universe()
except Exception:
    _search_df = pd.DataFrame()

if not _search_df.empty:
    search_text = st.sidebar.text_input(
        "Search NSE stock",
        key="nse_stock_search",
        placeholder="e.g. Tata Motors, Reliance, TCS",
        help="Search the official NSE securities list by symbol or company name.",
    ).strip().lower()

    search_results = _search_df
    if search_text:
        mask = (
            search_results["symbol"].str.lower().str.contains(search_text, regex=False, na=False)
            | search_results["company"].str.lower().str.contains(search_text, regex=False, na=False)
        )
        search_results = search_results.loc[mask].head(40)

    if search_results.empty:
        st.sidebar.caption("No NSE stock matched that search.")
    else:
        options = [
            f"{row.symbol} — {row.company}" if row.company and row.company.lower() != "nan"
            else row.symbol
            for row in search_results.itertuples()
        ]
        selected_search = st.sidebar.selectbox(
            "Matching stocks",
            options,
            key="nse_stock_search_result",
        )
        selected_symbol = selected_search.split(" — ", 1)[0].strip().upper()
        if st.sidebar.button("Add to watchlist", width="stretch"):
            current = []
            for item in st.session_state.get("active_watchlist", default_watchlist).split(","):
                value = item.strip().upper()
                if not value:
                    continue
                if not value.endswith((".NS", ".BO")) and not value.startswith("^"):
                    value = f"{value}.NS"
                current.append(value)
            selected_ns = f"{selected_symbol}.NS"
            if selected_ns not in current:
                current.append(selected_ns)
            new_watchlist = ", ".join(current)
            st.session_state["active_watchlist"] = new_watchlist
            st.session_state["watchlist_input"] = new_watchlist
            st.session_state["watchlist_revision"] = int(st.session_state.get("watchlist_revision", 0)) + 1
            st.rerun()
else:
    st.sidebar.caption("NSE stock search is temporarily unavailable; symbol entry remains available below.")

with st.sidebar.form("watchlist_form", clear_on_submit=False, border=False):
    watchlist_input = st.text_area(
        "NSE/BSE symbols (suffix optional)",
        st.session_state.get("active_watchlist", default_watchlist),
        key="watchlist_input",
        height=90,
        help="Example: BHARTIARTL, RELIANCE.NS, TCS.BO. Bare symbols default to NSE (.NS)."
    )
    analyze_watchlist = st.form_submit_button("Analyze", type="primary", width="stretch")

if analyze_watchlist:
    cleaned = ", ".join([part.strip() for part in watchlist_input.split(",") if part.strip()])
    if cleaned:
        st.session_state["active_watchlist"] = cleaned
        st.session_state["watchlist_revision"] = int(st.session_state.get("watchlist_revision", 0)) + 1
        st.session_state["analysis_requested"] = True

raw_symbols = st.session_state.get("active_watchlist", default_watchlist)

if interface_mode == "Live Test":
    st.sidebar.caption(f"Active analysis watchlist: {raw_symbols}")

def normalize_symbol(value: str) -> str:
    """Normalize user input for Yahoo Finance. Bare Indian symbols default to NSE."""
    s = value.strip().upper()
    if not s:
        return s
    if s.startswith("^"):
        return s
    if s.endswith((".NS", ".BO")):
        return s
    # The dashboard is Indian-market focused; bare symbols are treated as NSE.
    return f"{s}.NS"

symbols = [normalize_symbol(s) for s in raw_symbols.split(",") if s.strip()]


@st.cache_data(ttl=21600, show_spinner=False)
def _delivery_opening_stats(
    symbol: str,
    reference_date: str,
    reference_close: float,
    predicted_close: float = np.nan,
) -> dict:
    """Statistical next-session OHLC projection for display; never modifies the ML model."""
    try:
        end = pd.Timestamp(reference_date).date()
        start = end - pd.Timedelta(days=450)
        # Use the persistent official NSE CM-UDiFF archive first.
        # This keeps the OHLC projection fast and avoids a second burst of
        # hosted NSE HTTP requests.
        hist = fetch_nse_drive_bulk_symbol_history(
            symbol,
            max(start, end - pd.Timedelta(days=180)),
            end,
            max_calendar_days=180,
        )
        if hist is None or hist.empty:
            hist = get_historical(
                f"{symbol}.NS",
                period="450d",
                interval="1d",
                force_refresh=False,
            )
        if hist is None or hist.empty:
            hist = nse_fetch_equity_history(
                symbol,
                start,
                end,
                sync_drive=False,
            )
        if hist is None or hist.empty:
            return {"status": "NO_DATA"}

        x = hist.copy()
        if "date" not in x.columns:
            x["date"] = pd.to_datetime(x.index, errors="coerce")
        x.columns = [str(col).strip().lower() for col in x.columns]
        required = {"date", "open", "high", "low", "close"}
        if not required.issubset(x.columns):
            return {"status": "NO_DATA"}
        x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
        for col in ("open", "high", "low", "close"):
            x[col] = pd.to_numeric(x[col], errors="coerce")
        x = (
            x.dropna(subset=["date", "open", "high", "low", "close"])
             .sort_values("date")
             .drop_duplicates("date", keep="last")
        )
        if len(x) < 20:
            return {"status": "INSUFFICIENT_HISTORY"}

        x["prev_close"] = x["close"].shift(1)
        x["gap"] = x["open"] / x["prev_close"] - 1.0
        ref_ts = pd.Timestamp(reference_date)
        recent = x.loc[x["date"] < ref_ts].copy().tail(60)
        gaps = recent["gap"].dropna()
        if gaps.empty or not np.isfinite(reference_close):
            return {"status": "NO_DATA"}

        median_gap = float(gaps.median())
        mean_gap = float(gaps.mean())
        gap_std = float(gaps.std(ddof=1)) if len(gaps) >= 2 else np.nan
        expected_open = reference_close * (1.0 + median_gap)
        open_low = (
            reference_close * (1.0 + median_gap - gap_std)
            if np.isfinite(gap_std) else np.nan
        )
        open_high = (
            reference_close * (1.0 + median_gap + gap_std)
            if np.isfinite(gap_std) else np.nan
        )

        upper_base = recent[["open", "close"]].max(axis=1)
        lower_base = recent[["open", "close"]].min(axis=1)
        up_excursion = (
            recent["high"] / upper_base - 1.0
        ).replace([np.inf, -np.inf], np.nan).dropna()
        down_excursion = (
            1.0 - recent["low"] / lower_base
        ).replace([np.inf, -np.inf], np.nan).dropna()

        median_up_excursion = (
            float(up_excursion.median()) if not up_excursion.empty else np.nan
        )
        median_down_excursion = (
            float(down_excursion.median()) if not down_excursion.empty else np.nan
        )

        pc_num = pd.to_numeric(predicted_close, errors="coerce")
        pc = float(predicted_close) if np.isfinite(pc_num) else np.nan
        anchor_high = max(expected_open, pc) if np.isfinite(pc) else expected_open
        anchor_low = min(expected_open, pc) if np.isfinite(pc) else expected_open

        expected_high = (
            anchor_high * (1.0 + median_up_excursion)
            if np.isfinite(median_up_excursion) else np.nan
        )
        expected_low = (
            anchor_low * (1.0 - median_down_excursion)
            if np.isfinite(median_down_excursion) else np.nan
        )

        if np.isfinite(expected_high):
            expected_high = max(
                expected_high,
                expected_open,
                pc if np.isfinite(pc) else expected_open,
            )
        if np.isfinite(expected_low):
            expected_low = min(
                expected_low,
                expected_open,
                pc if np.isfinite(pc) else expected_open,
            )

        return {
            "status": "PASS",
            "samples": int(len(recent)),
            "median_gap": median_gap,
            "mean_gap": mean_gap,
            "gap_std": gap_std,
            "expected_open": float(expected_open),
            "open_low": float(open_low) if np.isfinite(open_low) else np.nan,
            "open_high": float(open_high) if np.isfinite(open_high) else np.nan,
            "expected_high": float(expected_high) if np.isfinite(expected_high) else np.nan,
            "expected_low": float(expected_low) if np.isfinite(expected_low) else np.nan,
            "method": "60-session median gap + empirical daily high/low excursion around model close",
        }
    except Exception as exc:
        return {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}


@st.cache_data(ttl=30, show_spinner=False)
def _cached_nse_ltp(symbol: str):
    """Fresh official NSE quote snapshot for display only; never feeds the locked Delivery model."""
    try:
        return get_latest_price(symbol)
    except Exception:
        return None


def _intraday_price_envelope(price_df: pd.DataFrame, decision_ts, minutes: int = 30) -> pd.DataFrame:
    """Create a numeric realized-volatility envelope, not an invented ML price path."""
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    x = price_df.copy()
    x.columns = [str(col).strip().lower() for col in x.columns]
    aliases = {"ltp": "close", "adj close": "close", "adj_close": "close", "vol": "volume"}
    x = x.rename(columns=aliases)
    for col in ("high", "low", "close"):
        if col not in x.columns:
            return pd.DataFrame()
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["high", "low", "close"])
    if x.empty:
        return pd.DataFrame()

    prev_close = x["close"].shift(1)
    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_1m = float(
        tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean().iloc[-1]
    ) if len(tr) >= 14 else np.nan
    last_price = float(x["close"].iloc[-1])

    if not np.isfinite(atr_1m) or atr_1m <= 0 or not np.isfinite(last_price):
        return pd.DataFrame()

    ts = pd.Timestamp(decision_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Kolkata")
    else:
        ts = ts.tz_convert("Asia/Kolkata")

    rows = []
    for minute in range(1, minutes + 1):
        distance = atr_1m * np.sqrt(minute)
        rows.append({
            "minute_ahead": minute,
            "time_IST": (ts + pd.Timedelta(minutes=minute)).strftime("%H:%M"),
            "current_price_reference": last_price,
            "lower_price": max(0.0, last_price - distance),
            "upper_price": last_price + distance,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=45, show_spinner=False)
def _cached_live_news() -> list[dict]:
    try:
        return tag_news_items(get_news())
    except Exception:
        return []



@st.cache_data(ttl=60, show_spinner=False)
def _cached_live_benchmark_summary(benchmark_symbol: str) -> dict:
    """Display-only live benchmark context; it does not alter the locked model."""
    try:
        frame = _cached_live_market_context(benchmark_symbol)
        if frame is None or frame.empty:
            return {"status": "NO_DATA"}
        x = frame.copy()
        x.columns = [str(col).strip().lower() for col in x.columns]
        if "close" not in x.columns:
            return {"status": "NO_DATA"}
        x["close"] = pd.to_numeric(x["close"], errors="coerce")
        x = x.dropna(subset=["close"])
        if x.empty:
            return {"status": "NO_DATA"}
        last = float(x["close"].iloc[-1])
        baseline = float(x["close"].iloc[max(0, len(x) - 31)])
        return {
            "status": "PASS",
            "last": last,
            "change_30m": last / baseline - 1.0 if baseline else np.nan,
            "benchmark": benchmark_symbol,
        }
    except Exception:
        return {"status": "ERROR"}

@st.cache_data(ttl=30, show_spinner=False)
def _cached_live_market_context(benchmark_symbol: str) -> pd.DataFrame:
    try:
        frame = get_historical(benchmark_symbol, period="5d", interval="1m")
        return frame if frame is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()



@st.cache_data(ttl=20, show_spinner=False)
def _live_5m_price_frame(symbol: str) -> tuple[pd.DataFrame, str]:
    """Get live 1-minute bars with an ordered, fail-fast provider chain."""
    root = str(symbol).replace(".NS", "").replace(".BO", "").upper()

    try:
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        cached = load_persistent_1m(
            root,
            start=(now - pd.Timedelta(days=3)),
            end=now,
        )
        if cached is not None and not cached.empty:
            latest_cached = pd.Timestamp(cached.index.max())
            if latest_cached.tzinfo is None:
                latest_cached = latest_cached.tz_localize("Asia/Kolkata")
            else:
                latest_cached = latest_cached.tz_convert("Asia/Kolkata")
            # Only use the local cache as a live source when it is today's data
            # and is recent enough to represent the current session.
            age_minutes = (now - latest_cached).total_seconds() / 60.0
            if latest_cached.date() == now.date() and age_minutes <= 5.0:
                return cached, "PERSISTENT_LOCAL_CACHE"
    except Exception:
        pass

    try:
        frame = nse_fetch_intraday_1m_current(
            root,
            sync_drive=False,
            retries=1,
        )
        if frame is not None and not frame.empty:
            return frame, "NSE_OFFICIAL_CHART"
    except Exception:
        pass

    try:
        fallback, source = fetch_live_intraday_1m(root, days=3)
        if fallback is not None and not fallback.empty:
            return fallback, source
    except Exception:
        pass

    return pd.DataFrame(), "NO_LIVE_1M_DATA"
@st.cache_data(ttl=30, show_spinner=False)
def _cached_live_announcements(symbol: str) -> list[dict]:
    try:
        return get_corporate_announcements(symbol) or []
    except Exception:
        return []


@st.cache_data(ttl=20, show_spinner=False)
def _cached_live_option_chain(symbol: str):
    try:
        raw = get_option_chain(symbol, is_index=False)
        return fo_summarize(raw) if raw else {}
    except Exception:
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def _cached_beta_status() -> dict:
    """Return the current Beta V2 training/manifest state for UI diagnostics."""
    return beta_status()


if interface_mode == "Live Test":
    st.sidebar.subheader("Live mode")
    live_mode = st.sidebar.radio(
        "Prediction type",
        ["Delivery", "Intraday 5-minute"],
        index=0,
        key="live_test_mode",
    )
    refresh_live = st.sidebar.button(
        "Refresh live data",
        type="primary",
        width="stretch",
        help="Fetch fresh production inputs and run the selected model. No orders are placed.",
    )
    if refresh_live:
        st.cache_data.clear()
        st.session_state["live_test_revision"] = int(
            st.session_state.get("live_test_revision", 0)
        ) + 1

    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    if now_ist.hour < 9 or (now_ist.hour == 9 and now_ist.minute < 15):
        session_label = "PRE-OPEN / BEFORE REGULAR SESSION"
    elif now_ist.hour > 15 or (now_ist.hour == 15 and now_ist.minute > 30):
        session_label = "MARKET CLOSED"
    elif now_ist.hour == 15 and now_ist.minute == 30:
        session_label = "REGULAR SESSION CLOSE"
    else:
        session_label = "REGULAR SESSION OPEN"

    st.title("Indian Market — Live Test")
    st.caption(
        "Production decision-support only · no order placement · "
        "NSE is the primary exchange-data source."
    )

    s1, s2, s3 = st.columns(3)
    s1.metric("NSE session", session_label)
    s2.metric("IST time", now_ist.strftime("%d %b %Y %H:%M:%S"))
    s3.metric("Symbols selected", str(len(symbols)))

    if live_mode == "Delivery":
        beta_state = _cached_beta_status()
        delivery_state = beta_state.get("delivery", {}) or {}
        delivery_status = str(delivery_state.get("status") or "NOT_TRAINED").upper()

        if delivery_status == "PROMOTED":
            st.success("Beta V2 Delivery model is active with a validated champion.")
        elif delivery_status == "REJECTED":
            st.warning(
                "The latest Beta V2 Delivery challenger was rejected by the validation gate. "
                "An existing validated champion remains eligible when present."
            )
        else:
            st.warning(
                f"Beta V2 Delivery status: {delivery_status}. "
                "Prediction is emitted only when a validated champion is available."
            )

        gate_info = delivery_state.get("gate") or {}
        if isinstance(gate_info, dict) and gate_info.get("reasons"):
            st.caption(
                "Latest training gate: "
                + " · ".join(map(str, gate_info["reasons"]))
            )

        st.caption(
            "The Live Delivery route calls the Beta V2 runtime adapter directly. "
            "Its canonical point-in-time global/news/F&O inputs are resolved from the persisted "
            "Beta V2 data layer; the UI's live advisory feeds are not substituted into the model."
        )

        beta_delivery_results = {}
        beta_delivery_errors = {}

        for sym in symbols:
            if not sym.endswith(".NS"):
                beta_delivery_results[sym] = {
                    "prediction_status": "BLOCKED",
                    "symbol": sym,
                    "reason": "BETA_V2_DELIVERY_REQUIRES_NSE_SYMBOL",
                }
                continue

            try:
                beta_delivery_results[sym] = beta_delivery_predict(
                    sym,
                    None,
                    None,
                    None,
                )
            except Exception as exc:
                beta_delivery_errors[sym] = {
                    "symbol": sym,
                    "prediction_status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }

        table_rows = []
        for sym in symbols:
            result = beta_delivery_results.get(sym, {})
            p_up = pd.to_numeric(result.get("p"), errors="coerce")
            coverage = pd.to_numeric(
                result.get("feature_coverage"), errors="coerce"
            )
            table_rows.append({
                "Symbol": sym,
                "Status": result.get("prediction_status", "ERROR"),
                "Signal": result.get("signal", "—"),
                "P(UP)": f"{float(p_up):.1%}" if pd.notna(p_up) else "—",
                "Feature coverage": (
                    f"{float(coverage):.1%}"
                    if pd.notna(coverage) else "—"
                ),
                "Reference date": str(result.get("date") or "—"),
                "Model": str(result.get("model_file") or "—"),
                "Reason": str(
                    result.get("signal_reason")
                    or result.get("reason")
                    or result.get("error")
                    or "—"
                ),
            })

        st.subheader("Beta V2 Delivery — next-session directional signal")
        if table_rows:
            st.dataframe(
                pd.DataFrame(table_rows),
                width="stretch",
                hide_index=True,
            )

        if beta_delivery_errors:
            with st.expander("Beta V2 Delivery runtime errors", expanded=True):
                st.dataframe(
                    pd.DataFrame(beta_delivery_errors.values()),
                    width="stretch",
                    hide_index=True,
                )

        with st.expander("Beta V2 Delivery runtime status", expanded=False):
            st.json(beta_state)

        st.caption(
            "P(UP) is the calibrated probability that the stock beats the same-session "
            "cross-sectional median over the next session. BUY/SELL are emitted only inside "
            "the validated Wilson-bound action band; otherwise the model explicitly abstains."
        )

    elif live_mode == "Intraday 5-minute":
        beta_state = _cached_beta_status()
        intraday_state = beta_state.get("intraday", {}) or {}
        manifest_status = str(intraday_state.get("status") or "NOT_TRAINED").upper()

        if manifest_status == "CANDIDATE_ONLY":
            st.info(
                "Beta V2 Intraday manifest is present. Each horizon requires its own validated "
                "champion before BUY/SELL can be emitted."
            )
        else:
            st.warning(
                f"Beta V2 Intraday status: {manifest_status}. "
                "Run the Beta V2 Intraday trainer before expecting trained horizon models."
            )

        universe_df = _searchable_nse_universe()
        if universe_df is None or universe_df.empty:
            candidates = [s for s in symbols if s.endswith(".NS")]
            candidate_labels = {s: s for s in candidates}
        else:
            candidates = [
                f"{s}.NS"
                for s in universe_df["symbol"].astype(str).str.upper()
            ]
            candidate_labels = {
                f"{row.symbol}.NS": (
                    f"{row.symbol} — {row.company}"
                    if str(row.company).strip()
                    and str(row.company).lower() != "nan"
                    else str(row.symbol)
                )
                for row in universe_df.itertuples()
            }

        if not candidates:
            st.info("Official NSE equity universe is unavailable for Beta V2 Intraday selection.")
        else:
            live_default = st.session_state.get("live_intraday_selected_symbol")
            if live_default not in candidates:
                live_default = candidates[0]

            selected_symbol = st.selectbox(
                "Stock for continuous Beta V2 Intraday forecast",
                candidates,
                index=candidates.index(live_default),
                format_func=lambda s: candidate_labels.get(s, s),
                key="live_intraday_selected_symbol",
                help="Select any current NSE equity; the forecast refreshes when a new completed 1-minute bar arrives.",
            )
            auto_update = st.checkbox(
                "Continuous recalculation every new 1-minute bar",
                value=True,
                key="live_intraday_auto_update",
            )
            st.caption(
                "Only completed 1-minute bars are passed to Beta V2. "
                "The six independent horizons are +5/+10/+15/+20/+25/+30 minutes."
            )
            run_every = "60s" if auto_update else None

            @st.fragment(run_every=run_every, key="live_intraday_beta_v2_stream")
            def _render_live_beta_v2_intraday_stream():
                now = pd.Timestamp.now(tz="Asia/Kolkata")
                root = selected_symbol.replace(".NS", "").upper()
                in_session = (
                    pd.Timestamp("09:15").time()
                    <= now.time()
                    <= pd.Timestamp("15:30").time()
                )
                if not in_session:
                    st.info(
                        f"Beta V2 Intraday stream idle outside regular NSE hours · "
                        f"IST {now.strftime('%H:%M:%S')}"
                    )
                    return

                try:
                    raw, live_source = _live_5m_price_frame(selected_symbol)
                    if raw is None or raw.empty:
                        st.error(
                            "Live 1-minute market data is unavailable from NSE and the configured fallbacks."
                        )
                        return

                    cutoff = now.floor("min")
                    completed = raw.loc[raw.index < cutoff].copy()
                    if completed.empty:
                        st.info("Waiting for the first completed NSE 1-minute bar.")
                        return

                    latest_ts = completed.index[-1]
                    if len(completed) < 80:
                        st.info(
                            f"Beta V2 warm-up: {len(completed)}/80 completed 1-minute rows available."
                        )
                        return

                    benchmark_symbol = benchmark_for_symbol(selected_symbol)
                    market_df = _cached_live_market_context(benchmark_symbol)

                    state_key = f"beta_v2_intraday_last_ts::{root}"
                    result_key = f"beta_v2_intraday_result::{root}"

                    if (
                        st.session_state.get(state_key) == str(latest_ts)
                        and result_key in st.session_state
                    ):
                        predictions = st.session_state[result_key]
                        recalculated = False
                    else:
                        predictions = beta_intraday_predict(
                            selected_symbol,
                            completed,
                            market_df,
                        )
                        st.session_state[state_key] = str(latest_ts)
                        st.session_state[result_key] = predictions
                        recalculated = True

                    source_label = {
                        "NSE_OFFICIAL_CHART": "NSE official",
                        "PERSISTENT_LOCAL_CACHE": "persistent local cache",
                        "DHAN": "Dhan",
                        "YAHOO_FINANCE": "Yahoo Finance",
                        "YFINANCE": "yfinance",
                    }.get(live_source, "fallback provider")

                    if not predictions:
                        st.warning(
                            "Beta V2 returned no horizon rows. A validated horizon champion "
                            "may not be available yet."
                        )
                        return

                    table_rows = []
                    for item in predictions:
                        p_up = pd.to_numeric(
                            item.get("probability_up"), errors="coerce"
                        )
                        table_rows.append({
                            "Horizon": f"+{int(item.get('horizon_min', 0))}m",
                            "Signal": item.get("signal", "ABSTAIN"),
                            "P(UP)": (
                                f"{float(p_up):.1%}"
                                if pd.notna(p_up) else "—"
                            ),
                            "Reference price": (
                                f"₹{float(item.get('reference_close')):,.2f}"
                                if pd.notna(item.get("reference_close"))
                                else "—"
                            ),
                            "Decision": str(
                                item.get("decision_timestamp") or latest_ts
                            ),
                            "Reason": item.get(
                                "signal_reason",
                                item.get("reason", "—"),
                            ),
                        })

                    if recalculated:
                        st.success(
                            f"Beta V2 recalculated from completed 1-minute bar "
                            f"{latest_ts.strftime('%H:%M:%S')} IST · source: {source_label}"
                        )
                    else:
                        st.info(
                            f"Showing Beta V2 forecast from {latest_ts.strftime('%H:%M:%S')} IST · "
                            f"source: {source_label}"
                        )

                    h0 = predictions[0]
                    p0 = pd.to_numeric(h0.get("probability_up"), errors="coerce")
                    a, b, c = st.columns(3)
                    a.metric(
                        "Reference price ₹",
                        (
                            f"₹{float(h0.get('reference_close')):,.2f}"
                            if pd.notna(h0.get("reference_close"))
                            else "—"
                        ),
                    )
                    b.metric(
                        "Next +5m signal",
                        str(h0.get("signal", "ABSTAIN")),
                    )
                    c.metric(
                        "Next +5m P(UP)",
                        f"{float(p0):.1%}" if pd.notna(p0) else "—",
                    )

                    st.subheader("Beta V2 multi-horizon directional predictions")
                    st.dataframe(
                        pd.DataFrame(table_rows),
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "These are direct Beta V2 probabilities from the volume-free price/time "
                        "feature contract. No legacy precomputed volume features are passed."
                    )
                except Exception as exc:
                    st.error(
                        f"Beta V2 Intraday forecast failed: {type(exc).__name__}: {exc}"
                    )

            _render_live_beta_v2_intraday_stream()

    st.divider()
    st.subheader("What to watch tomorrow")
    st.write(
        "For the first live test, verify only: NSE data arrives → model is READY → "
        "signal is produced → readiness is explicit → prediction is persisted. "
        "Do not use the research/replay screens to judge live performance."
    )
    st.stop()

# These are populated later by the NSE official-universe section. Initialize
# them before the early input validation so first app render cannot raise
# NameError before the NSE universe has been fetched.
_nse_symbol_count = 0
_nse_symbol_col = None
_nse_universe = pd.DataFrame()

# Validate NSE symbols against the official securities universe when the
# universe file is available. This is an eligibility/provenance check only; it
# does not pretend that every NSE-listed security already has a validated ML model.
if _nse_symbol_count and _nse_symbol_col:
    _known_nse = set(_nse_universe[_nse_symbol_col].astype(str).str.strip().str.upper())
    _unknown_nse_inputs = [s for s in symbols if s.endswith(".NS") and s[:-3] not in _known_nse]
    if _unknown_nse_inputs:
        st.sidebar.warning("Not found in the current NSE securities file: " + ", ".join(_unknown_nse_inputs))

st.caption(f"Active watchlist: {', '.join(symbols)} · Analysis refresh {int(st.session_state.get('watchlist_revision', 0))}")

analysis_view = st.sidebar.radio(
    "View", ["Live Dashboard", "Historical Replay", "Prediction History"], index=0,
    help="Historical Replay reconstructs a past timestamp and compares the model/rule prediction with subsequent actual prices."
)

if analysis_view == "Historical Replay":
    st.markdown("## Historical Prediction Replay & Audit — Research Only")
    st.caption(
        "Pick a past date/time. The replay uses only information available up to that timestamp, then compares the prediction with what actually happened afterward. "
        "Unavailable historical news/F&O sources are shown as missing rather than replaced with current data."
    )
    with st.expander("Beta V2 runtime status", expanded=False):
        try:
            st.json(_cached_beta_status())
            st.caption(
                "Historical replay remains a research-only screen. Live prediction routing "
                "is handled exclusively by the Beta V2 runtime adapters."
            )
        except Exception as exc:
            st.error(f"Beta V2 status check failed: {type(exc).__name__}: {exc}")

    batch_audit = st.sidebar.checkbox("Batch historical audit", value=False, help="Run many historical 30-minute predictions and compare each with the subsequent actual move.")
    if batch_audit:
        st.sidebar.subheader("Batch audit settings")
        batch_start = st.sidebar.date_input("Audit start date", value=pd.Timestamp("2026-09-01").date(), min_value=pd.Timestamp("2018-01-01").date(), max_value=pd.Timestamp.now().date(), key="batch_start")
        batch_end = st.sidebar.date_input("Audit end date", value=pd.Timestamp("2026-09-25").date(), min_value=pd.Timestamp("2018-01-01").date(), max_value=pd.Timestamp.now().date(), key="batch_end")
        batch_start_time = st.sidebar.time_input("First decision time (IST)", value=pd.Timestamp("10:00").time(), step=60, key="batch_start_time")
        batch_end_time = st.sidebar.time_input("Last decision time (IST)", value=pd.Timestamp("14:00").time(), step=60, key="batch_end_time")
        batch_step = st.sidebar.selectbox("Decision interval", [30, 60], index=0, format_func=lambda x: f"Every {x} minutes", key="batch_step")
        batch_symbols = [normalize_symbol(x) for x in st.sidebar.text_area("Batch symbols", ", ".join(symbols[:10]), key="batch_symbols").split(",") if x.strip()]
        run_batch = st.sidebar.button("Run Batch Audit", type="primary", key="run_batch_audit")
        st.markdown("## Batch Historical Prediction Audit")
        st.caption("Each 30-minute prediction is reconstructed point-in-time across the selected universe. The production target is UP when the next 30-minute return is positive. Predictions with less than 70% information-group readiness are retained but excluded from the headline eligible accuracy.")
        if run_batch:
            with st.spinner("Running point-in-time batch audit…"):
                try:
                    audit_df, audit_summary = run_batch_intraday_audit(batch_symbols, batch_start, batch_end, batch_start_time, batch_end_time, batch_step)
                    st.success(f"Batch audit completed: {audit_summary['total_predictions']:,} prediction rows.")
                    q1,q2,q3,q4 = st.columns(4)
                    q1.metric("Predictions", f"{audit_summary['total_predictions']:,}")
                    q2.metric("Actuals available", f"{audit_summary['eligible_predictions']:,}")
                    q3.metric("Directional accuracy", "N/A" if audit_summary['directional_accuracy'] is None else f"{audit_summary['directional_accuracy']:.1%}")
                    q4.metric("Data-limited", f"{audit_summary['data_limited_predictions']:,}")
                    st.write({
                        "full_data_eligible_predictions": audit_summary.get("full_data_eligible_predictions", 0),
                        "full_data_directional_accuracy": audit_summary.get("full_data_directional_accuracy"),
                        "mean_brier_error": audit_summary.get("mean_brier_error"),
                        "average_actual_move_pct": audit_summary.get("average_actual_move_pct"),
                        "average_signed_move_pct": audit_summary.get("average_signed_move_pct"),
                    })
                    if audit_summary['directional_accuracy'] is not None:
                        st.write({k:v for k,v in audit_summary.items() if k not in {"symbols_requested","symbols_scored"}})
                    if not audit_df.empty:
                        st.markdown("### Prediction vs actual ledger")
                        st.dataframe(audit_df.sort_values("timestamp", ascending=False), width="stretch", hide_index=True)
                        csv = audit_df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download batch audit CSV", csv, file_name="historical_prediction_audit.csv", mime="text/csv")
                        mistakes = audit_df[(audit_df["direction_match"] == False) & (audit_df["audit_eligible"])]
                        if not mistakes.empty:
                            with st.expander("Eligible wrong-direction predictions", expanded=False):
                                st.dataframe(mistakes.sort_values("timestamp", ascending=False), width="stretch", hide_index=True)
                except Exception as exc:
                    st.error(f"Batch historical audit failed: {type(exc).__name__}: {exc}")
        st.stop()

    replay_mode = st.sidebar.radio("Replay mode", ["Delivery", "Intraday", "F&O (underlying replay)"], index=1)
    replay_symbol = normalize_symbol(st.sidebar.text_input("Replay stock", symbols[0] if symbols else "BHARTIARTL.NS"))
    replay_date = st.sidebar.date_input(
        "Historical date", value=pd.Timestamp("2026-09-25").date(),
        min_value=pd.Timestamp("2018-01-01").date(), max_value=pd.Timestamp.now().date(),
    )
    if replay_mode == "Intraday" or replay_mode.startswith("F&O"):
        replay_time = st.sidebar.time_input("Historical time (IST)", value=pd.Timestamp("10:30").time(), step=60)
        replay_horizon = st.sidebar.selectbox("Production ML horizon", [30], index=0, format_func=lambda x: f"{x} minutes")
    else:
        replay_time = pd.Timestamp("15:30").time()
        replay_horizon = st.sidebar.selectbox("Holding horizon", [1, 3, 5, 10, 20], index=2, format_func=lambda x: f"{x} trading day(s)")

    replay_run = st.sidebar.button("Run Historical Replay", type="primary")

    if replay_run:
        from datetime import time as _time
        decision_ts = pd.Timestamp.combine(pd.Timestamp(replay_date).date(), replay_time)
        if replay_mode == "Delivery":
            with st.spinner("Reconstructing the historical daily information set and checking forward actuals…"):
                # Pull several calendar years around the target date; local 1-minute archives are preferred.
                price_df = historical_price(replay_symbol, decision_ts - pd.Timedelta(days=420), decision_ts + pd.Timedelta(days=45), interval="1m")
                benchmark_symbol = benchmark_for_symbol(replay_symbol)
                try:
                    benchmark_df = get_historical(benchmark_symbol, period="10y", interval="1d")
                except Exception:
                    # Historical replay must remain usable when a benchmark
                    # provider is temporarily unavailable; missing context is
                    # surfaced in the readiness panel instead of crashing.
                    benchmark_df = pd.DataFrame()
                hist_news = historical_news(replay_symbol, decision_ts)
                hist_ann = historical_announcements(replay_symbol, decision_ts)
                ann_news = announcement_to_news(hist_ann)
                all_hist_news = tag_news_items(hist_news) + ann_news
                try:
                    delivery_parts = replay_rule_delivery(
                        replay_symbol, price_df, benchmark_df, decision_ts,
                        tech_weight=tech_weight, news_weight=news_weight, market_weight=market_weight,
                        horizon_days=int(replay_horizon),
                    )
                    hist = price_df[price_df.index <= decision_ts]
                    hist_daily = __import__("free_data").resample_daily(hist)
                    mkt_hist = benchmark_df[benchmark_df.index <= decision_ts] if not benchmark_df.empty else benchmark_df
                    t_score = delivery_parts["t_score"]
                    m_score = delivery_parts["m_score"]
                    n_score = news_metrics(all_hist_news)["score"] if all_hist_news else 0.0
                    ev_risk = event_risk_score(hist_news, hist_ann)
                    label, combined = combined_signal(t_score, n_score, m_score, ev_risk, tech_weight, news_weight, market_weight)
                    entry_info = _entry_exit_metrics(hist_daily, decision_ts, "Delivery", int(replay_horizon))
                    outcome = actual_outcome(price_df, decision_ts, label, "Delivery", int(replay_horizon), entry_info)
                    readiness_groups = {
                        "price/market": bool(not hist.empty and not mkt_hist.empty),
                        "historical news": bool(hist_news),
                        "corporate announcements": bool(hist_ann),
                        "F&O historical chain": False,
                    }
                    st.markdown(f"### {replay_symbol} — Delivery replay")
                    st.write(f"**Decision timestamp:** {decision_ts} IST")
                    a,b,c,d = st.columns(4)
                    a.metric("Prediction", label)
                    b.metric("Rule score", f"{combined:+.2f}")
                    c.metric("Decision price", f"₹{entry_info.get('entry_reference', float('nan')):,.2f}")
                    d.metric("Data readiness", f"{sum(readiness_groups.values())}/{len(readiness_groups)} groups")
                    e1,e2,e3,e4 = st.columns(4)
                    e1.metric("Buy zone", f"₹{entry_info.get('buy_zone_low', float('nan')):,.2f}–₹{entry_info.get('buy_zone_high', float('nan')):,.2f}")
                    e2.metric("Target", f"₹{entry_info.get('entry_reference', 0)+entry_info.get('target_distance', 0):,.2f}" if label == "BUY" else f"₹{entry_info.get('entry_reference', 0)-entry_info.get('target_distance', 0):,.2f}")
                    e3.metric("Stop", f"₹{entry_info.get('entry_reference', 0)-entry_info.get('stop_distance', 0):,.2f}" if label == "BUY" else f"₹{entry_info.get('entry_reference', 0)+entry_info.get('stop_distance', 0):,.2f}")
                    e4.metric("Historical news", "Available" if hist_news else "Not available")
                    st.markdown("#### Prediction vs actual")
                    if outcome.get("status") == "OK":
                        o1,o2,o3,o4 = st.columns(4)
                        o1.metric("Actual price", f"₹{outcome['actual_price']:,.2f}")
                        o2.metric("Actual move", f"{outcome['actual_move_pct']:+.2f}%")
                        o3.metric("Actual direction", outcome["actual_direction"])
                        o4.metric("Direction match", "YES" if outcome.get("correct_direction") else "NO" if outcome.get("correct_direction") is not None else "N/A")
                        st.write({
                            "target_hit": outcome.get("target_hit"), "stop_hit": outcome.get("stop_hit"),
                            "first_touch": outcome.get("first_touch"), "horizon": outcome.get("horizon_label"),
                            "actual_entry": outcome.get("actual_entry"),
                        })
                    else:
                        st.warning(outcome.get("status", "Future data unavailable."))
                    st.markdown("#### Information available at the replay timestamp")
                    st.dataframe(pd.DataFrame([{
                        "group": k, "available": v
                    } for k,v in readiness_groups.items()]), width="stretch", hide_index=True)
                    if hist_news:
                        st.markdown("#### Historical news found")
                        for n in hist_news[:15]:
                            st.write(f"**{n.get('title','')}** — {n.get('published','')} — {n.get('source','')}")
                            if n.get("link"): st.caption(n["link"])
                    else:
                        st.info("No timestamped historical news was retrieved for this date from the public RSS search source.")
                    if hist_ann:
                        st.markdown("#### Historical corporate announcements")
                        for a in hist_ann[:15]: st.write(f"{a['date']}: {a['subject']}")
                    st.caption("Research only. Note: the current Delivery replay path uses the legacy rule-based replay function; it is not the active ALL-NSE 59-feature production Delivery model. Use the persistent production ledger for live-model validation.")
                    ledger = {
                        "timestamp": str(decision_ts), "symbol": replay_symbol, "mode": "Delivery",
                        "prediction": label, "score": combined, "decision_price": entry_info.get("entry_reference"),
                        "target": (entry_info.get("entry_reference", 0)+entry_info.get("target_distance", 0)) if label == "BUY" else (entry_info.get("entry_reference", 0)-entry_info.get("target_distance", 0)),
                        "stop": (entry_info.get("entry_reference", 0)-entry_info.get("stop_distance", 0)) if label == "BUY" else (entry_info.get("entry_reference", 0)+entry_info.get("stop_distance", 0)),
                        "actual_move_pct": outcome.get("actual_move_pct"), "correct": outcome.get("correct_direction"),
                        "news_available": bool(hist_news), "announcements_available": bool(hist_ann), "fo_historical_available": False,
                    }
                    st.session_state.setdefault("replay_ledger", []).append(ledger)
                    st.download_button("Export replay record", pd.DataFrame(st.session_state["replay_ledger"]).to_csv(index=False).encode("utf-8"), file_name="historical_replay_ledger.csv", mime="text/csv")
                except Exception as exc:
                    st.error(f"Historical Delivery replay failed: {type(exc).__name__}: {exc}")

        else:
            with st.spinner("Reconstructing minute-level history and running the production 30-minute model…"):
                start = decision_ts - pd.Timedelta(days=20)
                end = decision_ts + pd.Timedelta(minutes=int(replay_horizon) + 5)
                price_df = historical_price(replay_symbol, start, end, interval="1m")
                if price_df.empty or len(price_df) < 60:
                    st.error("Historical minute data could not be loaded for this symbol/date. Download the relevant 1-minute archive in the Free Historical Data manager or choose a date covered by the public archive.")
                else:
                    benchmark_symbol = benchmark_for_symbol(replay_symbol)
                    # NIFTY 5-minute index data is optional; if unavailable the production model explicitly receives missing market context.
                    benchmark_df = historical_benchmark(benchmark_symbol, decision_ts)
                    if not benchmark_df.empty:
                        benchmark_df = benchmark_df[benchmark_df.index <= decision_ts]
                    hist_news = tag_news_items(historical_news(replay_symbol, decision_ts))
                    hist_ann = historical_announcements(replay_symbol, decision_ts)
                    ann_news = announcement_to_news(hist_ann)
                    item = [{
                        "symbol": replay_symbol, "price_df": price_df[price_df.index <= decision_ts],
                        "market_df": benchmark_df, "matched_news": hist_news, "announcements": hist_ann,
                        "event_risk": event_risk_score(hist_news, hist_ann), "fo": {}, "decision_ts": decision_ts,
                    }]
                    try:
                        st.info(
                            "Beta V2 historical Intraday replay is not connected to a "
                            "point-in-time historical inference adapter in this screen."
                        )
                        st.write({
                            "decision_timestamp": str(decision_ts),
                            "symbol": replay_symbol,
                            "status": "BETA_V2_HISTORICAL_REPLAY_NOT_CONNECTED",
                            "historical_rows_available": int(
                                len(price_df[price_df.index <= decision_ts])
                            ),
                            "market_rows_available": int(len(benchmark_df)),
                            "historical_news_available": bool(hist_news),
                            "historical_announcements_available": bool(hist_ann),
                        })
                except Exception as exc:
                        st.error(f"Historical {replay_mode} replay failed: {type(exc).__name__}: {exc}")

    if "replay_ledger" in st.session_state and st.session_state["replay_ledger"]:
        with st.expander("Replay ledger from this session", expanded=False):
            st.dataframe(pd.DataFrame(st.session_state["replay_ledger"]), width="stretch", hide_index=True)
    st.stop()

if analysis_view == "Prediction History":
    render_prediction_history()
    st.stop()

style = st.sidebar.radio("Mode", ["Delivery", "Intraday"], index=0)
if style == "Delivery":
    # The next-session model needs roughly 50+ trading bars. 120 calendar
    # days gives sufficient warm-up without repeatedly downloading a full year
    # during an interactive stock search.
    interval, period, fast, slow = "1d", "120d", 20, 50
    stop_atr, target_atr, max_hold = 1.8, 3.0, 20
else:
    interval, period, fast, slow = "1m", "5d", 9, 21
    stop_atr, target_atr, max_hold = 1.2, 2.0, 30

st.sidebar.subheader("Signal weights")
tech_weight = st.sidebar.slider("Price/technical evidence", 0.30, 0.75, 0.55, 0.05)
news_weight = st.sidebar.slider("Stock-specific news", 0.05, 0.40, 0.25, 0.05)
market_weight = st.sidebar.slider("Market regime", 0.10, 0.35, 0.20, 0.05)

st.sidebar.subheader("Risk model")
cost_bps = st.sidebar.number_input("Estimated round-trip cost (bps)", min_value=0.0, max_value=100.0, value=15.0, step=1.0,
                                   help="Modeling assumption, not your broker's exact fee schedule.")
slippage_bps = st.sidebar.number_input("Slippage per side (bps)", min_value=0.0, max_value=50.0, value=5.0, step=1.0)

if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    st.session_state["watchlist_revision"] = int(st.session_state.get("watchlist_revision", 0)) + 1

status = "Market OPEN" if market_is_open() else "Market CLOSED (last available data)"
st.sidebar.info(status)

# ------------------------------------------------ Persistent historical store ---
@st.cache_data(ttl=300, show_spinner=False)
def cached_persistent_storage_summary():
    return persistent_storage_summary()

pstore = cached_persistent_storage_summary()
st.sidebar.subheader("Permanent historical store")
drive_info = pstore.get("google_drive", {}) or {}
if drive_info.get("configured") and drive_info.get("reachable", True) is not False:
    st.sidebar.success("Google Drive persistent store configured")
    if drive_info.get("root_name"):
        st.sidebar.caption(f"Drive folder: {drive_info['root_name']}")
elif drive_info.get("configured"):
    st.sidebar.warning("Google Drive configured but not reachable from this deployment")
    st.sidebar.caption(str(drive_info.get("reason", "Drive check failed")))
else:
    st.sidebar.info("Local persistent store active. Add Google Drive secrets for durable cloud storage.")
st.sidebar.caption(f"Local persistent 1m: {pstore.get('local_files', 0)} partitions · {pstore.get('local_gb', 0):.4f} GB")

if st.sidebar.button("Capture current 1-minute data", help="Fetch the latest official NSE 1-minute chart data for the current watchlist and persist it. Reported volume is never fabricated."):
    capture_results = []
    with st.spinner("Capturing latest official NSE 1-minute bars…"):
        for symbol in symbols:
            try:
                company_root = symbol.replace(".NS", "").replace(".BO", "").upper()
                df_1m = nse_fetch_intraday_1m_current(company_root, sync_drive=True)
                capture_results.append({"symbol": symbol, "status": "OK" if df_1m is not None and not df_1m.empty else "NO_DATA", "rows_returned": 0 if df_1m is None else int(len(df_1m)), "source": "NSE_OFFICIAL_CHART_1M"})
            except Exception as exc:
                capture_results.append({"symbol": symbol, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "source": "NSE_OFFICIAL_CHART_1M"})
    st.session_state["persistent_capture_results"] = pd.DataFrame(capture_results)
    cached_persistent_storage_summary.clear()

if st.sidebar.button("Sync local 1-minute archive to Google Drive"):
    with st.spinner("Syncing persistent 1-minute partitions to Google Drive…"):
        st.session_state["persistent_sync_result"] = persistent_sync_pending(max_files=500)
    cached_persistent_storage_summary.clear()

if st.sidebar.button("Scan permanent 1-minute archive"):
    st.session_state["persistent_coverage"] = persistent_scan_coverage(symbols)

if "persistent_capture_results" in st.session_state:
    with st.expander("Latest 1-minute capture", expanded=False):
        st.dataframe(st.session_state["persistent_capture_results"], width="stretch", hide_index=True)
if "persistent_sync_result" in st.session_state:
    with st.expander("Google Drive sync result", expanded=False):
        st.write(st.session_state["persistent_sync_result"])
if "persistent_coverage" in st.session_state and not st.session_state["persistent_coverage"].empty:
    with st.expander("Permanent 1-minute archive coverage", expanded=False):
        st.dataframe(st.session_state["persistent_coverage"], width="stretch", hide_index=True)

# --------------------------------------------------------------- DhanHQ ---
# --------------------------------------------------------------- NSE official data layer ---
st.sidebar.subheader("NSE official data layer")

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_nse_universe():
    return fetch_nse_equity_universe(force_refresh=False)

try:
    _nse_universe = cached_nse_universe()
    _nse_symbol_col = next((c for c in _nse_universe.columns if str(c).strip().upper() in {"SYMBOL", "CH_SYMBOL"}), None)
    _nse_symbol_count = int(_nse_universe[_nse_symbol_col].astype(str).str.strip().replace({"nan":""}).loc[lambda x: x.ne("")].nunique()) if _nse_symbol_col else 0
    st.sidebar.success(f"NSE official equity universe: {_nse_symbol_count:,} symbols")
    st.sidebar.caption("Universe source: NSE Securities available for Equity segment")
except Exception as exc:
    _nse_universe = pd.DataFrame()
    _nse_symbol_col = None
    _nse_symbol_count = 0
    st.sidebar.warning(f"NSE equity-universe file unavailable: {type(exc).__name__}")

nse_col1, nse_col2 = st.sidebar.columns(2)
if nse_col1.button("Pull NSE EOD"):
    nse_rows = []
    with st.spinner("Pulling official NSE EOD data…"):
        for sym in symbols:
            try:
                root = sym.replace(".NS", "").replace(".BO", "")
                end = pd.Timestamp.now(tz="Asia/Kolkata").date()
                start = end - pd.Timedelta(days=365)
                df_nse = nse_fetch_equity_history(root, start.date() if hasattr(start, 'date') else start, end, sync_drive=True)
                nse_rows.append({"symbol":root,"status":"OK","rows":len(df_nse),"source":"NSE_OFFICIAL"})
            except Exception as exc:
                nse_rows.append({"symbol":root,"status":"ERROR","error":f"{type(exc).__name__}: {exc}"})
    st.session_state["nse_pull_result"] = pd.DataFrame(nse_rows)

if nse_col2.button("Capture NSE 1-minute"):
    nse_rows = []
    with st.spinner("Capturing current NSE quote-chart data…"):
        for sym in symbols:
            try:
                root = sym.replace(".NS", "").replace(".BO", "")
                df_nse = nse_fetch_intraday_1m_current(root, sync_drive=True)
                nse_rows.append({"symbol":root,"status":"OK","rows":len(df_nse),"source":"NSE_OFFICIAL_CHART"})
            except Exception as exc:
                nse_rows.append({"symbol":root,"status":"ERROR","error":f"{type(exc).__name__}: {exc}"})
    st.session_state["nse_1m_result"] = pd.DataFrame(nse_rows)

if "nse_pull_result" in st.session_state:
    with st.expander("NSE EOD pull", expanded=False):
        st.dataframe(st.session_state["nse_pull_result"], width="stretch", hide_index=True)
if "nse_1m_result" in st.session_state:
    with st.expander("NSE 1-minute capture", expanded=False):
        st.dataframe(st.session_state["nse_1m_result"], width="stretch", hide_index=True)

st.sidebar.caption("NSE is primary for official exchange data. The public chart path is LTP-based; missing minute volume is never fabricated. Historical data and normalized archives are mirrored to Google Drive when configured.")

st.sidebar.subheader("DhanHQ data provider")
dhan_cfg = DhanConfig.from_env()
dhan_stat = dhan_status(dhan_cfg)
if dhan_stat.get("profile"):
    prof = dhan_stat["profile"]
    data_plan = prof.get("dataPlan", "Unknown")
    data_valid = prof.get("dataValidity", "")
    st.sidebar.success(f"Dhan connected · Data plan: {data_plan}")
    if data_valid:
        st.sidebar.caption(f"Data validity: {data_valid}")
elif dhan_stat.get("configured"):
    st.sidebar.warning("Dhan credentials detected, but no valid access token/profile response is available.")
else:
    st.sidebar.info("Dhan is not configured. The dashboard will use the existing free fallback sources.")

# Remote/cloud deployments cannot complete Dhan's localhost OAuth callback because
# the callback would terminate on the cloud server rather than on the user's device.
# For browser deployment, configure DHAN_ACCESS_TOKEN (and DHAN_CLIENT_ID for
# profile/quote endpoints) in Streamlit Secrets. Local .env/OAuth remains available
# when running this project on a laptop.
if dhan_cfg.has_api_key_flow() and not dhan_stat.get("access_token_loaded"):
    st.sidebar.info(
        "Remote deployment: set DHAN_ACCESS_TOKEN in Streamlit Secrets to use Dhan live data. "
        "The local OAuth callback is intentionally not launched from the hosted dashboard."
    )

# Dhan historical download controls
dhan_interval = st.sidebar.selectbox("Dhan historical interval", ["1m", "5m", "15m", "60m", "1d"], index=1)
dhan_days = st.sidebar.number_input("Dhan history days", min_value=1, max_value=1825, value=365, step=30)
dhan_oi = st.sidebar.checkbox("Include OI where applicable", value=False)
if st.sidebar.button("Download selected history from Dhan"):
    if not dhan_stat.get("access_token_loaded"):
        st.sidebar.error("Authenticate with Dhan first.")
    else:
        from datetime import datetime, timedelta
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=int(dhan_days))
        selected_roots = [normalize_symbol(x).replace(".NS", "") for x in raw_symbols.split(",") if x.strip()]
        results = []
        with st.spinner(f"Downloading Dhan {dhan_interval} history for {len(selected_roots)} symbols…"):
            for root in selected_roots:
                try:
                    results.append(dhan_download_symbol(root, start_dt.strftime("%Y-%m-%d %H:%M:%S"), end_dt.strftime("%Y-%m-%d %H:%M:%S"), dhan_interval, overwrite=False, oi=dhan_oi))
                except Exception as exc:
                    results.append({"symbol": root, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
        ok_count = sum(r.get("status") == "OK" for r in results)
        st.sidebar.success(f"Dhan download complete: {ok_count}/{len(results)} symbols")
        st.session_state["dhan_download_results"] = pd.DataFrame(results)

if "dhan_download_results" in st.session_state:
    with st.expander("Dhan download results", expanded=False):
        st.dataframe(st.session_state["dhan_download_results"], width="stretch")

# ------------------------------------------------------------------ audit ---
st.sidebar.subheader("1-Year Prediction Audit")
default_audit_text = ", ".join(s for s, _ in DEFAULT_AUDIT_UNIVERSE)
audit_text = st.sidebar.text_area(
    "Audit universe (NSE symbols)",
    default_audit_text,
    help="Tests each symbol over roughly the last year. Bare symbols are treated as NSE. The audit uses price + market history only because the free RSS feeds do not provide a timestamped historical news archive.",
)
audit_horizon = st.sidebar.selectbox("Primary audit horizon", [1, 3, 5, 10, 20], index=2)
audit_run = st.sidebar.button("Run 1-Year Prediction Audit")

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_universe_audit(symbol_text, audit_days, tech_w, news_w, market_w, threshold):
    symbols_for_audit = []
    for value in symbol_text.split(","):
        v = normalize_symbol(value)
        if v:
            symbols_for_audit.append(v)
    cfg = AuditConfig(
        audit_days=int(audit_days),
        tech_weight=float(tech_w),
        news_weight=float(news_w),
        market_weight=float(market_w),
        signal_threshold=float(threshold),
    )
    return run_universe_audit(symbols=symbols_for_audit, config=cfg, fetcher=local_history_provider)

if audit_run:
    with st.spinner("Downloading roughly two years of daily history and replaying the model causally…"):
        st.session_state["audit_result"] = cached_universe_audit(
            audit_text, 365, 0.60, 0.20, 0.20, 0.25
        )

if "audit_result" in st.session_state:
    result = st.session_state["audit_result"]
    overall_all = result.get("overall", {})
    records_all = result.get("records")
    stocks_all = result.get("stocks")
    sens_all = result.get("threshold_sensitivity")

    st.markdown("# 1-Year Prediction Audit")
    st.caption(
        "Causal replay: signal at day T close → entry at day T+1 open. "
        "BUY is counted correct when the forward return is positive; SELL when it is negative. "
        "HOLD is excluded from directional hit rate."
    )
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Stocks tested", len(result.get("symbols_tested", [])))
    a2.metric("Actionable signals", overall_all.get("actionable", 0))
    a3.metric(f"{audit_horizon}D hit rate", f"{overall_all.get(f'hit_rate_{audit_horizon}d_pct', 0):.1f}%")
    a4.metric(f"{audit_horizon}D signed edge", f"{overall_all.get(f'avg_signed_edge_{audit_horizon}d_pct', 0):+.3f}%")
    a5.metric("Score Spearman", f"{overall_all.get(f'score_corr_{audit_horizon}d', 0):+.2f}")

    if result.get("diagnostics"):
        with st.expander("What the audit is saying", expanded=True):
            for note in result["diagnostics"]:
                st.write("• " + note)

    if result.get("errors"):
        with st.expander(f"Data errors / unavailable symbols ({len(result['errors'])})"):
            st.write(result["errors"])

    if records_all is not None and not records_all.empty:
        horizon_cols = []
        for h in [1, 3, 5, 10, 20]:
            horizon_cols += [f"fwd_{h}d_pct", f"correct_{h}d"]
        display_cols = ["symbol", "decision_date", "signal", "combined_score", "technical_score", "market_score"] + horizon_cols
        with st.expander("All individual prediction dates", expanded=False):
            st.dataframe(records_all[display_cols], width="stretch")
            st.download_button(
                "Download prediction replay CSV",
                records_all.to_csv(index=False).encode("utf-8"),
                file_name="stock_prediction_audit_replay.csv",
                mime="text/csv",
            )

    if stocks_all is not None and not stocks_all.empty:
        st.markdown("### Stock-by-stock results")
        stock_display = ["symbol", "size_bucket", "decision_bars", "buy_signals", "sell_signals"]
        stock_display += [f"hit_rate_{h}d_pct" for h in [1, 3, 5, 10, 20]]
        stock_display += [f"avg_signed_edge_{h}d_pct" for h in [1, 3, 5, 10, 20]]
        st.dataframe(stocks_all[stock_display], width="stretch")

        bucket = stocks_all.groupby("size_bucket", dropna=False).agg(
            stocks=("symbol", "count"),
            signals=("actionable", "sum"),
            hit_rate_5d=("hit_rate_5d_pct", "mean"),
            edge_5d=("avg_signed_edge_5d_pct", "mean"),
        ).reset_index()
        st.markdown("### Results by universe bucket")
        st.dataframe(bucket.round(3), width="stretch")

    if sens_all is not None and not sens_all.empty:
        st.markdown("### Signal-threshold sensitivity")
        st.dataframe(sens_all.round(3), width="stretch")
        st.line_chart(sens_all.set_index("threshold")[["hit_rate_pct", "avg_signed_edge_pct"]])

    st.info(
        "This audit deliberately does not inject today's RSS news into historical dates. "
        "It therefore validates the causal price + market model, not the full live-news model. "
        "Use the results to decide what to test next, not as a guarantee of future returns."
    )

# --------------------------------------------------------- free data manager
st.sidebar.subheader("Legacy Third-party Research Archive")
st.sidebar.caption("Optional research-only archives. They are not the primary exchange data layer; NSE official data is primary.")
data_choice = st.sidebar.selectbox(
    "Download universe",
    ["Current watchlist", "19-stock audit core", "NIFTY 500"],
    index=1,
)
data_sources = st.sidebar.multiselect(
    "Equity 1-minute archives",
    ["2018-2025 (long archive)", "2026 (YTD/current repository)"],
    default=["2026 (YTD/current repository)"],
    help="The 2018-2025 archive contains 1-minute history from 2018 through 31 Dec 2025. The 2026 repository is a separate current-year archive.",
)
allow_big = st.sidebar.checkbox(
    "I understand NIFTY 500 can require substantial disk space",
    value=False,
)
col_dl1, col_dl2 = st.sidebar.columns(2)
if col_dl1.button("Download equity"):
    if data_choice == "NIFTY 500" and not allow_big:
        st.sidebar.error("Tick the disk-space confirmation before downloading NIFTY 500.")
    else:
        if data_choice == "Current watchlist":
            selected_syms = [normalize_symbol(x) for x in raw_symbols.split(",") if x.strip()]
        elif data_choice == "19-stock audit core":
            selected_syms = [f"{x}.NS" for x in CORE_SYMBOLS]
        else:
            try:
                selected_syms = [f"{x}.NS" for x in discover_repo_symbols("2026") if not str(x).startswith("NIFTY") and not str(x).startswith("BANKNIFTY") and not str(x).startswith("FINNIFTY") and not str(x).startswith("MIDCAP")][:600]
            except Exception as exc:
                st.sidebar.error(f"Could not discover NIFTY 500 file list: {exc}")
                selected_syms = []
        sources = []
        if "2018-2025 (long archive)" in data_sources:
            sources.append("2018_2025")
        if "2026 (YTD/current repository)" in data_sources:
            sources.append("2026")
        if not sources:
            st.sidebar.warning("Select at least one archive.")
        elif selected_syms:
            with st.spinner(f"Downloading {len(selected_syms)} symbol file(s)…"):
                result_rows = download_equity_universe(selected_syms, sources=sources, workers=4)
            ok_count = sum(1 for r in result_rows if r.get("status") == "ok")
            st.sidebar.success(f"Equity downloads complete: {ok_count}/{len(result_rows)} files ready.")

if col_dl2.button("Download F&O"):
    with st.spinner("Downloading and extracting the free F&O 1-minute archive (~464 MB compressed)…"):
        fo_result = download_fno_archive()
    if fo_result.get("status") == "ok":
        st.sidebar.success(f"F&O archive ready: {fo_result.get('extracted_files', 0)} files extracted.")
    else:
        st.sidebar.error(fo_result.get("message", "F&O download failed"))

if st.sidebar.button("Scan local archive"):
    ensure_data_dirs()
    st.session_state["coverage"] = scan_coverage()
    st.session_state["storage"] = storage_summary()

if st.sidebar.button("Check source availability"):
    with st.spinner("Checking public source/repository availability…"):
        st.session_state["source_check"] = online_source_check()

if st.sidebar.button("Validate local data"):
    ensure_data_dirs()
    with st.spinner("Checking OHLCV integrity of local archives…"):
        st.session_state["quality"] = scan_quality()

if st.sidebar.button("Reconcile to NSE (recent window)"):
    ensure_data_dirs()
    with st.spinner("Comparing local daily aggregates with official NSE history…"):
        st.session_state["quality"] = scan_quality(symbols=CORE_SYMBOLS, reconcile=True, sample_days=30)

if "storage" in st.session_state:
    st.sidebar.write(st.session_state["storage"])

with st.expander("Legacy research archive & NSE authenticity check", expanded=False):
    st.write(
        "NSE is the primary exchange-data source. Cached NSE bulk/EOD files are treated as the persisted exchange layer; "
        "direct NSE requests are used when available. Other public providers remain explicit fallbacks and are never labelled as NSE."
    )
    st.markdown("### Source provenance")
    st.dataframe(source_provenance_table(), width="stretch", hide_index=True)
    if "source_check" in st.session_state:
        st.markdown("### Online source availability")
        st.dataframe(st.session_state["source_check"], width="stretch", hide_index=True)
        st.download_button("Export source-availability check", st.session_state["source_check"].to_csv(index=False).encode("utf-8"), file_name="source_availability_check.csv", mime="text/csv")
    if "coverage" in st.session_state and not st.session_state["coverage"].empty:
        st.markdown("### Local coverage")
        st.dataframe(st.session_state["coverage"], width="stretch")
        st.download_button("Export local-data manifest", pd.DataFrame(st.session_state["coverage"]).to_csv(index=False).encode("utf-8"), file_name="local_historical_coverage.csv", mime="text/csv")
    else:
        st.info("Use 'Scan local archive' in the sidebar after your first download.")
    if "quality" in st.session_state and not st.session_state["quality"].empty:
        st.markdown("### Data-quality / NSE reconciliation")
        st.dataframe(st.session_state["quality"], width="stretch")
        st.download_button("Export data-quality report", st.session_state["quality"].to_csv(index=False).encode("utf-8"), file_name="historical_data_quality.csv", mime="text/csv")

st.sidebar.caption(
    "Build: 2026-09-29-NSE-FIRST11"
)
st.sidebar.caption(
    "NSE official data is the primary exchange-data layer. Dhan is optional for live/broker-specific fields; other public providers are fallbacks only and their provenance is preserved. No order APIs are used."
)

@st.cache_data(ttl=60)
def load_news():
    return get_news()

@st.cache_data(ttl=120, show_spinner=False)
def load_history(symbol, period, interval, revision=0):
    try:
        return get_historical(symbol, period=period, interval=interval)
    except Exception as exc:
        st.warning(f"{symbol}: historical data could not be loaded ({type(exc).__name__}: {exc})")
        return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False)
def load_benchmark(symbol, period, interval, revision=0):
    try:
        return get_historical(symbol, period=period, interval=interval)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=90)
def load_option_chain(nse_symbol, is_index):
    return get_option_chain(nse_symbol, is_index=is_index)

@st.cache_data(ttl=180)
def load_announcements(symbol):
    return get_corporate_announcements(symbol)

# News is not part of the live 5-minute model path. Keep it lazy so a
# Delivery or Intraday-5-minute page load does not wait on RSS/network I/O.
news_items = []

@st.cache_data(ttl=60, show_spinner=False)
def cached_delivery_validation():
    """Read persisted Delivery validation results without fresh per-symbol NSE calls."""
    records = [
        r for r in load_records()
        if str(r.get("mode", "")).lower() == "delivery"
    ][-100:]

    if not records:
        return pd.DataFrame(), {"status": "NO_DATA"}, {
            "status": "NO_DATA",
            "retraining_candidate": False,
        }

    ids = {str(r.get("prediction_id")) for r in records if r.get("prediction_id")}
    validation_rows = [
        r for r in load_validation_records()
        if str(r.get("prediction_id")) in ids
    ]

    validated = pd.DataFrame(validation_rows)
    if validated.empty:
        return validated, {
            "status": "NO_DATA",
            "pending_predictions": len(records),
        }, {
            "status": "INSUFFICIENT_HISTORY",
            "sample_count": 0,
            "retraining_candidate": False,
            "reason": "No completed D+1 Delivery validations are available yet.",
        }

    return (
        validated,
        rolling_7day_scorecard(validated),
        assess_delivery_health(validated),
    )

# ---------------------------------------------------------------- Research context
# No legacy production-model routing remains in Research / Diagnostics.
# The existing renderer continues to show non-ML analytical context only.
beta_delivery_results = {}
beta_intraday_results = {}
direct_delivery_results = {}
direct_delivery_errors = {}
direct_delivery_runtime = None
production_results = {}
delivery_results = {}

for rec in live_records:
    symbol = rec["symbol"]
    st.markdown(f"## {symbol}")
    raw_df = rec["raw_df"]
    df = rec["df"]
    market_df = rec["market_df"]
    benchmark_symbol = rec["benchmark_symbol"]
    matched_news = rec["matched_news"]
    news_stat = rec["news_stat"]
    news_perception_label = rec["news_perception_label"]
    news_freshness = rec["news_freshness"]
    announcements = rec["announcements"]
    event_risk = rec["event_risk"]
    live_news_score = rec["live_news_score"]
    fo = rec["fo"]
    t_score = rec["t_score"]
    m_score = rec["m_score"]

    # The legacy technical/chart frame can legitimately be empty even when
    # the dedicated ALL-NSE Delivery model has produced a valid prediction.
    # Never let that optional UI context crash or suppress the ML result.
    if df is None or df.empty or "Close" not in df.columns:

        if style == "Delivery":

            dlv = (
                direct_delivery_results.get(symbol)
                or delivery_results.get(symbol)
            )

            st.markdown(
                f"## {symbol}"
            )

            st.markdown(
                "### Live ALL-NSE Delivery ML — next-session signal"
            )

            if dlv and dlv.get("prediction_status") == "PASS":

                d1, d2, d3, d4, d5 = st.columns(5)

                d1.metric(
                    "Delivery Signal",
                    dlv.get("signal", "HOLD"),
                )

                d2.metric(
                    "P(up) next session",
                    f"{dlv.get('probability_up_next_session', np.nan):.1%}",
                )

                d3.metric(
                    "Expected move",
                    f"{dlv.get('predicted_return_next_session', np.nan):+.2%}",
                )

                d4.metric(
                    "Confidence",
                    dlv.get("confidence", "UNKNOWN"),
                )

                d5.metric(
                    "Model inputs",
                    str(dlv.get("feature_count", 59)),
                )

                st.write({
                    "model_version":
                        dlv.get("model_version"),
                    "reference_session":
                        dlv.get("reference_session"),
                    "feature_source":
                        dlv.get("feature_source"),
                    "PIT_snapshot_date":
                        dlv.get("pit_snapshot_date"),
                    "data_readiness":
                        dlv.get("data_readiness_score"),
                    "non_missing_features":
                        f"{dlv.get('non_missing_feature_count', 0)}/{dlv.get('feature_count', 59)}",
                    "trainable_NSE_symbols":
                        dlv.get("trainable_symbol_count"),
                    "NSE_universe_symbols":
                        dlv.get("universe_symbol_count"),
                })

                st.info(
                    "The live ALL-NSE Delivery prediction is valid. "
                    "The optional legacy technical/chart context was unavailable "
                    "for this request and was skipped."
                )

            elif dlv and dlv.get("prediction_status") == "BLOCKED":

                st.error(
                    f"Delivery prediction blocked: "
                    f"{dlv.get('reason', 'unknown')}"
                )

            else:

                st.warning(
                    "Delivery ML prediction is unavailable for this symbol."
                )

        else:

            st.warning(
                "The technical history needed for the legacy dashboard "
                "context is unavailable for this symbol."
            )

        # Never execute df.iloc[-1], technical_components(), line_chart(),
        # or the legacy backtest blocks on an empty frame.
        continue

    label, combined = combined_signal(
        t_score, live_news_score, m_score, event_risk,
        tech_weight, news_weight, market_weight,
    )
    last = df.iloc[-1]
    latest_close = float(last["Close"])
    previous_close = float(df["Close"].iloc[-2])
    pct_change = (latest_close - previous_close) / previous_close * 100 if previous_close else 0.0

    primary = (
        direct_delivery_results.get(symbol)
        if style == "Delivery"
        else production_results.get(symbol)
    )
    primary_signal = primary.get("signal") if primary else "UNAVAILABLE"
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Primary Signal", primary_signal)
    c2.metric("Last", f"₹{latest_close:,.2f}", f"bar {pct_change:+.2f}%")
    c3.metric("Technical context", f"{t_score:+.2f}")
    c4.metric("News perception", news_perception_label, f"{news_stat['count']} RSS matches · {news_freshness}")
    c5.metric("Market regime", f"{m_score:+.2f}", benchmark_symbol.replace("^", ""))

    ml = production_results.get(symbol)
    if style == "Intraday":
        st.markdown("### Production ML — next 30-minute direction")
        if ml:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ML Signal", ml["signal"])
            m2.metric("P(up) 30m", f"{ml['probability_up_30m']:.1%}")
            m3.metric("Confidence", ml["confidence"])
            m4.metric("Data readiness", f"{ml['data_readiness_score']:.0%}")
            st.caption(
                "Certified logistic-regression production model · 137-feature contract · target_30m. "
                "The probability is the model output, not a guarantee of profit. Missing optional live groups are explicitly represented and handled by the trained imputer."
            )
        else:
            st.warning("Production ML prediction unavailable for this symbol; the rule-based context remains available.")
    else:
        dlv = direct_delivery_results.get(symbol) or delivery_results.get(symbol)
        st.markdown("### Live ALL-NSE Delivery ML — next-session signal")
        if dlv and dlv.get("prediction_status") == "PASS":
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Delivery Signal", dlv["signal"])
            d2.metric("P(up) next session", f"{dlv['probability_up_next_session']:.1%}")
            d3.metric("Expected move", f"{dlv['predicted_return_next_session']:+.2%}")
            d4.metric("Confidence", dlv["confidence"])
            d5.metric("Model inputs", f"{dlv['feature_count']}")
            st.caption(
                "LIVE ALL-NSE Delivery candidate · 59-feature point-in-time NSE contract · "
                f"{int(dlv.get('trainable_symbol_count', 0)):,} trainable stocks from the "
                f"{int(dlv.get('universe_symbol_count', 0)):,}-stock official NSE equity universe. "
                "Prediction is stored automatically for actual-vs-predicted validation."
            )
        elif dlv and dlv.get("prediction_status") == "BLOCKED":
            st.error(
                f"Delivery prediction blocked: {dlv.get('reason', 'unknown')}"
            )
            st.caption(
                f"ALL-NSE Delivery roster: "
                f"{int(dlv.get('trainable_symbol_count', 0)):,} trainable / "
                f"{int(dlv.get('universe_symbol_count', 0)):,} NSE universe "
                f"with a {int(dlv.get('minimum_history_sessions', 60))}-session minimum-history gate."
            )
        else:
            st.warning("Delivery production prediction unavailable for this symbol.")

    if style == "Delivery":
        with st.expander("Continuous 7-day prediction validation", expanded=True):
            try:
                vdf, vscore, vhealth = cached_delivery_validation()
                st.write({
                    "rolling_7_completed_predictions": vscore.get("completed_predictions", 0),
                    "rolling_7_directional_accuracy": vscore.get("directional_accuracy_7_predictions"),
                    "mean_absolute_price_error_pct": vscore.get("mean_absolute_percentage_price_error"),
                    "model_health": vhealth.get("status"),
                    "retraining_candidate": vhealth.get("retraining_candidate", False),
                })
                root = symbol.replace(".NS", "").replace(".BO", "").upper()
                if not vdf.empty and "symbol" in vdf.columns:
                    sub = vdf[vdf["symbol"].astype(str).str.upper() == root].sort_values("decision_timestamp", ascending=False).head(7)
                    if not sub.empty:
                        st.dataframe(sub, width="stretch", hide_index=True)
                st.caption("D+1 is the target-aligned next-session validation. D+2 through D+7 are preserved as follow-through observations, not counted as extra next-session targets.")
            except Exception as exc:
                st.warning(f"Continuous validation unavailable: {type(exc).__name__}: {exc}")

    if event_risk > 0:
        st.warning(f"Trigger/event risk is elevated ({event_risk:.2f}). The engine reduces conviction when material events are present; it does not assume those events are bullish or bearish.")
    if abs(pct_change) >= 3.0:
        st.warning(f"Large move: {pct_change:+.2f}% on the latest bar. Check the trigger/news section before acting.")

    comp = technical_components(df, fast=fast, slow=slow)
    with st.expander("Why the rule-based signal looks this way", expanded=True):
        st.write({k: round(v, 3) for k, v in comp.items()})
        st.write({
            "technical_weighted": round(tech_weight * t_score, 3),
            "news_weighted": round(news_weight * live_news_score, 3),
            "market_weighted": round(market_weight * m_score, 3),
            "event_risk": round(event_risk, 3),
        })
        st.caption("News perception is a headline-level VADER/keyword score. The Production ML probability is shown separately and is not blended into this legacy rule score.")

    st.line_chart(df[["Close", f"SMA{fast}", f"SMA{slow}"]].tail(300))

    with st.expander("Stock-specific news & triggers"):
        if matched_news:
            for n in matched_news[:15]:
                tags = ", ".join(n.get("tags", [])) or "routine"
                st.write(f"**{n['title']}** — `{tags}` — {n.get('source', '')}")
                if n.get("link"):
                    st.caption(n["link"])
        else:
            st.info("No stock-specific headlines were found in the current RSS pull.")
        if announcements:
            st.markdown("**Recent NSE corporate announcements**")
            for a in announcements:
                st.write(f"- {a['date']}: {a['subject']}")
        else:
            st.caption("No corporate announcements were returned by NSE's endpoint in this pull.")

    with st.expander("Historical validation / backtest"):
        trades_df, summary = backtest(
            raw_df, style=style, market_df=market_df, fast=fast, slow=slow,
            stop_atr=stop_atr, target_atr=target_atr, max_hold=max_hold,
            cost_bps=cost_bps, slippage_bps=slippage_bps,
        )
        st.write(summary)
        if summary.get("trades", 0):
            baseline = random_baseline(raw_df, trade_durations=trades_df["holding_bars"].tolist(), cost_bps=cost_bps, slippage_bps=slippage_bps)
            holdout = holdout_backtest(raw_df, style, market_df=market_df, fast=fast, slow=slow, stop_atr=stop_atr, target_atr=target_atr, max_hold=max_hold, cost_bps=cost_bps, slippage_bps=slippage_bps)
            st.write("Matched-duration random benchmark:", baseline)
            st.write("Last 30% holdout period:", holdout["summary"])
            st.caption("Validation uses price/market data only. Reliable historical news archives are not included in this free build, so the live news component is not falsely presented as historically validated.")
            st.dataframe(trades_df, width="stretch")

    with st.expander("F&O context"):
        if not fo:
            st.info("No F&O snapshot was returned. NSE may have rate-limited the request or the security may not have a listed derivative.")
        else:
            a, b, c, d = st.columns(4)
            a.metric("Spot", fo.get("spot")); b.metric("PCR", fo.get("pcr")); c.metric("Max pain", fo.get("max_pain")); d.metric("Expiry", str(fo.get("expiry")))
            st.write(f"ATM call OI read: **{fo.get('call_read') or 'not available'}**")
            st.write(f"ATM put OI read: **{fo.get('put_read') or 'not available'}**")
            st.caption("F&O figures are contextual inputs; PCR/Max Pain are not treated as validated directional predictors.")
            near_atm = fo.get("near_atm_chain")
            if near_atm is not None and not near_atm.empty:
                st.dataframe(near_atm, width="stretch")

