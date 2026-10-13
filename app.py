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
from live_signal_guards import (
    assess_horizon_completeness,
    latest_completed_bar_status,
    target_session_is_underway,
)

from prediction_validation import validate_ledger, rolling_7day_scorecard
from model_health import assess_delivery_health
from historical_audit import run_batch_intraday_audit
from historical_replay import (
    historical_price, historical_benchmark, historical_news, historical_announcements,
    actual_outcome, replay_rule_delivery, _entry_exit_metrics,
)
from prediction_history_ui import render_delivery_forecast, render_prediction_history
from daily_ohlc_forecast import forecast_next_sessions
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
from market_session_guard import (
    latest_completed_nse_session,
    nse_session_label,
    parse_reference_date,
)

from market_watchlist import (
    build_stock_watchlist,
    fetch_watchlist_articles,
    target_session_date as resolve_watch_target_session,
)

try:
    from indstocks_client import (
        configured as _indstocks_configured,
        historical_1m as _indstocks_historical_1m,
        latest_quote as _indstocks_latest_quote,
        historical_1m as _indstocks_test_history,
        profile as _indstocks_profile,
    )
except Exception:
    _indstocks_configured = _indstocks_historical_1m = None
    _indstocks_latest_quote = _indstocks_test_history = _indstocks_profile = None

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
        "Development and research tools. Live Test uses the Beta V2 self-driving Delivery and Intraday runtime adapters. "
        "Research / Diagnostics remains separate from live model serving. No orders are placed."
    )

INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}

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


default_watchlist = ""


def normalize_symbol(value: str) -> str:
    """Normalize user input for Yahoo Finance. Bare Indian symbols default to NSE."""
    s = value.strip().upper()
    if not s:
        return s
    if s.startswith("^"):
        return s
    if s.endswith((".NS", ".BO")):
        return s
    return f"{s}.NS"


@st.cache_data(ttl=900, show_spinner=False)
def _today_stock_watchlist() -> dict:
    """Cache the internet-sourced, source-attributed daily NSE watchlist for 15 minutes."""
    try:
        universe = _searchable_nse_universe()
    except Exception:
        universe = pd.DataFrame()
    sources = fetch_watchlist_articles()
    return build_stock_watchlist(universe, sources, max_stocks=5)


if "watchlist_revision" not in st.session_state:
    st.session_state["watchlist_revision"] = 0
if "active_watchlist" not in st.session_state:
    st.session_state["active_watchlist"] = ""
# Clear the former built-in three-stock starter list from existing sessions.
if st.session_state.get("active_watchlist", "").strip().upper() == "BHARTIARTL.NS, RELIANCE.NS, SBIN.NS":
    st.session_state["active_watchlist"] = ""

_search_df = pd.DataFrame()
if interface_mode == "Research / Diagnostics":
    st.sidebar.subheader("Manual research universe")
    st.sidebar.caption("Live Test uses the automatically sourced daily watchlist. Manual symbols are only for research tools.")
    try:
        _search_df = _searchable_nse_universe()
    except Exception:
        _search_df = pd.DataFrame()

    if not _search_df.empty:
        search_text = st.sidebar.text_input(
            "Find NSE stock",
            key="nse_stock_search",
            placeholder="Company name or NSE symbol",
        ).strip().lower()
        search_results = _search_df
        if search_text:
            mask = (
                search_results["symbol"].str.lower().str.contains(search_text, regex=False, na=False)
                | search_results["company"].str.lower().str.contains(search_text, regex=False, na=False)
            )
            search_results = search_results.loc[mask].head(40)

        if not search_results.empty:
            options = [
                f"{row.symbol} — {row.company}" if row.company and row.company.lower() != "nan"
                else row.symbol
                for row in search_results.itertuples()
            ]
            selected_search = st.sidebar.selectbox("Matching stocks", options, key="nse_stock_search_result")
            selected_symbol = selected_search.split(" — ", 1)[0].strip().upper()
            if st.sidebar.button("Add to research symbols", width="stretch"):
                current = [
                    value.strip().upper()
                    for value in st.session_state.get("active_watchlist", "").split(",")
                    if value.strip()
                ]
                selected_ns = f"{selected_symbol}.NS"
                if selected_ns not in current:
                    current.append(selected_ns)
                st.session_state["active_watchlist"] = ", ".join(current)
                st.session_state["watchlist_input"] = st.session_state["active_watchlist"]
                st.session_state["watchlist_revision"] += 1
                st.rerun()

    with st.sidebar.form("watchlist_form", clear_on_submit=False, border=False):
        watchlist_input = st.text_area(
            "Research symbols (comma-separated)",
            st.session_state.get("active_watchlist", ""),
            key="watchlist_input",
            height=78,
            placeholder="Example: TCS, INFY, HDFCBANK",
        )
        analyze_watchlist = st.form_submit_button("Apply research symbols", width="stretch")

    if analyze_watchlist:
        cleaned = ", ".join(part.strip() for part in watchlist_input.split(",") if part.strip())
        st.session_state["active_watchlist"] = cleaned
        st.session_state["watchlist_revision"] += 1

raw_symbols = st.session_state.get("active_watchlist", "") if interface_mode == "Research / Diagnostics" else ""
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
    """Fetch fresh intraday benchmark candles, with explicit source diagnostics.

    The old implementation swallowed every exception and returned an empty
    frame. For index tickers such as ^NSEI, the generic historical adapter may
    have no local archive and its first Yahoo chart path can return no candles.
    Try the primary path first, then yfinance directly; keep the freshness guard
    fail-closed if neither source yields completed current-session candles.
    """
    import json
    import time

    started = time.perf_counter()
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    attempts: list[dict[str, object]] = []
    primary_frame = pd.DataFrame()

    def _freshness(frame: pd.DataFrame) -> dict:
        return latest_completed_bar_status(
            frame,
            now,
            max_age_seconds=5 * 60,
            require_current_session=True,
        )

    try:
        candidate = get_historical(
            benchmark_symbol,
            period="5d",
            interval="1m",
            force_refresh=True,
        )
        if candidate is not None:
            primary_frame = candidate
        state = _freshness(primary_frame)
        attempts.append({
            "source": "get_historical",
            "rows": int(len(primary_frame)),
            "status": state.get("status"),
            "reason": state.get("reason"),
            "latest_candle_ist": state.get("latest_candle_ist"),
        })
        if not primary_frame.empty and state.get("ok"):
            primary_frame.attrs["market_context_source"] = "get_historical"
            print("[MARKET_CONTEXT_FETCH] " + json.dumps({
                "benchmark": benchmark_symbol,
                "selected_source": "get_historical",
                "rows": int(len(primary_frame)),
                "freshness": state,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            }, default=str, sort_keys=True))
            return primary_frame
    except Exception as exc:
        attempts.append({
            "source": "get_historical",
            "status": "EXCEPTION",
            "error": f"{type(exc).__name__}: {exc}",
        })

    fallback_frame = pd.DataFrame()
    try:
        import yfinance as yf
        from data_fetcher import _normalize_historical_frame

        raw = yf.download(
            str(benchmark_symbol).strip(),
            period="1d",
            interval="1m",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=10,
        )
        fallback_frame = _normalize_historical_frame(raw)
        state = _freshness(fallback_frame)
        attempts.append({
            "source": "yfinance_direct",
            "rows": int(len(fallback_frame)),
            "status": state.get("status"),
            "reason": state.get("reason"),
            "latest_candle_ist": state.get("latest_candle_ist"),
        })
        if not fallback_frame.empty and state.get("ok"):
            fallback_frame.attrs["market_context_source"] = "yfinance_direct"
            print("[MARKET_CONTEXT_FETCH] " + json.dumps({
                "benchmark": benchmark_symbol,
                "selected_source": "yfinance_direct",
                "rows": int(len(fallback_frame)),
                "freshness": state,
                "attempts": attempts,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            }, default=str, sort_keys=True))
            return fallback_frame
    except Exception as exc:
        attempts.append({
            "source": "yfinance_direct",
            "status": "EXCEPTION",
            "error": f"{type(exc).__name__}: {exc}",
        })

    # Preserve a stale primary frame when available so the UI can explain that
    # its timestamps are stale, rather than disguising it as a successful fetch.
    selected = primary_frame if not primary_frame.empty else fallback_frame
    selected_state = _freshness(selected)
    print("[MARKET_CONTEXT_FETCH] " + json.dumps({
        "benchmark": benchmark_symbol,
        "selected_source": (
            selected.attrs.get("market_context_source", "stale_or_unverified")
            if not selected.empty else "none"
        ),
        "rows": int(len(selected)),
        "freshness": selected_state,
        "attempts": attempts,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "action": "fail_closed_no_fresh_benchmark_context",
    }, default=str, sort_keys=True))
    return selected



@st.cache_data(ttl=20, show_spinner=False)
def _live_5m_price_frame(symbol: str) -> tuple[pd.DataFrame, str]:
    """Get live 1-minute bars with broker-first routing when INDstocks is configured."""
    root = str(symbol).replace(".NS", "").replace(".BO", "").upper()

    # Prefer the authenticated broker feed for live model input when configured.
    # The official NSE chart and persistent cache remain fallbacks if the broker
    # cannot authenticate or return usable candles.
    if _indstocks_configured is not None and _indstocks_historical_1m is not None:
        try:
            if _indstocks_configured():
                broker_frame = _indstocks_historical_1m(root, days=3)
                if broker_frame is not None and not broker_frame.empty:
                    persist_needed = True
                    try:
                        now = pd.Timestamp.now(tz="Asia/Kolkata")
                        existing = load_persistent_1m(
                            root,
                            start=(now - pd.Timedelta(days=3)),
                            end=now,
                        )
                        if existing is not None and not existing.empty:
                            existing_last = pd.Timestamp(existing.index.max())
                            broker_last = pd.Timestamp(broker_frame.index.max())
                            if existing_last.tzinfo is None:
                                existing_last = existing_last.tz_localize("Asia/Kolkata")
                            else:
                                existing_last = existing_last.tz_convert("Asia/Kolkata")
                            if broker_last.tzinfo is None:
                                broker_last = broker_last.tz_localize("Asia/Kolkata")
                            else:
                                broker_last = broker_last.tz_convert("Asia/Kolkata")
                            persist_needed = (
                                broker_last > existing_last + pd.Timedelta(seconds=30)
                                or len(broker_frame) > len(existing)
                            )
                    except Exception:
                        persist_needed = True
                    if persist_needed:
                        try:
                            persist_1m_frame(
                                root,
                                broker_frame,
                                source="INDstocks API live 1m",
                                sync_drive=True,
                            )
                        except Exception:
                            # A failed archive write must not discard good live model input.
                            pass
                    return broker_frame, "INDSTOCKS_API"
        except Exception:
            pass

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
            # Persist authenticated broker candles when INDstocks is the
            # selected fallback, preserving daily partitions and Drive copies.
            if source == "INDSTOCKS_API":
                try:
                    persist_1m_frame(
                        root,
                        fallback,
                        source="INDstocks API historical 1m",
                        sync_drive=True,
                    )
                except Exception:
                    # Persistence failure must not invalidate a usable live frame.
                    pass
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


@st.cache_data(ttl=180, show_spinner=False)
def _cached_watch_delivery_prediction(symbol: str, refresh_revision: int) -> dict:
    """Delivery result for one internet-sourced stock; cache briefly between dashboard ticks."""
    try:
        result = dict(beta_delivery_predict(symbol, None, None, None))
        if str(result.get("prediction_status", "")).upper() == "PASS":
            now = pd.Timestamp.now(tz="Asia/Kolkata")
            expected_reference = latest_completed_nse_session(now.to_pydatetime())
            observed_reference = parse_reference_date(result.get("date"))
            if expected_reference is not None and observed_reference != expected_reference:
                result["prediction_status"] = "STALE"
                result["signal"] = "STALE"
                result["signal_reason"] = (
                    f"Reference snapshot {observed_reference or 'missing'}; "
                    f"expected latest completed NSE session {expected_reference}."
                )
            else:
                try:
                    from production_data_freshness_gate import HOLIDAYS as _delivery_holidays
                    known_delivery_holidays = set(_delivery_holidays)
                except Exception:
                    known_delivery_holidays = set()
                if target_session_is_underway(
                    result.get("target_session_date"),
                    now,
                    known_delivery_holidays,
                ):
                    # Preserve what the previous-close model actually forecast.
                    # The machine status remains IN_PROGRESS so clients cannot
                    # mistake it for a newly calculated intraday signal.
                    result["prior_close_forecast_signal"] = result.get("signal")
                    result["prior_close_probability_up"] = result.get("p")
                    result["target_session_underway"] = True
                    result["prediction_status"] = "IN_PROGRESS"
                    result["signal"] = "IN_PROGRESS"
                    result["signal_reason"] = (
                        "TARGET_SESSION_UNDERWAY: showing the prior-close Delivery forecast "
                        "for this active session. It is not recalculated from current-session "
                        "price action; use the separate intraday forecast for live decisions."
                    )
        return result
    except Exception as exc:
        return {
            "prediction_status": "ERROR",
            "signal": "UNAVAILABLE",
            "reason": f"{type(exc).__name__}: {exc}",
        }



@st.cache_data(ttl=3600, show_spinner=False)
def _cached_daily_ohlc_history(symbol: str, refresh_revision: int = 0) -> pd.DataFrame:
    """Load daily OHLCV history for one selected stock, never for the whole watchlist."""
    root = str(symbol).replace(".NS", "").replace(".BO", "").strip().upper()
    period = "450d"
    try:
        frame = get_historical(
            f"{root}.NS",
            period=period,
            interval="1d",
            force_refresh=bool(refresh_revision),
        )
        if frame is not None and not frame.empty and len(frame) >= 160:
            return frame
    except Exception:
        pass

    now = pd.Timestamp.now(tz="Asia/Kolkata")
    start = (now - pd.Timedelta(days=450)).date()
    end = now.date()
    try:
        frame = nse_fetch_equity_history(root, start, end, sync_drive=False)
        if frame is not None and not frame.empty:
            return frame
    except Exception:
        pass

    try:
        frame = fetch_nse_drive_bulk_symbol_history(
            root,
            start,
            end,
            max_calendar_days=240,
        )
        if frame is not None and not frame.empty:
            return frame
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_7day_ohlc_forecast(symbol: str, refresh_revision: int = 0) -> dict:
    """Fit/evaluate the selected symbol's direct daily OHLC horizon models once per cache key."""
    frame = _cached_daily_ohlc_history(symbol, refresh_revision)
    return forecast_next_sessions(frame, n_sessions=7)


def _next_nse_session_dates(as_of_date, count: int = 7) -> list:
    """Return future weekdays excluding the project's known NSE holiday dates."""
    try:
        from production_data_freshness_gate import HOLIDAYS
        holidays = set()
        for item in HOLIDAYS:
            try:
                holidays.add(pd.Timestamp(item).date())
            except Exception:
                continue
    except Exception:
        holidays = set()

    cursor = pd.Timestamp(as_of_date).date()
    out = []
    while len(out) < count:
        cursor += pd.Timedelta(days=1)
        if cursor.weekday() < 5 and cursor not in holidays:
            out.append(cursor)
    return out


def _auto_watch_intraday(symbol: str) -> dict:
    """Use only fresh, completed one-minute candles for intraday watchlist forecasts."""
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    try:
        raw, source = _live_5m_price_frame(symbol)
    except Exception as exc:
        return {"status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"}
    if raw is None or raw.empty:
        return {"status": "NO_LIVE_DATA", "reason": "No live one-minute candles returned."}

    frame = raw.copy()
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
        valid = ~idx.isna()
        frame = frame.loc[valid].copy()
        idx = idx[valid]
        idx = idx.tz_localize("Asia/Kolkata") if idx.tz is None else idx.tz_convert("Asia/Kolkata")
        frame.index = idx
        frame = frame.sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
    except Exception as exc:
        return {"status": "INVALID_TIMESTAMPS", "reason": f"{type(exc).__name__}: {exc}"}

    # Never call a delayed archive a live feed. A completed candle older than
    # five minutes is stale during regular trading hours and cannot issue a signal.
    completed = frame.loc[frame.index < now.floor("min")].copy()
    if completed.empty:
        return {"status": "WAITING_FOR_BAR", "reason": "Waiting for first completed one-minute candle."}
    latest_ts = pd.Timestamp(completed.index[-1])
    age_seconds = max(0.0, (now - latest_ts).total_seconds())
    if age_seconds > 5 * 60:
        return {
            "status": "STALE",
            "reason": f"Latest candle is {age_seconds / 60:.1f} minutes old.",
            "latest_candle_ist": latest_ts.isoformat(),
            "source": source,
        }
    if len(completed) < 80:
        return {
            "status": "WARMING_UP",
            "reason": f"{len(completed)}/80 completed one-minute rows available.",
            "latest_candle_ist": latest_ts.isoformat(),
            "source": source,
        }

    root = symbol.replace(".NS", "").replace(".BO", "").upper()
    state_key = f"auto_watch_intraday::{root}"

    try:
        benchmark = benchmark_for_symbol(symbol)
        market_frame = _cached_live_market_context(benchmark)
        market_freshness = latest_completed_bar_status(
            market_frame,
            now,
            max_age_seconds=5 * 60,
            require_current_session=True,
        )
        if not market_freshness.get("ok"):
            return {
                "status": "STALE_MARKET_CONTEXT",
                "reason": (
                    f"{benchmark} benchmark context is not fresh enough: "
                    f"{market_freshness.get('reason', market_freshness.get('status'))}"
                ),
                "latest_candle_ist": latest_ts.isoformat(),
                "market_latest_candle_ist": market_freshness.get("latest_candle_ist"),
                "source": source,
            }

        market_latest_ts = market_freshness.get("latest_candle_ist")
        refresh_revision = int(st.session_state.get("watchlist_refresh_revision", 0))
        cached = st.session_state.get(state_key)
        if (
            isinstance(cached, dict)
            and cached.get("latest_candle_ist") == latest_ts.isoformat()
            and cached.get("market_latest_candle_ist") == market_latest_ts
            and cached.get("refresh_revision") == refresh_revision
        ):
            return cached

        predictions = beta_intraday_predict(symbol, completed, market_frame)
        by_horizon = {
            int(item.get("horizon_min", 0)): item
            for item in predictions
            if int(item.get("horizon_min", 0)) in {5, 15, 30}
        }
        completeness = assess_horizon_completeness(by_horizon, (5, 15, 30))
        out = {
            "status": completeness["status"],
            "reason": completeness.get("reason"),
            "latest_candle_ist": latest_ts.isoformat(),
            "market_latest_candle_ist": market_latest_ts,
            "source": source,
            "age_seconds": age_seconds,
            "refresh_revision": refresh_revision,
            "horizons": by_horizon,
        }
        st.session_state[state_key] = out
        return out
    except Exception as exc:
        return {
            "status": "ERROR",
            "reason": f"{type(exc).__name__}: {exc}",
            "latest_candle_ist": latest_ts.isoformat(),
            "source": source,
        }


if interface_mode == "Live Test":
    # A single-stock workflow modelled on modern broker terminals: compact market
    # status, searchable watchlist, and explicit Delivery/Intraday workspaces.
    st.markdown(
        """
        <style>
        .block-container {max-width: 1500px; padding-top: 1.25rem; padding-bottom: 2.5rem;}
        .trading-hero {border-radius: 18px; padding: 1.35rem 1.5rem; margin: 0.15rem 0 1rem;
            background: linear-gradient(115deg,#101d35 0%,#162b4d 56%,#1a4661 100%); color: #f8fafc;}
        .trading-eyebrow {font-size: .72rem; letter-spacing: .16em; text-transform: uppercase;
            color: #9dd7d7; font-weight: 750; margin-bottom: .35rem;}
        .trading-title {font-size: 2rem; line-height: 1.15; font-weight: 760; margin: 0;}
        .trading-subtitle {font-size: .92rem; color: #d2dfed; margin-top: .55rem;}
        div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e2e8f0;
            border-radius: 13px; padding: .8rem 1rem; box-shadow: 0 2px 8px rgba(15,23,42,.035);}
        div[data-testid="stMetricLabel"] {color: #64748b; font-weight: 600;}
        div[data-testid="stMetricValue"] {color: #10213d; font-weight: 750;}
        div.stButton > button {border-radius: 10px; min-height: 2.55rem; font-weight: 650;}
        div[data-testid="stRadio"] > div {gap: .45rem;}
        div[data-testid="stRadio"] label {border: 1px solid #dbe4ee; padding: .35rem .8rem;
            border-radius: 10px; background: #fff;}
        </style>
        <div class="trading-hero">
          <div class="trading-eyebrow">NSE · Market intelligence</div>
          <div class="trading-title">Indian Market <span style="color:#8be0d3">Trading Desk</span></div>
          <div class="trading-subtitle">One selected stock at a time · seven-session Delivery outlook · minute-level Intraday signals</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    try:
        from production_data_freshness_gate import HOLIDAYS as _ui_holidays
        known_holidays = {pd.Timestamp(x).date() for x in _ui_holidays}
    except Exception:
        known_holidays = set()
    market_open = (
        now_ist.weekday() < 5
        and now_ist.date() not in known_holidays
        and pd.Timestamp("09:15").time() <= now_ist.time() < pd.Timestamp("15:30").time()
    )
    target_session = resolve_watch_target_session(now_ist.to_pydatetime(), known_holidays)

    refresh_col, market_col, session_col, time_col = st.columns([1.35, 1, 1.15, 1])
    with refresh_col:
        refresh_pressed = st.button(
            "↻ Refresh market data",
            type="primary",
            width="stretch",
            help="Refresh the shortlist and selected-stock inputs only. It does not clear unrelated application caches.",
        )
    if refresh_pressed:
        st.session_state["trading_refresh_revision"] = (
            int(st.session_state.get("trading_refresh_revision", 0)) + 1
        )
        for cache_owner in (
            _today_stock_watchlist,
            _cached_watch_delivery_prediction,
            _cached_daily_ohlc_history,
            _cached_7day_ohlc_forecast,
            _live_5m_price_frame,
            _cached_live_market_context,
        ):
            try:
                cache_owner.clear()
            except Exception:
                pass
        st.rerun()

    refresh_revision = int(st.session_state.get("trading_refresh_revision", 0))
    watch_data = _today_stock_watchlist()
    stocks = watch_data.get("stocks", []) or []
    sources = watch_data.get("sources", []) or []
    with market_col:
        st.metric("NSE market", "OPEN" if market_open else "CLOSED")
    with session_col:
        st.metric("Next watch session", target_session.strftime("%d %b %Y"))
    with time_col:
        st.metric("IST time", now_ist.strftime("%H:%M"))

    # The universe and watchlist are cached, but no model is run while the user
    # is only viewing the shortlist. Forecasts are computed for the chosen symbol.
    try:
        universe = _searchable_nse_universe()
    except Exception:
        universe = pd.DataFrame()

    company_by_symbol: dict[str, str] = {}
    if universe is not None and not universe.empty:
        company_by_symbol = {
            str(row.symbol).strip().upper(): str(row.company).strip()
            for row in universe.itertuples(index=False)
            if str(getattr(row, "symbol", "")).strip()
        }
    watch_symbols: list[str] = []
    for stock in stocks:
        root = str(stock.get("symbol") or "").strip().upper().replace(".NS", "")
        if root and root not in watch_symbols:
            watch_symbols.append(root)
    all_symbols = list(dict.fromkeys(watch_symbols + list(company_by_symbol)))
    if not all_symbols:
        all_symbols = watch_symbols

    def _stock_label(root: str) -> str:
        company = company_by_symbol.get(root, "")
        return f"{root} — {company}" if company and company.lower() != "nan" else root

    labels = [_stock_label(root) for root in all_symbols]
    default_root = (
        str(st.session_state.get("trading_selected_root", "")).strip().upper()
        or (watch_symbols[0] if watch_symbols else (all_symbols[0] if all_symbols else ""))
    )
    default_label = _stock_label(default_root) if default_root in all_symbols else (labels[0] if labels else None)

    left, right = st.columns([2.5, 1])
    with left:
        if labels:
            selected_label = st.selectbox(
                "Selected stock",
                labels,
                index=labels.index(default_label) if default_label in labels else 0,
                key="trading_stock_selector",
                help="Type a symbol or company name to search. Only this stock is forecast in Delivery or Intraday.",
            )
            selected_root = selected_label.split(" — ", 1)[0].strip().upper()
            st.session_state["trading_selected_root"] = selected_root
            selected_symbol = f"{selected_root}.NS"
        else:
            selected_root = ""
            selected_symbol = ""
            st.warning("The NSE symbol list and source-backed watchlist are unavailable. Retry Refresh market data.")
    with right:
        st.caption(
            f"News scan: {int(watch_data.get('article_count', 0))} articles · "
            f"{sum(1 for item in sources if item.get('status') == 'PASS')}/{len(sources)} publishers healthy"
        )

    workspace = st.radio(
        "Workspace",
        ["Market overview", "Delivery · 7 sessions", "Intraday · 1 minute"],
        horizontal=True,
        label_visibility="collapsed",
        key="trading_workspace",
    )

    if workspace == "Market overview":
        st.subheader("Today's watchlist")
        st.caption(
            "News is used to discover stocks, not to create a buy/sell signal. "
            "Select a row above, then open Delivery or Intraday when you need that stock's forecast."
        )
        if stocks:
            watch_rows = []
            for item in stocks:
                root = str(item.get("symbol") or "").strip().upper()
                if not root:
                    continue
                watch_rows.append({
                    "Stock": root,
                    "Company": company_by_symbol.get(root, root),
                    "Publishers": int(item.get("source_count", 0)),
                    "Mentions": int(item.get("mention_count", 0)),
                    "Coverage": "News watchlist",
                })
            st.dataframe(
                pd.DataFrame(watch_rows),
                width="stretch",
                hide_index=True,
                height=min(340, 46 + 38 * max(1, len(watch_rows))),
            )
            selected_watch = next(
                (item for item in stocks if str(item.get("symbol") or "").strip().upper() == selected_root),
                None,
            )
            headlines = (selected_watch or {}).get("headlines", []) or []
            if headlines:
                with st.expander(f"Recent headlines for {selected_root}", expanded=False):
                    for article in headlines[:5]:
                        title = str(article.get("title", "Market article")).replace("[", "(").replace("]", ")")
                        url = str(article.get("url", ""))
                        publisher = str(article.get("publisher", article.get("domain", "Publisher")))
                        published = str(article.get("published_utc", ""))[:16].replace("T", " ")
                        if url.startswith(("https://", "http://")):
                            st.markdown(f"[{title}]({url})")
                        else:
                            st.write(title)
                        st.caption(f"{publisher} · {published} UTC")
        else:
            st.info(
                watch_data.get("message")
                or "No recent source-backed stocks are available. No default picks will be substituted."
            )
        with st.expander("Data source health and methodology", expanded=False):
            st.dataframe(pd.DataFrame(sources), width="stretch", hide_index=True)
            st.caption(
                "Watchlist refreshes when requested and is cached for 15 minutes. "
                "Model inference runs only in the selected Delivery or Intraday workspace."
            )

    elif workspace == "Delivery · 7 sessions":
        st.subheader(f"{selected_root} · Seven-session OHLC outlook" if selected_root else "Seven-session OHLC outlook")
        st.caption(
            "Daily opening, high, low and closing estimates for the next seven expected NSE sessions. "
            "The OHLC path is a horizon-conditioned model trained on historical daily bars and checked "
            "against a chronological holdout. Validation metrics are shown; a forecast is not a guarantee."
        )
        if not selected_symbol:
            st.stop()

        with st.spinner(f"Loading daily history and evaluating the seven-session OHLC model for {selected_root}…"):
            try:
                ohlc_forecast = _cached_7day_ohlc_forecast(selected_symbol, refresh_revision)
            except Exception as exc:
                ohlc_forecast = {
                    "status": "ERROR",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "forecasts": [],
                }
            try:
                delivery = _cached_watch_delivery_prediction(selected_symbol, refresh_revision)
            except Exception as exc:
                delivery = {
                    "prediction_status": "ERROR",
                    "signal": "UNAVAILABLE",
                    "reason": f"{type(exc).__name__}: {exc}",
                }

        forecast_status = str(ohlc_forecast.get("status", "UNKNOWN")).upper()
        as_of = ohlc_forecast.get("as_of_date")
        expected_reference = latest_completed_nse_session(now_ist.to_pydatetime())
        if forecast_status == "PASS" and expected_reference is not None:
            if str(as_of) != str(expected_reference):
                st.warning(
                    f"Daily history is stale: last close {as_of or 'missing'}, "
                    f"latest completed NSE session expected {expected_reference}. "
                    "The seven-session OHLC projections are withheld until history refreshes."
                )
                forecast_status = "STALE_HISTORY"

        direction_signal = str(
            delivery.get("prior_close_forecast_signal")
            or delivery.get("signal")
            or delivery.get("prediction_status")
            or "UNAVAILABLE"
        )
        p_up = pd.to_numeric(
            delivery.get("prior_close_probability_up", delivery.get("p")),
            errors="coerce",
        )
        active_coverage = pd.to_numeric(delivery.get("feature_coverage"), errors="coerce")
        predicted_delivery_close = pd.to_numeric(delivery.get("predicted_price"), errors="coerce")
        delivery_return_status = str(delivery.get("price_forecast_status") or "RETRAIN_REQUIRED")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Beta V2 direction", direction_signal)
        m2.metric("Probability up", f"{float(p_up):.1%}" if pd.notna(p_up) else "—")
        m3.metric("Active-model coverage", f"{float(active_coverage):.1%}" if pd.notna(active_coverage) else "—")
        m4.metric(
            "Beta V2 next close",
            f"₹{float(predicted_delivery_close):,.2f}" if pd.notna(predicted_delivery_close) else "Not available",
        )
        st.caption(
            f"Beta V2 status: {delivery.get('prediction_status', 'UNKNOWN')} · "
            f"reference session: {delivery.get('date') or '—'} · "
            f"return-head status: {delivery_return_status}"
        )
        if delivery.get("signal_reason") or delivery.get("reason"):
            st.caption(str(delivery.get("signal_reason") or delivery.get("reason")))

        if forecast_status != "PASS":
            st.error(
                "Seven-session OHLC forecast is unavailable. "
                + str(ohlc_forecast.get("reason") or f"Forecast status: {forecast_status}")
            )
        else:
            validation = ohlc_forecast.get("validation", []) or []
            overall_skill = pd.to_numeric(
                ohlc_forecast.get("overall_mean_mae_skill_vs_median"),
                errors="coerce",
            )
            v1, v2, v3 = st.columns(3)
            v1.metric("History through", str(as_of or "—"))
            v2.metric("Chronological holdout", f"{ohlc_forecast.get('validation_start', '—')} → {ohlc_forecast.get('validation_end', '—')}")
            v3.metric("Mean MAE skill vs baseline", f"{float(overall_skill):+.1%}" if pd.notna(overall_skill) else "Unavailable")
            if pd.isna(overall_skill) or float(overall_skill) <= 0:
                st.warning(
                    "LOW VALIDATION SKILL: the OHLC path did not beat the median-return baseline overall. "
                    "Values are shown for transparency, but they are not qualified as trading-grade forecasts."
                )
            else:
                st.info(
                    "Positive average holdout skill means the model beat the median baseline on average in this historical split; "
                    "it does not establish future profitability."
                )

            session_dates = _next_nse_session_dates(as_of, 7)
            rows = []
            chart_rows = []
            for index, prediction in enumerate(ohlc_forecast.get("forecasts", []) or []):
                if index >= len(session_dates):
                    break
                session_date = session_dates[index]
                skill = pd.to_numeric(prediction.get("mean_mae_skill_vs_median"), errors="coerce")
                row = {
                    "Session": f"Day +{prediction.get('horizon', index + 1)}",
                    "Date": session_date.strftime("%a, %d %b"),
                    "Open forecast": f"₹{float(prediction['open']):,.2f}",
                    "High forecast": f"₹{float(prediction['high']):,.2f}",
                    "Low forecast": f"₹{float(prediction['low']):,.2f}",
                    "Close forecast": f"₹{float(prediction['close']):,.2f}",
                    "MAE skill vs baseline": f"{float(skill):+.1%}" if pd.notna(skill) else "—",
                    "Validation": str(prediction.get("validation_status", "UNKNOWN")).replace("_", " "),
                }
                rows.append(row)
                chart_rows.append({"Session": pd.Timestamp(session_date), "Predicted close": float(prediction["close"])})

            st.dataframe(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True,
                height=330,
            )
            if chart_rows:
                chart = pd.DataFrame(chart_rows).set_index("Session")
                st.line_chart(chart, height=220, y_label="Projected close (₹)")
            st.caption(
                f"Model: {ohlc_forecast.get('model', '—')} · "
                f"training samples: {ohlc_forecast.get('training_samples', '—')} · "
                f"holdout samples: {ohlc_forecast.get('validation_samples', '—')} · "
                f"purge: {ohlc_forecast.get('purge_sessions', 7)} sessions."
            )
            with st.expander("How to read this forecast", expanded=False):
                st.write(str(ohlc_forecast.get("note", "")))
                st.write(
                    "Opening, high, low and closing values are direct model estimates anchored on the latest observed close. "
                    "Each session has a separate holdout skill check against a training-median baseline. "
                    "The table is an OHLC forecast, not a broker signal and not a promise of a particular price."
                )

    else:
        st.subheader(f"{selected_root} · Intraday signal desk" if selected_root else "Intraday signal desk")
        st.caption(
            "Only the selected symbol is recalculated. In-session updates target each new completed one-minute candle; "
            "stale stock or benchmark input is withheld rather than guessed."
        )
        auto_refresh = st.checkbox(
            "Auto-refresh this stock every 60 seconds",
            value=True,
            key=f"intraday_auto_refresh::{selected_root or 'none'}",
            help="Auto-refresh runs only in this Intraday workspace and only during regular NSE market hours.",
        )
        if not selected_symbol:
            st.stop()
        if not market_open:
            st.info("Market is closed. Intraday inference will resume during the regular NSE session; no all-stock scans are running.")
        else:
            @st.fragment(
                run_every="60s" if auto_refresh and market_open else None,
                key="selected_intraday_prediction_panel",
            )
            def _render_selected_intraday_panel():
                now = pd.Timestamp.now(tz="Asia/Kolkata")
                try:
                    raw, source = _live_5m_price_frame(selected_symbol)
                except Exception as exc:
                    st.error(f"Market-data request failed: {type(exc).__name__}: {exc}")
                    return
                if raw is None or raw.empty:
                    st.error("No live one-minute candles are available from the configured providers.")
                    return

                frame = raw.copy()
                frame.columns = [str(col).strip().lower() for col in frame.columns]
                aliases = {"ltp": "close", "adj close": "close", "adj_close": "close", "vol": "volume"}
                frame = frame.rename(columns=aliases)
                try:
                    idx = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
                    valid = ~idx.isna()
                    frame = frame.loc[valid].copy()
                    idx = idx[valid]
                    idx = idx.tz_localize("Asia/Kolkata") if idx.tz is None else idx.tz_convert("Asia/Kolkata")
                    frame.index = idx
                    frame = frame.sort_index()
                    frame = frame[~frame.index.duplicated(keep="last")]
                except Exception as exc:
                    st.error(f"Invalid candle timestamps: {type(exc).__name__}: {exc}")
                    return

                completed = frame.loc[frame.index < now.floor("min")].copy()
                if completed.empty:
                    st.info("Waiting for the first completed one-minute candle.")
                    return
                latest_candle = pd.Timestamp(completed.index[-1])
                age_seconds = max(0.0, (now - latest_candle).total_seconds())
                if latest_candle.date() != now.date() or age_seconds > 300:
                    st.warning(
                        f"Intraday inference withheld: newest completed candle is {latest_candle.strftime('%H:%M:%S')} IST "
                        f"({age_seconds / 60:.1f} minutes old). A current-session candle under five minutes old is required."
                    )
                    return
                if len(completed) < 80:
                    st.info(f"Model warm-up: {len(completed)}/80 completed one-minute candles available.")
                    return

                benchmark = benchmark_for_symbol(selected_symbol)
                try:
                    market_frame = _cached_live_market_context(benchmark)
                    market_freshness = latest_completed_bar_status(
                        market_frame,
                        now,
                        max_age_seconds=300,
                        require_current_session=True,
                    )
                except Exception as exc:
                    st.error(f"Benchmark validation failed: {type(exc).__name__}: {exc}")
                    return
                if not market_freshness.get("ok"):
                    st.warning(
                        f"Intraday inference withheld because {benchmark} context is not fresh: "
                        f"{market_freshness.get('reason') or market_freshness.get('status')}."
                    )
                    return

                market_latest = str(market_freshness.get("latest_candle_ist"))
                prediction_key = f"intraday_predictions::{selected_root}::{refresh_revision}"
                cached_result = st.session_state.get(prediction_key, {})
                if (
                    cached_result.get("latest_candle") == latest_candle.isoformat()
                    and cached_result.get("market_latest") == market_latest
                ):
                    predictions = cached_result.get("predictions", [])
                    recalculated = False
                else:
                    try:
                        predictions = beta_intraday_predict(
                            selected_symbol,
                            completed,
                            market_frame,
                            decision_interval_minutes=1,
                        )
                    except Exception as exc:
                        st.error(f"Intraday model failed: {type(exc).__name__}: {exc}")
                        return
                    st.session_state[prediction_key] = {
                        "latest_candle": latest_candle.isoformat(),
                        "market_latest": market_latest,
                        "predictions": predictions,
                    }
                    recalculated = True

                if not predictions:
                    st.warning("The model returned no eligible horizon predictions.")
                    return

                source_label = {
                    "NSE_OFFICIAL_CHART": "NSE official",
                    "PERSISTENT_LOCAL_CACHE": "Persistent cache",
                    "INDSTOCKS_API": "INDstocks API",
                    "DHAN": "Dhan",
                    "YAHOO_FINANCE": "Yahoo Finance",
                    "YFINANCE": "Yahoo Finance",
                }.get(source, source)
                p1, p2, p3, p4 = st.columns(4)
                first = predictions[0]
                reference0 = pd.to_numeric(first.get("reference_close"), errors="coerce")
                predicted0 = pd.to_numeric(first.get("predicted_price"), errors="coerce")
                move0 = float(predicted0) - float(reference0) if pd.notna(predicted0) and pd.notna(reference0) else np.nan
                p1.metric("Reference price", f"₹{float(reference0):,.2f}" if pd.notna(reference0) else "—")
                p2.metric("Next +5m signal", str(first.get("signal") or "ABSTAIN"))
                p3.metric("Predicted +5m price", f"₹{float(predicted0):,.2f}" if pd.notna(predicted0) else "Unavailable",
                          delta=f"₹{move0:+,.2f}" if pd.notna(move0) else None)
                p4.metric("Latest 1m candle", latest_candle.strftime("%H:%M:%S"))

                st.caption(
                    f"{'Recalculated' if recalculated else 'Unchanged for latest candle'} · "
                    f"source: {source_label} · benchmark: {benchmark} · "
                    f"benchmark candle: {market_latest} · input age: {age_seconds:.0f}s"
                )

                rows = []
                for item in predictions:
                    prob = pd.to_numeric(item.get("probability_up"), errors="coerce")
                    ref_price = pd.to_numeric(item.get("reference_close"), errors="coerce")
                    price = pd.to_numeric(item.get("predicted_price"), errors="coerce")
                    expected_return = pd.to_numeric(item.get("expected_return"), errors="coerce")
                    price_status = str(item.get("price_forecast_status") or "RETRAIN_REQUIRED")
                    move = float(price) - float(ref_price) if pd.notna(price) and pd.notna(ref_price) else np.nan
                    coverage = pd.to_numeric(item.get("feature_coverage"), errors="coerce")
                    rows.append({
                        "Horizon": f"+{int(item.get('horizon_min', 0))} min",
                        "Signal": item.get("signal", "ABSTAIN"),
                        "P(UP)": f"{float(prob):.1%}" if pd.notna(prob) else "—",
                        "Reference price": f"₹{float(ref_price):,.2f}" if pd.notna(ref_price) else "—",
                        "Predicted price": f"₹{float(price):,.2f}" if pd.notna(price) else "—",
                        "Expected move": f"₹{move:+,.2f}" if pd.notna(move) else "—",
                        "Feature coverage": f"{float(coverage):.1%}" if pd.notna(coverage) else "—",
                        "Price-head status": price_status,
                        "Reason": str(item.get("signal_reason") or item.get("reason") or "—"),
                    })
                st.subheader("Multi-horizon forecast")
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                weak_heads = [
                    str(item.get("price_forecast_status") or "RETRAIN_REQUIRED")
                    for item in predictions
                    if pd.isna(pd.to_numeric(item.get("predicted_price"), errors="coerce"))
                    or str(item.get("price_forecast_status") or "") == "MAE_NOT_BETTER_THAN_MEDIAN_BASELINE"
                ]
                if weak_heads:
                    st.warning(
                        "Some price horizons are unavailable or below the median-baseline validation gate. "
                        "Read the per-horizon Reason and Price-head status columns; do not substitute probability for price."
                    )
                with st.expander("Model diagnostics", expanded=False):
                    st.json({
                        "stock_source": source_label,
                        "latest_completed_candle_ist": latest_candle.isoformat(),
                        "benchmark": benchmark,
                        "benchmark_freshness": market_freshness,
                        "horizons": predictions,
                    })

            _render_selected_intraday_panel()

    st.caption(
        "Decision-support only · no orders are placed · the model may abstain when data or validation is inadequate."
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
live_records = []
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

