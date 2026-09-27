"""Indian-market stock signal dashboard: delivery, intraday and F&O context.

The dashboard produces decision-support signals only. It does not place orders.
"""
from __future__ import annotations

import pandas as pd
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
from production_model import build_feature_rows, predict as production_predict, model_status as production_model_status

st.set_page_config(page_title="Indian Stock Signal Dashboard", layout="wide")
st.title("Indian Stock Signal Dashboard")
st.caption(
    "Rule-based decision support for NSE/BSE cash equity with F&O context. "
    "Signals are not guarantees and no orders are placed. In cash-equity mode, SELL means exit/avoid a long position; the backtest is long-only."
)

st.info(
    "Production ML model: CONNECTED for Intraday mode. It predicts the next 30-minute direction using the certified 137-feature production contract. "
    "Delivery mode continues to use the existing rule-based engine because the certified model target is 30-minute direction."
)
with st.expander("Step 52 PIT walk-forward audit", expanded=False):
    st.write(
        f"Frozen pilot: {', '.join(STEP52_SYMBOLS)} · {STEP52_JOINED_ROWS:,} joined rows · "
        f"{STEP52_PREDICTORS} predictors · latest holdout date {STEP52_LAST_TEST_DATE}."
    )
    st.dataframe(pd.DataFrame(STEP52_AUDIT), use_container_width=True, hide_index=True)
    st.caption(
        "The Step 52 model has not been connected to live signals. Its out-of-sample results do not establish a tradable edge. "
        "No order execution is available in this dashboard."
    )

INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}

default_watchlist = "BHARTIARTL.NS, RELIANCE.NS, HDFCBANK.NS, TCS.NS, INFY.NS"
st.sidebar.header("Watchlist")
raw_symbols = st.sidebar.text_area(
    "NSE/BSE symbols (suffix optional)", default_watchlist,
    help="Example: BHARTIARTL, RELIANCE.NS, TCS.BO. Bare symbols default to NSE (.NS)."
)
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

style = st.sidebar.radio("Mode", ["Delivery", "Intraday"])
if style == "Delivery":
    interval, period, fast, slow = "1d", "1y", 20, 50
    stop_atr, target_atr, max_hold = 1.8, 3.0, 20
else:
    interval, period, fast, slow = "5m", "5d", 9, 21
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

status = "Market OPEN" if market_is_open() else "Market CLOSED (last available data)"
st.sidebar.info(status)

# --------------------------------------------------------------- DhanHQ ---
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
        st.dataframe(st.session_state["dhan_download_results"], use_container_width=True)

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
            st.dataframe(records_all[display_cols], use_container_width=True)
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
        st.dataframe(stocks_all[stock_display], use_container_width=True)

        bucket = stocks_all.groupby("size_bucket", dropna=False).agg(
            stocks=("symbol", "count"),
            signals=("actionable", "sum"),
            hit_rate_5d=("hit_rate_5d_pct", "mean"),
            edge_5d=("avg_signed_edge_5d_pct", "mean"),
        ).reset_index()
        st.markdown("### Results by universe bucket")
        st.dataframe(bucket.round(3), use_container_width=True)

    if sens_all is not None and not sens_all.empty:
        st.markdown("### Signal-threshold sensitivity")
        st.dataframe(sens_all.round(3), use_container_width=True)
        st.line_chart(sens_all.set_index("threshold")[["hit_rate_pct", "avg_signed_edge_pct"]])

    st.info(
        "This audit deliberately does not inject today's RSS news into historical dates. "
        "It therefore validates the causal price + market model, not the full live-news model. "
        "Use the results to decide what to test next, not as a guarantee of future returns."
    )

# --------------------------------------------------------- free data manager
st.sidebar.subheader("Free Historical Data")
st.sidebar.caption("Downloads public research archives to this laptop. Live quotes are still fetched separately.")
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

with st.expander("Free historical-data manager & authenticity check", expanded=False):
    st.write(
        "Public research archives are treated as third-party inputs, not as exchange-authoritative truth. "
        "Use **Validate local data** for bar integrity, then **Reconcile to NSE** to compare daily aggregates "
        "against NSE's official historical equity endpoint. A failed NSE request is recorded as an error, not a pass."
    )
    st.markdown("### Source provenance")
    st.dataframe(source_provenance_table(), use_container_width=True, hide_index=True)
    if "source_check" in st.session_state:
        st.markdown("### Online source availability")
        st.dataframe(st.session_state["source_check"], use_container_width=True, hide_index=True)
        st.download_button("Export source-availability check", st.session_state["source_check"].to_csv(index=False).encode("utf-8"), file_name="source_availability_check.csv", mime="text/csv")
    if "coverage" in st.session_state and not st.session_state["coverage"].empty:
        st.markdown("### Local coverage")
        st.dataframe(st.session_state["coverage"], use_container_width=True)
        st.download_button("Export local-data manifest", pd.DataFrame(st.session_state["coverage"]).to_csv(index=False).encode("utf-8"), file_name="local_historical_coverage.csv", mime="text/csv")
    else:
        st.info("Use 'Scan local archive' in the sidebar after your first download.")
    if "quality" in st.session_state and not st.session_state["quality"].empty:
        st.markdown("### Data-quality / NSE reconciliation")
        st.dataframe(st.session_state["quality"], use_container_width=True)
        st.download_button("Export data-quality report", st.session_state["quality"].to_csv(index=False).encode("utf-8"), file_name="historical_data_quality.csv", mime="text/csv")

st.sidebar.caption(
    "When a valid Dhan access token is configured, NSE cash historical/live quote requests prefer Dhan. "
    "Otherwise the dashboard falls back to the existing free sources. No order APIs are used."
)

@st.cache_data(ttl=60)
def load_news():
    return get_news()

@st.cache_data(ttl=60)
def load_history(symbol, period, interval):
    return get_historical(symbol, period=period, interval=interval)

@st.cache_data(ttl=60)
def load_benchmark(symbol, period, interval):
    return get_historical(symbol, period=period, interval=interval)

@st.cache_data(ttl=90)
def load_option_chain(nse_symbol, is_index):
    return get_option_chain(nse_symbol, is_index=is_index)

@st.cache_data(ttl=180)
def load_announcements(symbol):
    return get_corporate_announcements(symbol)

news_items = tag_news_items(load_news())

# ---------------------------------------------------------------- Production ML
# Fetch all watchlist inputs first so the model can compute cross-sectional
# ranks/breadth across the user's actual watchlist rather than one symbol at a time.
live_records = []
for symbol in symbols:
    raw_df = load_history(symbol, period, interval)
    if raw_df.empty or len(raw_df) < slow + 15:
        continue
    company_root = symbol.replace(".NS", "").replace(".BO", "")
    benchmark_symbol = benchmark_for_symbol(symbol)
    market_df = load_benchmark(benchmark_symbol, period, interval)
    df = add_indicators(raw_df, fast=fast, slow=slow)
    mkt_df = add_indicators(market_df, fast=fast, slow=slow) if not market_df.empty else market_df
    aliases = stock_news_keywords(symbol)
    matched_news = news_for_symbol(aliases, news_items)
    news_stat = news_metrics(matched_news)
    news_perception_label, news_freshness = news_perception(news_stat)
    announcements_raw = load_announcements(company_root)
    announcements = []
    for item in announcements_raw[:5]:
        if not isinstance(item, dict):
            continue
        announcements.append({
            "subject": item.get("subject") or item.get("desc") or item.get("attchmntText") or "Corporate announcement",
            "date": item.get("an_dt") or item.get("date") or item.get("sort_date") or "",
            "link": item.get("attchmntFile") or item.get("link") or "",
        })
    event_items = announcement_to_news(announcements)
    all_event_news = matched_news + event_items
    event_risk = event_risk_score(matched_news, announcements)
    live_news_score = news_metrics(all_event_news)["score"] if all_event_news else 0.0
    if company_root in INDEX_SYMBOLS:
        fo_symbol, is_index = company_root, True
    else:
        fo_symbol, is_index = company_root, False
    raw_chain = load_option_chain(fo_symbol, is_index)
    fo = fo_summarize(raw_chain) if raw_chain else {}
    live_records.append({
        "symbol": symbol, "raw_df": raw_df, "df": df, "market_df": market_df,
        "mkt_df": mkt_df, "benchmark_symbol": benchmark_symbol,
        "matched_news": matched_news, "news_stat": news_stat,
        "news_perception_label": news_perception_label, "news_freshness": news_freshness,
        "announcements": announcements, "event_risk": event_risk,
        "live_news_score": live_news_score, "fo": fo,
        "t_score": technical_score(df, fast=fast, slow=slow),
        "m_score": market_regime_score(mkt_df, fast=fast, slow=slow) if not mkt_df.empty else 0.0,
    })

production_results = {}
if style == "Intraday" and live_records:
    try:
        feature_items = [
            {
                "symbol": r["symbol"], "price_df": r["raw_df"], "market_df": r["market_df"],
                "matched_news": r["matched_news"], "announcements": r["announcements"],
                "event_risk": r["event_risk"], "fo": r["fo"],
            }
            for r in live_records
        ]
        feature_frame = build_feature_rows(feature_items)
        production_results = {
            x["symbol"]: x for x in production_predict(feature_frame)
        }
    except Exception as exc:
        st.error(f"Production ML engine could not produce a live prediction: {type(exc).__name__}: {exc}")
        production_results = {}

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

    label, combined = combined_signal(
        t_score, live_news_score, m_score, event_risk,
        tech_weight, news_weight, market_weight,
    )
    last = df.iloc[-1]
    latest_close = float(last["Close"])
    previous_close = float(df["Close"].iloc[-2])
    pct_change = (latest_close - previous_close) / previous_close * 100 if previous_close else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rule Signal", label, f"score {combined:+.2f}")
    c2.metric("Last", f"₹{latest_close:,.2f}", f"bar {pct_change:+.2f}%")
    c3.metric("Technical", f"{t_score:+.2f}")
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
            st.warning("Production ML prediction unavailable for this symbol; the rule-based signal remains available.")
    else:
        st.caption("Production ML is an intraday 30-minute model; Delivery mode remains rule-based.")

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
            st.dataframe(trades_df, use_container_width=True)

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
                st.dataframe(near_atm, use_container_width=True)

