"""One-year prediction audit for the Indian stock signal engine.

The audit replays the model causally across a diversified NSE universe:
* signal is formed at the close of day T using only data available through T
* simulated entry price is the next day's OPEN
* forward outcomes are measured from that entry across 1/3/5/10/20 trading days
* BUY correctness = positive forward return; SELL correctness = negative forward return
* HOLD signals are excluded from directional hit-rate metrics

Important limitation: this free build does not have a timestamped historical news archive,
so the one-year replay validates the price + market components of the model. It does not
pretend to backfill today's RSS news sentiment into historical dates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from signals import add_indicators, combined_signal


DEFAULT_AUDIT_UNIVERSE = [
    ("RELIANCE.NS", "large/liquid"),
    ("HDFCBANK.NS", "large/liquid"),
    ("ICICIBANK.NS", "large/liquid"),
    ("TCS.NS", "large/liquid"),
    ("INFY.NS", "large/liquid"),
    ("BHARTIARTL.NS", "large/liquid"),
    ("SBIN.NS", "large/liquid"),
    ("LT.NS", "large/liquid"),
    ("MARUTI.NS", "large/liquid"),
    ("SUNPHARMA.NS", "large/liquid"),
    ("BEL.NS", "mid/high-beta"),
    ("HAL.NS", "mid/high-beta"),
    ("TRENT.NS", "mid/high-beta"),
    ("JIOFIN.NS", "mid/high-beta"),
    ("IDFCFIRSTB.NS", "mid/high-beta"),
    ("SUZLON.NS", "smaller/high-volatility"),
    ("RVNL.NS", "smaller/high-volatility"),
    ("IRFC.NS", "smaller/high-volatility"),
    ("RPOWER.NS", "smaller/high-volatility"),
]

DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
DEFAULT_THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40)

_BANKING = {"HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BANKBARODA", "PNB"}


def benchmark_for_audit(symbol: str) -> str:
    root = symbol.upper().replace(".NS", "").replace(".BO", "")
    return "^NSEBANK" if root in _BANKING else "^NSEI"


@dataclass(frozen=True)
class AuditConfig:
    audit_days: int = 365
    fast: int = 20
    slow: int = 50
    rsi_period: int = 14
    tech_weight: float = 0.60
    news_weight: float = 0.20
    market_weight: float = 0.20
    signal_threshold: float = 0.25
    horizons: tuple[int, ...] = DEFAULT_HORIZONS


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in out.columns:
            out[c] = np.nan
    out = out[needed].apply(pd.to_numeric, errors="coerce")
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def _market_score_series(market_df: pd.DataFrame, fast: int, slow: int, rsi_period: int) -> pd.Series:
    if market_df is None or market_df.empty:
        return pd.Series(dtype=float)
    m = normalize_ohlcv(market_df)
    if m.empty:
        return pd.Series(dtype=float)
    m = add_indicators(m, fast=fast, slow=slow, rsi_period=rsi_period)
    scores = pd.Series(0.0, index=m.index, dtype=float)
    valid = m[f"SMA{slow}"].notna() & m[f"RSI{rsi_period}"].notna() & m["MACD_hist"].notna()
    for idx in m.index[valid]:
        row = m.loc[idx]
        score = 0.0
        score += 0.35 if row["Close"] > row[f"SMA{slow}"] else -0.35
        score += 0.30 if row[f"SMA{fast}"] > row[f"SMA{slow}"] else -0.30
        score += 0.20 if row["MACD_hist"] > 0 else -0.20
        rsi = float(row[f"RSI{rsi_period}"])
        if rsi > 55:
            score += 0.15
        elif rsi < 45:
            score -= 0.15
        scores.loc[idx] = max(-1.0, min(1.0, score))
    return scores


def _technical_score_from_row(row: pd.Series, fast: int, slow: int, rsi_period: int) -> float:
    cols = [f"SMA{fast}", f"SMA{slow}", f"RSI{rsi_period}", "VolumeRatio", "MACD_hist", "ATR14", "RollingHigh", "RollingLow", "TrendSlope"]
    if any(c not in row.index for c in cols):
        return 0.0
    if pd.isna(row[f"SMA{slow}"]) or pd.isna(row[f"RSI{rsi_period}"]):
        return 0.0

    trend = 0.5 if row[f"SMA{fast}"] > row[f"SMA{slow}"] else -0.5
    trend += 0.25 if row["MACD_hist"] > 0 else -0.25
    trend += 0.25 if row["TrendSlope"] > 0 else -0.25

    rsi = float(row[f"RSI{rsi_period}"])
    momentum = max(-1.0, min(1.0, (rsi - 50.0) / 20.0))
    if rsi > 72 or rsi < 28:
        momentum *= 0.35

    vr = float(row["VolumeRatio"]) if pd.notna(row["VolumeRatio"]) else 1.0
    volume = 0.0
    if vr >= 1.8:
        volume = 0.8 if momentum >= 0 else -0.8
    elif vr >= 1.3:
        volume = 0.4 if momentum >= 0 else -0.4

    breakout = 0.0
    if pd.notna(row["RollingHigh"]) and row["Close"] > row["RollingHigh"]:
        breakout = 1.0
    elif pd.notna(row["RollingLow"]) and row["Close"] < row["RollingLow"]:
        breakout = -1.0

    volatility = 0.0
    atr = float(row["ATR14"]) if pd.notna(row["ATR14"]) else 0.0
    if atr > 0 and pd.notna(row.get("Return1")):
        bar_atr = abs(float(row["Return1"])) * float(row["Close"]) / atr
        if bar_atr >= 2.0:
            volatility = -0.5

    return round(max(-1.0, min(1.0, 0.40 * trend + 0.25 * momentum + 0.15 * volume + 0.15 * breakout + 0.05 * volatility)), 4)


def _label_from_score(score: float, threshold: float) -> str:
    if score >= threshold:
        return "BUY"
    if score <= -threshold:
        return "SELL"
    return "HOLD"


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    """Spearman correlation without scipy (keeps the app self-contained)."""
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 3 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return 0.0
    xr = z.iloc[:, 0].rank(method="average")
    yr = z.iloc[:, 1].rank(method="average")
    value = xr.corr(yr)
    return 0.0 if pd.isna(value) else float(value)


def audit_symbol(
    symbol: str,
    price_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: AuditConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    cfg = config or AuditConfig()
    prices = normalize_ohlcv(price_df)
    market = normalize_ohlcv(market_df)
    if len(prices) < cfg.slow + max(cfg.horizons) + 20:
        return pd.DataFrame(), {"symbol": symbol, "status": "insufficient_history"}

    ind = add_indicators(prices, fast=cfg.fast, slow=cfg.slow, rsi_period=cfg.rsi_period)
    market_scores = _market_score_series(market, cfg.fast, cfg.slow, cfg.rsi_period)

    latest_date = ind.index[-1]
    start_date = latest_date - pd.Timedelta(days=cfg.audit_days)
    usable = max(cfg.slow, 26, 14) + 2
    start_i = max(usable, int(ind.index.searchsorted(start_date)))
    max_h = max(cfg.horizons)
    end_i = len(ind) - max_h - 1
    rows: list[dict] = []

    for i in range(start_i, end_i + 1):
        decision_date = ind.index[i]
        entry_i = i + 1
        entry = float(ind.iloc[entry_i]["Open"])
        if not np.isfinite(entry) or entry <= 0:
            continue

        tech = _technical_score_from_row(ind.iloc[i], cfg.fast, cfg.slow, cfg.rsi_period)
        mkt = float(market_scores.get(decision_date, 0.0))
        _, score = combined_signal(
            tech,
            0.0,
            mkt,
            event_risk=0.0,
            tech_weight=cfg.tech_weight,
            news_weight=cfg.news_weight,
            market_weight=cfg.market_weight,
        )
        label = _label_from_score(score, cfg.signal_threshold)

        rec = {
            "symbol": symbol,
            "decision_date": decision_date.date().isoformat(),
            "entry_date": ind.index[entry_i].date().isoformat(),
            "entry_price": entry,
            "technical_score": tech,
            "market_score": mkt,
            "combined_score": score,
            "signal": label,
            "size_bucket": None,
        }
        for h in cfg.horizons:
            j = entry_i + h - 1
            future_close = float(ind.iloc[j]["Close"])
            fwd = future_close / entry - 1.0
            rec[f"fwd_{h}d_pct"] = fwd * 100.0
            rec[f"correct_{h}d"] = bool((label == "BUY" and fwd > 0) or (label == "SELL" and fwd < 0)) if label != "HOLD" else None
            rec[f"signed_edge_{h}d_pct"] = fwd * 100.0 if label == "BUY" else (-fwd * 100.0 if label == "SELL" else np.nan)
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {"symbol": symbol, "status": "no_decisions"}

    summaries = {
        "symbol": symbol,
        "status": "ok",
        "start_date": out["decision_date"].iloc[0],
        "end_date": out["decision_date"].iloc[-1],
        "decision_bars": len(out),
        "buy_signals": int((out["signal"] == "BUY").sum()),
        "sell_signals": int((out["signal"] == "SELL").sum()),
        "hold_signals": int((out["signal"] == "HOLD").sum()),
    }
    for h in cfg.horizons:
        actionable = out[out["signal"] != "HOLD"]
        if actionable.empty:
            summaries[f"hit_rate_{h}d_pct"] = 0.0
            summaries[f"avg_signed_edge_{h}d_pct"] = 0.0
        else:
            summaries[f"hit_rate_{h}d_pct"] = round(float(actionable[f"correct_{h}d"].mean() * 100), 2)
            summaries[f"avg_signed_edge_{h}d_pct"] = round(float(actionable[f"signed_edge_{h}d_pct"].mean()), 4)
        summaries[f"score_corr_{h}d"] = round(_safe_corr(out["combined_score"], out[f"fwd_{h}d_pct"]), 4)
    return out, summaries


def summarize_signals(records: pd.DataFrame, horizon: int = 5) -> dict:
    if records is None or records.empty:
        return {"decision_bars": 0, "actionable": 0}
    actionable = records[records["signal"] != "HOLD"]
    buy = actionable[actionable["signal"] == "BUY"]
    sell = actionable[actionable["signal"] == "SELL"]
    result = {
        "decision_bars": int(len(records)),
        "actionable": int(len(actionable)),
        "coverage_pct": round(float(len(actionable) / len(records) * 100), 2),
        "buy_signals": int(len(buy)),
        "sell_signals": int(len(sell)),
        "hold_signals": int((records["signal"] == "HOLD").sum()),
        "buy_hit_rate_pct": round(float(buy[f"correct_{horizon}d"].mean() * 100), 2) if not buy.empty else 0.0,
        "sell_hit_rate_pct": round(float(sell[f"correct_{horizon}d"].mean() * 100), 2) if not sell.empty else 0.0,
        "balanced_accuracy_pct": round(float((
            (buy[f"correct_{horizon}d"].mean() if not buy.empty else np.nan)
            + (sell[f"correct_{horizon}d"].mean() if not sell.empty else np.nan)
        ) / 2 * 100), 2) if not buy.empty and not sell.empty else 0.0,
        "avg_signed_edge_pct": round(float(actionable[f"signed_edge_{horizon}d_pct"].mean()), 4) if not actionable.empty else 0.0,
        "score_spearman": round(_safe_corr(records["combined_score"], records[f"fwd_{horizon}d_pct"]), 4),
        "base_positive_rate_pct": round(float((actionable[f"fwd_{horizon}d_pct"] > 0).mean() * 100), 2) if not actionable.empty else 0.0,
        "base_negative_rate_pct": round(float((actionable[f"fwd_{horizon}d_pct"] < 0).mean() * 100), 2) if not actionable.empty else 0.0,
    }
    return result


def aggregate_audit(
    per_symbol_records: dict[str, pd.DataFrame],
    buckets: dict[str, str],
    horizon: int = 5,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    frames = []
    stock_rows = []
    for symbol, frame in per_symbol_records.items():
        if frame is None or frame.empty:
            continue
        x = frame.copy()
        x["size_bucket"] = buckets.get(symbol, "unclassified")
        frames.append(x)
        row = {
            "symbol": symbol,
            "size_bucket": buckets.get(symbol, "unclassified"),
            "decision_bars": int(len(x)),
            "buy_signals": int((x["signal"] == "BUY").sum()),
            "sell_signals": int((x["signal"] == "SELL").sum()),
            "hold_signals": int((x["signal"] == "HOLD").sum()),
        }
        actionable = x[x["signal"] != "HOLD"]
        row["actionable"] = int(len(actionable))
        row["coverage_pct"] = round(float(len(actionable) / len(x) * 100), 2)
        for h in DEFAULT_HORIZONS:
            if actionable.empty:
                row[f"hit_rate_{h}d_pct"] = 0.0
                row[f"avg_signed_edge_{h}d_pct"] = 0.0
            else:
                row[f"hit_rate_{h}d_pct"] = round(float(actionable[f"correct_{h}d"].mean() * 100), 2)
                row[f"avg_signed_edge_{h}d_pct"] = round(float(actionable[f"signed_edge_{h}d_pct"].mean()), 4)
            row[f"score_corr_{h}d"] = round(_safe_corr(x["combined_score"], x[f"fwd_{h}d_pct"]), 4)
        stock_rows.append(row)
    all_records = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    overall = summarize_signals(all_records, horizon=horizon)
    stock_df = pd.DataFrame(stock_rows)
    return all_records, overall, stock_df


def threshold_sensitivity(records: pd.DataFrame, thresholds: Iterable[float] = DEFAULT_THRESHOLDS, horizon: int = 5) -> pd.DataFrame:
    if records is None or records.empty:
        return pd.DataFrame()
    rows = []
    for th in thresholds:
        x = records.copy()
        x["signal_test"] = x["combined_score"].apply(lambda s: _label_from_score(float(s), float(th)))
        actionable = x[x["signal_test"] != "HOLD"]
        if actionable.empty:
            rows.append({"threshold": th, "signals": 0, "coverage_pct": 0.0, "hit_rate_pct": 0.0, "avg_signed_edge_pct": 0.0})
            continue
        signed = np.where(
            actionable["signal_test"] == "BUY",
            actionable[f"fwd_{horizon}d_pct"],
            -actionable[f"fwd_{horizon}d_pct"],
        )
        correct = signed > 0
        rows.append({
            "threshold": th,
            "signals": int(len(actionable)),
            "coverage_pct": round(float(len(actionable) / len(x) * 100), 2),
            "hit_rate_pct": round(float(correct.mean() * 100), 2),
            "avg_signed_edge_pct": round(float(np.mean(signed)), 4),
        })
    return pd.DataFrame(rows)


def diagnostics(overall: dict, sensitivity: pd.DataFrame, stock_df: pd.DataFrame, horizon: int = 5) -> list[str]:
    notes: list[str] = []
    hit = overall.get(f"hit_rate_{horizon}d_pct", 0.0)
    edge = overall.get("avg_signed_edge_pct", 0.0)
    corr = overall.get("score_spearman", 0.0)
    if hit < 50:
        notes.append(f"At {horizon} days the current threshold is below a 50% directional hit rate ({hit:.1f}%). Treat the present threshold as unvalidated.")
    elif edge <= 0:
        notes.append(f"Directional hit rate is at/above 50%, but average signed edge is not positive ({edge:.3f}%). Accuracy alone is not enough; inspect trade magnitude and costs.")
    else:
        notes.append(f"At {horizon} days the model shows positive average signed edge ({edge:.3f}%) with a {hit:.1f}% hit rate in this replay.")
    if corr < 0.10:
        notes.append("Score-to-return monotonicity is weak. A continuous/normalized score and relative-strength features are candidates for improvement.")
    elif corr > 0.20:
        notes.append("Score has meaningful positive monotonicity with forward returns in this sample; preserving score magnitude is more useful than a binary crossover alone.")
    if not sensitivity.empty:
        best_hit = sensitivity.loc[sensitivity["hit_rate_pct"].idxmax()]
        current = sensitivity.iloc[(sensitivity["threshold"] - 0.25).abs().argmin()]
        if best_hit["threshold"] > 0.25 and best_hit["signals"] >= max(20, int(0.35 * current["signals"])):
            notes.append(f"A stricter threshold ({best_hit['threshold']:.2f}) has the highest observed hit rate ({best_hit['hit_rate_pct']:.1f}%) while retaining {int(best_hit['signals'])} signals; this supports testing a higher action threshold out-of-sample.")
    if not stock_df.empty:
        by_bucket = stock_df.groupby("size_bucket")[f"hit_rate_{horizon}d_pct"].mean().sort_values()
        if len(by_bucket) >= 2 and float(by_bucket.iloc[-1] - by_bucket.iloc[0]) >= 10:
            notes.append("Performance differs materially by stock bucket. Keep a separate liquidity/volatility filter for smaller/high-volatility stocks rather than one universal risk setting.")
    return notes


def run_universe_audit(
    symbols: list[str] | None = None,
    config: AuditConfig | None = None,
    fetcher=None,
) -> dict:
    cfg = config or AuditConfig()
    universe = symbols or [s for s, _ in DEFAULT_AUDIT_UNIVERSE]
    bucket_map = {s: bucket for s, bucket in DEFAULT_AUDIT_UNIVERSE}
    records: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    symbol_summaries: list[dict] = []

    if fetcher is None:
        from data_fetcher import get_historical
        fetcher = get_historical

    benchmark_cache: dict[str, pd.DataFrame] = {}
    for symbol in universe:
        try:
            px = fetcher(symbol, period="2y", interval="1d")
            if px is None or px.empty:
                errors[symbol] = "no price history returned"
                continue
            bench_symbol = benchmark_for_audit(symbol)
            if bench_symbol not in benchmark_cache:
                benchmark_cache[bench_symbol] = fetcher(bench_symbol, period="2y", interval="1d")
            rec, summ = audit_symbol(symbol, px, benchmark_cache[bench_symbol], cfg)
            if rec.empty:
                errors[symbol] = summ.get("status", "audit returned no rows")
                continue
            records[symbol] = rec
            symbol_summaries.append(summ)
        except Exception as exc:
            errors[symbol] = f"{type(exc).__name__}: {exc}"

    all_records, overall, stock_df = aggregate_audit(records, bucket_map, horizon=5)
    sensitivity = threshold_sensitivity(all_records, DEFAULT_THRESHOLDS, horizon=5)
    notes = diagnostics(overall, sensitivity, stock_df, horizon=5)
    return {
        "records": all_records,
        "overall": overall,
        "stocks": stock_df.sort_values("symbol") if not stock_df.empty else stock_df,
        "threshold_sensitivity": sensitivity,
        "diagnostics": notes,
        "errors": errors,
        "symbols_requested": universe,
        "symbols_tested": list(records.keys()),
        "symbol_summaries": symbol_summaries,
    }

if __name__ == "__main__":
    result = run_universe_audit()
    print("1-YEAR PREDICTION AUDIT")
    print("stocks tested:", len(result["symbols_tested"]))
    print("overall 5D:", result["overall"])
    print("\nstock results:")
    print(result["stocks"].to_string(index=False) if not result["stocks"].empty else "No stock results")
    print("\nthreshold sensitivity:")
    print(result["threshold_sensitivity"].to_string(index=False) if not result["threshold_sensitivity"].empty else "No threshold results")
    print("\ndiagnostics:")
    for note in result["diagnostics"]:
        print("-", note)
    if result["errors"]:
        print("\nerrors:")
        for k, v in result["errors"].items():
            print(f"- {k}: {v}")
