"""Batch point-in-time prediction-vs-actual audit.

This module replays the certified 30-minute production model across a selected
stock universe and historical date/time window. It loads each stock once for
the requested window, constructs cross-sectional features at each timestamp,
runs the production model, and compares the forecast with the subsequent
30-minute outcome.

Historical news/announcements are best-effort and never replaced with current
information. Historical F&O/fundamentals remain unavailable unless a timestamped
archive is supplied.
"""
from __future__ import annotations

from datetime import time
from typing import Iterable

import numpy as np
import pandas as pd

from historical_replay import (
    _entry_exit_metrics,
    actual_outcome,
    historical_announcements,
    historical_benchmark,
    historical_news,
    historical_price,
)
from production_model import build_feature_rows, predict as production_predict
from triggers import announcement_to_news, event_risk_score, tag_news_items
from data_fetcher import benchmark_for_symbol


DEFAULT_START = time(10, 0)
DEFAULT_END = time(14, 0)


def _timestamps_for_day(day: pd.Timestamp, start: time, end: time, step_minutes: int = 30) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp.combine(day.date(), start)
    end_ts = pd.Timestamp.combine(day.date(), end)
    return list(pd.date_range(start_ts, end_ts, freq=f"{int(step_minutes)}min"))


def _actual_binary(outcome: dict) -> float:
    d = outcome.get("actual_direction")
    if d == "UP":
        return 1.0
    if d == "DOWN":
        return 0.0
    return np.nan


def run_batch_intraday_audit(
    symbols: Iterable[str],
    start_date,
    end_date,
    start_time: time = DEFAULT_START,
    end_time: time = DEFAULT_END,
    step_minutes: int = 30,
    max_symbols: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Run a causal 30-minute replay over many symbols/timestamps."""
    symbols = [str(s).strip().upper() for s in symbols if str(s).strip()][:max_symbols]
    if not symbols:
        raise ValueError("No symbols supplied")
    start_day = pd.Timestamp(start_date).normalize()
    end_day = pd.Timestamp(end_date).normalize()
    if end_day < start_day:
        raise ValueError("End date must be on/after start date")
    if step_minutes not in {15, 30, 60}:
        raise ValueError("Audit step must be 15, 30 or 60 minutes")
    if start_time >= end_time:
        raise ValueError("Audit start time must be before end time")

    # Load price history once per symbol. A 20-day warmup supports the 20/50-ish
    # features and the production contract's return/RSI/ATR calculations.
    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = historical_price(
                symbol,
                start_day - pd.Timedelta(days=25),
                end_day + pd.Timedelta(days=2),
                interval="1m",
            )
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            data[symbol] = df

    if not data:
        raise ValueError("No historical minute data could be loaded for the selected universe/date range.")

    # One benchmark source for the audit universe. If benchmark data is absent,
    # the production model's explicit missing-value path is retained.
    bench_symbol = benchmark_for_symbol(next(iter(data)))
    benchmark = historical_benchmark(bench_symbol, end_day + pd.Timedelta(days=2))
    if not benchmark.empty:
        benchmark = benchmark[benchmark.index <= end_day + pd.Timedelta(days=2)]

    # Cache timestamped public news/announcement windows by symbol+day; each
    # timestamp then filters those rows point-in-time without another request.
    news_cache: dict[tuple[str, object], list[dict]] = {}
    ann_cache: dict[tuple[str, object], list[dict]] = {}

    rows: list[dict] = []
    for day in pd.date_range(start_day, end_day, freq="D"):
        if day.weekday() >= 5:
            continue
        slots = _timestamps_for_day(day, start_time, end_time, step_minutes)
        day_key = day.date()
        for ts in slots:
            active_items = []
            active_symbols = []
            for symbol, full_df in data.items():
                hist = full_df[full_df.index <= ts]
                if len(hist) < 60:
                    continue
                if (symbol, day_key) not in news_cache:
                    try:
                        news_cache[(symbol, day_key)] = historical_news(symbol, ts + pd.Timedelta(hours=8))
                    except Exception:
                        news_cache[(symbol, day_key)] = []
                if (symbol, day_key) not in ann_cache:
                    try:
                        ann_cache[(symbol, day_key)] = historical_announcements(symbol, ts + pd.Timedelta(hours=8))
                    except Exception:
                        ann_cache[(symbol, day_key)] = []
                raw_news = []
                for n in news_cache[(symbol, day_key)]:
                    try:
                        p = pd.Timestamp(n.get("published"))
                        if p.tzinfo is not None:
                            p = p.tz_convert("Asia/Kolkata").tz_localize(None)
                        if p <= ts:
                            raw_news.append(n)
                    except Exception:
                        continue
                hist_news = tag_news_items(raw_news)
                hist_ann = [a for a in ann_cache[(symbol, day_key)] if pd.to_datetime(a.get("date"), errors="coerce") <= ts]
                active_items.append({
                    "symbol": symbol,
                    "price_df": hist,
                    "market_df": benchmark[benchmark.index <= ts] if not benchmark.empty else benchmark,
                    "matched_news": hist_news + announcement_to_news(hist_ann),
                    "announcements": hist_ann,
                    "event_risk": event_risk_score(hist_news, hist_ann),
                    "fo": {},
                    "decision_ts": ts,
                })
                active_symbols.append(symbol)

            if not active_items:
                continue
            try:
                feature_frame = build_feature_rows(active_items)
                preds = production_predict(feature_frame)
            except Exception:
                continue

            for idx, item in enumerate(active_items):
                symbol = item["symbol"]
                full_df = data[symbol]
                ml = preds[idx]
                entry = _entry_exit_metrics(item["price_df"], ts, "Intraday", 30)
                outcome = actual_outcome(full_df, ts, ml["signal"], "Intraday", 30, entry)
                actual_binary = _actual_binary(outcome)
                probability = float(ml["probability_up_30m"])
                brier = (probability - actual_binary) ** 2 if np.isfinite(actual_binary) else np.nan
                direction_match = outcome.get("correct_direction")
                signed_move = np.nan
                if outcome.get("status") == "OK":
                    mv = float(outcome.get("actual_move_pct", np.nan))
                    if ml["signal"] == "BUY":
                        signed_move = mv
                    elif ml["signal"] == "SELL":
                        signed_move = -mv
                    elif ml["signal"] == "HOLD":
                        signed_move = 0.0
                readiness = float(ml.get("data_readiness_score", np.nan))
                rows.append({
                    "timestamp": ts,
                    "symbol": symbol,
                    "prediction": ml["signal"],
                    "probability_up_30m": probability,
                    "model_probability_confidence": ml["confidence"],
                    "data_readiness": readiness,
                    # A prediction-vs-actual result is valid whenever the future
                    # market outcome exists. Full-data eligibility is tracked
                    # separately so missing optional information does not make
                    # the entire historical audit disappear.
                    "audit_eligible": bool(direction_match is not None),
                    "full_data_eligible": bool(direction_match is not None and readiness >= 0.70),
                    "decision_price": entry.get("entry_reference"),
                    "actual_price": outcome.get("actual_price"),
                    "actual_move_pct": outcome.get("actual_move_pct"),
                    "actual_direction": outcome.get("actual_direction"),
                    "direction_match": direction_match,
                    "target_hit": outcome.get("target_hit"),
                    "stop_hit": outcome.get("stop_hit"),
                    "first_touch": outcome.get("first_touch"),
                    "signed_move_pct": signed_move,
                    "brier_error": brier,
                    "historical_news": bool(item["matched_news"]),
                    "corporate_announcements": bool(item["announcements"]),
                    "historical_fno": False,
                    "historical_fundamentals": False,
                })

    result = pd.DataFrame(rows)
    summary = {
        "status": "PASS" if not result.empty else "NO_RESULTS",
        "symbols_requested": symbols,
        "symbols_scored": sorted(result["symbol"].unique().tolist()) if not result.empty else [],
        "start_date": str(start_day.date()),
        "end_date": str(end_day.date()),
        "step_minutes": step_minutes,
        "total_predictions": int(len(result)),
        "eligible_predictions": int(result["audit_eligible"].sum()) if not result.empty else 0,
        "full_data_eligible_predictions": int(result["full_data_eligible"].sum()) if not result.empty else 0,
        "directional_accuracy": None,
        "full_data_directional_accuracy": None,
        "average_actual_move_pct": None,
        "average_signed_move_pct": None,
        "mean_brier_error": None,
        "data_limited_predictions": 0,
    }
    if not result.empty:
        summary["data_limited_predictions"] = int((result["data_readiness"] < 0.70).sum())
        eligible = result[result["audit_eligible"]].copy()
        if not eligible.empty:
            summary["directional_accuracy"] = float(eligible["direction_match"].mean())
        full = result[result["full_data_eligible"]].copy()
        if not full.empty:
            summary["full_data_directional_accuracy"] = float(full["direction_match"].mean())
            summary["average_actual_move_pct"] = float(eligible["actual_move_pct"].mean())
            summary["average_signed_move_pct"] = float(eligible["signed_move_pct"].mean())
            summary["mean_brier_error"] = float(eligible["brier_error"].mean())
    return result, summary
