#!/usr/bin/env python3
"""Shared-date, purged cross-sectional benchmark for daily OHLC forecasts.

This is a read-only evaluation utility. It fetches daily histories via the
repository's NSE-first provider chain, evaluates the existing daily OHLC model
against the same training-median baseline on a common NSE-session holdout, and
writes local JSON/CSV reports. The official NSE snapshot layer may cache
immutable source CSVs on mounted Drive to resume an interrupted download; the
audit does not modify research datasets, production models, prediction ledgers,
or place orders.

Run:
  ./py.sh cross_sectional_daily_ohlc_audit.py --period 900d
  ./py.sh cross_sectional_daily_ohlc_audit.py --period 900d --diverse-symbols-file my_30_pit_symbols.txt

Supply the exact list of the previous PIT-panel sample with
--diverse-symbols-file to reproduce that cohort. The built-in 30-symbol
fallback is a fixed, sector-diverse sample; it is not claimed to be the same
sample that produced an earlier report.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from daily_ohlc_forecast import (
    HORIZONS,
    TARGETS,
    _cohere_log_return_predictions,
    _features,
    _new_model,
    _normalise_daily,
)

IST = ZoneInfo("Asia/Kolkata")
CORE_SYMBOLS = ["SBIN", "RELIANCE", "TCS", "INFY", "HDFCBANK"]
DEFAULT_DIVERSE_SYMBOLS = [
    "ICICIBANK", "AXISBANK", "KOTAKBANK", "BAJFINANCE", "BAJAJFINSV",
    "BHARTIARTL", "HCLTECH", "WIPRO", "TECHM", "LT",
    "ITC", "HINDUNILVR", "ASIANPAINT", "TITAN", "TRENT",
    "MARUTI", "TATAMOTORS", "M&M", "SUNPHARMA", "DRREDDY",
    "CIPLA", "ONGC", "NTPC", "POWERGRID", "TATASTEEL",
    "HINDALCO", "ADANIPORTS", "ULTRACEMCO", "APOLLOHOSP", "EICHERMOT",
]
MIN_HISTORY_ROWS = 200
MIN_COMMON_HOLDOUT_DATES = 20
PURGE_SESSIONS = max(HORIZONS) + 1
BLOCK_LENGTH_SESSIONS = max(HORIZONS)
SEED = 20261014


def normalize_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    for suffix in (".NS", ".BO", "-EQ", ".EQ"):
        if symbol.endswith(suffix):
            symbol = symbol[:-len(suffix)]
    return symbol


def read_symbol_file(path: str | Path) -> list[str]:
    values: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        values.extend(item.strip() for item in line.replace(";", ",").split(",") if item.strip())
    result = list(dict.fromkeys(normalize_symbol(value) for value in values))
    return [value for value in result if value]


def completed_session_cutoff(now_ist: datetime | pd.Timestamp | None = None):
    """Never use today's possibly-incomplete daily candle before 16:00 IST."""
    now = pd.Timestamp(now_ist) if now_ist is not None else pd.Timestamp.now(tz=IST)
    if now.tzinfo is None:
        now = now.tz_localize(IST)
    else:
        now = now.tz_convert(IST)
    return now.date() if now.time() >= time(16, 0) else (now.date() - timedelta(days=1))


def load_daily_history(symbol: str, period: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load history with diagnostic errors; never request Drive persistence for the index."""
    from data_fetcher import _period_days, get_historical

    ticker = "^NSEI" if symbol == "^NSEI" else f"{normalize_symbol(symbol)}.NS"
    provider = "repository_NSE_first_provider_chain"
    provider_errors: list[str] = []
    source_diag_data: dict[str, Any] = {}

    if ticker == "^NSEI":
        # get_historical("^NSEI") requests sync_drive=True in the official index
        # adapter. This audit uses the last completed-session cutoff and explicitly
        # disables Drive sync so its benchmark fetch remains read-only.
        from nse_data_layer import fetch_index_history

        end_date = completed_session_cutoff()
        start_date = end_date - timedelta(days=_period_days(period))
        try:
            index_frame = fetch_index_history(
                ticker,
                start_date,
                end_date,
                sync_drive=False,
                allow_latest_missing_end=True,
            )
            if index_frame is None or index_frame.empty:
                daily = pd.DataFrame()
                provider_errors.append(
                    f"official NSE index adapter returned no rows for {start_date}..{end_date}"
                )
            else:
                index_frame = index_frame.rename(
                    columns={
                        "open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume",
                    }
                )
                if "date" in index_frame.columns:
                    index_frame = index_frame.set_index("date")
                daily = _normalise_daily(index_frame)
            provider = "NSE_official_index_adapter_sync_drive_false"
        except Exception as exc:
            daily = pd.DataFrame()
            provider = "NSE_official_index_adapter_failed"
            provider_errors.append(
                f"official NSE index adapter: {type(exc).__name__}: {exc}"
            )
    else:
        try:
            frame = get_historical(
                ticker, period=period, interval="1d", force_refresh=False
            )
            daily = _normalise_daily(frame)
            if daily.empty:
                provider_errors.append(
                    "repository historical provider chain returned no usable daily rows"
                )
        except Exception as exc:
            daily = pd.DataFrame()
            provider_errors.append(
                f"repository historical provider chain: {type(exc).__name__}: {exc}"
            )
        try:
            from data_fetcher import source_diagnostics as get_source_diagnostics

            diagnostics = get_source_diagnostics()
            if diagnostics:
                # Keep the source-level diagnostic for investigation; do not include
                # credentials or request headers.
                source_diag_data = diagnostics if isinstance(diagnostics, dict) else {
                    "detail": str(diagnostics)
                }
        except Exception as exc:
            source_diag_data = {
                "diagnostic_read_error": f"{type(exc).__name__}: {exc}"
            }

    # Some provider paths return an empty or recent-only slice. Prefer whichever
    # history source gives more rows, and record failures instead of silently hiding them.
    if len(daily) < MIN_HISTORY_ROWS:
        try:
            from data_fetcher import _yahoo_history

            fallback = _normalise_daily(_yahoo_history(ticker, period, "1d"))
            if len(fallback) > len(daily):
                daily = fallback
                provider = "Yahoo_reference_fallback_after_short_provider_history"
            elif fallback.empty:
                provider_errors.append("Yahoo reference history returned no usable daily rows")
            else:
                provider_errors.append(
                    f"Yahoo reference history had only {len(fallback)} usable rows; "
                    f"kept provider history with {len(daily)} rows"
                )
        except Exception as exc:
            provider_errors.append(
                f"Yahoo reference history: {type(exc).__name__}: {exc}"
            )

    meta = {
        "requested_symbol": ticker,
        "provider_path": provider,
        "history_rows": int(len(daily)),
        "first_date": daily.index.min().date().isoformat() if not daily.empty else None,
        "last_date": daily.index.max().date().isoformat() if not daily.empty else None,
    }
    if provider_errors:
        meta["provider_errors"] = provider_errors
    if source_diag_data:
        meta["source_diagnostics"] = source_diag_data
    return daily, meta

def prepare_symbol_data(
    daily: pd.DataFrame,
    session_calendar: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Align OHLC to the exchange calendar and preserve missing sessions as missing.

    Targets use positional NSE sessions, not the next available stock row. If a
    stock is missing a bar on an expected session, that row cannot be silently
    relabelled as a longer horizon.
    """
    normalized = _normalise_daily(daily)
    aligned = normalized.reindex(session_calendar)
    features = _features(aligned)
    close = pd.to_numeric(aligned["close"], errors="coerce")
    eligible = (
        features["ma_deviation_50"].notna().to_numpy()
        & features["volatility_20"].notna().to_numpy()
        & np.isfinite(close.to_numpy(dtype=float, na_value=np.nan))
        & (close.to_numpy(dtype=float, na_value=np.nan) > 0)
    )
    positions = np.arange(len(session_calendar), dtype=int)
    anchor = close.to_numpy(dtype=float, na_value=np.nan)
    stacked_blocks: list[pd.DataFrame] = []
    complete_origin = eligible.copy()

    for horizon in HORIZONS:
        block = features.loc[eligible].copy()
        selected_positions = positions[eligible]
        selected_anchor = anchor[eligible]
        block["origin_position"] = selected_positions
        block["forecast_horizon"] = float(horizon)
        block["anchor_close"] = selected_anchor

        horizon_complete = np.ones(len(session_calendar), dtype=bool)
        for target in TARGETS:
            future = pd.to_numeric(aligned[target], errors="coerce").shift(-horizon)
            future_values = future.to_numpy(dtype=float, na_value=np.nan)
            horizon_complete &= np.isfinite(future_values) & (future_values > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                target_values = np.log(future_values[eligible] / selected_anchor)
            block[f"target_{target}"] = target_values

        complete_origin &= horizon_complete
        target_columns = [f"target_{target}" for target in TARGETS]
        block = block.replace([np.inf, -np.inf], np.nan)
        block = block.dropna(subset=["anchor_close", *target_columns])
        stacked_blocks.append(block)

    stacked = pd.concat(stacked_blocks, axis=0, ignore_index=True)
    stacked = stacked.sort_values(["origin_position", "forecast_horizon"]).reset_index(drop=True)
    return {
        "aligned": aligned,
        "features": features,
        "feature_names": list(features.columns) + ["forecast_horizon"],
        "stacked": stacked,
        "complete_origin": complete_origin,
    }


def select_shared_holdout_positions(
    complete_by_symbol: dict[str, np.ndarray],
    *,
    session_count: int,
    holdout_fraction: float = 0.20,
    horizon_count: int = max(HORIZONS),
) -> tuple[int, np.ndarray]:
    """Return one shared holdout start and complete dates with all target horizons matured."""
    if not complete_by_symbol:
        raise ValueError("No prepared symbols available.")
    if not 0.10 <= holdout_fraction <= 0.40:
        raise ValueError("holdout_fraction must be between 0.10 and 0.40.")
    split_position = int(np.floor(session_count * (1.0 - holdout_fraction)))
    if split_position <= PURGE_SESSIONS or session_count - horizon_count <= split_position:
        raise ValueError("Session calendar is too short for the requested purged holdout.")
    candidates = np.arange(split_position, session_count - horizon_count, dtype=int)
    masks = []
    for symbol, values in complete_by_symbol.items():
        flags = np.asarray(values, dtype=bool)
        if len(flags) != session_count:
            raise ValueError(f"{symbol} complete-origin mask does not match the session calendar.")
        masks.append(flags[candidates])
    shared = np.logical_and.reduce(masks)
    return split_position, candidates[shared]


def _skill(model_errors: np.ndarray, baseline_errors: np.ndarray) -> np.ndarray:
    model_mae = np.mean(model_errors, axis=0)
    baseline_mae = np.mean(baseline_errors, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 1.0 - model_mae / baseline_mae
    return np.where(np.isfinite(result) & (baseline_mae > 0), result, np.nan)


def moving_block_bootstrap_ci(
    model_error_tensor: np.ndarray,
    baseline_error_tensor: np.ndarray,
    symbol_indices: list[int],
    *,
    replicates: int = 1000,
    seed: int = SEED,
    block_length: int = BLOCK_LENGTH_SESSIONS,
) -> dict[str, Any]:
    """Moving-block bootstrap across issue dates; keep each date's full cross-section together.

    Input shape: (common issue dates, symbols, horizons, OHLC targets). The same
    sampled dates apply to every stock, preserving cross-sectional dependence.
    """
    if replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    date_count = model_error_tensor.shape[0]
    if date_count < 2:
        return {"replicates": 0, "block_length_sessions": 0, "reason": "too_few_dates"}
    block_length = max(1, min(int(block_length), date_count))
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=float)
    medians = np.empty(replicates, dtype=float)
    positive_rates = np.empty(replicates, dtype=float)

    for rep in range(replicates):
        picked: list[int] = []
        while len(picked) < date_count:
            start = int(rng.integers(0, date_count - block_length + 1))
            picked.extend(range(start, start + block_length))
        sampled = np.asarray(picked[:date_count], dtype=int)

        model = model_error_tensor[sampled][:, symbol_indices, :, :]
        baseline = baseline_error_tensor[sampled][:, symbol_indices, :, :]
        cell_skill = _skill(model, baseline)
        stock_skill = np.nanmean(cell_skill, axis=(1, 2))
        means[rep] = float(np.nanmean(stock_skill))
        medians[rep] = float(np.nanmedian(stock_skill))
        positive_rates[rep] = float(np.mean(stock_skill > 0))

    def interval(values: np.ndarray) -> list[float]:
        return [float(v) for v in np.nanpercentile(values, [2.5, 97.5])]

    return {
        "replicates": int(replicates),
        "block_length_sessions": int(block_length),
        "mean_stock_skill_ci95": interval(means),
        "median_stock_skill_ci95": interval(medians),
        "positive_stock_rate_ci95": interval(positive_rates),
        "bootstrap_method": "moving-block bootstrap over common issue dates; full cross-section sampled together",
    }


def _score_one_symbol(
    symbol: str,
    prepared: dict[str, Any],
    *,
    training_cutoff_position: int,
    evaluation_positions: np.ndarray,
) -> list[dict[str, Any]]:
    samples = prepared["stacked"]
    feature_names = prepared["feature_names"]
    train = samples.loc[samples["origin_position"] < training_cutoff_position].copy()
    valid = samples.loc[samples["origin_position"].isin(set(map(int, evaluation_positions)))].copy()
    if len(train) < 200:
        raise ValueError(f"{symbol}: only {len(train)} labelled training rows remain after the horizon purge.")
    expected_valid_rows = len(evaluation_positions) * len(HORIZONS)
    if len(valid) != expected_valid_rows:
        raise ValueError(
            f"{symbol}: expected {expected_valid_rows} common-holdout rows, received {len(valid)}."
        )

    X_train = train[feature_names].apply(pd.to_numeric, errors="coerce")
    X_valid = valid[feature_names].apply(pd.to_numeric, errors="coerce")
    target_columns = [f"target_{target}" for target in TARGETS]
    models = {}
    for target in TARGETS:
        model = _new_model(len(train))
        model.fit(X_train, train[f"target_{target}"].to_numpy(dtype=float))
        models[target] = model

    raw_model = np.column_stack([
        models[target].predict(X_valid) for target in TARGETS
    ])
    horizon_values = valid["forecast_horizon"].to_numpy(dtype=int)
    baseline = np.full_like(raw_model, np.nan, dtype=float)
    for row_idx, horizon in enumerate(horizon_values):
        horizon_train = train.loc[train["forecast_horizon"].eq(float(horizon))]
        for target_idx, target in enumerate(TARGETS):
            values = horizon_train[f"target_{target}"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                raise ValueError(f"{symbol}: no purged training observations for horizon {horizon}/{target}.")
            baseline[row_idx, target_idx] = float(np.median(values))

    anchors = valid["anchor_close"].to_numpy(dtype=float)
    model_scored = _cohere_log_return_predictions(raw_model, anchors)
    baseline_scored = _cohere_log_return_predictions(baseline, anchors)
    actual = valid[target_columns].to_numpy(dtype=float)
    if not (
        np.isfinite(model_scored).all()
        and np.isfinite(baseline_scored).all()
        and np.isfinite(actual).all()
    ):
        raise ValueError(f"{symbol}: non-finite model, baseline, or actual holdout values.")

    output: list[dict[str, Any]] = []
    for row_idx, (_, row) in enumerate(valid.iterrows()):
        date = pd.Timestamp(row.name) if isinstance(row.name, pd.Timestamp) else None
        position = int(row["origin_position"])
        for target_idx, target in enumerate(TARGETS):
            model_error = abs(float(actual[row_idx, target_idx]) - float(model_scored[row_idx, target_idx]))
            baseline_error = abs(float(actual[row_idx, target_idx]) - float(baseline_scored[row_idx, target_idx]))
            output.append({
                "symbol": symbol,
                "origin_position": position,
                "origin_date": str(pd.Timestamp(row.get("origin_date", "")).date())
                    if row.get("origin_date") is not None else None,
                "horizon": int(row["forecast_horizon"]),
                "target": target,
                "target_date_position": position + int(row["forecast_horizon"]),
                "anchor_close": float(row["anchor_close"]),
                "actual_log_return": float(actual[row_idx, target_idx]),
                "model_log_return_after_coherence": float(model_scored[row_idx, target_idx]),
                "baseline_log_return_after_coherence": float(baseline_scored[row_idx, target_idx]),
                "model_abs_error": float(model_error),
                "baseline_abs_error": float(baseline_error),
            })
    return output


def summarize_observations(
    observations: pd.DataFrame,
    *,
    symbols: list[str],
    evaluation_dates: list[str],
    core_symbols: list[str],
    diverse_symbols: list[str],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    dates = list(evaluation_dates)
    symbol_order = list(symbols)
    horizon_order = list(HORIZONS)
    target_order = list(TARGETS)
    date_lookup = {value: index for index, value in enumerate(dates)}
    symbol_lookup = {value: index for index, value in enumerate(symbol_order)}
    horizon_lookup = {value: index for index, value in enumerate(horizon_order)}
    target_lookup = {value: index for index, value in enumerate(target_order)}
    shape = (len(dates), len(symbol_order), len(horizon_order), len(target_order))
    model_tensor = np.full(shape, np.nan, dtype=float)
    baseline_tensor = np.full(shape, np.nan, dtype=float)

    for row in observations.itertuples(index=False):
        di = date_lookup[str(row.origin_date)]
        si = symbol_lookup[str(row.symbol)]
        hi = horizon_lookup[int(row.horizon)]
        ti = target_lookup[str(row.target)]
        model_tensor[di, si, hi, ti] = float(row.model_abs_error)
        baseline_tensor[di, si, hi, ti] = float(row.baseline_abs_error)

    if not (np.isfinite(model_tensor).all() and np.isfinite(baseline_tensor).all()):
        raise ValueError("Audit observations do not form a complete common date × stock × horizon × target tensor.")

    symbol_skill = _skill(model_tensor, baseline_tensor)
    per_stock_overall = np.mean(symbol_skill, axis=(1, 2))
    stock_rows: list[dict[str, Any]] = []
    by_target_by_symbol: dict[str, dict[str, float]] = {}
    for si, symbol in enumerate(symbol_order):
        target_means = {
            target: float(np.mean(symbol_skill[si, :, ti]))
            for ti, target in enumerate(target_order)
        }
        beats_by_horizon = [
            int(np.sum(symbol_skill[si, hi, :] > 0))
            for hi in range(len(horizon_order))
        ]
        stock_rows.append({
            "symbol": symbol,
            "overall_skill": float(per_stock_overall[si]),
            "open_skill": target_means["open"],
            "high_skill": target_means["high"],
            "low_skill": target_means["low"],
            "close_skill": target_means["close"],
            "beats_per_horizon": beats_by_horizon,
            "evaluation_origins": len(dates),
            "horizon_target_cells": len(HORIZONS) * len(TARGETS),
        })
        by_target_by_symbol[symbol] = target_means

    def summarize_group(group_name: str, group_symbols: list[str]) -> dict[str, Any]:
        indices = [symbol_lookup[symbol] for symbol in group_symbols if symbol in symbol_lookup]
        if not indices:
            return {"status": "NO_SYMBOLS", "requested_symbols": group_symbols}
        skills = per_stock_overall[indices]
        ci = moving_block_bootstrap_ci(
            model_tensor, baseline_tensor, indices,
            replicates=bootstrap_replicates, seed=SEED + len(indices),
        )
        model_mean = float(np.mean(model_tensor[:, indices, :, :]))
        baseline_mean = float(np.mean(baseline_tensor[:, indices, :, :]))
        return {
            "group": group_name,
            "symbols": [symbol_order[index] for index in indices],
            "stock_count": len(indices),
            "evaluation_origins_per_stock": len(dates),
            "mean_stock_skill": float(np.mean(skills)),
            "median_stock_skill": float(np.median(skills)),
            "positive_skill_stocks": int(np.sum(skills > 0)),
            "positive_skill_rate": float(np.mean(skills > 0)),
            "pooled_mae_skill": float(1.0 - model_mean / baseline_mean) if baseline_mean > 0 else None,
            **ci,
        }

    per_horizon = []
    per_target_horizon = []
    for hi, horizon in enumerate(horizon_order):
        horizon_values = symbol_skill[:, hi, :].mean(axis=1)
        per_horizon.append({
            "horizon": horizon,
            "mean_stock_skill": float(np.mean(horizon_values)),
            "median_stock_skill": float(np.median(horizon_values)),
            "positive_skill_stocks": int(np.sum(horizon_values > 0)),
            "stock_count": len(symbol_order),
            "targets_evaluated_per_stock": len(target_order),
            "evaluation_origins_per_stock": len(dates),
        })
        for ti, target in enumerate(target_order):
            target_values = symbol_skill[:, hi, ti]
            per_target_horizon.append({
                "horizon": horizon,
                "target": target,
                "mean_stock_skill": float(np.mean(target_values)),
                "median_stock_skill": float(np.median(target_values)),
                "positive_skill_stocks": int(np.sum(target_values > 0)),
                "stock_count": len(symbol_order),
                "evaluation_origins_per_stock": len(dates),
            })

    core_present = [symbol for symbol in core_symbols if symbol in symbol_lookup]
    diverse_present = [symbol for symbol in diverse_symbols if symbol in symbol_lookup]
    return {
        "overall": {
            "stock_count": len(symbol_order),
            "evaluation_origins_per_stock": len(dates),
            "evaluation_origin_start": dates[0],
            "evaluation_origin_end": dates[-1],
            "mean_stock_skill": float(np.mean(per_stock_overall)),
            "median_stock_skill": float(np.median(per_stock_overall)),
            "positive_skill_stocks": int(np.sum(per_stock_overall > 0)),
            "positive_skill_rate": float(np.mean(per_stock_overall > 0)),
        },
        "cohorts": {
            "core_5": summarize_group("core_5", core_present),
            "diverse_30": summarize_group("diverse_30", diverse_present),
        },
        "per_symbol": stock_rows,
        "per_horizon": per_horizon,
        "per_target_horizon": per_target_horizon,
        "definition": {
            "mae_skill": "1 - MAE(model) / MAE(training-median baseline)",
            "overall_stock_skill": "unweighted mean of 28 individual stock × horizon × OHLC MAE-skill ratios",
            "beats_per_horizon": "count of the four OHLC target skills > 0 for that stock and horizon",
            "error_space": "log-return after identical OHLC-coherence projection for model and baseline",
            "baseline": "per-stock, per-horizon, per-target median of purged training labels",
        },
    }


def run_audit(
    *,
    period: str = "900d",
    diverse_symbols: list[str] | None = None,
    holdout_fraction: float = 0.20,
    bootstrap_replicates: int = 1000,
    now_ist: datetime | pd.Timestamp | None = None,
    history_loader=load_daily_history,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diverse = list(dict.fromkeys(normalize_symbol(value) for value in (diverse_symbols or DEFAULT_DIVERSE_SYMBOLS)))
    if len(diverse) != 30:
        raise ValueError(f"Expected exactly 30 diverse symbols for the requested benchmark; received {len(diverse)}.")
    if len(set(CORE_SYMBOLS)) != len(CORE_SYMBOLS):
        raise ValueError("Core symbol list must be unique.")
    all_symbols = list(dict.fromkeys(CORE_SYMBOLS + diverse))
    overlaps = sorted(set(CORE_SYMBOLS) & set(diverse))
    now = pd.Timestamp(now_ist) if now_ist is not None else pd.Timestamp.now(tz=IST)
    if now.tzinfo is None:
        now = now.tz_localize(IST)
    else:
        now = now.tz_convert(IST)
    cutoff_date = completed_session_cutoff(now)

    fetch_meta: dict[str, Any] = {}
    raw_history: dict[str, pd.DataFrame] = {}
    for symbol in ["^NSEI", *all_symbols]:
        try:
            daily, meta = history_loader(symbol, period)
        except Exception as exc:
            daily = pd.DataFrame()
            meta = {
                "requested_symbol": symbol,
                "provider_path": "history_loader_exception",
                "history_rows": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if daily is None or daily.empty:
            fetch_meta[symbol] = {**meta, "status": "NO_HISTORY"}
            raw_history[symbol] = pd.DataFrame()
            continue
        daily = _normalise_daily(daily)
        daily = daily.loc[[stamp.date() <= cutoff_date for stamp in daily.index]]
        fetch_meta[symbol] = {
            **meta,
            "status": "PASS" if len(daily) >= MIN_HISTORY_ROWS else "INSUFFICIENT_HISTORY",
            "history_rows_after_completed_session_cutoff": int(len(daily)),
            "first_completed_date": daily.index.min().date().isoformat() if not daily.empty else None,
            "last_completed_date": daily.index.max().date().isoformat() if not daily.empty else None,
        }
        raw_history[symbol] = daily

    missing = [symbol for symbol in ["^NSEI", *all_symbols]
               if raw_history.get(symbol) is None or len(raw_history[symbol]) < MIN_HISTORY_ROWS]
    if missing:
        report = {
            "status": "BLOCKED_INCOMPLETE_HISTORY",
            "generated_at_ist": now.isoformat(),
            "completed_session_cutoff": cutoff_date.isoformat(),
            "period_requested": period,
            "requested_symbol_count": len(all_symbols),
            "missing_or_short_history_symbols": missing,
            "history": fetch_meta,
            "reason": "The controlled comparison requires adequate history for the NIFTY session calendar and every requested stock; no incomplete cohort result is reported.",
            "read_only": True,
            "model_weights_changed": False,
            "orders_placed": False,
        }
        empty = pd.DataFrame()
        return report, empty, empty, empty

    calendar = pd.DatetimeIndex(raw_history["^NSEI"].index).normalize().unique().sort_values()
    calendar = calendar[calendar.date <= cutoff_date]
    if len(calendar) < MIN_HISTORY_ROWS:
        report = {
            "status": "BLOCKED_SHORT_SESSION_CALENDAR",
            "generated_at_ist": now.isoformat(),
            "completed_session_cutoff": cutoff_date.isoformat(),
            "history": fetch_meta,
            "reason": f"Only {len(calendar)} completed NIFTY sessions are available; at least {MIN_HISTORY_ROWS} are required.",
            "read_only": True,
            "model_weights_changed": False,
            "orders_placed": False,
        }
        empty = pd.DataFrame()
        return report, empty, empty, empty

    prepared: dict[str, dict[str, Any]] = {}
    complete_flags: dict[str, np.ndarray] = {}
    for symbol in all_symbols:
        prepared[symbol] = prepare_symbol_data(raw_history[symbol], calendar)
        complete_flags[symbol] = prepared[symbol]["complete_origin"]

    split_position, evaluation_positions = select_shared_holdout_positions(
        complete_flags,
        session_count=len(calendar),
        holdout_fraction=holdout_fraction,
        horizon_count=max(HORIZONS),
    )
    if len(evaluation_positions) < MIN_COMMON_HOLDOUT_DATES:
        report = {
            "status": "BLOCKED_TOO_FEW_COMMON_HOLDOUT_DATES",
            "generated_at_ist": now.isoformat(),
            "completed_session_cutoff": cutoff_date.isoformat(),
            "period_requested": period,
            "session_count": len(calendar),
            "split_position": split_position,
            "requested_holdout_origins": max(0, len(calendar) - max(HORIZONS) - split_position),
            "common_holdout_origins": int(len(evaluation_positions)),
            "history": fetch_meta,
            "reason": f"Only {len(evaluation_positions)} shared holdout origin dates remain across all 35 stocks; at least {MIN_COMMON_HOLDOUT_DATES} are required.",
            "read_only": True,
            "model_weights_changed": False,
            "orders_placed": False,
        }
        empty = pd.DataFrame()
        return report, empty, empty, empty

    first_evaluation_position = int(evaluation_positions[0])
    training_cutoff_position = first_evaluation_position - PURGE_SESSIONS
    if training_cutoff_position <= 0:
        raise ValueError("Holdout begins too early to apply the seven-session label purge.")
    evaluation_dates = [calendar[pos].date().isoformat() for pos in evaluation_positions]
    observation_rows: list[dict[str, Any]] = []
    training_meta: dict[str, Any] = {}
    for index, symbol in enumerate(all_symbols, start=1):
        print(f"[{index}/{len(all_symbols)}] Fitting purged holdout for {symbol}...", flush=True)
        rows = _score_one_symbol(
            symbol,
            prepared[symbol],
            training_cutoff_position=training_cutoff_position,
            evaluation_positions=evaluation_positions,
        )
        observation_rows.extend(rows)
        train_rows = int((prepared[symbol]["stacked"]["origin_position"] < training_cutoff_position).sum())
        training_meta[symbol] = {
            "purged_training_rows": train_rows,
            "purge_cutoff_position_exclusive": training_cutoff_position,
            "max_training_label_target_position": training_cutoff_position - 1 + max(HORIZONS),
            "first_validation_origin_position": first_evaluation_position,
            "validation_origin_count": len(evaluation_positions),
            "validation_rows_per_horizon": len(evaluation_positions),
            "model_refit_policy": "one fit before the shared holdout; weights frozen during holdout",
        }

    observations = pd.DataFrame(observation_rows)
    observations["origin_date"] = observations["origin_position"].map(
        {position: date for position, date in zip(evaluation_positions, evaluation_dates)}
    )
    observations["target_date"] = observations.apply(
        lambda row: calendar[int(row["target_date_position"])].date().isoformat(),
        axis=1,
    )
    symbols_in_order = all_symbols
    summary = summarize_observations(
        observations,
        symbols=symbols_in_order,
        evaluation_dates=evaluation_dates,
        core_symbols=CORE_SYMBOLS,
        diverse_symbols=diverse,
        bootstrap_replicates=bootstrap_replicates,
    )
    report = {
        "status": "COMPLETE_COMMON_HOLDOUT",
        "generated_at_ist": now.isoformat(),
        "completed_session_cutoff": cutoff_date.isoformat(),
        "period_requested": period,
        "session_count": int(len(calendar)),
        "session_calendar_start": calendar[0].date().isoformat(),
        "session_calendar_end": calendar[-1].date().isoformat(),
        "shared_holdout_start_position": int(evaluation_positions[0]),
        "shared_holdout_end_position": int(evaluation_positions[-1]),
        "validation_origin_start": evaluation_dates[0],
        "validation_origin_end": evaluation_dates[-1],
        "validation_origin_count": len(evaluation_dates),
        "validation_outcome_end": calendar[-1].date().isoformat(),
        "holdout_fraction": float(holdout_fraction),
        "purge_sessions": int(PURGE_SESSIONS),
        "horizons": list(HORIZONS),
        "targets": list(TARGETS),
        "requested_core_symbols": CORE_SYMBOLS,
        "requested_diverse_symbols": diverse,
        "core_diverse_overlap": overlaps,
        "diverse_sample_selection": (
            "user-supplied --diverse-symbols-file"
            if diverse_symbols is not None else "fixed built-in sector-diverse 30; not asserted to match the previous PIT-panel sample"
        ),
        "common_date_policy": "same issue dates for all symbols; date only included if all 35 symbols have an eligible anchor and all seven future-session OHLC outcomes",
        "model": "existing horizon-conditioned HistGradientBoostingRegressor and existing OHLC-coherence transform",
        "validation_method": (
            "fixed chronological holdout; one model per target per stock is fitted only on labels whose outcomes end before "
            "the common holdout, then kept frozen across holdout issue dates. This is not an expanding/refitting walk-forward."
        ),
        "training_meta": training_meta,
        "history": fetch_meta,
        "metrics": summary,
        "safety": {
            "read_only": True,
            "model_weights_changed": False,
            "production_registry_changed": False,
            "prediction_ledger_changed": False,
            "drive_written": False,
            "orders_placed": False,
        },
        "interpretation": [
            "Positive MAE skill means lower model absolute error than the training-median baseline; it is not investment return or directional accuracy.",
            "The confidence intervals use a seven-session moving-block bootstrap over shared issue dates, preserving each sampled date's cross-section.",
            "The baseline receives the same OHLC-coherence projection as the model before scoring.",
            "Today's incomplete daily candle is excluded before 16:00 IST.",
            "Missing stock sessions stay missing on the NIFTY session calendar; targets are never shifted to the next available stock row.",
            "Even positive validation skill does not approve the model for live trading; paper forward validation and trading-cost backtests remain required.",
        ],
    }
    stock_summary = pd.DataFrame(summary["per_symbol"])
    per_horizon = pd.DataFrame(summary["per_horizon"])
    per_target_horizon = pd.DataFrame(summary["per_target_horizon"])
    return report, observations, stock_summary, pd.concat(
        [per_horizon.assign(metric_type="horizon"), per_target_horizon.assign(metric_type="target_horizon")],
        ignore_index=True,
        sort=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="900d", help="History period accepted by the configured historical providers (default: 900d).")
    parser.add_argument("--diverse-symbols-file", help="Optional file containing exactly 30 diverse NSE symbols, one per line or comma-separated.")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--out-dir", default="diagnostics/cross_sectional_daily_ohlc_audit")
    args = parser.parse_args()
    try:
        diverse = read_symbol_file(args.diverse_symbols_file) if args.diverse_symbols_file else None
        report, observations, stock_summary, horizon_summary = run_audit(
            period=args.period,
            diverse_symbols=diverse,
            holdout_fraction=args.holdout_fraction,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    except Exception as exc:
        print(json.dumps({
            "status": "AUDIT_ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "read_only": True,
            "model_weights_changed": False,
            "orders_placed": False,
        }, indent=2), file=sys.stderr)
        return 2

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cross_sectional_daily_ohlc_audit.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    if not observations.empty:
        observations.to_csv(out / "cross_sectional_daily_ohlc_observations.csv", index=False)
        stock_summary.to_csv(out / "cross_sectional_daily_ohlc_by_symbol.csv", index=False)
        horizon_summary.to_csv(out / "cross_sectional_daily_ohlc_by_horizon.csv", index=False)

    # Keep console output compact while exposing the cohort comparison and CIs.
    console = {
        "status": report.get("status"),
        "completed_session_cutoff": report.get("completed_session_cutoff"),
        "common_holdout": {
            "start": report.get("validation_origin_start"),
            "end": report.get("validation_origin_end"),
            "origins": report.get("validation_origin_count"),
            "purge_sessions": report.get("purge_sessions"),
        },
        "cohorts": report.get("metrics", {}).get("cohorts"),
        "per_symbol": report.get("metrics", {}).get("per_symbol"),
        "report_json": str(out / "cross_sectional_daily_ohlc_audit.json"),
        "observations_csv": str(out / "cross_sectional_daily_ohlc_observations.csv") if not observations.empty else None,
    }
    print(json.dumps(console, indent=2, default=str))
    return 0 if report.get("status") == "COMPLETE_COMMON_HOLDOUT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
