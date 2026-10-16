#!/usr/bin/env python3
"""Read-only rolling-origin audit for seven-session daily OHLC forecasts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_ohlc_forecast import HORIZONS, TARGETS, _cohere_log_return_predictions, _normalise_daily, _new_model
from cross_sectional_daily_ohlc_audit import completed_session_cutoff, load_daily_history, prepare_symbol_data

BPS = 10_000.0


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)) * BPS)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))) * BPS)


def run_rolling_origin_audit(
    frame: pd.DataFrame,
    session_calendar: pd.DatetimeIndex | list[object],
    *,
    symbol: str = "UNKNOWN",
    min_train_origins: int = 160,
    step_sessions: int = 5,
    max_origins: int = 24,
    min_evaluation_origins: int = 10,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Refit at rolling issue dates, purging all +1..+7 labels at every fold.

    Returns a JSON-safe report, detailed per-origin/horizon predictions, and
    per-target/per-horizon MAE/RMSE/skill metrics. Calendar must be NSE sessions.
    """
    if min_train_origins < 80:
        raise ValueError("min_train_origins must be at least 80.")
    if min(step_sessions, max_origins, min_evaluation_origins) < 1:
        raise ValueError("step_sessions, max_origins and min_evaluation_origins must be positive.")

    daily = _normalise_daily(frame)
    cutoff = completed_session_cutoff()
    if not daily.empty:
        daily = daily.loc[[stamp.date() <= cutoff for stamp in daily.index]]
    if daily.empty:
        return ({
            "status": "BLOCKED_NO_HISTORY", "symbol": str(symbol).upper(),
            "reason": "No valid completed daily OHLC rows were available.",
            "read_only": True, "production_model_changed": False, "orders_placed": False,
        }, pd.DataFrame(), pd.DataFrame())

    calendar = pd.DatetimeIndex(pd.to_datetime(session_calendar, errors="coerce")).normalize()
    calendar = calendar[~calendar.isna()].unique().sort_values()
    calendar = calendar[(calendar >= daily.index.min()) & (calendar <= daily.index.max())]
    if len(calendar) < min_train_origins + max(HORIZONS) + 55:
        return ({
            "status": "BLOCKED_SHORT_SESSION_CALENDAR", "symbol": str(symbol).upper(),
            "history_rows": int(len(daily)), "calendar_sessions": int(len(calendar)),
            "reason": "Insufficient exchange sessions for feature warm-up, training and seven matured horizons.",
            "read_only": True, "production_model_changed": False, "orders_placed": False,
        }, pd.DataFrame(), pd.DataFrame())

    prepared = prepare_symbol_data(daily, calendar)
    stacked = prepared["stacked"]
    feature_names = prepared["feature_names"]
    complete = np.asarray(prepared["complete_origin"], dtype=bool)
    if len(complete) != len(calendar):
        raise ValueError("Prepared completeness mask does not match the NSE calendar.")

    viable: list[int] = []
    for pos in range(max(HORIZONS) + 1, len(calendar) - max(HORIZONS)):
        if not complete[pos]:
            continue
        train = stacked.loc[stacked["origin_position"] < pos - max(HORIZONS)]
        if train["origin_position"].nunique() >= min_train_origins:
            viable.append(pos)
    if not viable:
        return ({
            "status": "BLOCKED_NO_VIABLE_ORIGINS", "symbol": str(symbol).upper(),
            "history_rows": int(len(daily)), "calendar_sessions": int(len(calendar)),
            "min_train_origins": int(min_train_origins),
            "reason": "No issue dates have enough matured history and seven complete future outcomes.",
            "read_only": True, "production_model_changed": False, "orders_placed": False,
        }, pd.DataFrame(), pd.DataFrame())

    first = viable[0]
    positions = [p for p in viable if (p - first) % step_sessions == 0]
    if len(positions) > max_origins:
        indices = np.linspace(0, len(positions) - 1, num=max_origins).round().astype(int)
        positions = list(dict.fromkeys(positions[i] for i in indices))

    # Always score the most recent fully matured origin. A regular five-session
    # grid can otherwise stop one to four sessions early (e.g. target 14 Oct while
    # completed 15 Oct outcomes are available), which misleads users into thinking
    # the latest source date is missing. Preserve the cap by replacing the last
    # selected origin when necessary.
    latest_viable_origin = viable[-1]
    if latest_viable_origin not in positions:
        if len(positions) >= max_origins:
            positions[-1] = latest_viable_origin
        else:
            positions.append(latest_viable_origin)
        positions = sorted(set(positions))

    target_cols = [f"target_{t}" for t in TARGETS]
    rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []

    for fold_no, pos in enumerate(positions, start=1):
        # The largest training label ends before the current origin: p + 7 < pos.
        train = stacked.loc[stacked["origin_position"] < pos - max(HORIZONS)].copy()
        valid = stacked.loc[stacked["origin_position"].eq(pos)].sort_values("forecast_horizon").reset_index(drop=True)
        if train["origin_position"].nunique() < min_train_origins:
            continue
        if len(valid) != len(HORIZONS) or valid["forecast_horizon"].astype(int).tolist() != list(HORIZONS):
            continue

        x_train = train[feature_names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        x_valid = valid[feature_names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        models = {}
        for target in TARGETS:
            model = _new_model(len(train))
            model.fit(x_train, train[f"target_{target}"].to_numpy(dtype=float))
            models[target] = model

        raw = np.column_stack([models[t].predict(x_valid) for t in TARGETS]).astype(float)
        median = np.full_like(raw, np.nan)
        horizons = valid["forecast_horizon"].to_numpy(dtype=int)
        for r, horizon in enumerate(horizons):
            train_h = train.loc[train["forecast_horizon"].eq(float(horizon))]
            for c, target in enumerate(TARGETS):
                vals = train_h[f"target_{target}"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if not len(vals):
                    raise ValueError(f"No purged training values for horizon {horizon}/{target}.")
                median[r, c] = float(np.median(vals))

        anchors = valid["anchor_close"].to_numpy(dtype=float)
        coherent = _cohere_log_return_predictions(raw, anchors)
        coherent_median = _cohere_log_return_predictions(median, anchors)
        zero_return = _cohere_log_return_predictions(np.zeros_like(raw), anchors)
        actual = valid[target_cols].to_numpy(dtype=float)
        if not all(np.isfinite(a).all() for a in (raw, coherent, median, coherent_median, zero_return, actual)):
            raise ValueError(f"Non-finite values in rolling fold at {calendar[pos].date()}.")

        raw_px = anchors[:, None] * np.exp(raw)
        coherent_px = anchors[:, None] * np.exp(coherent)
        median_px = anchors[:, None] * np.exp(coherent_median)
        prev_px = anchors[:, None] * np.exp(zero_return)
        actual_px = anchors[:, None] * np.exp(actual)
        high_corrected = coherent_px[:, 1] > raw_px[:, 1] + 1e-8
        low_corrected = coherent_px[:, 2] < raw_px[:, 2] - 1e-8
        any_corrected = high_corrected | low_corrected
        high_equals_close = np.isclose(coherent_px[:, 1], coherent_px[:, 3], rtol=1e-9, atol=1e-8)
        raw_high_below_close = raw_px[:, 1] < raw_px[:, 3] - 1e-8
        corrections.append({
            "origin_date": calendar[pos].date().isoformat(),
            "horizon_count": len(HORIZONS),
            "any_correction_count": int(any_corrected.sum()),
            "high_corrected_count": int(high_corrected.sum()),
            "low_corrected_count": int(low_corrected.sum()),
            "high_equals_close_count": int(high_equals_close.sum()),
            "raw_high_below_close_count": int(raw_high_below_close.sum()),
            "mean_high_correction_bps": float(np.mean((coherent_px[:, 1] - raw_px[:, 1]) / anchors) * BPS),
            "mean_low_correction_bps": float(np.mean((raw_px[:, 2] - coherent_px[:, 2]) / anchors) * BPS),
        })

        max_train_origin = int(train["origin_position"].max())
        max_train_label = int((train["origin_position"] + train["forecast_horizon"].astype(int)).max())
        if max_train_label >= pos:
            raise AssertionError(f"Label leakage: max target {max_train_label} >= issue position {pos}.")
        origin_date = calendar[pos].date().isoformat()
        folds.append({
            "fold": fold_no, "origin_position": int(pos), "origin_date": origin_date,
            "target_dates": [calendar[pos + h].date().isoformat() for h in HORIZONS],
            "training_rows": int(len(train)),
            "training_origin_count": int(train["origin_position"].nunique()),
            "max_training_origin_date": calendar[max_train_origin].date().isoformat(),
            "max_training_label_date": calendar[max_train_label].date().isoformat(),
            "leakage_check_pass": True,
        })

        for r, horizon in enumerate(horizons):
            target_date = calendar[pos + int(horizon)].date().isoformat()
            row: dict[str, Any] = {
                "symbol": str(symbol).strip().upper(), "fold": fold_no,
                "origin_date": origin_date, "target_date": target_date,
                "horizon": int(horizon), "anchor_close": float(anchors[r]),
                "high_corrected": bool(high_corrected[r]), "low_corrected": bool(low_corrected[r]),
                "any_ohlc_corrected": bool(any_corrected[r]),
                "high_equals_close_after_coherence": bool(high_equals_close[r]),
                "raw_high_below_close": bool(raw_high_below_close[r]),
            }
            for c, target in enumerate(TARGETS):
                row[f"actual_{target}_price"] = float(actual_px[r, c])
                row[f"raw_{target}_forecast_price"] = float(raw_px[r, c])
                row[f"coherent_{target}_forecast_price"] = float(coherent_px[r, c])
                row[f"median_{target}_baseline_price"] = float(median_px[r, c])
                row[f"previous_close_{target}_baseline_price"] = float(prev_px[r, c])
                row[f"actual_{target}_log_return"] = float(actual[r, c])
                row[f"raw_{target}_log_return"] = float(raw[r, c])
                row[f"coherent_{target}_log_return"] = float(coherent[r, c])
            rows.append(row)

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        return ({
            "status": "BLOCKED_NO_VALID_FOLDS", "symbol": str(symbol).upper(),
            "reason": "No complete rolling-origin folds could be scored.",
            "read_only": True, "production_model_changed": False, "orders_placed": False,
        }, predictions, pd.DataFrame())

    metric_rows: list[dict[str, Any]] = []
    for scope, subset in [
        ("per_horizon_target", predictions),
        ("all_horizons_target", predictions),
    ]:
        groups = (
            [(int(h), t, predictions.loc[predictions["horizon"].eq(h)]) for h in HORIZONS for t in TARGETS]
            if scope == "per_horizon_target"
            else [("ALL", t, predictions) for t in TARGETS]
        )
        for horizon, target, sub in groups:
            actual = sub[f"actual_{target}_log_return"].to_numpy(dtype=float)
            raw = sub[f"raw_{target}_log_return"].to_numpy(dtype=float)
            coherent = sub[f"coherent_{target}_log_return"].to_numpy(dtype=float)
            anchor = sub["anchor_close"].to_numpy(dtype=float)
            median = np.log(sub[f"median_{target}_baseline_price"].to_numpy(dtype=float) / anchor)
            previous = np.log(sub[f"previous_close_{target}_baseline_price"].to_numpy(dtype=float) / anchor)
            model_mae, median_mae, previous_mae = _mae(actual, coherent), _mae(actual, median), _mae(actual, previous)
            metric: dict[str, Any] = {
                "scope": scope, "horizon": horizon, "target": target,
                "origin_count": int(sub["origin_date"].nunique()),
                "raw_mae_bps": _mae(actual, raw), "raw_rmse_bps": _rmse(actual, raw),
                "coherent_mae_bps": model_mae, "coherent_rmse_bps": _rmse(actual, coherent),
                "median_baseline_mae_bps": median_mae, "median_baseline_rmse_bps": _rmse(actual, median),
                "previous_close_baseline_mae_bps": previous_mae,
                "previous_close_baseline_rmse_bps": _rmse(actual, previous),
                "coherent_mae_skill_vs_median_pct": 100 * (1 - model_mae / median_mae) if median_mae > 0 else None,
                "coherent_mae_skill_vs_previous_close_pct": 100 * (1 - model_mae / previous_mae) if previous_mae > 0 else None,
            }
            if target == "close":
                metric["raw_directional_accuracy_pct"] = float(np.mean(np.sign(actual) == np.sign(raw)) * 100)
                metric["coherent_directional_accuracy_pct"] = float(np.mean(np.sign(actual) == np.sign(coherent)) * 100)
                metric["median_baseline_directional_accuracy_pct"] = float(np.mean(np.sign(actual) == np.sign(median)) * 100)
            metric_rows.append(metric)
    metrics = pd.DataFrame(metric_rows)
    correction_df = pd.DataFrame(corrections)
    total_rows = len(predictions)
    close_summary = metrics.loc[metrics["scope"].eq("all_horizons_target") & metrics["target"].eq("close")].iloc[0].to_dict()
    report = {
        "status": "COMPLETE" if predictions["origin_date"].nunique() >= min_evaluation_origins else "INSUFFICIENT_EVALUATION_ORIGINS",
        "symbol": str(symbol).strip().upper(),
        "cutoff_completed_session": cutoff.isoformat(),
        "history_first_date": daily.index.min().date().isoformat(),
        "history_last_date": daily.index.max().date().isoformat(),
        "calendar_first_date": calendar.min().date().isoformat(),
        "calendar_last_date": calendar.max().date().isoformat(),
        "history_rows": int(len(daily)), "calendar_sessions": int(len(calendar)),
        "missing_stock_sessions_on_calendar": int((~calendar.isin(daily.index)).sum()),
        "min_train_origins": int(min_train_origins), "step_sessions": int(step_sessions),
        "max_origins": int(max_origins), "rolling_origin_count": int(predictions["origin_date"].nunique()),
        "latest_forecast_origin_date": str(predictions["origin_date"].max()),
        "latest_evaluated_target_date": str(predictions["target_date"].max()),
        "forecast_rows": int(total_rows), "horizons": list(HORIZONS), "targets": list(TARGETS),
        "aggregate_close_metric": close_summary,
        "coherence_corrections": {
            "origin_horizon_rows": int(total_rows),
            "rows_any_correction": int(correction_df["any_correction_count"].sum()),
            "rows_high_corrected": int(correction_df["high_corrected_count"].sum()),
            "rows_low_corrected": int(correction_df["low_corrected_count"].sum()),
            "rows_high_equal_close_after_coherence": int(correction_df["high_equals_close_count"].sum()),
            "rows_raw_high_below_close": int(correction_df["raw_high_below_close_count"].sum()),
            "high_equals_close_after_coherence_pct": 100 * float(correction_df["high_equals_close_count"].sum()) / max(1, total_rows),
            "raw_high_below_close_pct": 100 * float(correction_df["raw_high_below_close_count"].sum()) / max(1, total_rows),
            "mean_high_correction_bps": float(correction_df["mean_high_correction_bps"].mean()),
            "mean_low_correction_bps": float(correction_df["mean_low_correction_bps"].mean()),
        },
        "folds": folds,
        "validation_method": "expanding-window rolling origin; model refit at each sampled issue date; seven-session purge at every fold",
        "safety": {
            "read_only": True, "production_model_changed": False,
            "production_registry_changed": False, "prediction_ledger_changed": False,
            "drive_written": False, "orders_placed": False,
        },
        "interpretation": [
            "Positive MAE skill means lower log-return absolute error than the named baseline, not positive trading profit.",
            "Raw forecasts are before coherence; coherent forecasts are the values displayed by the dashboard.",
            "High=Close frequency quantifies cases where the OHLC-coherence rule raises the predicted high to the close.",
            "Missing stock sessions stay missing on the NSE calendar; targets are never shifted to a later observed stock row.",
            "Even a positive audit does not authorize real-money trading; paper validation and transaction-cost backtests remain required.",
        ],
    }
    if report["status"] != "COMPLETE":
        report["reason"] = f"Only {report['rolling_origin_count']} origins scored; at least {min_evaluation_origins} are required."
    return report, predictions, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="TCS", help="NSE symbol, e.g. TCS or RELIANCE")
    parser.add_argument("--period", default="900d")
    parser.add_argument("--min-train-origins", type=int, default=160)
    parser.add_argument("--step-sessions", type=int, default=5)
    parser.add_argument("--max-origins", type=int, default=24)
    parser.add_argument("--out-dir", default="diagnostics/rolling_daily_ohlc_audit")
    args = parser.parse_args()
    try:
        history, source_meta = load_daily_history(args.symbol, args.period)
        calendar_frame, calendar_meta = load_daily_history("^NSEI", args.period)
        calendar = pd.DatetimeIndex(calendar_frame.index).normalize().unique().sort_values()
        report, predictions, metrics = run_rolling_origin_audit(
            history, calendar, symbol=args.symbol, min_train_origins=args.min_train_origins,
            step_sessions=args.step_sessions, max_origins=args.max_origins,
        )
        report["source"] = {"stock": source_meta, "nse_calendar": calendar_meta}
    except Exception as exc:
        print(json.dumps({"status": "AUDIT_ERROR", "error": f"{type(exc).__name__}: {exc}",
                          "read_only": True, "production_model_changed": False, "orders_placed": False}, indent=2))
        return 2
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rolling_daily_ohlc_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if not predictions.empty:
        predictions.to_csv(out / "rolling_daily_ohlc_predictions.csv", index=False)
        metrics.to_csv(out / "rolling_daily_ohlc_metrics.csv", index=False)
    print(json.dumps({
        "status": report["status"], "symbol": report.get("symbol"),
        "rolling_origin_count": report.get("rolling_origin_count", 0),
        "aggregate_close_metric": report.get("aggregate_close_metric"),
        "coherence_corrections": report.get("coherence_corrections"),
        "output_dir": str(out), "read_only": True, "production_model_changed": False,
        "orders_placed": False,
    }, indent=2, default=str))
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
