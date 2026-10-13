"""Opt-in runtime timeframe telemetry and time-tolerant model-selection fitness."""
from __future__ import annotations

import json
import os
from typing import Any, Iterable

import numpy as np
import pandas as pd


def runtime_diagnostics_enabled() -> bool:
    """Read opt-in from environment first, then Streamlit Cloud secrets.

    Colab environment variables do not propagate to a separately hosted
    Streamlit deployment. Defining QUANT_TIMEFRAME_DIAGNOSTICS="true" in the
    deployed app's Secrets enables telemetry there without enabling it by
    default for ordinary installations.
    """
    env_value = os.getenv("QUANT_TIMEFRAME_DIAGNOSTICS")
    if env_value is not None:
        return str(env_value).strip().lower() in {"1", "true", "yes", "on"}

    try:
        import streamlit as st
        setting = st.secrets.get("QUANT_TIMEFRAME_DIAGNOSTICS", "0")
    except Exception:
        setting = "0"

    return str(setting).strip().lower() in {"1", "true", "yes", "on"}


def report_runtime_cadence(
    stage: str,
    timestamps: Iterable[Any],
    *,
    session_timezone: str = "Asia/Kolkata",
) -> dict[str, Any]:
    """Measure timestamp gaps in an actual in-memory inference input; no file search."""
    try:
        values = list(timestamps)
    except TypeError:
        values = []
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    ts = (
        pd.Series(parsed).dropna().drop_duplicates().sort_values()
        .reset_index(drop=True)
    )
    if ts.empty:
        report = {
            "stage": stage, "status": "NO_VALID_TIMESTAMPS",
            "timestamp_count": 0, "latest_timestamp_utc": None,
            "median_interval_minutes": None, "common_intervals_minutes": {},
        }
        print("[QUANT_TIMEFRAME] " + json.dumps(report, default=str, sort_keys=True))
        return report

    local_ts = ts.dt.tz_convert(session_timezone)
    table = pd.DataFrame({"ts_utc": ts, "session": local_ts.dt.date})
    gaps = (
        table.groupby("session", sort=False)["ts_utc"].diff()
        .dt.total_seconds().div(60.0).dropna()
    )
    positive = gaps[gaps > 0]
    common = {
        f"{float(gap):g}": int(count)
        for gap, count in positive.round(3).value_counts().head(8).items()
    } if not positive.empty else {}
    latest = pd.Timestamp(ts.iloc[-1])
    report = {
        "stage": stage,
        "status": "PASS",
        "timestamp_count": int(len(ts)),
        "first_timestamp_utc": pd.Timestamp(ts.iloc[0]).isoformat(),
        "latest_timestamp_utc": latest.isoformat(),
        "first_timestamp_local": pd.Timestamp(ts.iloc[0]).tz_convert(session_timezone).isoformat(),
        "latest_timestamp_local": latest.tz_convert(session_timezone).isoformat(),
        "median_interval_minutes": round(float(positive.median()), 3) if not positive.empty else None,
        "min_interval_minutes": round(float(positive.min()), 3) if not positive.empty else None,
        "max_interval_minutes": round(float(positive.max()), 3) if not positive.empty else None,
        "common_intervals_minutes": common,
        "note": (
            "Observed intervals are measured from this in-memory input only. "
            "Missing bars and provider timestamp semantics can affect the estimate."
        ),
    }
    print("[QUANT_TIMEFRAME] " + json.dumps(report, default=str, sort_keys=True))
    return report


def runtime_event(event: str, **fields: Any) -> None:
    """Print structured runtime telemetry."""
    print("[QUANT_RUNTIME] " + json.dumps(
        {"event": event, **fields}, default=str, sort_keys=True
    ))


def time_tolerant_fitness(
    y_true: Iterable[int],
    p_up: Iterable[float],
    predicted_event_ts: Iterable[Any],
    actual_event_ts: Iterable[Any],
    feature_importance: dict[str, float],
    feature_age_minutes: dict[str, float],
    *,
    feature_reaction_lag_minutes: dict[str, float] | None = None,
    net_returns: Iterable[float] | None = None,
    tolerance_minutes: float = 5.0,
    timing_decay_minutes: float = 10.0,
    freshness_budget_minutes: float = 5.0,
    lag_scale_minutes: float = 15.0,
    lambda_timing: float = 0.50,
    lambda_freshness: float = 0.25,
    lambda_indicator_lag: float = 0.25,
    lambda_net_return: float = 10.0,
    minimum_timing_coverage: float = 0.90,
    minimum_feature_coverage: float = 0.80,
) -> dict[str, Any]:
    """Evaluate a candidate model; lower loss is better.

    Event timestamps must have a consistent definition (such as first threshold
    crossing). Feature importance and empirical reaction lag must be computed
    on training folds only. Feature age is actual source age at decision time.
    net_returns must be realised strategy returns after realistic costs.

    This is an evaluation/model-selection objective, not a gradient-based
    replacement for sklearn LogisticRegression.fit() or HistGradientBoosting.fit().
    """
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(p_up), dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("y_true and p_up must have equal non-zero lengths.")
    if not np.isfinite(y).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("y_true must contain only finite 0/1 labels.")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_up must contain finite probabilities in [0, 1].")
    if tolerance_minutes < 0 or timing_decay_minutes <= 0:
        raise ValueError("Invalid time tolerance or timing decay.")
    if freshness_budget_minutes < 0 or lag_scale_minutes <= 0:
        raise ValueError("Invalid freshness budget or lag scale.")

    eps = 1e-7
    p_safe = np.clip(p, eps, 1.0 - eps)
    direction_log_loss = float(-np.mean(
        y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe)
    ))

    predicted = pd.to_datetime(
        pd.Series(list(predicted_event_ts)), errors="coerce", utc=True
    )
    actual = pd.to_datetime(
        pd.Series(list(actual_event_ts)), errors="coerce", utc=True
    )
    if len(predicted) != len(y) or len(actual) != len(y):
        raise ValueError("Event timestamp arrays must match y_true length.")
    valid_time = predicted.notna() & actual.notna()
    time_coverage = float(valid_time.mean())
    timing_penalty = None
    timing_mae = None
    if valid_time.any():
        errors = (
            (predicted[valid_time] - actual[valid_time]).abs()
            .dt.total_seconds().to_numpy(dtype=float) / 60.0
        )
        excess = np.maximum(errors - tolerance_minutes, 0.0)
        timing_penalty = float(np.mean(
            1.0 - np.exp(-excess / timing_decay_minutes)
        ))
        timing_mae = float(np.mean(errors))

    common = sorted(set(feature_importance) & set(feature_age_minutes))
    importance = np.asarray([
        max(0.0, float(feature_importance[n])) for n in common
    ], dtype=float)
    ages = np.asarray([float(feature_age_minutes[n]) for n in common], dtype=float)
    finite_ages = np.isfinite(ages) & (ages >= 0.0)
    age_coverage = float(finite_ages.mean()) if len(common) else 0.0
    freshness_penalty = None
    scored_features: list[str] = []
    if len(common) and importance.sum() > 0 and age_coverage >= minimum_feature_coverage:
        weights = importance / importance.sum()
        age_excess = np.maximum(
            np.nan_to_num(ages, nan=2.0 * lag_scale_minutes) - freshness_budget_minutes,
            0.0,
        )
        freshness_penalty = float(np.sum(
            weights * np.clip(age_excess / lag_scale_minutes, 0.0, 1.0)
        ))
        scored_features = common

    reaction_penalty = None
    reaction_coverage = 0.0
    lag_map = feature_reaction_lag_minutes or {}
    lag_common = sorted(set(feature_importance) & set(lag_map))
    lag_weights_raw = np.asarray([
        max(0.0, float(feature_importance[n])) for n in lag_common
    ], dtype=float)
    reaction = np.asarray([float(lag_map[n]) for n in lag_common], dtype=float)
    valid_reaction = np.isfinite(reaction) & (reaction >= 0.0)
    reaction_coverage = float(valid_reaction.mean()) if len(lag_common) else 0.0
    if (
        lag_common and lag_weights_raw.sum() > 0
        and reaction_coverage >= minimum_feature_coverage
    ):
        lag_weights = lag_weights_raw / lag_weights_raw.sum()
        reaction = np.nan_to_num(reaction, nan=2.0 * lag_scale_minutes)
        reaction_penalty = float(np.sum(
            lag_weights * np.clip(reaction / lag_scale_minutes, 0.0, 1.0)
        ))
        scored_features = sorted(set(scored_features) | set(lag_common))

    mean_net_return = None
    net_return_component = None
    net_coverage = 0.0
    if net_returns is not None:
        returns = np.asarray(list(net_returns), dtype=float)
        if len(returns) != len(y):
            raise ValueError("net_returns length must match y_true.")
        finite_returns = returns[np.isfinite(returns)]
        net_coverage = float(len(finite_returns) / len(returns))
        if len(finite_returns):
            mean_net_return = float(np.mean(finite_returns))
            # Profitable net performance lowers loss; negative performance raises it.
            net_return_component = -lambda_net_return * mean_net_return

    missing = []
    if timing_penalty is None or time_coverage < minimum_timing_coverage:
        missing.append("insufficient_event_timestamp_coverage")
    if freshness_penalty is None:
        missing.append("insufficient_feature_age_coverage")
    if feature_reaction_lag_minutes is not None and reaction_penalty is None:
        missing.append("insufficient_empirical_indicator_lag_coverage")

    eligible = not missing
    loss = None
    if eligible:
        loss = direction_log_loss + lambda_timing * timing_penalty
        loss += lambda_freshness * freshness_penalty
        if reaction_penalty is not None:
            loss += lambda_indicator_lag * reaction_penalty
        if net_return_component is not None:
            loss += net_return_component

    return {
        "eligible_for_comparison": eligible,
        "fitness_loss": float(loss) if loss is not None else None,
        "lower_is_better": True,
        "direction_log_loss": direction_log_loss,
        "time_tolerance_penalty": timing_penalty,
        "timing_mae_minutes": timing_mae,
        "timing_coverage": time_coverage,
        "tolerance_minutes": float(tolerance_minutes),
        "freshness_penalty": freshness_penalty,
        "feature_age_coverage": age_coverage,
        "empirical_indicator_lag_penalty": reaction_penalty,
        "empirical_indicator_lag_coverage": reaction_coverage,
        "features_scored_for_lag": scored_features,
        "mean_net_return_after_costs": mean_net_return,
        "net_return_component": net_return_component,
        "net_return_coverage": net_coverage,
        "missing_components": missing,
        "warning": (
            "Use chronological/purged validation folds. Derive feature importance "
            "and reaction lag from training folds and net returns after costs. "
            "This score does not guarantee live execution profitability."
        ),
    }
