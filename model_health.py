"""Descriptive model-health monitoring and retraining flags."""
from __future__ import annotations

from typing import Any

import pandas as pd


def assess_delivery_health(validation_df: pd.DataFrame, rolling_window: int = 7, min_samples: int = 7) -> dict[str, Any]:
    if validation_df.empty:
        return {"status": "NO_DATA", "sample_count": 0, "retraining_candidate": False}
    if "validation_status" not in validation_df.columns:
        return {"status": "NO_DATA", "sample_count": 0, "retraining_candidate": False}

    x = validation_df[
        validation_df["validation_status"].astype(str).str.strip().str.upper().eq("VALIDATED")
    ].copy()
    if x.empty:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "sample_count": 0,
            "directional_accuracy": None,
            "mean_absolute_percentage_error": None,
            "retraining_candidate": False,
            "reason": f"Need at least {min_samples} completed delivery predictions for the rolling health gate.",
        }

    if "decision_timestamp" in x.columns:
        x["decision_timestamp"] = pd.to_datetime(x["decision_timestamp"], errors="coerce")
        x = x.dropna(subset=["decision_timestamp"]).sort_values("decision_timestamp").tail(rolling_window)
    else:
        x = x.tail(rolling_window)

    signal = x.get("signal", pd.Series(index=x.index, dtype="object")).astype(str).str.strip().str.upper()
    actual = x.get("actual_direction_d1", pd.Series(index=x.index, dtype="object")).astype(str).str.strip().str.upper()

    matches = pd.Series(
        ((signal == "BUY") & actual.eq("UP"))
        | ((signal == "SELL") & actual.eq("DOWN"))
        | ((signal == "HOLD") & actual.eq("FLAT")),
        index=x.index,
        dtype="float64",
    )
    usable = actual.isin(["UP", "DOWN", "FLAT"]) & signal.isin(["BUY", "SELL", "HOLD"])
    matches = matches[usable].dropna()

    mape = pd.to_numeric(
        x.get("percentage_error_d1", pd.Series(index=x.index, dtype="float64")),
        errors="coerce",
    ).abs().mean()

    if len(matches) < min_samples:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "sample_count": int(len(matches)),
            "directional_accuracy": float(matches.mean()) if len(matches) else None,
            "mean_absolute_percentage_error": float(mape) if pd.notna(mape) else None,
            "retraining_candidate": False,
            "reason": f"Need at least {min_samples} completed delivery predictions for the rolling health gate.",
        }

    accuracy = float(matches.mean())

    action = signal.isin(["BUY", "SELL"])
    action_correct = (
        ((signal == "BUY") & actual.eq("UP"))
        | ((signal == "SELL") & actual.eq("DOWN"))
    )
    action_precision = float(action_correct[action].mean()) if bool(action.any()) else None
    action_coverage = float(action.mean()) if len(action) else None

    probability = pd.to_numeric(
        x.get("probability_up_next_session", pd.Series(index=x.index, dtype="float64")),
        errors="coerce",
    )
    actual_up = actual.map({"UP": 1.0, "DOWN": 0.0})
    brier = ((probability - actual_up) ** 2).dropna().mean()

    primary_health = (
        action_precision
        if action_precision is not None and int(action.sum()) >= 10
        else accuracy
    )

    return {
        "status": "MONITOR",
        "sample_count": int(len(matches)),
        "directional_accuracy": accuracy,
        "action_precision": action_precision,
        "action_coverage": action_coverage,
        "brier_score": float(brier) if pd.notna(brier) else None,
        "mean_absolute_percentage_error": float(mape) if pd.notna(mape) else None,
        "retraining_candidate": primary_health < 0.50,
        "health_metric_used": "action_precision" if action_precision is not None and int(action.sum()) >= 10 else "directional_accuracy",
        "rule": "candidate flag when the primary OOS health metric falls below 50%; candidate still requires full retraining + walk-forward + validation before promotion.",
    }
