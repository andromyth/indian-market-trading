"""Descriptive model-health monitoring and retraining flags."""
from __future__ import annotations

from typing import Any

import pandas as pd


def assess_delivery_health(validation_df: pd.DataFrame, rolling_window: int = 7, min_samples: int = 7) -> dict[str, Any]:
    if validation_df.empty:
        return {"status": "NO_DATA", "sample_count": 0, "retraining_candidate": False}
    if "validation_status" not in validation_df.columns:
        return {"status": "NO_DATA", "sample_count": 0, "retraining_candidate": False}
    x = validation_df[validation_df["validation_status"] == "VALIDATED"].copy()
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
    matches = pd.to_numeric(x.get("direction_match_d1"), errors="coerce").dropna()
    mape = pd.to_numeric(x.get("percentage_error_d1"), errors="coerce").abs().mean()
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
    return {
        "status": "MONITOR",
        "sample_count": int(len(matches)),
        "directional_accuracy": accuracy,
        "mean_absolute_percentage_error": float(mape) if pd.notna(mape) else None,
        # This is intentionally only a flag. It never changes the production model.
        "retraining_candidate": accuracy < 0.50,
        "rule": "candidate flag when rolling directional accuracy falls below 50%; candidate still requires full retraining + walk-forward + 7-prediction validation before promotion.",
    }
