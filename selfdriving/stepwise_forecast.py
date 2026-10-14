"""Presentation transform for a stepwise view of direct multi-horizon forecasts.

This does not re-run a model recursively. It lays each direct horizon endpoint
out as a price path: the first step is measured from the latest observed price,
and each subsequent step is measured from the immediately previous horizon's
predicted endpoint. Original model-origin references/returns remain in separate
fields so the UI cannot silently erase the distinction.
"""
from __future__ import annotations

import math
from typing import Any, Iterable


def _finite_positive(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def stepwise_horizon_path(
    predictions: Iterable[dict[str, Any]],
    *,
    current_reference_price: Any = None,
) -> list[dict[str, Any]]:
    """Return sorted copies whose displayed reference is the prior path endpoint.

    The original signal and probability are retained as direct horizon model
    outputs from the current decision timestamp. This function changes only
    presentation fields; it does not produce a recursively re-inferred forecast.
    """
    rows = [dict(item) for item in predictions]
    rows.sort(key=lambda row: int(row.get("horizon_min") or 0))

    base = _finite_positive(current_reference_price)
    if base is None:
        for row in rows:
            base = _finite_positive(row.get("reference_close"))
            if base is not None:
                break

    previous_endpoint: float | None = None
    previous_horizon: int | None = None
    for position, row in enumerate(rows):
        horizon = int(row.get("horizon_min") or 0)
        original_reference = row.get("reference_close")
        original_return = row.get("expected_return")
        endpoint = _finite_positive(row.get("predicted_price"))

        row["model_reference_close"] = original_reference
        row["model_expected_return_from_now"] = original_return
        row["model_expected_move_from_now"] = (
            endpoint - _finite_positive(original_reference)
            if endpoint is not None and _finite_positive(original_reference) is not None
            else None
        )
        row["forecast_view"] = "STEPWISE_PATH_FROM_DIRECT_HORIZON_ENDPOINTS"

        if position == 0:
            reference = base
            row["reference_price_source"] = (
                "LATEST_OBSERVED_MARKET_PRICE"
                if _finite_positive(current_reference_price) is not None
                else "MODEL_ORIGIN_REFERENCE_FALLBACK"
            )
            row["step_interval"] = f"Now → +{horizon} min"
        else:
            reference = previous_endpoint
            row["reference_price_source"] = (
                f"PREVIOUS_PLUS_{previous_horizon}_MIN_FORECAST_ENDPOINT"
                if reference is not None
                else "BLOCKED_PREVIOUS_HORIZON_PRICE_UNAVAILABLE"
            )
            row["step_interval"] = (
                f"+{previous_horizon} → +{horizon} min"
                if previous_horizon is not None
                else f"Previous → +{horizon} min"
            )

        if reference is not None and endpoint is not None:
            move = endpoint - reference
            step_return = endpoint / reference - 1.0
            direction = "UP" if move > 0 else ("DOWN" if move < 0 else "FLAT")
            row["reference_close"] = reference
            row["expected_move"] = move
            row["expected_return"] = step_return
            row["step_direction"] = direction
        else:
            row["reference_close"] = None if reference is None else reference
            row["expected_move"] = None
            row["expected_return"] = None
            row["step_direction"] = "UNAVAILABLE"

        previous_endpoint = endpoint
        previous_horizon = horizon

    return rows
