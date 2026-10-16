"""Shared, explicit interpretation of production Delivery quality-gate state."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

CORE_READINESS = {
    "ready_market_price": "MARKET_PRICE",
    "ready_market_context": "MARKET_CONTEXT",
}
OPTIONAL_READINESS = {
    "ready_news": "NEWS",
    "ready_events": "EVENTS",
    "ready_fno": "FNO",
    "ready_fundamentals": "FUNDAMENTALS",
}


def _known_binary(value: Any) -> float:
    """Return 0/1 for an explicit readiness value; NaN means unknown."""
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "pass", "ready"}:
                return 1.0
            if token in {"0", "false", "no", "fail", "not_ready"}:
                return 0.0
            return np.nan
        number = float(value)
        return number if number in (0.0, 1.0) else np.nan
    except (TypeError, ValueError):
        return np.nan


def quality_gate_reasons(row: Any) -> list[str]:
    """Explain why a row cannot be actioned without treating unknown as false data."""
    reasons: list[str] = []
    for column, label in CORE_READINESS.items():
        value = _known_binary(row.get(column, np.nan))
        if not np.isfinite(value):
            reasons.append(f"{label}_READINESS_UNKNOWN")
        elif value != 1.0:
            reasons.append(f"{label}_NOT_READY")
    for column, label in OPTIONAL_READINESS.items():
        if not np.isfinite(_known_binary(row.get(column, np.nan))):
            reasons.append(f"{label}_STATE_UNKNOWN")
    return reasons


def apply_quality_gate(raw_signals: Iterable[Any], gate_values: Iterable[Any]) -> list[str]:
    """Never show a blocked model result as HOLD; expose the actual data status."""
    output: list[str] = []
    for signal, gate_value in zip(raw_signals, gate_values):
        gate = _known_binary(gate_value)
        if not np.isfinite(gate) or gate != 1.0:
            output.append("DATA INSUFFICIENT")
        else:
            output.append(str(signal))
    return output
