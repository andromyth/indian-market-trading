"""Fail-closed live-input and feature-coverage guards for stock signals.

These guards validate data availability, not model profitability. They are
intentionally small and dependency-light so they can be unit-tested offline.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
MIN_LIVE_FEATURE_COVERAGE = 0.70
MAX_LIVE_BAR_AGE_SECONDS = 5 * 60


def feature_coverage_fraction(frame: pd.DataFrame | None, features: Iterable[str]) -> float:
    """Return finite numeric coverage for the final row over named model features.

    Missing columns count as missing inputs. NaN, +/-inf and non-numeric values
    also count as missing; the function never imputes feature values.
    """
    columns = list(dict.fromkeys(str(c) for c in features if str(c)))
    if frame is None or frame.empty or not columns:
        return 0.0
    selected = frame.reindex(columns=columns)
    numeric = selected.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(values)
    if finite.shape[0] == 0 or finite.shape[1] == 0:
        return 0.0
    return float(finite[-1].mean())


def latest_completed_bar_status(
    frame: pd.DataFrame | None,
    now: Any = None,
    *,
    max_age_seconds: int = MAX_LIVE_BAR_AGE_SECONDS,
    require_current_session: bool = True,
) -> dict[str, Any]:
    """Check that the latest completed bar is from today and sufficiently fresh.

    Naive timestamps are interpreted as India Standard Time. The timestamp at
    the current minute is excluded because that candle may still be forming.
    """
    now_ts = pd.Timestamp.now(tz=IST) if now is None else pd.Timestamp(now)
    now_ts = now_ts.tz_localize(IST) if now_ts.tzinfo is None else now_ts.tz_convert(IST)

    if frame is None or frame.empty:
        return {"ok": False, "status": "NO_DATA", "reason": "No benchmark candles returned."}

    try:
        idx = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
        idx = idx[~idx.isna()]
        if idx.empty:
            return {"ok": False, "status": "INVALID_TIMESTAMPS", "reason": "No valid benchmark candle timestamps."}
        idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
    except Exception as exc:
        return {
            "ok": False,
            "status": "INVALID_TIMESTAMPS",
            "reason": f"Cannot normalize benchmark timestamps: {type(exc).__name__}: {exc}",
        }

    completed = idx[idx < now_ts.floor("min")]
    if completed.empty:
        return {"ok": False, "status": "WAITING_FOR_BAR", "reason": "No completed benchmark candle is available."}

    latest = pd.Timestamp(completed.max())
    age_seconds = max(0.0, float((now_ts - latest).total_seconds()))
    base = {
        "latest_candle_ist": latest.isoformat(),
        "age_seconds": age_seconds,
        "max_age_seconds": int(max_age_seconds),
    }

    if require_current_session and latest.date() != now_ts.date():
        return {
            **base,
            "ok": False,
            "status": "STALE_SESSION",
            "reason": f"Benchmark candle is from {latest.date()}, not the current session {now_ts.date()}.",
        }
    if age_seconds > max_age_seconds:
        return {
            **base,
            "ok": False,
            "status": "STALE",
            "reason": f"Latest benchmark candle is {age_seconds / 60:.1f} minutes old; maximum allowed is {max_age_seconds / 60:.1f} minutes.",
        }

    return {**base, "ok": True, "status": "PASS", "reason": None}


def assess_horizon_completeness(
    horizons: Mapping[Any, Mapping[str, Any]],
    required_horizons: Iterable[int] = (5, 15, 30),
) -> dict[str, Any]:
    """Require a numeric model probability for each horizon shown as live output."""
    required = list(required_horizons)
    missing: list[int] = []
    details: list[str] = []
    all_missing_because_no_champion = True

    for horizon in required:
        item = horizons.get(horizon, horizons.get(str(horizon), {})) or {}
        probability = pd.to_numeric(item.get("probability_up"), errors="coerce")
        valid_probability = pd.notna(probability) and np.isfinite(float(probability))
        if valid_probability:
            all_missing_because_no_champion = False
            continue
        missing.append(int(horizon))
        reason = str(item.get("reason") or item.get("signal_reason") or "NO_PROBABILITY")
        details.append(f"+{horizon}m: {reason}")
        if reason != "NO_CHAMPION":
            all_missing_because_no_champion = False

    if not missing:
        return {"status": "PASS", "missing_horizons": [], "reason": None}

    status = "NO_VALIDATED_MODEL" if all_missing_because_no_champion else "INCOMPLETE_HORIZONS"
    return {
        "status": status,
        "missing_horizons": missing,
        "reason": "Required live horizon output unavailable — " + "; ".join(details),
    }
