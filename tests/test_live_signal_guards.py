import pandas as pd

from live_signal_guards import (
    MIN_LIVE_FEATURE_COVERAGE,
    assess_horizon_completeness,
    feature_coverage_fraction,
    latest_completed_bar_status,
)


def test_live_bar_guard_accepts_fresh_completed_current_session_bar():
    now = pd.Timestamp("2026-10-12 12:47:30", tz="Asia/Kolkata")
    frame = pd.DataFrame(
        {"close": [858.0]},
        index=pd.DatetimeIndex(["2026-10-12 12:45:00"], tz="Asia/Kolkata"),
    )
    result = latest_completed_bar_status(frame, now)
    assert result["ok"] is True
    assert result["status"] == "PASS"
    assert result["age_seconds"] == 150.0


def test_live_bar_guard_rejects_old_bar():
    now = pd.Timestamp("2026-10-12 12:47:30", tz="Asia/Kolkata")
    frame = pd.DataFrame(
        {"close": [858.0]},
        index=pd.DatetimeIndex(["2026-10-12 12:40:00"], tz="Asia/Kolkata"),
    )
    result = latest_completed_bar_status(frame, now)
    assert result["ok"] is False
    assert result["status"] == "STALE"


def test_live_bar_guard_rejects_previous_session_even_if_clock_is_wrong():
    now = pd.Timestamp("2026-10-12 12:47:30", tz="Asia/Kolkata")
    frame = pd.DataFrame(
        {"close": [858.0]},
        index=pd.DatetimeIndex(["2026-10-09 15:29:00"], tz="Asia/Kolkata"),
    )
    result = latest_completed_bar_status(frame, now)
    assert result["ok"] is False
    assert result["status"] == "STALE_SESSION"


def test_feature_coverage_counts_missing_columns_non_numeric_and_infinite_as_missing():
    frame = pd.DataFrame({"a": [1.0], "b": [float("inf")], "c": ["bad"]})
    coverage = feature_coverage_fraction(frame, ["a", "b", "c", "missing"])
    assert coverage == 0.25
    assert MIN_LIVE_FEATURE_COVERAGE == 0.70


def test_horizon_guard_requires_all_displayed_horizons():
    horizons = {
        5: {"probability_up": 0.61, "signal": "ABSTAIN"},
        15: {"probability_up": 0.58, "signal": "HOLD"},
        30: {"signal": "ABSTAIN", "reason": "NO_CHAMPION"},
    }
    result = assess_horizon_completeness(horizons)
    assert result["status"] == "INCOMPLETE_HORIZONS"
    assert result["missing_horizons"] == [30]


def test_horizon_guard_does_not_mark_missing_champions_as_pass():
    horizons = {
        5: {"signal": "ABSTAIN", "reason": "NO_CHAMPION"},
        15: {"signal": "ABSTAIN", "reason": "NO_CHAMPION"},
        30: {"signal": "ABSTAIN", "reason": "NO_CHAMPION"},
    }
    result = assess_horizon_completeness(horizons)
    assert result["status"] == "NO_VALIDATED_MODEL"
