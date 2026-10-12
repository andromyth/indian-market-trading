import pandas as pd

from delivery_production_refresh import readiness_summary


def test_readiness_summary_handles_nullable_float_na_without_crashing():
    values = pd.Series([1.0, pd.NA, 0.0], dtype="Float64")

    assert readiness_summary(values) == {
        "known_pct": 66.667,
        "ready_pct": 33.333,
        "unknown_rows": 1,
    }


def test_readiness_summary_all_unknown_rows_are_never_counted_as_ready():
    values = pd.Series([pd.NA, pd.NA], dtype="Float64")

    assert readiness_summary(values) == {
        "known_pct": 0.0,
        "ready_pct": 0.0,
        "unknown_rows": 2,
    }


def test_readiness_summary_empty_input_has_deterministic_zero_percentages():
    assert readiness_summary(pd.Series([], dtype="Float64")) == {
        "known_pct": 0.0,
        "ready_pct": 0.0,
        "unknown_rows": 0,
    }
