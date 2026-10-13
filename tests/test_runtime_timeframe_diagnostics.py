import pandas as pd

from selfdriving.runtime_timeframe_diagnostics import (
    report_runtime_cadence,
    time_tolerant_fitness,
)


def test_runtime_cadence_measures_intraday_intervals(capsys):
    ts = pd.to_datetime([
        "2026-10-12T09:15:00+05:30",
        "2026-10-12T09:16:00+05:30",
        "2026-10-12T09:17:00+05:30",
        "2026-10-12T09:22:00+05:30",
    ])
    report = report_runtime_cadence("test", ts)
    assert report["median_interval_minutes"] == 1.0
    assert report["timestamp_count"] == 4
    assert "[QUANT_TIMEFRAME]" in capsys.readouterr().out


def test_time_tolerant_fitness_scores_timing_and_lag():
    actual = pd.to_datetime([
        "2026-10-12T09:35:00+05:30", "2026-10-12T09:40:00+05:30",
        "2026-10-12T09:45:00+05:30", "2026-10-12T09:50:00+05:30",
    ])
    predicted = pd.to_datetime([
        "2026-10-12T09:36:00+05:30", "2026-10-12T09:41:00+05:30",
        "2026-10-12T09:49:00+05:30", "2026-10-12T09:51:00+05:30",
    ])
    result = time_tolerant_fitness(
        [1, 0, 1, 0], [0.8, 0.2, 0.7, 0.3], predicted, actual,
        feature_importance={"ret_1m": 0.7, "ma50_ratio": 0.3},
        feature_age_minutes={"ret_1m": 0.5, "ma50_ratio": 8.0},
        feature_reaction_lag_minutes={"ret_1m": 1.0, "ma50_ratio": 12.0},
        net_returns=[0.002, -0.001, 0.001, 0.0005],
        tolerance_minutes=2,
    )
    assert result["eligible_for_comparison"] is True
    assert result["fitness_loss"] is not None
    assert result["time_tolerance_penalty"] >= 0
    assert result["freshness_penalty"] >= 0
    assert result["empirical_indicator_lag_penalty"] >= 0


def test_time_tolerant_fitness_rejects_missing_event_timestamps():
    result = time_tolerant_fitness(
        [1, 0], [0.7, 0.3], [None, None], [None, None],
        feature_importance={"ret_1m": 1.0},
        feature_age_minutes={"ret_1m": 1.0},
    )
    assert result["eligible_for_comparison"] is False
    assert result["fitness_loss"] is None
    assert "insufficient_event_timestamp_coverage" in result["missing_components"]
