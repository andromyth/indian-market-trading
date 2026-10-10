from __future__ import annotations

import numpy as np
import pandas as pd

from delivery_policy_update import TARGET_HOLDOUT_PRECISION, _precision, _wilson_lower_bound


def test_wilson_lower_bound_is_below_observed_precision_for_finite_sample() -> None:
    assert _wilson_lower_bound(24, 30) < 24 / 30
    assert _wilson_lower_bound(30, 30) < 1.0
    assert np.isnan(_wilson_lower_bound(0, 0))


def test_action_precision_counts_only_buy_and_sell_signals() -> None:
    frame = pd.DataFrame({
        "probability_up_next_session": [0.9, 0.9, 0.9, 0.1, 0.1, 0.5],
        "predicted_return_next_session": [0.01, 0.01, 0.01, -0.01, -0.01, 0.0],
        "actual_direction_d1": ["UP", "UP", "DOWN", "DOWN", "UP", "UP"],
    })
    score = _precision(frame, 0.8, 0.2, 0.0015)
    assert score["actions"] == 5
    assert score["precision"] == 0.6
    assert score["wilson_lower_95"] < score["precision"]
    assert score["coverage"] == 5 / 6


def test_policy_target_is_explicitly_high_precision() -> None:
    assert TARGET_HOLDOUT_PRECISION == 0.80
