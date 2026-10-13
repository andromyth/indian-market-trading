from __future__ import annotations

import numpy as np
import pandas as pd

from selfdriving.sd_walkforward import (
    ModelConfig,
    fit_bundle,
    fit_return_regressor,
)


def _synthetic_return_panel(n_dates: int = 100, rows_per_date: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(22)
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="B")
    x = rng.normal(size=n_dates * rows_per_date)
    noise = rng.normal(scale=0.001, size=len(x))
    returns = 0.004 * x + noise
    return pd.DataFrame({
        "date": np.repeat(dates, rows_per_date),
        "symbol": [f"S{i % rows_per_date}" for i in range(len(x))],
        "signal_x": x,
        "fwd_ret": returns,
        "y_return": returns,
        "y_up_excess": (returns > np.median(returns)).astype("int8"),
    })


def test_return_regressor_produces_continuous_return_and_tail_metrics():
    frame = _synthetic_return_panel()
    model, metrics = fit_return_regressor(
        frame, ["signal_x"], "fwd_ret"
    )

    assert model is not None
    pred = model.predict(frame[["signal_x"]].iloc[:10])
    assert len(pred) == 10
    assert np.isfinite(pred).all()
    assert metrics["validation_rows"] > 0
    assert metrics["validation_mae"] >= 0
    assert metrics["baseline_mae"] >= 0
    assert metrics["status"] in {
        "MAE_BEATS_MEDIAN_BASELINE",
        "MAE_NOT_BETTER_THAN_MEDIAN_BASELINE",
    }


def test_fit_bundle_attaches_return_head_when_forward_return_exists():
    frame = _synthetic_return_panel(n_dates=80, rows_per_date=10)
    cfg = ModelConfig(
        name="test_return_head",
        hgb_params={
            "max_depth": 2,
            "learning_rate": 0.1,
            "max_iter": 8,
            "min_samples_leaf": 10,
            "l2_regularization": 1.0,
            "max_bins": 31,
            "early_stopping": False,
            "random_state": 7,
        },
        logit_C=0.1,
        blend_grid=(0.0, 0.5, 1.0),
        val_frac=0.15,
        calibrate=False,
    )
    bundle = fit_bundle(
        frame,
        ["signal_x"],
        "y_up_excess",
        cfg,
    )

    assert getattr(bundle, "return_model", None) is not None
    estimates = bundle.predict_expected_return(frame.iloc[:5])
    assert estimates is not None
    assert len(estimates) == 5
    assert np.isfinite(estimates).all()
    assert "return_forecast" in bundle.meta


def test_legacy_style_bundle_without_return_head_returns_none():
    from selfdriving.sd_walkforward import Bundle

    bundle = Bundle(
        features=["signal_x"],
        hgb=None,
        logit=None,
        w_hgb=0.5,
        iso=None,
        meta={"target": "y_up_excess"},
    )
    assert bundle.predict_expected_return(pd.DataFrame({"signal_x": [1.0]})) is None
