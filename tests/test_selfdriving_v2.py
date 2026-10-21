import numpy as np
import pandas as pd

from selfdriving.sd_intraday_v2 import (
    BASE_FEATURES,
    add_cross_sectional_targets,
    feature_columns,
    _session_rows,
    _dedupe_columns,
)
from selfdriving.sd_registry import GateRules, gate
from selfdriving.sd_v2_panel import add_delivery_targets
from selfdriving.sd_github_job_v2 import _update_forecast_ledger


def test_beta_v2_delivery_target_uses_next_observed_session():
    panel = pd.DataFrame({
        "symbol": ["A", "A", "A", "B", "B", "B"],
        "date": pd.to_datetime([
            "2026-01-01", "2026-01-02", "2026-01-05",
            "2026-01-01", "2026-01-02", "2026-01-05",
        ]),
        "close": [100.0, 102.0, 101.0, 100.0, 99.0, 101.0],
    })
    out = add_delivery_targets(panel)
    a = out[out["symbol"].eq("A")].sort_values("date")
    assert np.isclose(a.iloc[0]["y_return"], 0.02)
    assert np.isclose(a.iloc[1]["y_return"], 101.0 / 102.0 - 1.0)
    assert np.isnan(a.iloc[2]["y_return"])


def test_beta_v2_intraday_has_no_volume_dependency():
    assert not any("volume" in c.lower() for c in BASE_FEATURES)
    assert not any("volume" in c.lower() for c in feature_columns())


def test_beta_v2_intraday_labels_do_not_cross_session():
    idx = pd.date_range(
        "2026-01-05 09:15",
        "2026-01-05 15:29",
        freq="1min",
        tz="Asia/Kolkata",
    )
    px = 100.0 + np.linspace(0, 1, len(idx))
    bars = pd.DataFrame({
        "symbol": "A",
        "ts": idx.tz_convert("UTC"),
        "open": px,
        "high": px + 0.05,
        "low": px - 0.05,
        "close": px,
    })
    rows = _session_rows(bars, None, (5, 30))
    assert rows
    for r in rows:
        decision = pd.Timestamp(r["decision_timestamp"]).tz_convert("Asia/Kolkata")
        target = pd.Timestamp(r["target_timestamp"]).tz_convert("Asia/Kolkata")
        assert decision.date() == target.date()
        assert target.time() <= pd.Timestamp("15:30").time()


def test_beta_v2_excess_target_is_cross_sectional_by_decision_time():
    ts = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    p = pd.DataFrame({
        "symbol": ["A", "B", "C"],
        "decision_timestamp": [ts, ts, ts],
        "date": [pd.Timestamp("2026-01-05")] * 3,
        "horizon_min": [30, 30, 30],
        "fwd_ret": [0.03, 0.01, -0.02],
        "ret_5m": [0.1, 0.2, 0.3],
    })
    out = add_cross_sectional_targets(p)
    med = 0.01
    assert np.allclose(
        out["fwd_ret_excess"].to_numpy(),
        np.array([0.02, 0.0, -0.03]),
    )
    assert out.loc[out["symbol"].eq("A"), "y_up_excess"].iloc[0] == 1
    assert out.loc[out["symbol"].eq("C"), "y_up_excess"].iloc[0] == 0


def test_beta_v2_gate_requires_valid_champion_pair():
    challenger = {
        "dates": 100,
        "auc": 0.60,
        "ic_t": 3.0,
        "fold_auc": [0.55, 0.58, 0.60],
        "brier": 0.20,
    }
    decision = gate(
        {"brier": 0.25},
        challenger,
        {"status": "UNAVAILABLE", "reason": "feature mismatch"},
        GateRules(),
    )
    assert decision["promote"] is False
    assert any("paired comparison" in x for x in decision["reasons"])



def test_delivery_forecast_revision_replaces_current_and_archives_prior(tmp_path):
    from datetime import date

    ledger = tmp_path / "predictions.parquet"
    history = tmp_path / "predictions_history.parquet"
    inference = pd.DataFrame({"symbol": ["ABC"], "feature_x": [1.0]})
    first_prediction = pd.DataFrame([{
        "symbol": "ABC",
        "date": pd.Timestamp("2026-10-09"),
        "p": 0.70,
        "signal": "BUY",
        "model_file": "champion.joblib",
        "feature_coverage": 1.0,
    }])

    current, first_status = _update_forecast_ledger(
        first_prediction, inference, ["feature_x"], ledger, history, date(2026, 10, 12)
    )
    assert len(current) == 1
    assert current.iloc[0]["signal"] == "BUY"
    assert first_status["new_or_revised_forecasts"] == 1
    assert first_status["unchanged_forecasts"] == 0
    assert not history.exists() or pd.read_parquet(history).empty

    # Same features and score are idempotent: no duplicate forecast and no
    # additional scored sample is created by an identical scheduled rerun.
    current, repeat_status = _update_forecast_ledger(
        first_prediction, inference, ["feature_x"], ledger, history, date(2026, 10, 12)
    )
    assert len(current) == 1
    assert repeat_status["unchanged_forecasts"] == 1
    assert repeat_status["superseded_versions_archived"] == 0
    assert not history.exists() or pd.read_parquet(history).empty

    # A new input fingerprint and changed model output supersede, rather than
    # erase or double-count, the earlier issued recommendation.
    changed_inference = pd.DataFrame({"symbol": ["ABC"], "feature_x": [2.0]})
    revised_prediction = first_prediction.copy()
    revised_prediction.loc[0, "p"] = 0.52
    revised_prediction.loc[0, "signal"] = "HOLD"
    current, revised_status = _update_forecast_ledger(
        revised_prediction, changed_inference, ["feature_x"], ledger, history, date(2026, 10, 12)
    )
    assert len(current) == 1
    assert current.iloc[0]["signal"] == "HOLD"
    assert revised_status["new_or_revised_forecasts"] == 1
    assert revised_status["superseded_versions_archived"] == 1
    prior = pd.read_parquet(history)
    assert len(prior) == 1
    assert prior.iloc[0]["signal"] == "BUY"
    assert prior.iloc[0]["revision_status"] == "SUPERSEDED"


def test_intraday_duplicate_column_normalization_works_with_pandas_index():
    # Some provider frames (notably Yahoo-style OHLC/MultiIndex frames after
    # normalization) can expose duplicate labels such as Close/Adj Close -> Close.
    # pandas Index has no .eq() method; boolean equality is cross-version safe.
    frame = pd.DataFrame(
        [
            [np.nan, 100.0, 10.0],
            [101.0, np.nan, 11.0],
        ],
        columns=["close", "close", "open"],
    )

    normalized = _dedupe_columns(frame)

    assert list(normalized.columns) == ["close", "open"]
    assert normalized.columns.is_unique
    assert np.allclose(normalized["close"].to_numpy(), [100.0, 101.0])


def test_live_inference_minimum_does_not_relax_training_minimum():
    from datetime import date

    idx = pd.date_range(
        "2026-10-13 09:15",
        periods=16,
        freq="1min",
        tz="Asia/Kolkata",
    )
    px = 100.0 + np.arange(len(idx)) * 0.02
    bars = pd.DataFrame({
        "symbol": "A",
        "ts": idx.tz_convert("UTC"),
        "open": px,
        "high": px + 0.05,
        "low": px - 0.05,
        "close": px,
    })
    live = _session_rows(
        bars, None, (5, 30), include_future=False, decision_interval_minutes=1
    )
    train = _session_rows(bars, None, (5, 30), include_future=True)
    assert live
    assert train == []
