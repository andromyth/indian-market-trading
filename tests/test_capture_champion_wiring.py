from __future__ import annotations

from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import all_nse_minute_prediction_capture as cap

IST = ZoneInfo("Asia/Kolkata")


def test_edge_rule_holds_when_edge_below_cost():
    assert cap._apply_edge_rule("BUY", 0.0005, 0.0011) == "HOLD"
    assert cap._apply_edge_rule("SELL", -0.0005, 0.0011) == "HOLD"


def test_edge_rule_allows_clear_edge():
    assert cap._apply_edge_rule("BUY", 0.003, 0.0011) == "BUY"
    assert cap._apply_edge_rule("SELL", -0.003, 0.0011) == "SELL"


def test_edge_rule_identity_without_hurdle():
    assert cap._apply_edge_rule("BUY", None, None) == "BUY"
    assert cap._apply_edge_rule("SELL", -0.5, 0.0) == "SELL"
    assert cap._apply_edge_rule("HOLD", 0.01, 0.0011) == "HOLD"


def test_edge_rule_holds_on_missing_expected_return():
    assert cap._apply_edge_rule("BUY", float("nan"), 0.0011) == "HOLD"


def test_closed_bar_detection():
    assert cap._is_closed_5m_bar(pd.Timestamp("2026-11-20 10:35", tz=IST))
    assert not cap._is_closed_5m_bar(pd.Timestamp("2026-11-20 10:35:02", tz=IST))
    assert not cap._is_closed_5m_bar(pd.Timestamp("2026-11-20 10:36", tz=IST))


def test_model_api_helpers_read_and_fail_open():
    api = SimpleNamespace(signal_min_edge=lambda: 0.0011, serving_mode=lambda: "PAPER")
    assert cap._model_signal_min_edge(api) == 0.0011
    assert cap._model_serving_mode(api) == "PAPER"
    assert cap._model_signal_min_edge(SimpleNamespace()) is None
    assert cap._model_serving_mode(SimpleNamespace()) is None

    def boom():
        raise RuntimeError("unavailable")

    bad = SimpleNamespace(signal_min_edge=boom, serving_mode=boom)
    assert cap._model_signal_min_edge(bad) is None
    assert cap._model_serving_mode(bad) is None


def _predictor(tmp_path):
    return cap.AllNSEMinutePredictor(tmp_path)


def test_ensure_model_prefers_champion(tmp_path):
    predictor = _predictor(tmp_path)
    predictor._classifier = predictor._regressor = None
    champion = SimpleNamespace(
        __name__="champion_model_api",
        model_status=lambda: {"manifest_sha256_match": True, "training_symbols": ["aaa.ns"]},
        _models=lambda: ("champion-clf", "champion-reg"),
    )
    predictor._model_api_candidates = lambda: iter([champion])

    predictor._ensure_model()

    assert predictor._model_api is champion
    assert predictor._classifier == "champion-clf"
    assert predictor._regressor == "champion-reg"
    assert predictor._trained_symbols == {"AAA"}


def test_ensure_model_falls_back_to_legacy(tmp_path):
    predictor = _predictor(tmp_path)
    predictor._classifier = predictor._regressor = None
    champion = SimpleNamespace(
        __name__="champion_model_api",
        model_status=lambda: {"manifest_sha256_match": False, "error": "no champion"},
        _models=lambda: pytest.fail("a broken champion must not be loaded"),
    )
    legacy = SimpleNamespace(
        __name__="all_nse_intraday_model",
        model_status=lambda: {"manifest_sha256_match": True, "training_symbols": []},
        _models=lambda: ("legacy-clf", "legacy-reg"),
    )
    predictor._model_api_candidates = lambda: iter([champion, legacy])

    predictor._ensure_model()

    assert predictor._model_api is legacy
    assert predictor._classifier == "legacy-clf"


def test_ensure_model_raises_when_nothing_usable(tmp_path):
    predictor = _predictor(tmp_path)
    predictor._classifier = predictor._regressor = None
    broken = SimpleNamespace(
        __name__="broken",
        model_status=lambda: {"manifest_sha256_match": False, "error": "nope"},
        _models=lambda: None,
    )
    predictor._model_api_candidates = lambda: iter([broken])

    with pytest.raises(RuntimeError, match="No usable ALL-NSE model API"):
        predictor._ensure_model()


def test_champion_api_is_none_without_drive(tmp_path, monkeypatch):
    predictor = _predictor(tmp_path)
    monkeypatch.delenv("ALL_NSE_MODEL_API", raising=False)
    monkeypatch.setattr("persistent_store.gdrive_config", lambda: {"configured": False})
    assert predictor._champion_api() is None


def test_champion_api_respects_legacy_switch(tmp_path, monkeypatch):
    predictor = _predictor(tmp_path)
    monkeypatch.setenv("ALL_NSE_MODEL_API", "legacy")
    assert predictor._champion_api() is None
