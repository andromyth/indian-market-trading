from __future__ import annotations

import pytest

from all_nse_intraday_model import FEATURES, _declared_features


def test_minutes_to_close_is_a_first_class_feature() -> None:
    assert "minutes_to_close" in FEATURES


def test_manifest_can_pin_the_legacy_34_feature_contract() -> None:
    legacy = [name for name in FEATURES if name != "minutes_to_close"]
    assert len(legacy) == 34
    assert _declared_features({"features": legacy}) == legacy


def test_missing_manifest_features_defaults_to_the_adapter_contract() -> None:
    assert _declared_features({}) == FEATURES


def test_unknown_manifest_feature_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="cannot construct"):
        _declared_features({"features": ["not_a_real_feature"]})
