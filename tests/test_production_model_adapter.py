import pandas as pd
import numpy as np

import production_model as pm


def test_production_manifest_contract_is_137():
    assert len(pm.FEATURES) == 137
    assert len(set(pm.FEATURES)) == 137


def test_price_feature_builder_uses_defined_high_low_variables():
    idx = pd.date_range("2026-01-01", periods=80, freq="D")
    base = np.linspace(100.0, 120.0, len(idx))
    df = pd.DataFrame(
        {
            "Open": base,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base + 0.5,
            "Volume": np.arange(len(idx)) + 1000,
        },
        index=idx,
    )
    out = pm._price_features(df)
    assert out["high"] == float(df["High"].iloc[-1])
    assert out["low"] == float(df["Low"].iloc[-1])
    assert np.isfinite(out["atr14"])


def test_feature_row_builder_preserves_137_feature_surface():
    idx = pd.date_range("2026-01-01", periods=80, freq="D")
    base = np.linspace(100.0, 120.0, len(idx))
    price = pd.DataFrame(
        {
            "Open": base,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base + 0.5,
            "Volume": np.arange(len(idx)) + 1000,
        },
        index=idx,
    )
    market = price.copy()
    frame = pm.build_feature_rows(
        [{
            "symbol": "ABC",
            "price_df": price,
            "market_df": market,
            "matched_news": [],
            "announcements": [],
            "event_risk": 0.0,
            "fo": {},
            "decision_ts": str(idx[-1]),
        }]
    )
    assert list(frame.columns) == ["_symbol", "_decision_ts"] + pm.FEATURES
    assert frame.shape[1] == 139
