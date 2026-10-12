import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from delivery_factor_weightage import build_factor_weightage_report
from multifactor_data_layer_v2 import FEATURES_59, GROUPS


def _fit_small_models():
    rng = np.random.default_rng(20261012)
    n = 180
    x = pd.DataFrame(
        {name: rng.normal(size=n) for name in FEATURES_59},
        columns=FEATURES_59,
    )
    x["news_available"] = rng.integers(0, 2, size=n).astype(float)
    x["event_available"] = rng.integers(0, 2, size=n).astype(float)
    x["event_high_impact"] = rng.integers(0, 2, size=n).astype(float)
    x.loc[::9, "news_available"] = np.nan
    x.loc[::11, "fundamental_revenue"] = np.nan
    x.loc[::13, "fno_pcr"] = np.nan

    news = x["news_available"].fillna(0).to_numpy()
    ret = pd.to_numeric(x["ret_1"], errors="coerce").to_numpy()
    market = pd.to_numeric(x["market_ret_1"], errors="coerce").to_numpy()
    event = x["event_high_impact"].to_numpy()
    y = ((ret + 0.4 * market + 0.15 * news + 0.1 * event) > 0).astype(int)
    y_return = 0.01 * ret + 0.006 * market + 0.002 * news + 0.003 * event + rng.normal(0, 0.002, n)

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler(with_mean=False)),
        ("model", LogisticRegression(C=0.5, class_weight="balanced", max_iter=300)),
    ])
    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("model", HistGradientBoostingRegressor(
            max_iter=6, learning_rate=0.08, max_leaf_nodes=5,
            min_samples_leaf=5, l2_regularization=2.0, random_state=42,
        )),
    ])
    clf.fit(x, y)
    reg.fit(x, y_return)
    return x, clf, reg


def test_weightage_audit_reports_separate_classifier_regressor_and_gate_measures():
    x, clf, reg = _fit_small_models()
    symbols = [f"SYM{i:03d}" for i in range(len(x))]
    p = clf.predict_proba(x)[:, 1]
    r = reg.predict(x)
    raw = np.where((p >= 0.6) & (r > 0.0015), "BUY",
                   np.where((p <= 0.4) & (r < -0.0015), "SELL", "HOLD"))
    gate = np.ones(len(x), dtype=bool)
    gate[::10] = False
    final = np.where((~gate) & np.isin(raw, ["BUY", "SELL"]), "HOLD", raw)

    report = build_factor_weightage_report(
        classifier=clf,
        regressor=reg,
        matrix=x,
        symbols=symbols,
        probability_up=p,
        predicted_return=r,
        raw_signals=raw,
        final_signals=final,
        quality_gate=gate,
        snapshot_date="2026-10-12",
        prediction_date="2026-10-13",
        model_version="test-model",
        signal_policy={"buy_probability_threshold": 0.6, "sell_probability_threshold": 0.4},
    )

    assert report["status"] == "COMPLETE"
    assert report["feature_count"] == 59
    assert set(report["feature_group_contract"]) == set(GROUPS)
    classifier_weights = report["classifier"]["parameter_weightage"]
    assert abs(sum(v["coefficient_magnitude_share_pct"] for v in classifier_weights.values()) - 100.0) < 0.05
    assert report["regressor"]["status"] == "PASS"
    assert set(report["regressor"]["tree_split_gain_share_pct"]) >= set(GROUPS)
    assert len(report["per_symbol_predictions"]) == len(x)
    assert len(report["per_symbol_classifier_contributions"]) == len(x)
    assert len(report["per_symbol_regressor_sensitivity"]) == len(x)
    assert report["data_quality_audit"]["is_directional_model_feature"] is False
    assert report["data_quality_audit"]["gate_blocked_rows"] == int((~gate).sum())
    assert report["news_semantics"]["live_59_feature_inputs"] == ["news_available"]
    assert report["news_semantics"]["news_sentiment_directionally_modelled_by_this_contract"] is False
