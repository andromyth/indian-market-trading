"""Read-only introspection of the live ALL-NSE 59-feature Delivery models.

This module reports how the already-fitted classifier/regressor use source groups.
It never changes model weights, predictions, signal policy, or the quality gate.
Reported percentages are different diagnostics, not interchangeable causal weights.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from multifactor_data_layer_v2 import FEATURES_59, GROUPS


def _group_for_feature(feature: str, groups: dict[str, list[str]]) -> str:
    for group, names in groups.items():
        if feature in names:
            return group
    return "unmapped"


def _transformed_group_map(
    features: list[str],
    imputer: Any,
    groups: dict[str, list[str]],
    transformed_width: int,
) -> list[str]:
    """Map original and add_indicator columns back to their source group."""
    if transformed_width < len(features):
        raise ValueError(
            f"Imputer transformed only {transformed_width} columns for {len(features)} features."
        )
    mapped = [_group_for_feature(name, groups) for name in features]
    indicator = getattr(imputer, "indicator_", None)
    missing_feature_indices = list(getattr(indicator, "features_", [])) if indicator is not None else []
    for index in range(transformed_width - len(features)):
        if index < len(missing_feature_indices):
            original_index = int(missing_feature_indices[index])
            if 0 <= original_index < len(features):
                mapped.append(_group_for_feature(features[original_index], groups))
            else:
                mapped.append("unmapped")
        else:
            mapped.append("unmapped")
    return mapped


def _pct(values: dict[str, float]) -> dict[str, float]:
    denominator = float(sum(max(0.0, float(v)) for v in values.values()))
    if not np.isfinite(denominator) or denominator <= 0:
        return {name: 0.0 for name in values}
    return {name: round(100.0 * max(0.0, float(v)) / denominator, 3) for name, v in values.items()}


def _fit_classifier_diagnostics(clf: Any, X: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    steps = getattr(clf, "named_steps", {})
    imputer = steps.get("imputer")
    scale = steps.get("scale")
    linear = steps.get("model")
    if imputer is None or linear is None or not hasattr(linear, "coef_"):
        return {"status": "UNSUPPORTED_CLASSIFIER_PIPELINE"}

    transformed = imputer.transform(X.loc[:, features])
    baseline_values = np.asarray(getattr(imputer, "statistics_", []), dtype=float)
    if len(baseline_values) != len(features):
        return {"status": "UNSUPPORTED_IMPUTER_STATISTICS"}
    baseline_frame = pd.DataFrame([baseline_values], columns=features)
    baseline = imputer.transform(baseline_frame)
    if scale is not None:
        transformed = scale.transform(transformed)
        baseline = scale.transform(baseline)

    coefficients = np.asarray(linear.coef_, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[0] != 1:
        return {"status": "UNSUPPORTED_CLASSIFIER_COEFFICIENT_SHAPE"}
    coef = coefficients[0]
    if transformed.shape[1] != len(coef) or baseline.shape[1] != len(coef):
        return {"status": "CLASSIFIER_TRANSFORM_WIDTH_MISMATCH"}

    transformed_groups = _transformed_group_map(
        features, imputer, GROUPS, transformed.shape[1]
    )
    if len(transformed_groups) != len(coef):
        return {"status": "CLASSIFIER_GROUP_MAPPING_MISMATCH"}

    group_names = list(GROUPS.keys())
    parameter_magnitude = {name: 0.0 for name in group_names}
    for idx, group in enumerate(transformed_groups):
        parameter_magnitude.setdefault(group, 0.0)
        parameter_magnitude[group] += abs(float(coef[idx]))

    # Additive per-row change in log-odds relative to a row of fitted imputer
    # medians with all missingness indicators off. This preserves the exact
    # linear decomposition; it is not a probability percentage or a win rate.
    delta = np.asarray(transformed - baseline, dtype=float)
    contribution_by_group: dict[str, np.ndarray] = {
        name: np.zeros(len(X), dtype=float) for name in parameter_magnitude
    }
    for idx, group in enumerate(transformed_groups):
        contribution_by_group.setdefault(group, np.zeros(len(X), dtype=float))
        contribution_by_group[group] += delta[:, idx] * float(coef[idx])

    abs_by_group = {
        name: np.abs(values) for name, values in contribution_by_group.items()
    }
    denominator = np.zeros(len(X), dtype=float)
    for values in abs_by_group.values():
        denominator += values
    share_by_group = {
        name: np.divide(
            values * 100.0,
            denominator,
            out=np.zeros_like(values),
            where=denominator > 0,
        )
        for name, values in abs_by_group.items()
    }
    summary = {
        name: {
            "coefficient_magnitude_share_pct": _pct(parameter_magnitude).get(name, 0.0),
            "mean_signed_logit_delta_vs_median": round(float(np.mean(contribution_by_group[name])), 6),
            "mean_abs_logit_delta_vs_median": round(float(np.mean(abs_by_group[name])), 6),
            "mean_per_symbol_abs_logit_share_pct": round(float(np.mean(share_by_group[name])), 3),
        }
        for name in parameter_magnitude
    }
    per_symbol = []
    symbols = X.index.to_list()
    for row_number, idx in enumerate(symbols):
        per_symbol.append({
            "row_index": int(row_number),
            "symbol": str(idx),
            "group_signed_logit_delta_vs_median": {
                name: round(float(values[row_number]), 6)
                for name, values in contribution_by_group.items()
            },
            "group_abs_logit_share_pct": {
                name: round(float(values[row_number]), 3)
                for name, values in share_by_group.items()
            },
        })
    return {
        "status": "PASS",
        "baseline": "fitted median imputer values; missingness indicators off",
        "parameter_weightage": summary,
        "per_row": per_symbol,
        "caveat": (
            "Coefficient-magnitude share describes fitted logistic parameters; the logit deltas are additive "
            "relative to the stated baseline. Neither is a causal weight or a guaranteed outcome percentage."
        ),
    }


def _fit_regressor_diagnostics(reg: Any, X: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    steps = getattr(reg, "named_steps", {})
    imputer = steps.get("imputer")
    estimator = steps.get("model")
    if imputer is None or estimator is None:
        return {"status": "UNSUPPORTED_REGRESSOR_PIPELINE"}

    predictors = getattr(estimator, "_predictors", None)
    if predictors is None:
        return {"status": "UNSUPPORTED_REGRESSOR_TYPE"}

    width = len(features)
    indicator = getattr(imputer, "indicator_", None)
    missing_feature_indices = list(getattr(indicator, "features_", [])) if indicator is not None else []
    group_names = list(GROUPS.keys())
    gain_by_group = {name: 0.0 for name in group_names}
    gain_by_group["unmapped"] = 0.0
    total_splits = 0

    # HistGradientBoostingRegressor does not expose feature_importances_. Sum
    # its fitted split gains and map missingness indicators to their source.
    for stage in predictors:
        for tree in stage:
            nodes = getattr(tree, "nodes", None)
            dtype_names = getattr(getattr(nodes, "dtype", None), "names", None) or ()
            if nodes is None or "feature_idx" not in dtype_names or "gain" not in dtype_names:
                continue
            for node in nodes:
                feature_idx = int(node["feature_idx"])
                gain = float(node["gain"])
                if feature_idx < 0 or not np.isfinite(gain) or gain <= 0:
                    continue
                if feature_idx < width:
                    feature = features[feature_idx]
                else:
                    indicator_idx = feature_idx - width
                    if indicator_idx >= len(missing_feature_indices):
                        gain_by_group["unmapped"] += gain
                        continue
                    original_idx = int(missing_feature_indices[indicator_idx])
                    if not 0 <= original_idx < width:
                        gain_by_group["unmapped"] += gain
                        continue
                    feature = features[original_idx]
                group = _group_for_feature(feature, GROUPS)
                gain_by_group.setdefault(group, 0.0)
                gain_by_group[group] += gain
                total_splits += 1

    gain_share = _pct(gain_by_group)
    stats = getattr(imputer, "statistics_", None)
    if stats is None or len(stats) != width:
        return {
            "status": "TREE_GAIN_AVAILABLE_BASELINE_UNAVAILABLE",
            "tree_split_gain_share_pct": gain_share,
            "positive_gain_split_count": total_splits,
        }

    base = np.asarray(stats, dtype=float)
    baseline_frame = pd.DataFrame([base], columns=features)
    baseline_pred = np.asarray(reg.predict(X.loc[:, features]), dtype=float)
    counterfactual_delta: dict[str, np.ndarray] = {}
    for group, group_features in GROUPS.items():
        counter = X.loc[:, features].copy()
        for feature in group_features:
            if feature in counter.columns:
                counter[feature] = base[features.index(feature)]
        counter_pred = np.asarray(reg.predict(counter), dtype=float)
        counterfactual_delta[group] = baseline_pred - counter_pred

    abs_delta = {name: np.abs(values) for name, values in counterfactual_delta.items()}
    totals = np.zeros(len(X), dtype=float)
    for values in abs_delta.values():
        totals += values
    share = {
        name: np.divide(values * 100.0, totals, out=np.zeros_like(values), where=totals > 0)
        for name, values in abs_delta.items()
    }
    summary = {
        name: {
            "tree_split_gain_share_pct": gain_share.get(name, 0.0),
            "mean_signed_counterfactual_delta_bps": round(float(np.mean(values) * 10000.0), 4),
            "mean_abs_counterfactual_delta_bps": round(float(np.mean(abs_delta[name]) * 10000.0), 4),
            "mean_per_symbol_abs_counterfactual_share_pct": round(float(np.mean(share[name])), 3),
        }
        for name, values in counterfactual_delta.items()
    }
    per_row = []
    for row_number, symbol in enumerate(X.index.to_list()):
        per_row.append({
            "row_index": int(row_number),
            "symbol": str(symbol),
            "group_counterfactual_return_delta_bps": {
                name: round(float(values[row_number] * 10000.0), 4)
                for name, values in counterfactual_delta.items()
            },
            "group_abs_counterfactual_share_pct": {
                name: round(float(values[row_number]), 3)
                for name, values in share.items()
            },
        })
    return {
        "status": "PASS",
        "tree_split_gain_share_pct": gain_share,
        "positive_gain_split_count": int(total_splits),
        "median_baseline": "replace one feature group at a time with that group's fitted imputer medians",
        "current_snapshot_group_sensitivity": summary,
        "per_row": per_row,
        "caveat": (
            "Tree split gain is a training-fit diagnostic. Group replacement deltas are local sensitivity proxies; "
            "they are not additive because tree interactions and correlated features can redistribute influence."
        ),
    }


def build_factor_weightage_report(
    classifier: Any,
    regressor: Any,
    matrix: pd.DataFrame,
    symbols: list[str],
    probability_up: np.ndarray,
    predicted_return: np.ndarray,
    raw_signals: np.ndarray,
    final_signals: np.ndarray,
    quality_gate: np.ndarray,
    *,
    snapshot_date: str,
    prediction_date: str,
    model_version: str,
    signal_policy: dict[str, Any],
) -> dict[str, Any]:
    """Report fitted model/group influence and quality-gate effects without changing predictions."""
    features = list(matrix.columns)
    if len(features) != 59:
        raise ValueError(f"Expected the locked 59-feature input matrix; received {len(features)}.")
    if features != FEATURES_59:
        raise ValueError("Prediction matrix does not match the locked FEATURES_59 order.")
    if not (
        len(symbols) == len(matrix) == len(probability_up) == len(predicted_return)
        == len(raw_signals) == len(final_signals) == len(quality_gate)
    ):
        raise ValueError("Factor-weightage inputs are not row-aligned.")

    # Use symbols as the index so per-row report entries have stock identities.
    X = matrix.copy()
    X.index = pd.Index([str(s).strip().upper() for s in symbols], name="symbol")

    classifier_report = _fit_classifier_diagnostics(classifier, X, features)
    regressor_report = _fit_regressor_diagnostics(regressor, X, features)
    raw = np.asarray(raw_signals, dtype=object)
    final = np.asarray(final_signals, dtype=object)
    gate = np.asarray(quality_gate, dtype=bool)
    blocked_candidates = (~gate) & np.isin(raw, ["BUY", "SELL"])

    feature_group_contract = {
        group: {
            "feature_count": len(group_features),
            "feature_share_of_59_pct": round(100.0 * len(group_features) / len(features), 3),
            "features": list(group_features),
        }
        for group, group_features in GROUPS.items()
    }
    return {
        "schema_version": "1.0",
        "status": "COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_date": snapshot_date,
        "prediction_for_date": prediction_date,
        "model_version": model_version,
        "feature_count": len(features),
        "symbols_scored": len(X),
        "signal_policy": dict(signal_policy),
        "feature_group_contract": feature_group_contract,
        "classifier": classifier_report,
        "regressor": regressor_report,
        "news_semantics": {
            "live_59_feature_inputs": ["news_available"],
            "news_sentiment_directionally_modelled_by_this_contract": False,
            "news_availability_only_is_not_a_positive_or_negative_sentiment_signal": True,
        },
        "data_quality_audit": {
            "is_directional_model_feature": False,
            "role": "hard action gate after model scoring; blocked BUY/SELL candidates are converted to HOLD",
            "gate_pass_rows": int(gate.sum()),
            "gate_blocked_rows": int((~gate).sum()),
            "gate_pass_pct": round(float(gate.mean() * 100.0), 3) if len(gate) else 0.0,
            "directional_candidates_blocked_by_gate": int(blocked_candidates.sum()),
            "final_signal_counts": pd.Series(final).value_counts().to_dict(),
        },
        "prediction_outputs": {
            "mean_probability_up": round(float(np.nanmean(probability_up)), 6) if len(probability_up) else None,
            "mean_predicted_return_pct": round(float(np.nanmean(predicted_return) * 100.0), 6) if len(predicted_return) else None,
            "raw_signal_counts_before_gate": pd.Series(raw).value_counts().to_dict(),
        },
        "methodology_warning": (
            "Feature-count shares are not learned importance. Classifier coefficient/logit diagnostics, "
            "regressor tree-gain/local-sensitivity diagnostics, and data-quality veto counts are separate quantities. "
            "None should be treated as a probability of profit or a production source-weight until confirmed with "
            "purged out-of-sample grouped ablation and cost-aware validation."
        ),
        "per_symbol_classifier_contributions": classifier_report.get("per_row", []),
        "per_symbol_regressor_sensitivity": regressor_report.get("per_row", []),
    }
