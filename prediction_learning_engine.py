#!/usr/bin/env python3
"""Prediction-learning engine: diagnose matured forecast errors and validate challengers.

This module is deliberately separate from the production inference code. It reads
persisted ledgers and their actual target bars, reports root-cause evidence, and
fits candidate probability/return corrections using a chronological train/holdout
split. It never edits the locked model, thresholds, prediction ledger, or places orders.

The correction profiles are proposals only. A candidate is marked eligible for
human review only when it clears minimum-sample and untouched temporal holdout gates.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from audit_autonomous_intraday_outcomes import (
    WORKER_BARS_ROOT,
    WORKER_LEDGER_ROOT,
    _download_drive_bytes,
    _normalise_bars,
    _read_ledger,
    _root_symbol,
    _safe_symbol,
    evaluate_record,
)

IST = ZoneInfo("Asia/Kolkata")
LEARNING_ROOT = f"{WORKER_LEDGER_ROOT}/learning"
MIN_GROUP_SAMPLES = 25
MIN_CALIBRATION_TRAIN = 200
MIN_CALIBRATION_TEST = 60
MIN_RETURN_TRAIN = 200
MIN_RETURN_TEST = 60


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _parse_ts(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    return pd.Timestamp(ts) if not pd.isna(ts) else pd.NaT


def forecast_identity(record: dict[str, Any]) -> str:
    """Stable identity for de-duplicating retried/appended forecast ledger rows."""
    explicit = str(record.get("forecast_key") or "").strip()
    if explicit:
        return explicit
    return "|".join([
        _root_symbol(record.get("symbol")),
        str(record.get("decision_timestamp") or ""),
        str(record.get("horizon_min") or ""),
        str(record.get("model_version") or ""),
    ])


def deduplicate_forecasts(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the first exact forecast identity; expose duplicate count and samples."""
    seen: dict[str, dict[str, Any]] = {}
    duplicate_counts: Counter[str] = Counter()
    malformed_identity = 0
    for record in records:
        identity = forecast_identity(record)
        if not identity.strip("|"):
            malformed_identity += 1
            continue
        if identity in seen:
            duplicate_counts[identity] += 1
            continue
        seen[identity] = record
    return list(seen.values()), {
        "input_rows": len(records),
        "unique_forecasts": len(seen),
        "duplicate_rows_removed": int(sum(duplicate_counts.values())),
        "duplicate_forecast_keys": int(len(duplicate_counts)),
        "malformed_identity_rows": malformed_identity,
        "duplicate_examples": [
            {"forecast_key": key, "duplicate_count": count}
            for key, count in duplicate_counts.most_common(20)
        ],
    }


def chronological_split(
    frame: pd.DataFrame,
    *,
    time_column: str = "target_timestamp_utc",
    train_fraction: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """Split by unique timestamps so same-target observations cannot cross the split."""
    if frame.empty:
        return frame.copy(), frame.copy(), None
    x = frame.copy()
    parsed = pd.to_datetime(x.get(time_column), errors="coerce", utc=True)
    x["_learning_sort_ts"] = parsed
    x = x.loc[x["_learning_sort_ts"].notna()].sort_values("_learning_sort_ts")
    unique_ts = sorted(pd.Timestamp(v) for v in x["_learning_sort_ts"].unique())
    if len(unique_ts) < 2:
        return x.drop(columns=["_learning_sort_ts"]), x.iloc[0:0].drop(columns=["_learning_sort_ts"]), None
    fraction = float(np.clip(train_fraction, 0.50, 0.85))
    split_index = int(math.floor(len(unique_ts) * fraction))
    split_index = min(max(split_index, 1), len(unique_ts) - 1)
    cutoff = unique_ts[split_index]
    train = x.loc[x["_learning_sort_ts"] < cutoff].copy()
    test = x.loc[x["_learning_sort_ts"] >= cutoff].copy()
    return (
        train.drop(columns=["_learning_sort_ts"]),
        test.drop(columns=["_learning_sort_ts"]),
        cutoff.isoformat(),
    )


def _binary_label(actual_return: float) -> float:
    return 1.0 if np.isfinite(actual_return) and actual_return > 0 else 0.0


def probability_metrics(probability: Any, label: Any) -> dict[str, Any]:
    p = pd.to_numeric(pd.Series(probability), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(label), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(p) & np.isfinite(y) & (p >= 0) & (p <= 1) & np.isin(y, [0, 1])
    p, y = p[valid], y[valid]
    if len(p) == 0:
        return {"n": 0, "brier": None, "log_loss": None, "accuracy_at_0_5": None, "mean_probability": None, "actual_up_rate": None}
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
    return {
        "n": int(len(p)),
        "brier": brier,
        "log_loss": logloss,
        "accuracy_at_0_5": float(np.mean((p >= 0.5) == (y == 1))),
        "mean_probability": float(np.mean(p)),
        "actual_up_rate": float(np.mean(y)),
        "calibration_gap_mean_probability_minus_up_rate": float(np.mean(p) - np.mean(y)),
    }


def _fit_probability_calibrator(train: pd.DataFrame, test: pd.DataFrame, split_time: str | None) -> dict[str, Any]:
    needed = ["probability_up", "actual_return"]
    if any(column not in train or column not in test for column in needed):
        return {"status": "NOT_AVAILABLE", "reason": "missing probability or actual-return columns"}
    tr = train[needed].apply(pd.to_numeric, errors="coerce").dropna()
    te = test[needed].apply(pd.to_numeric, errors="coerce").dropna()
    tr = tr.loc[tr["probability_up"].between(0, 1)]
    te = te.loc[te["probability_up"].between(0, 1)]
    tr["label"] = (tr["actual_return"] > 0).astype(int)
    te["label"] = (te["actual_return"] > 0).astype(int)
    if len(tr) < MIN_CALIBRATION_TRAIN or len(te) < MIN_CALIBRATION_TEST:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "training_rows": int(len(tr)),
            "holdout_rows": int(len(te)),
            "minimum_training_rows": MIN_CALIBRATION_TRAIN,
            "minimum_holdout_rows": MIN_CALIBRATION_TEST,
            "reason": "not enough matured rows for a credible chronological calibration test",
        }
    if tr["label"].nunique() < 2 or te["label"].nunique() < 2:
        return {"status": "INSUFFICIENT_CLASS_VARIATION", "training_rows": len(tr), "holdout_rows": len(te)}
    from sklearn.linear_model import LogisticRegression

    eps = 1e-5
    def to_logit(values: pd.Series) -> np.ndarray:
        clipped = np.clip(values.to_numpy(dtype=float), eps, 1 - eps)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(to_logit(tr["probability_up"]), tr["label"])
    calibrated_test = model.predict_proba(to_logit(te["probability_up"]))[:, 1]
    raw_metrics = probability_metrics(te["probability_up"], te["label"])
    calibrated_metrics = probability_metrics(calibrated_test, te["label"])
    raw_brier = raw_metrics.get("brier")
    cal_brier = calibrated_metrics.get("brier")
    raw_logloss = raw_metrics.get("log_loss")
    cal_logloss = calibrated_metrics.get("log_loss")
    brier_gain_pct = (
        100.0 * (raw_brier - cal_brier) / raw_brier if raw_brier and cal_brier is not None else None
    )
    logloss_change_pct = (
        100.0 * (cal_logloss - raw_logloss) / raw_logloss if raw_logloss and cal_logloss is not None else None
    )
    eligible = bool(
        brier_gain_pct is not None
        and brier_gain_pct >= 2.0
        and logloss_change_pct is not None
        and logloss_change_pct <= 1.0
    )
    return {
        "status": "ELIGIBLE_FOR_REVIEW" if eligible else "REJECTED_BY_HOLDOUT_GATE",
        "candidate_type": "logistic_probability_calibration",
        "target": "P(actual 30-minute-or-recorded-horizon return > 0)",
        "split_method": "chronological unique target timestamps; train precedes untouched holdout",
        "split_timestamp_utc": split_time,
        "training_rows": int(len(tr)),
        "holdout_rows": int(len(te)),
        "training_actual_up_rate": float(tr["label"].mean()),
        "holdout_actual_up_rate": float(te["label"].mean()),
        "raw_holdout_metrics": raw_metrics,
        "calibrated_holdout_metrics": calibrated_metrics,
        "brier_improvement_pct": brier_gain_pct,
        "log_loss_change_pct": logloss_change_pct,
        "candidate_coefficients": {
            "intercept": float(model.intercept_[0]),
            "logit_probability_coefficient": float(model.coef_[0, 0]),
            "application": "p_new = sigmoid(intercept + coefficient * logit(p_raw))",
        },
        "promotion_gate": {
            "minimum_training_rows": MIN_CALIBRATION_TRAIN,
            "minimum_holdout_rows": MIN_CALIBRATION_TEST,
            "minimum_holdout_brier_improvement_pct": 2.0,
            "maximum_holdout_logloss_degradation_pct": 1.0,
            "applied_automatically": False,
        },
        "interpretation": "A candidate is eligible for review only if it improves unseen holdout probability scores; this is not proof of future performance.",
    }


def _fit_return_correction(train: pd.DataFrame, test: pd.DataFrame, split_time: str | None) -> dict[str, Any]:
    required = ["expected_return", "actual_return"]
    if any(column not in train or column not in test for column in required):
        return {"status": "NOT_AVAILABLE", "reason": "expected_return or actual_return missing"}
    tr = train[required].apply(pd.to_numeric, errors="coerce").dropna()
    te = test[required].apply(pd.to_numeric, errors="coerce").dropna()
    if len(tr) < MIN_RETURN_TRAIN or len(te) < MIN_RETURN_TEST:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "training_rows": int(len(tr)),
            "holdout_rows": int(len(te)),
            "minimum_training_rows": MIN_RETURN_TRAIN,
            "minimum_holdout_rows": MIN_RETURN_TEST,
        }
    X = np.column_stack([np.ones(len(tr)), tr["expected_return"].to_numpy(dtype=float)])
    intercept, slope = np.linalg.lstsq(X, tr["actual_return"].to_numpy(dtype=float), rcond=None)[0]
    actual = te["actual_return"].to_numpy(dtype=float)
    predicted = te["expected_return"].to_numpy(dtype=float)
    corrected = intercept + slope * predicted
    no_change = np.zeros_like(actual)
    base_mae = float(np.mean(np.abs(actual - predicted)))
    corrected_mae = float(np.mean(np.abs(actual - corrected)))
    baseline_mae = float(np.mean(np.abs(actual - no_change)))
    improvement_pct = 100.0 * (base_mae - corrected_mae) / base_mae if base_mae > 0 else None
    beats_baseline = corrected_mae < baseline_mae
    eligible = bool(improvement_pct is not None and improvement_pct >= 2.0 and beats_baseline)
    return {
        "status": "ELIGIBLE_FOR_REVIEW" if eligible else "REJECTED_BY_HOLDOUT_GATE",
        "candidate_type": "linear_expected_return_bias_correction",
        "split_method": "chronological unique target timestamps; train precedes untouched holdout",
        "split_timestamp_utc": split_time,
        "training_rows": int(len(tr)),
        "holdout_rows": int(len(te)),
        "coefficients": {
            "intercept_return": float(intercept),
            "expected_return_slope": float(slope),
            "application": "corrected_return = intercept_return + expected_return_slope * raw_expected_return",
        },
        "holdout_mae": {
            "uncorrected_model": base_mae,
            "corrected_candidate": corrected_mae,
            "no_change_baseline": baseline_mae,
        },
        "improvement_vs_uncorrected_model_pct": improvement_pct,
        "beats_no_change_baseline": bool(beats_baseline),
        "promotion_gate": {
            "minimum_training_rows": MIN_RETURN_TRAIN,
            "minimum_holdout_rows": MIN_RETURN_TEST,
            "minimum_holdout_mae_improvement_pct": 2.0,
            "must_beat_no_change_baseline": True,
            "applied_automatically": False,
        },
    }


def _feature_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("diagnostic_feature_snapshot")
    if isinstance(value, dict):
        return value
    # JSONL string compatibility for rows exported through tabular tools.
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _prediction_error_type(row: dict[str, Any]) -> str:
    signal = str(row.get("signal") or "UNKNOWN").upper()
    actual = _num(row.get("actual_return"))
    p_up = _num(row.get("probability_up"))
    if not np.isfinite(actual):
        return "UNSCORABLE"
    if signal == "BUY" and actual <= 0:
        return "BUY_DIRECTION_WRONG"
    if signal == "SELL" and actual >= 0:
        return "SELL_DIRECTION_WRONG"
    if signal in {"HOLD", "ABSTAIN"}:
        if np.isfinite(p_up) and ((p_up >= 0.5) != (actual > 0)):
            return "PROBABILITY_DIRECTION_WRONG"
        return "NO_ACTION_OR_ABSTAIN"
    if np.isfinite(p_up) and ((p_up >= 0.5) != (actual > 0)):
        return "PROBABILITY_DIRECTION_WRONG"
    return "DIRECTION_CORRECT"


def _time_bucket(value: Any) -> str:
    ts = _parse_ts(value)
    if pd.isna(ts):
        return "UNKNOWN"
    local = ts.tz_convert(IST)
    minute = local.hour * 60 + local.minute
    bucket_start = (minute // 30) * 30
    hh, mm = divmod(bucket_start, 60)
    eh, em = divmod(bucket_start + 30, 60)
    return f"{hh:02d}:{mm:02d}-{eh:02d}:{em:02d}"


def _group_metrics(frame: pd.DataFrame, group_column: str, *, min_samples: int = MIN_GROUP_SAMPLES) -> list[dict[str, Any]]:
    if frame.empty or group_column not in frame:
        return []
    results = []
    valid = frame.loc[frame["validation_status"].eq("VALIDATED")].copy()
    if valid.empty:
        return []
    for key, group in valid.groupby(group_column, dropna=False):
        if len(group) < min_samples:
            continue
        action = group.loc[group["signal"].isin(["BUY", "SELL"])]
        direction_values = action["direction_match"].dropna().astype(float)
        brier = pd.to_numeric(group.get("brier_score"), errors="coerce").dropna()
        model_mae = pd.to_numeric(group.get("predicted_price_abs_error"), errors="coerce").dropna()
        base_mae = pd.to_numeric(group.get("baseline_price_abs_error"), errors="coerce").dropna()
        results.append({
            "dimension": group_column,
            "value": str(key),
            "validated_samples": int(len(group)),
            "action_samples": int(len(action)),
            "action_direction_accuracy": float(direction_values.mean()) if len(direction_values) else None,
            "mean_brier_score": float(brier.mean()) if len(brier) else None,
            "model_price_mae": float(model_mae.mean()) if len(model_mae) else None,
            "no_change_baseline_price_mae": float(base_mae.mean()) if len(base_mae) else None,
            "mean_actual_return_pct": float(pd.to_numeric(group["actual_return"], errors="coerce").mean() * 100.0),
            "mean_signed_action_return_before_costs_pct": (
                float(pd.to_numeric(action["signed_signal_return_before_costs"], errors="coerce").mean() * 100.0)
                if len(action) else None
            ),
        })
    return results


def _calibration_bins(frame: pd.DataFrame) -> list[dict[str, Any]]:
    valid = frame.loc[
        frame["validation_status"].eq("VALIDATED")
        & pd.to_numeric(frame["probability_up"], errors="coerce").between(0, 1)
    ].copy()
    if valid.empty:
        return []
    cuts = [0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.000001]
    valid["probability_bin"] = pd.cut(
        valid["probability_up"], bins=cuts, include_lowest=True, right=True
    ).astype(str)
    rows = []
    for key, group in valid.groupby("probability_bin", observed=False):
        if group.empty:
            continue
        rows.append({
            "probability_bin": key,
            "samples": int(len(group)),
            "mean_probability_up": float(pd.to_numeric(group["probability_up"], errors="coerce").mean()),
            "actual_up_rate": float((pd.to_numeric(group["actual_return"], errors="coerce") > 0).mean()),
            "brier_score": float(pd.to_numeric(group["brier_score"], errors="coerce").mean()),
            "calibration_gap": float(
                pd.to_numeric(group["probability_up"], errors="coerce").mean()
                - (pd.to_numeric(group["actual_return"], errors="coerce") > 0).mean()
            ),
        })
    return rows


def _feature_error_associations(frame: pd.DataFrame, min_samples: int = 100) -> list[dict[str, Any]]:
    """Rank associations between logged inputs and residuals (not causal explanations)."""
    if frame.empty or "diagnostic_feature_snapshot" not in frame:
        return []
    valid = frame.loc[frame["validation_status"].eq("VALIDATED")].copy()
    observations: list[dict[str, Any]] = []
    for _, row in valid.iterrows():
        snap = row.get("diagnostic_feature_snapshot")
        if not isinstance(snap, dict):
            snap = _feature_snapshot({"diagnostic_feature_snapshot": snap})
        actual = _num(row.get("actual_return"))
        probability = _num(row.get("probability_up"))
        expected = _num(row.get("expected_return"))
        if not snap or not np.isfinite(actual):
            continue
        record: dict[str, Any] = {"features": snap}
        if np.isfinite(probability) and 0 <= probability <= 1:
            record["classification_probability_residual"] = _binary_label(actual) - probability
        if np.isfinite(expected):
            record["expected_return_residual"] = actual - expected
        if len(record) > 1:
            observations.append(record)
    if len(observations) < min_samples:
        return []
    feature_names = sorted({
        str(name)
        for record in observations
        for name in record["features"]
        if not str(name).startswith("_")
    })
    results = []
    for residual_name in ("classification_probability_residual", "expected_return_residual"):
        eligible = [record for record in observations if residual_name in record]
        if len(eligible) < min_samples:
            continue
        residual = np.asarray([record[residual_name] for record in eligible], dtype=float)
        for name in feature_names:
            values = np.asarray([_num(record["features"].get(name)) for record in eligible], dtype=float)
            mask = np.isfinite(values) & np.isfinite(residual)
            if int(mask.sum()) < min_samples or float(np.nanstd(values[mask])) <= 1e-12:
                continue
            correlation = float(pd.Series(values[mask]).corr(pd.Series(residual[mask]), method="spearman"))
            if np.isfinite(correlation):
                results.append({
                    "feature": name,
                    "residual_type": residual_name,
                    "samples": int(mask.sum()),
                    "spearman_correlation_with_residual": correlation,
                    "absolute_correlation": abs(correlation),
                    "interpretation": "observational association only; not causal feature importance",
                })
    return sorted(results, key=lambda item: item["absolute_correlation"], reverse=True)[:100]


def _root_cause_findings(frame: pd.DataFrame, groups: list[dict[str, Any]], calibration: list[dict[str, Any]], dedup: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if frame.empty:
        return [{
            "priority": "CRITICAL",
            "finding": "NO_OUTCOME_DATA",
            "evidence": "No prediction rows were loaded from the requested Drive date range.",
            "recommended_action": "Verify Drive credentials, ledger paths, service segment and outcome-bar persistence before assessing model accuracy.",
        }]
    valid = frame.loc[frame["validation_status"].eq("VALIDATED")].copy()
    invalid = len(frame) - len(valid)
    if invalid:
        findings.append({
            "priority": "HIGH",
            "finding": "OUTCOME_COVERAGE_GAP",
            "samples_affected": int(invalid),
            "evidence": f"{invalid} of {len(frame)} predictions were not validated against target bars.",
            "recommended_action": "Repair missing bar partitions, target alignment or delayed persistence; do not label waiting/unavailable outcomes as incorrect predictions.",
        })
    if dedup.get("duplicate_rows_removed", 0):
        findings.append({
            "priority": "HIGH",
            "finding": "DUPLICATE_FORECAST_RECORDS",
            "samples_affected": dedup["duplicate_rows_removed"],
            "evidence": f"{dedup['duplicate_rows_removed']} duplicate ledger rows were removed before scoring.",
            "recommended_action": "Review worker retry/restart append semantics and enforce forecast_key idempotency at write time.",
        })
    if valid.empty:
        return findings + [{
            "priority": "CRITICAL",
            "finding": "NO_MATURED_OUTCOMES",
            "evidence": "No forecast targets have matching persisted bars yet.",
            "recommended_action": "Do not calibrate or change decision thresholds until matured target-bar coverage is available.",
        }]
    action = valid.loc[valid["signal"].isin(["BUY", "SELL"])]
    action_accuracy = pd.to_numeric(action["direction_match"], errors="coerce").mean() if len(action) else np.nan
    brier = pd.to_numeric(valid["brier_score"], errors="coerce").mean()
    price_mae = pd.to_numeric(valid["predicted_price_abs_error"], errors="coerce").mean()
    baseline_mae = pd.to_numeric(valid["baseline_price_abs_error"], errors="coerce").mean()
    if np.isfinite(action_accuracy) and action_accuracy < 0.55 and len(action) >= 50:
        findings.append({
            "priority": "HIGH",
            "finding": "ACTION_DIRECTION_EDGE_WEAK",
            "samples_affected": int(len(action)),
            "metric": float(action_accuracy),
            "evidence": f"Matured BUY/SELL directional accuracy is {action_accuracy:.1%} across {len(action)} forecasts (overlapping horizons; pre-cost).",
            "recommended_action": "Evaluate horizon-specific thresholds on a chronological holdout; inspect symbol/time-of-day/regime slices before promoting any threshold change.",
        })
    if np.isfinite(brier) and brier >= 0.25 and len(valid) >= 50:
        findings.append({
            "priority": "HIGH",
            "finding": "PROBABILITY_CALIBRATION_OR_DISCRIMINATION_WEAK",
            "samples_affected": int(len(valid)),
            "metric": float(brier),
            "evidence": f"Mean Brier score is {brier:.4f}; 0.25 is the uninformative 50/50 reference for a balanced binary target (class balance must also be considered).",
            "recommended_action": "Review calibration bins and the held-out probability calibrator; poor Brier can reflect calibration, discrimination, or regime drift.",
        })
    if np.isfinite(price_mae) and np.isfinite(baseline_mae) and len(valid) >= 30 and price_mae >= baseline_mae:
        findings.append({
            "priority": "HIGH",
            "finding": "PRICE_FORECAST_LOSES_TO_NO_CHANGE_BASELINE",
            "samples_affected": int(len(valid)),
            "metric": {"model_mae": float(price_mae), "baseline_mae": float(baseline_mae)},
            "evidence": "The predicted price's mean absolute error is no better than using the reference price unchanged.",
            "recommended_action": "Inspect regression-head training/validation and expected-return bias before showing numeric predicted prices as dependable.",
        })
    if "input_age_seconds" in frame:
        stale = valid.loc[pd.to_numeric(valid["input_age_seconds"], errors="coerce") > 120]
        if len(stale) >= 10:
            findings.append({
                "priority": "HIGH",
                "finding": "STALE_INPUT_AT_DECISION",
                "samples_affected": int(len(stale)),
                "evidence": f"{len(stale)} validated forecasts were based on input bars older than 120 seconds.",
                "recommended_action": "Review upstream candle latency and worker tick overruns; stale records should be blocked before inference.",
            })
    if "feature_coverage" in valid:
        coverage = pd.to_numeric(valid["feature_coverage"], errors="coerce")
        low = valid.loc[coverage.notna() & (coverage < 0.8)]
        if len(low) >= 10:
            findings.append({
                "priority": "HIGH",
                "finding": "LOW_ACTIVE_FEATURE_COVERAGE",
                "samples_affected": int(len(low)),
                "evidence": f"{len(low)} validated forecasts were recorded with feature coverage below 80%.",
                "recommended_action": "Check availability and point-in-time alignment of the inputs responsible for those gaps; do not impute unavailable information with fabricated values.",
            })
    if "missing_model_features_count" in valid:
        missing = pd.to_numeric(valid["missing_model_features_count"], errors="coerce").fillna(0)
        if int((missing > 0).sum()):
            findings.append({
                "priority": "MEDIUM",
                "finding": "MISSING_FEATURE_CONTRACT_INPUTS",
                "samples_affected": int((missing > 0).sum()),
                "evidence": "Some forecast records explicitly list missing model input columns.",
                "recommended_action": "Investigate the named inputs and their source/coverage; compare the precise logged feature vector before deciding whether to retrain.",
            })
    weak_slices = [
        g for g in groups
        if g.get("action_samples", 0) >= MIN_GROUP_SAMPLES
        and g.get("action_direction_accuracy") is not None
        and g["action_direction_accuracy"] < 0.48
    ]
    weak_slices = sorted(weak_slices, key=lambda item: (item["action_direction_accuracy"], -item["action_samples"]))[:10]
    for item in weak_slices:
        findings.append({
            "priority": "MEDIUM",
            "finding": "UNDERPERFORMING_SEGMENT",
            "samples_affected": item["action_samples"],
            "metric": item["action_direction_accuracy"],
            "evidence": f"{item['dimension']}={item['value']} has {item['action_samples']} actionable predictions with {item['action_direction_accuracy']:.1%} direction accuracy.",
            "recommended_action": "Investigate a segment-specific data/model issue and validate any change on a later time period; do not tune on the same rows used for reporting.",
        })
    if not findings:
        findings.append({
            "priority": "INFO",
            "finding": "NO_SINGLE_DOMINANT_FAILURE_PROVED",
            "evidence": "The available matured sample did not cross the module's diagnostic alert gates.",
            "recommended_action": "Continue collecting matched forecasts and actuals; keep the model unchanged unless a candidate passes out-of-time validation.",
        })
    return findings


def _load_records(store: Any, start_date: date, end_date: date, segment: str) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    records: list[tuple[str, dict[str, Any]]] = []
    errors: list[str] = []
    ledgers_found = 0
    malformed = 0
    total_records = 0
    current = start_date
    while current <= end_date:
        remote = f"{WORKER_LEDGER_ROOT}/{current:%Y/%m/%d}/predictions_{segment}.jsonl"
        try:
            payload = _download_drive_bytes(store, remote)
            if payload is not None:
                ledgers_found += 1
                rows, bad = _read_ledger(payload)
                malformed += bad
                total_records += len(rows)
                records.extend((current.isoformat(), row) for row in rows)
        except Exception as exc:
            errors.append(f"{remote}: {type(exc).__name__}: {exc}")
        current += timedelta(days=1)
    return records, {
        "ledger_days_found": ledgers_found,
        "ledger_rows_loaded": total_records,
        "malformed_ledger_rows": malformed,
        "ledger_load_errors": errors,
    }


def _build_outcomes(store: Any, records: list[tuple[str, dict[str, Any]]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    deduped, dedup_report = deduplicate_forecasts([row for _, row in records])
    # Duplicate removal can shorten the row list; date is recovered from the ledger
    # row's decision date when available, with a stable identity lookup fallback.
    ledger_date_by_identity: dict[str, str] = {}
    for ledger_date, row in records:
        ledger_date_by_identity.setdefault(forecast_identity(row), ledger_date)
    bar_cache: dict[tuple[str, str], pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    load_errors: list[str] = []
    for record in deduped:
        identity = forecast_identity(record)
        ledger_date = ledger_date_by_identity.get(identity)
        if not ledger_date:
            decision = _parse_ts(record.get("decision_timestamp"))
            ledger_date = decision.tz_convert(IST).date().isoformat() if not pd.isna(decision) else None
        symbol = _root_symbol(record.get("symbol"))
        if not ledger_date or not symbol:
            continue
        cache_key = (ledger_date, symbol)
        if cache_key not in bar_cache:
            remote = f"{WORKER_BARS_ROOT}/{ledger_date.replace('-', '/')}/{_safe_symbol(symbol)}.parquet"
            try:
                payload = _download_drive_bytes(store, remote)
                bar_cache[cache_key] = _normalise_bars(pd.read_parquet(__import__("io").BytesIO(payload))) if payload is not None else pd.DataFrame()
            except Exception as exc:
                load_errors.append(f"{ledger_date}/{symbol}: {type(exc).__name__}: {exc}")
                bar_cache[cache_key] = pd.DataFrame()
        evaluated = evaluate_record(record, bar_cache[cache_key])
        decision_ts = _parse_ts(record.get("decision_timestamp"))
        observed_ts = _parse_ts(record.get("worker_observed_at_utc"))
        stock_ts = _parse_ts(record.get("stock_latest_candle_ist"))
        target_ts = _parse_ts(evaluated.get("target_timestamp_utc"))
        actual_ts = _parse_ts(evaluated.get("actual_timestamp_utc"))
        input_age = (
            float((observed_ts - stock_ts).total_seconds())
            if not pd.isna(observed_ts) and not pd.isna(stock_ts) else np.nan
        )
        snap = _feature_snapshot(record)
        missing_features = record.get("missing_model_features") or []
        if isinstance(missing_features, str):
            try:
                missing_features = json.loads(missing_features)
            except json.JSONDecodeError:
                missing_features = [missing_features]
        combined = {
            **evaluated,
            "forecast_key": identity,
            "ledger_date": ledger_date,
            "decision_time_ist": decision_ts.tz_convert(IST).isoformat() if not pd.isna(decision_ts) else None,
            "target_time_ist": target_ts.tz_convert(IST).isoformat() if not pd.isna(target_ts) else None,
            "decision_time_bucket_30m": _time_bucket(evaluated.get("decision_timestamp_utc")),
            "model_version": str(record.get("model_version") or "UNKNOWN"),
            "model_family": str(record.get("model_family") or "UNKNOWN"),
            "source": str(record.get("source") or "UNKNOWN"),
            "benchmark_symbol": str(record.get("benchmark_symbol") or "UNKNOWN"),
            "signal": str(record.get("signal") or "UNKNOWN").upper(),
            "probability_up": _num(record.get("probability_up")),
            "confidence": str(record.get("confidence") or "UNKNOWN"),
            "expected_return": _num(record.get("expected_return")),
            "predicted_price": _num(record.get("predicted_price")),
            "feature_coverage": _num(record.get("feature_coverage")),
            "active_model_feature_count": _num(record.get("active_model_feature_count")),
            "missing_model_features_count": len(missing_features) if isinstance(missing_features, list) else np.nan,
            "missing_model_features": json.dumps(missing_features, separators=(",", ":")),
            "diagnostic_feature_snapshot": snap,
            "input_age_seconds": input_age,
            "data_source_age_flag": (
                "STALE_GT_120S" if np.isfinite(input_age) and input_age > 120
                else "FRESH_OR_AGE_UNKNOWN"
            ),
            "prediction_error_type": _prediction_error_type({**evaluated, **record}),
            "probability_direction_correct": (
                bool((_num(record.get("probability_up")) >= 0.5) == (_num(evaluated.get("actual_return")) > 0))
                if np.isfinite(_num(record.get("probability_up"))) and np.isfinite(_num(evaluated.get("actual_return")))
                else None
            ),
            "probability_abs_error": (
                abs(_num(record.get("probability_up")) - _binary_label(_num(evaluated.get("actual_return"))))
                if np.isfinite(_num(record.get("probability_up"))) and np.isfinite(_num(evaluated.get("actual_return")))
                else np.nan
            ),
        }
        rows.append(combined)
    frame = pd.DataFrame(rows)
    return frame, {
        **dedup_report,
        "bar_partition_cache_entries": len(bar_cache),
        "bar_load_errors": load_errors,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def build_learning_report(
    frame: pd.DataFrame,
    ingest_report: dict[str, Any],
    *,
    train_fraction: float = 0.70,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Return report, segment metrics and calibration bins from matured comparisons."""
    if frame.empty:
        empty = pd.DataFrame()
        report = {
            "status": "NO_OUTCOME_ROWS",
            "read_only": True,
            "production_model_changed": False,
            "orders_placed": False,
            "ingestion": ingest_report,
            "finding_count": 1,
            "root_cause_findings": _root_cause_findings(frame, [], [], ingest_report),
            "next_step": "Resolve the input/persistence coverage problem before algorithm tuning.",
        }
        return report, empty, empty

    x = frame.copy()
    for column in ["actual_return", "probability_up", "expected_return", "feature_coverage", "predicted_price_abs_error", "baseline_price_abs_error", "brier_score"]:
        if column not in x:
            x[column] = np.nan
        x[column] = pd.to_numeric(x[column], errors="coerce")
    x["actual_label_up"] = (x["actual_return"] > 0).astype(float)
    x["probability_bin"] = pd.cut(
        x["probability_up"], bins=[0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.000001],
        include_lowest=True, right=True,
    ).astype(str)
    x["feature_coverage_band"] = pd.cut(
        x["feature_coverage"], bins=[-0.001, 0.5, 0.7, 0.8, 0.9, 1.001],
        labels=["LE_50", "50_70", "70_80", "80_90", "90_100"], include_lowest=True,
    ).astype("object").fillna("UNKNOWN")
    x["model_version"] = x.get("model_version", pd.Series("UNKNOWN", index=x.index)).fillna("UNKNOWN").astype(str)
    x["signal"] = x.get("signal", pd.Series("UNKNOWN", index=x.index)).fillna("UNKNOWN").astype(str).str.upper()

    validated = x.loc[x["validation_status"].eq("VALIDATED")].copy()
    action = validated.loc[validated["signal"].isin(["BUY", "SELL"])]
    baseline_metrics = probability_metrics(validated["probability_up"], validated["actual_label_up"])
    split_train, split_test, split_time = chronological_split(validated, train_fraction=train_fraction)

    group_rows: list[dict[str, Any]] = []
    for dimension in [
        "horizon_min", "symbol", "signal", "source", "model_version",
        "decision_time_bucket_30m", "feature_coverage_band", "confidence",
    ]:
        group_rows.extend(_group_metrics(x, dimension))
    groups_frame = pd.DataFrame(group_rows)
    groups_sorted = sorted(
        group_rows,
        key=lambda row: (
            row.get("action_direction_accuracy") is None,
            row.get("action_direction_accuracy") if row.get("action_direction_accuracy") is not None else 1.0,
            -row.get("action_samples", 0),
        ),
    )
    cal_bins = pd.DataFrame(_calibration_bins(validated))
    # Fit a distinct challenger for each horizon and model version. Mixing, for
    # example, 5-minute and 30-minute outcomes in one calibrator can hide horizon
    # differences and produce a misleading improvement.
    candidates: dict[str, Any] = {}
    if not validated.empty:
        for (horizon, version), horizon_group in validated.groupby(
            ["horizon_min", "model_version"], dropna=False
        ):
            tr, te, candidate_split_time = chronological_split(
                horizon_group, train_fraction=train_fraction
            )
            key = f"h{int(horizon) if pd.notna(horizon) else 'unknown'}__{str(version).replace('/', '_')}"
            candidates[key] = {
                "horizon_min": int(horizon) if pd.notna(horizon) else None,
                "model_version": str(version),
                "probability_calibration": _fit_probability_calibrator(tr, te, candidate_split_time),
                "return_bias_correction": _fit_return_correction(tr, te, candidate_split_time),
            }
    feature_associations = _feature_error_associations(validated)
    findings = _root_cause_findings(x, group_rows, [], ingest_report)
    for candidate_key, candidate in candidates.items():
        if candidate.get("probability_calibration", {}).get("status") == "ELIGIBLE_FOR_REVIEW":
            findings.append({
                "priority": "REVIEW",
                "finding": "PROBABILITY_CALIBRATION_CANDIDATE_PASSES_HOLDOUT_GATE",
                "evidence": {"candidate_key": candidate_key, **candidate["probability_calibration"]},
                "recommended_action": "Export as a horizon/model-specific challenger calibration profile and shadow-test before any activation.",
            })
        if candidate.get("return_bias_correction", {}).get("status") == "ELIGIBLE_FOR_REVIEW":
            findings.append({
                "priority": "REVIEW",
                "finding": "RETURN_CORRECTION_CANDIDATE_PASSES_HOLDOUT_GATE",
                "evidence": {"candidate_key": candidate_key, **candidate["return_bias_correction"]},
                "recommended_action": "Compare the horizon/model-specific candidate over another later period and transaction/slippage assumptions before activation.",
            })
    if not feature_associations:
        findings.append({
            "priority": "INFO",
            "finding": "FEATURE_LEVEL_ATTRIBUTION_NOT_YET_AVAILABLE",
            "evidence": "Matured outcomes can be segmented by input quality, but exact numeric feature snapshots are missing or do not yet meet the minimum sample threshold.",
            "recommended_action": "Exact numeric inference features are now logged for new forecasts. Accumulate at least 100 matured, joined observations with valid probability or return residuals before interpreting feature associations.",
        })
    report = {
        "status": "COMPLETED" if len(validated) else "NO_MATURED_OUTCOMES",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "production_model_changed": False,
        "production_thresholds_changed": False,
        "orders_placed": False,
        "scope": {
            "forecast_segment": "autonomous intraday worker",
            "matured_only_for_accuracy": True,
            "overlapping_forecasts_are_not_independent_trades": True,
            "feature_associations_are_not_causal_proof": True,
        },
        "ingestion": ingest_report,
        "sample_summary": {
            "unique_forecasts": int(len(x)),
            "validated_outcomes": int(len(validated)),
            "unvalidated_outcomes": int(len(x) - len(validated)),
            "action_predictions": int(len(action)),
            "distinct_symbols": int(validated["symbol"].nunique()) if len(validated) else 0,
            "horizons_minutes": sorted([int(v) for v in validated["horizon_min"].dropna().unique()]),
            "distinct_model_versions": sorted(validated["model_version"].astype(str).unique().tolist()),
        },
        "overall_matured_metrics": {
            **baseline_metrics,
            "action_direction_accuracy": (
                float(pd.to_numeric(action["direction_match"], errors="coerce").mean())
                if len(action) and pd.to_numeric(action["direction_match"], errors="coerce").notna().any()
                else None
            ),
            "buy_precision": (
                float(pd.to_numeric(action.loc[action["signal"].eq("BUY"), "direction_match"], errors="coerce").mean())
                if len(action.loc[action["signal"].eq("BUY")]) else None
            ),
            "sell_precision": (
                float(pd.to_numeric(action.loc[action["signal"].eq("SELL"), "direction_match"], errors="coerce").mean())
                if len(action.loc[action["signal"].eq("SELL")]) else None
            ),
            "model_price_mae": float(pd.to_numeric(validated["predicted_price_abs_error"], errors="coerce").mean()) if pd.to_numeric(validated["predicted_price_abs_error"], errors="coerce").notna().any() else None,
            "no_change_baseline_price_mae": float(pd.to_numeric(validated["baseline_price_abs_error"], errors="coerce").mean()) if pd.to_numeric(validated["baseline_price_abs_error"], errors="coerce").notna().any() else None,
        },
        "chronological_validation": {
            "split_timestamp_utc": split_time,
            "train_fraction": float(train_fraction),
            "training_rows": int(len(split_train)),
            "holdout_rows": int(len(split_test)),
            "method": "unique target timestamps; no random shuffle; later rows never influence fitting",
        },
        "probability_calibration_bins": _calibration_bins(validated),
        "diagnostic_slices": groups_sorted[:100],
        "feature_error_associations": feature_associations,
        "candidate_corrections": candidates,
        "root_cause_findings": sorted(
            findings,
            key=lambda item: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "REVIEW": 3, "INFO": 4}.get(item.get("priority"), 5),
        ),
        "model_update_policy": {
            "locked_production_artifact_modified": False,
            "candidate_profile_auto_applied": False,
            "requirements_before_promotion": [
                "sufficient matured outcomes and exact forecast-to-target matches",
                "an untouched chronological holdout with measurable improvement",
                "no worse log loss/calibration and no material per-horizon regression",
                "a later shadow period plus trading-cost and slippage evaluation",
                "explicit model promotion decision; no automatic live weight/threshold writes",
            ],
        },
        "next_step": (
            "If a correction candidate passes the holdout gate, shadow-test it and compare to the unchanged champion. "
            "Otherwise fix the highest-priority data/persistence or feature-quality issue before model tuning."
        ),
    }
    return _json_safe(report), groups_frame, cal_bins


def run_learning(
    start_date: date,
    end_date: date,
    *,
    segment: str = "service",
    output_dir: str | Path = "diagnostics/prediction_learning",
    sync_drive: bool = True,
    train_fraction: float = 0.70,
) -> dict[str, Any]:
    from persistent_store import GoogleDriveStore

    store = GoogleDriveStore()
    loaded, ingest = _load_records(store, start_date, end_date, segment)
    raw_records = [row for _, row in loaded]
    deduped, dedup_report = deduplicate_forecasts(raw_records)
    # Keep original ledger date for each forecast identity after de-duplication.
    date_by_identity: dict[str, str] = {}
    for ledger_date, record in loaded:
        date_by_identity.setdefault(forecast_identity(record), ledger_date)
    unique_loaded = [(date_by_identity.get(forecast_identity(row), ""), row) for row in deduped]
    frame, build_report = _build_outcomes(store, unique_loaded)
    build_report.update({key: value for key, value in ingest.items() if key not in build_report})
    # Preserve duplicate evidence from the entire input ledger, before deduplication
    # intentionally removes duplicate weights from the scored dataset.
    build_report.update({
        key: value for key, value in dedup_report.items()
        if key in {
            "input_rows", "unique_forecasts", "duplicate_rows_removed",
            "duplicate_forecast_keys", "malformed_identity_rows", "duplicate_examples",
        }
    })
    report, groups, bins = build_learning_report(frame, build_report, train_fraction=train_fraction)
    report["date_range"] = {"start": start_date.isoformat(), "end": end_date.isoformat()}
    report["segment"] = segment
    report["duplicate_audit"] = dedup_report
    report["outcome_loading"] = build_report

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{start_date:%Y%m%d}_{end_date:%Y%m%d}_{segment}"
    outcomes_path = out_dir / f"learning_outcomes_{tag}.csv"
    groups_path = out_dir / f"learning_slices_{tag}.csv"
    report_path = out_dir / f"prediction_learning_report_{tag}.json"
    profile_path = out_dir / f"candidate_corrections_{tag}.json"
    associations_path = out_dir / f"feature_error_associations_{tag}.csv"
    frame.to_csv(outcomes_path, index=False)
    groups.to_csv(groups_path, index=False)
    pd.DataFrame(report.get("feature_error_associations", [])).to_csv(associations_path, index=False)
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True, default=str), encoding="utf-8")
    profile = {
        "profile_status": "REVIEW_ONLY_NOT_APPLIED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_range": report["date_range"],
        "candidates": report.get("candidate_corrections", {}),
        "production_model_modified": False,
        "automatic_activation": False,
    }
    profile_path.write_text(json.dumps(_json_safe(profile), indent=2, sort_keys=True, default=str), encoding="utf-8")

    if sync_drive:
        remote_dir = f"{LEARNING_ROOT}/{end_date:%Y/%m/%d}"
        uploads = {}
        for key, path, mime in [
            ("report", report_path, "application/json"),
            ("candidate_profile", profile_path, "application/json"),
            ("outcomes", outcomes_path, "text/csv"),
            ("slices", groups_path, "text/csv"),
            ("feature_associations", associations_path, "text/csv"),
        ]:
            uploaded = store.upload_or_update(path, f"{remote_dir}/{path.name}", mime_type=mime)
            uploads[key] = uploaded
            if uploaded.get("status") != "OK":
                raise RuntimeError(f"Google Drive upload not confirmed for {key}: {uploaded}")
        latest_path = out_dir / "latest_learning_report.json"
        latest_path.write_text(json.dumps({
            "status": report.get("status"),
            "generated_at_utc": report.get("generated_at_utc"),
            "date_range": report.get("date_range"),
            "report_path": f"{remote_dir}/{report_path.name}",
            "candidate_profile_path": f"{remote_dir}/{profile_path.name}",
            "uploads": uploads,
            "production_model_modified": False,
        }, indent=2, sort_keys=True), encoding="utf-8")
        latest_upload = store.upload_or_update(
            latest_path, f"{LEARNING_ROOT}/latest_learning_report.json",
            mime_type="application/json",
        )
        if latest_upload.get("status") != "OK":
            raise RuntimeError(f"Google Drive latest learning manifest upload failed: {latest_upload}")
        report["google_drive_sync"] = {
            "status": "OK",
            "remote_directory": remote_dir,
            "latest_learning_report": f"{LEARNING_ROOT}/latest_learning_report.json",
            "uploads": uploads,
            "latest_manifest_upload": latest_upload,
        }
        report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True, default=str), encoding="utf-8")
        updated = store.upload_or_update(report_path, f"{remote_dir}/{report_path.name}", mime_type="application/json")
        if updated.get("status") != "OK":
            raise RuntimeError(f"Google Drive final report update failed: {updated}")
    return _json_safe(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="", help="First ledger date YYYY-MM-DD; default 29 calendar days ago.")
    parser.add_argument("--end-date", default="", help="Last ledger date YYYY-MM-DD; default today in IST.")
    parser.add_argument("--segment", default="service", choices=["service", "morning", "afternoon", "once"])
    parser.add_argument("--out-dir", default="diagnostics/prediction_learning")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--no-sync-drive", action="store_true", help="Write local files only.")
    args = parser.parse_args()
    today = datetime.now(IST).date()
    try:
        end_date = date.fromisoformat(args.end_date) if args.end_date else today
        start_date = date.fromisoformat(args.start_date) if args.start_date else end_date - timedelta(days=29)
    except ValueError:
        parser.error("dates must use YYYY-MM-DD")
    if end_date < start_date:
        parser.error("--end-date must be on or after --start-date")
    if not 0.50 <= args.train_fraction <= 0.85:
        parser.error("--train-fraction must be between 0.50 and 0.85")
    result = run_learning(
        start_date, end_date, segment=args.segment, output_dir=args.out_dir,
        sync_drive=not args.no_sync_drive, train_fraction=args.train_fraction,
    )
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") in {"COMPLETED", "NO_MATURED_OUTCOMES", "NO_OUTCOME_ROWS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
