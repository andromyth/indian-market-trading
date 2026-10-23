#!/usr/bin/env python3
"""Quality-gated outcome learning for the deployed full-NSE 30-minute predictor.

Partial capture files remain in Drive for operations/audit and keep their
PARTIAL status. Individual symbol rows can enter outcome evaluation only when
their prediction, feature readiness, same-slot quote, and timestamp checks pass;
a valid row never upgrades its containing slot to OK. This module changes only
the calibrated probability/return/signal overlay; it does not replace
classifier/regressor joblib weights or place orders.
"""
from __future__ import annotations

import argparse
import io
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from audit_all_universe_minute_snapshots import (
    _download_json,
    _find_folder,
    _list_children,
)
from prediction_learning_engine import _run_calibration_autopilot
from persistent_store import GoogleDriveStore

IST = ZoneInfo("Asia/Kolkata")
PREDICTION_ROOT = "production_data/all_nse_minute_predictions"
QUOTE_ROOT = "production_data/all_universe_minute_snapshots"
LEARNING_ROOT = f"{PREDICTION_ROOT}/learning"
MIME_FOLDER = "application/vnd.google-apps.folder"
MAX_TARGET_QUOTE_DISTANCE_SECONDS = 180
MAX_INPUT_AGE_SECONDS = 120
MIN_DATA_READINESS = 0.70
# Data-limited probabilities can be assessed in their own segment; almost-empty rows remain excluded.
MIN_LEARNING_READINESS = 0.35
READINESS_LOW_SUFFIX = "__READINESS_LOW"
READINESS_READY_SUFFIX = "__READINESS_READY"
READINESS_UNKNOWN_SUFFIX = "__READINESS_UNKNOWN"
MODEL_VERSION_PREFIX = "all_nse_intraday__"


def readiness_segment(value: Any) -> str:
    score = _number(value)
    if not np.isfinite(score) or score < 0 or score > 1:
        return "READINESS_UNKNOWN"
    return "READINESS_LOW" if score < MIN_DATA_READINESS else "READINESS_READY"


def calibration_model_version(base_version: Any, readiness: Any) -> str:
    """Scope calibration to a deployed model artifact and its readiness band."""
    base = str(base_version or "LEGACY_UNKNOWN").strip()
    for suffix in (READINESS_LOW_SUFFIX, READINESS_READY_SUFFIX, READINESS_UNKNOWN_SUFFIX):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    suffix = {
        "READINESS_LOW": READINESS_LOW_SUFFIX,
        "READINESS_READY": READINESS_READY_SUFFIX,
        "READINESS_UNKNOWN": READINESS_UNKNOWN_SUFFIX,
    }[readiness_segment(readiness)]
    return f"{base}{suffix}"


def _number(value: Any) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _true(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def bounded_calibration_sample(
    frame: pd.DataFrame,
    *,
    max_rows_per_session: int = 12000,
    expected_slots_per_session: int = 75,
) -> pd.DataFrame:
    """Deterministically sample calibration rows across each session's time slots.

    The complete clean outcome ledger remains available for audit/metrics. This
    bounded, stratified view prevents a multi-million-row month from making the
    bootstrap/calibrator step unbounded while retaining cross-sectional coverage.
    """
    if frame.empty or max_rows_per_session < 1:
        return frame.copy()
    required = {"symbol", "target_timestamp_utc"}
    if not required.issubset(frame.columns):
        return frame.copy()

    working = frame.copy()
    target = pd.to_datetime(working["target_timestamp_utc"], errors="coerce", utc=True)
    valid = target.notna()
    working = working.loc[valid].copy()
    if working.empty:
        return frame.iloc[0:0].copy()
    target = target.loc[valid].dt.tz_convert(IST)
    working["_learning_session"] = target.dt.strftime("%Y-%m-%d")
    working["_learning_slot"] = target.dt.floor("5min").astype(str)
    working["_learning_rank"] = pd.util.hash_pandas_object(
        working["symbol"].astype(str) + "|" + working["_learning_session"] + "|" + working["_learning_slot"],
        index=False,
    ).astype("uint64").to_numpy()
    per_slot = max(1, (max_rows_per_session + expected_slots_per_session - 1) // expected_slots_per_session)

    selected = []
    for _, slot_group in working.groupby(["_learning_session", "_learning_slot"], sort=True):
        if len(slot_group) > per_slot:
            slot_group = slot_group.nsmallest(per_slot, "_learning_rank")
        selected.append(slot_group)

    sampled = pd.concat(selected, ignore_index=False) if selected else working.iloc[0:0].copy()
    sampled_sessions = []
    for _, session_group in sampled.groupby("_learning_session", sort=True):
        if len(session_group) > max_rows_per_session:
            session_group = session_group.nsmallest(max_rows_per_session, "_learning_rank")
        sampled_sessions.append(session_group)
    result = pd.concat(sampled_sessions, ignore_index=False) if sampled_sessions else sampled.iloc[0:0].copy()
    return result.drop(columns=["_learning_session", "_learning_slot", "_learning_rank"]).reset_index(drop=True)


def prediction_manifest_is_clean(manifest: dict[str, Any]) -> bool:
    """Require the full-NSE inference slot to be complete and model-ready."""
    try:
        expected = int(manifest.get("expected_symbol_count") or 0)
        rows = int(manifest.get("row_count") or 0)
        predictions = int(manifest.get("prediction_count") or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        manifest.get("status") == "OK"
        and manifest.get("quality_gate_pass") is True
        and expected > 0
        and rows == expected
        and predictions == expected
        and int(manifest.get("missing_quote_count") or 0) == 0
        and int(manifest.get("data_limited_count") or 0) == 0
        and int(manifest.get("untrained_symbol_count") or 0) == 0
        and int(manifest.get("model_failure_count") or 0) == 0
        and manifest.get("trained_symbol_manifest_available") is True
        and manifest.get("history_ready") is True
        and int(manifest.get("history_seed_coverage_symbols") or 0) == expected
        and int(manifest.get("history_seed_failed_batches") or 0) == 0
    )


def quote_manifest_is_clean(manifest: dict[str, Any]) -> bool:
    """Require every official-universe quote row and every provider batch."""
    try:
        expected = int(manifest.get("expected_symbol_count") or 0)
        rows = int(manifest.get("row_count") or 0)
        quoted = int(manifest.get("quoted_symbol_count") or 0)
    except (TypeError, ValueError):
        return False
    pending_snapshot_errors = len((manifest.get("pending_snapshot_recovery") or {}).get("errors") or [])
    pending_manifest_errors = len((manifest.get("pending_manifest_recovery") or {}).get("errors") or [])
    return bool(
        manifest.get("status") == "OK"
        and expected > 0
        and rows == expected
        and quoted == expected
        and int(manifest.get("missing_quote_count") or 0) == 0
        and int(manifest.get("unmapped_symbol_count") or 0) == 0
        and pending_snapshot_errors == 0
        and pending_manifest_errors == 0
    )


def select_row_level_clean_predictions(
    frame: pd.DataFrame,
    quote_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep individually complete predictions from an OK or PARTIAL slot.

    A PARTIAL slot is never promoted to slot-level OK. Rows enter outcomes only
    when the producer marked them PASS, input readiness and quote checks pass,
    and the same-symbol quote from that slot corroborates the observation.
    Explicitly untrained (False) rows are excluded. UNKNOWN training membership
    remains labelled UNKNOWN rather than being presented as trained: this is a
    pooled feature model, and its inference contract allows PASS when per-symbol
    membership is not supplied. Those rows remain tied to the recorded model
    version and are counted separately in the learning report.
    """
    empty = frame.iloc[0:0].copy()
    base_report = {
        "slot_status": str(manifest.get("status") or "UNKNOWN").upper(),
        "input_rows": int(len(frame)),
        "quote_matched_rows": 0,
        "clean_rows": 0,
        "quarantined_rows": int(len(frame)),
        "condition_counts": {
            "prediction_status_pass": 0,
            "prediction_status_pass_data_limited": 0,
            "prediction_status_eligible": 0,
            "actual_quote_status_pass": 0,
            "trained_true": 0,
            "trained_false": 0,
            "trained_unknown": 0,
            "data_limited_false": 0,
            "data_limited_true": 0,
            "probability_valid": 0,
            "quote_age_valid": 0,
            "readiness_valid": 0,
            "target_timestamp_valid": 0,
            "symbol_valid": 0,
            "quote_matched": 0,
            "all_non_quote_conditions": 0,
            "all_conditions": 0,
        },
    }
    if frame.empty or str(manifest.get("status") or "").upper() not in {"OK", "PARTIAL"}:
        return empty, base_report
    required_quotes = {"symbol", "quote_observed_at_utc", "actual_observed_price", "quote_status"}
    if not required_quotes.issubset(set(quote_frame.columns)) or quote_frame.empty:
        return empty, base_report
    if "symbol" not in frame.columns:
        return empty, base_report

    pred = frame.copy().reset_index(drop=True)
    pred["_row_id_internal"] = np.arange(len(pred))
    pred["_symbol_norm_internal"] = (
        pred["symbol"].astype(str).str.strip().str.upper().str.replace(".NS", "", regex=False)
    )
    pred_ts = pd.to_datetime(
        pred.get("actual_observed_at_utc", pd.Series(pd.NaT, index=pred.index)),
        errors="coerce", utc=True,
    )
    pred_price = pd.to_numeric(
        pred.get("actual_market_value", pd.Series(np.nan, index=pred.index)),
        errors="coerce",
    )
    probability = pd.to_numeric(
        pred.get("probability_up_30m", pd.Series(np.nan, index=pred.index)),
        errors="coerce",
    )
    quote_age = pd.to_numeric(
        pred.get("actual_quote_age_seconds", pd.Series(np.nan, index=pred.index)),
        errors="coerce",
    )
    readiness = pd.to_numeric(
        pred.get("data_readiness_score", pd.Series(np.nan, index=pred.index)),
        errors="coerce",
    )
    target_ts = pd.to_datetime(
        pred.get("forecast_target_timestamp_utc", pd.Series(pd.NaT, index=pred.index)),
        errors="coerce", utc=True,
    )
    pred_status = pred.get("prediction_status", pd.Series("", index=pred.index)).astype(str).str.upper()
    actual_status = pred.get("actual_quote_status", pd.Series("", index=pred.index)).astype(str).str.upper()
    trained = pred.get("trained_for_symbol", pd.Series(False, index=pred.index)).map(_true)
    data_limited = pred.get("data_limited", pd.Series(True, index=pred.index)).map(_true)

    quotes = quote_frame[list(required_quotes)].copy()
    quotes["_symbol_norm_internal"] = (
        quotes["symbol"].astype(str).str.strip().str.upper().str.replace(".NS", "", regex=False)
    )
    if quotes["_symbol_norm_internal"].duplicated().any():
        return empty, base_report
    quotes["_quote_ts_internal"] = pd.to_datetime(
        quotes["quote_observed_at_utc"], errors="coerce", utc=True
    )
    quotes["_quote_price_internal"] = pd.to_numeric(
        quotes["actual_observed_price"], errors="coerce"
    )
    quotes["_quote_status_internal"] = quotes["quote_status"].astype(str).str.upper()
    merged = pred.merge(
        quotes[[
            "_symbol_norm_internal", "_quote_ts_internal", "_quote_price_internal",
            "_quote_status_internal",
        ]],
        on="_symbol_norm_internal",
        how="left",
        validate="many_to_one",
    )
    merged_pred_ts = pd.to_datetime(
        merged.get("actual_observed_at_utc", pd.Series(pd.NaT, index=merged.index)),
        errors="coerce", utc=True,
    )
    merged_quote_ts = pd.to_datetime(merged["_quote_ts_internal"], errors="coerce", utc=True)
    timestamp_delta = (merged_pred_ts - merged_quote_ts).dt.total_seconds().abs()
    merged_pred_price = pd.to_numeric(
        merged.get("actual_market_value", pd.Series(np.nan, index=merged.index)),
        errors="coerce",
    )
    merged_quote_price = pd.to_numeric(merged["_quote_price_internal"], errors="coerce")
    price_match = pd.Series(
        np.isclose(
            merged_pred_price.to_numpy(dtype=float),
            merged_quote_price.to_numpy(dtype=float),
            rtol=1e-7,
            atol=1e-4,
        ),
        index=merged.index,
    )
    merged_probability = pd.to_numeric(
        merged.get("probability_up_30m", pd.Series(np.nan, index=merged.index)),
        errors="coerce",
    )
    merged_quote_age = pd.to_numeric(
        merged.get("actual_quote_age_seconds", pd.Series(np.nan, index=merged.index)),
        errors="coerce",
    )
    merged_readiness = pd.to_numeric(
        merged.get("data_readiness_score", pd.Series(np.nan, index=merged.index)),
        errors="coerce",
    )
    quote_matched = (
        merged["_quote_status_internal"].eq("PASS")
        & merged_quote_price.gt(0)
        & merged_quote_ts.notna()
        & pred_ts.notna()
        & timestamp_delta.le(MAX_INPUT_AGE_SECONDS)
        & merged_pred_price.gt(0)
        & price_match
    )

    prediction_status = merged.get("prediction_status", pd.Series("", index=merged.index)).astype(str).str.upper()
    prediction_status_pass = prediction_status.eq("PASS")
    prediction_status_pass_data_limited = prediction_status.eq("PASS_DATA_LIMITED")
    prediction_status_eligible = prediction_status_pass | prediction_status_pass_data_limited
    actual_quote_status_pass = (
        merged.get("actual_quote_status", pd.Series("", index=merged.index))
        .astype(str).str.upper().eq("PASS")
    )
    trained_raw = merged.get("trained_for_symbol", pd.Series(None, index=merged.index))
    trained_true = trained_raw.map(_true)
    trained_text = trained_raw.map(lambda value: str(value).strip().lower())
    trained_false = trained_raw.map(
        lambda value: (
            (isinstance(value, (bool, np.bool_)) and not bool(value))
            or str(value).strip().lower() in {"false", "0", "no"}
        )
    )
    trained_unknown = ~(trained_true | trained_false)
    data_limited_flags = merged.get("data_limited", pd.Series(True, index=merged.index)).map(_true)
    data_limited_false = ~data_limited_flags
    probability_valid = merged_probability.between(0, 1)
    quote_age_valid = merged_quote_age.between(0, MAX_INPUT_AGE_SECONDS)
    readiness_valid = merged_readiness.ge(MIN_LEARNING_READINESS) & merged_readiness.le(1.0)
    target_timestamp_valid = (
        pd.to_datetime(
            merged.get("forecast_target_timestamp_utc", pd.Series(pd.NaT, index=merged.index)),
            errors="coerce", utc=True
        ).notna()
    )
    symbol_valid = merged["_symbol_norm_internal"].astype(str).str.strip().ne("")

    non_quote_valid = (
        prediction_status_eligible
        & actual_quote_status_pass
        & ~trained_false
        & probability_valid
        & quote_age_valid
        & readiness_valid
        & target_timestamp_valid
        & symbol_valid
    )
    valid = quote_matched & non_quote_valid

    base_report["condition_counts"] = {
        "prediction_status_pass": int(prediction_status_pass.sum()),
        "prediction_status_pass_data_limited": int(prediction_status_pass_data_limited.sum()),
        "prediction_status_eligible": int(prediction_status_eligible.sum()),
        "actual_quote_status_pass": int(actual_quote_status_pass.sum()),
        "trained_true": int(trained_true.sum()),
        "trained_false": int(trained_false.sum()),
        "trained_unknown": int(trained_unknown.sum()),
        "data_limited_false": int(data_limited_false.sum()),
        "data_limited_true": int(data_limited_flags.sum()),
        "probability_valid": int(probability_valid.sum()),
        "quote_age_valid": int(quote_age_valid.sum()),
        "readiness_valid": int(readiness_valid.sum()),
        "target_timestamp_valid": int(target_timestamp_valid.sum()),
        "symbol_valid": int(symbol_valid.sum()),
        "quote_matched": int(quote_matched.sum()),
        "all_non_quote_conditions": int(non_quote_valid.sum()),
        "all_conditions": int(valid.sum()),
    }
    base_report["quote_matched_rows"] = int(quote_matched.sum())
    row_ids = merged.loc[valid, "_row_id_internal"].astype(int).tolist()
    clean = frame.iloc[row_ids].copy()
    clean["symbol"] = (
        clean["symbol"].astype(str).str.strip().str.upper().str.replace(".NS", "", regex=False)
    )
    base_report["clean_rows"] = int(len(clean))
    base_report["quarantined_rows"] = int(len(frame) - len(clean))
    return clean, base_report


def match_matured_outcomes(
    predictions: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    max_target_distance_seconds: int = MAX_TARGET_QUOTE_DISTANCE_SECONDS,
) -> pd.DataFrame:
    """Vector-match matured forecasts to the nearest clean target-window quote.

    Source timestamps are provider observation times, not exchange-stamped
    one-minute candle closes. Unmatched targets remain pending, never scored.
    """
    output_columns = [
        "forecast_key", "symbol", "decision_timestamp_utc", "target_timestamp_utc",
        "actual_quote_timestamp_utc", "target_quote_distance_seconds", "reference_price",
        "actual_target_price", "actual_return", "probability_up", "probability_up_calibrated",
        "expected_return", "expected_return_calibrated", "signal", "signal_calibrated",
        "calibrated_signal_advisory",
        "calibration_profile_id", "model_version", "base_model_version", "readiness_segment",
        "model_family", "horizon_min", "input_age_seconds", "feature_coverage",
        "validation_status", "source", "quote_basis", "prediction_status",
        "trained_for_symbol", "trained_membership_status", "data_limited",
        "prediction_slot_status", "quote_slot_status", "slot_quality_gate_pass",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=output_columns)

    pred = predictions.copy()
    pred["symbol"] = pred["symbol"].astype(str).str.strip().str.upper().str.replace(".NS", "", regex=False)
    pred["_target_ts"] = pd.to_datetime(pred.get("forecast_target_timestamp_utc"), errors="coerce", utc=True)
    pred["_decision_ts"] = pd.to_datetime(pred.get("actual_observed_at_utc"), errors="coerce", utc=True)
    pred["_reference"] = pd.to_numeric(pred.get("actual_market_value"), errors="coerce")
    pred["_probability"] = pd.to_numeric(
        pred.get("probability_up_30m_raw", pred.get("probability_up_30m")), errors="coerce"
    )
    pred["_expected_return"] = pd.to_numeric(
        pred.get("expected_return_30m_raw", pred.get("expected_return_30m")), errors="coerce"
    )
    if "model_version" not in pred:
        pred["model_version"] = "LEGACY_UNKNOWN"
    pred["model_version"] = pred["model_version"].fillna("LEGACY_UNKNOWN").astype(str)
    if "base_model_version" not in pred:
        pred["base_model_version"] = pred["model_version"]
    pred["base_model_version"] = pred["base_model_version"].fillna(pred["model_version"]).astype(str)
    if "readiness_segment" not in pred:
        pred["readiness_segment"] = pd.to_numeric(
            pred.get("data_readiness_score", pd.Series(np.nan, index=pred.index)), errors="coerce"
        ).map(readiness_segment)
    for col in ("probability_up_30m_calibrated", "expected_return_30m_calibrated",
                "actual_quote_age_seconds", "data_readiness_score"):
        if col not in pred:
            pred[col] = np.nan
    for col in ("signal", "signal_raw", "raw_model_signal", "signal_calibrated", "calibrated_signal_advisory",
                "calibration_profile_id", "model_family", "forecast_key", "prediction_status",
                "trained_for_symbol", "data_limited"):
        if col not in pred:
            pred[col] = None

    quote_cols = ["symbol", "quote_observed_at_utc", "actual_observed_price", "quote_status"]
    if quotes.empty:
        q = pd.DataFrame(columns=["symbol", "_quote_ts", "_quote_price"])
    else:
        q = quotes[quote_cols].copy()
        q["symbol"] = q["symbol"].astype(str).str.strip().str.upper().str.replace(".NS", "", regex=False)
        q["_quote_ts"] = pd.to_datetime(q["quote_observed_at_utc"], errors="coerce", utc=True)
        q["_quote_price"] = pd.to_numeric(q["actual_observed_price"], errors="coerce")
        q = q.loc[
            q["_quote_ts"].notna()
            & q["_quote_price"].gt(0)
            & q["quote_status"].astype(str).str.upper().eq("PASS")
        ].sort_values(["_quote_ts", "symbol"]).drop_duplicates(["symbol", "_quote_ts"], keep="last")
        q = q[["symbol", "_quote_ts", "_quote_price"]]

    valid_left_mask = pred["_target_ts"].notna() & pred["symbol"].ne("")
    left_valid = pred.loc[valid_left_mask].sort_values(["_target_ts", "symbol"])
    left_invalid = pred.loc[~valid_left_mask].copy()
    if not left_valid.empty and not q.empty:
        matched = pd.merge_asof(
            left_valid,
            q.sort_values(["_quote_ts", "symbol"]),
            left_on="_target_ts",
            right_on="_quote_ts",
            by="symbol",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=max_target_distance_seconds),
        )
    else:
        matched = left_valid.copy()
        matched["_quote_ts"] = pd.NaT
        matched["_quote_price"] = np.nan
    if not left_invalid.empty:
        left_invalid["_quote_ts"] = pd.NaT
        left_invalid["_quote_price"] = np.nan
        merged = pd.concat([matched, left_invalid], ignore_index=True, sort=False)
    else:
        merged = matched

    reference = pd.to_numeric(merged["_reference"], errors="coerce")
    target_price = pd.to_numeric(merged["_quote_price"], errors="coerce")
    quote_ok = merged["_quote_ts"].notna() & reference.gt(0) & target_price.gt(0)
    merged["validation_status"] = np.where(quote_ok, "VALIDATED", "PENDING_TARGET_QUOTE")
    merged["actual_return"] = np.where(quote_ok, target_price / reference - 1.0, np.nan)
    target_distance_seconds = pd.Series(np.nan, index=merged.index, dtype=float)
    distance_mask = merged["_quote_ts"].notna() & merged["_target_ts"].notna()
    if distance_mask.any():
        target_distance_seconds.loc[distance_mask] = (
            merged.loc[distance_mask, "_quote_ts"] - merged.loc[distance_mask, "_target_ts"]
        ).dt.total_seconds().abs()
    merged["target_quote_distance_seconds"] = target_distance_seconds
    merged["decision_timestamp_utc"] = merged["_decision_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    merged["target_timestamp_utc"] = merged["_target_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    merged["actual_quote_timestamp_utc"] = merged["_quote_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    merged["reference_price"] = reference
    merged["actual_target_price"] = target_price
    merged["probability_up"] = merged["_probability"]
    merged["probability_up_calibrated"] = pd.to_numeric(merged["probability_up_30m_calibrated"], errors="coerce")
    merged["expected_return"] = merged["_expected_return"]
    merged["expected_return_calibrated"] = pd.to_numeric(merged["expected_return_30m_calibrated"], errors="coerce")
    signal = merged["signal_raw"].fillna("").astype(str).str.strip()
    fallback = merged["raw_model_signal"].fillna("").astype(str).str.strip()
    signal = signal.mask(signal.eq(""), fallback)
    signal = signal.mask(signal.eq(""), merged["signal"].fillna("").astype(str).str.strip())
    merged["signal"] = signal.mask(signal.eq(""), "HOLD").str.upper()
    merged["calibrated_signal_advisory"] = merged["calibrated_signal_advisory"].fillna("").astype(str).str.upper()
    merged["horizon_min"] = 30
    merged["input_age_seconds"] = pd.to_numeric(merged["actual_quote_age_seconds"], errors="coerce")
    merged["feature_coverage"] = pd.to_numeric(merged["data_readiness_score"], errors="coerce")
    merged["source"] = "all_nse_intraday_production"
    merged["quote_basis"] = "nearest clean provider quote observation within target tolerance; not exchange-stamped 1m OHLCV"
    merged["base_model_version"] = merged["base_model_version"].fillna(merged["model_version"]).astype(str)
    merged["readiness_segment"] = merged["readiness_segment"].fillna("READINESS_UNKNOWN").astype(str)
    merged["trained_membership_status"] = merged["trained_for_symbol"].map(
        lambda value: "TRAINED" if _true(value) else (
            "UNTRAINED" if (
                (isinstance(value, (bool, np.bool_)) and not bool(value))
                or str(value).strip().lower() in {"false", "0", "no"}
            ) else "UNKNOWN"
        )
    )
    merged["prediction_status"] = merged["prediction_status"].fillna("UNKNOWN").astype(str).str.upper()
    for column, default in (
        ("prediction_slot_status", "UNKNOWN"),
        ("quote_slot_status", "UNKNOWN"),
        ("slot_quality_gate_pass", False),
    ):
        if column not in merged:
            merged[column] = default
    merged["prediction_slot_status"] = merged["prediction_slot_status"].fillna("UNKNOWN").astype(str)
    merged["quote_slot_status"] = merged["quote_slot_status"].fillna("UNKNOWN").astype(str)
    merged["slot_quality_gate_pass"] = merged["slot_quality_gate_pass"].map(_true)
    return merged[output_columns].reset_index(drop=True)

def _day_folder_files(store: Any, root: str, day: date) -> list[dict[str, Any]]:
    folder_id = _find_folder(store, root.split("/") + [f"{day:%Y}", f"{day:%m}", f"{day:%d}"])
    return _list_children(store, folder_id) if folder_id else []


def _download_bytes(store: Any, file_id: str) -> bytes:
    payload = store.service.files().get_media(fileId=file_id).execute()
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if hasattr(payload, "getvalue"):
        return payload.getvalue()
    return str(payload).encode("utf-8")


def _file_map(files: list[dict[str, Any]], prefix: str, suffix: str) -> dict[str, dict[str, Any]]:
    result = {}
    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{8}}_\d{{4}}){re.escape(suffix)}$")
    for item in files:
        match = pattern.match(str(item.get("name") or ""))
        if match:
            result[match.group(1)] = item
    return result


def load_clean_session_data(
    store: Any, start_date: date, end_date: date
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prediction_frames: list[pd.DataFrame] = []
    quote_frames: list[pd.DataFrame] = []
    prediction_columns = [
        "symbol", "actual_observed_at_utc", "actual_market_value", "actual_quote_status",
        "actual_quote_age_seconds", "forecast_target_timestamp_utc", "forecast_key",
        "prediction_status", "trained_for_symbol", "data_limited", "base_model_version",
        "readiness_segment", "probability_up_30m", "probability_up_30m_raw", "expected_return_30m",
        "expected_return_30m_raw", "signal", "signal_raw", "raw_model_signal",
        "probability_up_30m_calibrated", "expected_return_30m_calibrated",
        "signal_calibrated", "calibrated_signal_advisory", "calibration_profile_id",
        "model_version", "base_model_version", "readiness_segment", "model_family",
        "data_readiness_score", "prediction_slot_status", "quote_slot_status",
        "slot_quality_gate_pass",
    ]
    report = {
        "calendar_days_scanned": (end_date - start_date).days + 1,
        "prediction_slots_seen": 0,
        "prediction_slots_eligible": 0,
        "prediction_slots_partial_evaluated": 0,
        "prediction_slots_quarantined": 0,
        "prediction_slots_row_quality_empty": 0,
        "quote_slots_seen": 0,
        "quote_slots_eligible": 0,
        "quote_slots_partial_evaluated": 0,
        "quote_slots_quarantined": 0,
        "prediction_rows_retained_for_diagnostics": 0,
        "prediction_rows_from_partial_slots": 0,
        "clean_prediction_rows_from_partial_slots": 0,
        "row_level_quality_rejections": 0,
        "row_gate_condition_counts": {},
        "clean_prediction_rows": 0,
        "quote_rows_retained": 0,
        "manifest_errors": [],
        "parquet_errors": [],
        "excluded_reasons": {},
    }

    current = start_date
    while current <= end_date:
        pred_files = _day_folder_files(store, PREDICTION_ROOT, current)
        quote_files = _day_folder_files(store, QUOTE_ROOT, current)
        pred_manifests = _file_map(pred_files, "manifest_", ".json")
        pred_parquets = _file_map(pred_files, "predictions_", ".parquet")
        quote_manifests = _file_map(quote_files, "manifest_", ".json")
        quote_parquets = _file_map(quote_files, "snapshot_", ".parquet")
        clean_quote_keys: set[str] = set()
        loaded_quote_keys: set[str] = set()
        quote_slot_frames: dict[str, pd.DataFrame] = {}
        quote_slot_manifests: dict[str, dict[str, Any]] = {}

        for key, mf in quote_manifests.items():
            report["quote_slots_seen"] += 1
            try:
                manifest = _download_json(store, mf["id"])
            except Exception as exc:
                report["manifest_errors"].append(f"quote/{current}/{key}: {type(exc).__name__}: {exc}")
                report["quote_slots_quarantined"] += 1
                report["excluded_reasons"]["quote_manifest_unreadable"] = report["excluded_reasons"].get("quote_manifest_unreadable", 0) + 1
                continue

            slot_status = str(manifest.get("status") or "").upper()
            if key not in quote_parquets or slot_status not in {"OK", "PARTIAL"}:
                report["quote_slots_quarantined"] += 1
                reason = "quote_snapshot_missing_or_manifest_not_persisted"
                report["excluded_reasons"][reason] = report["excluded_reasons"].get(reason, 0) + 1
                continue
            try:
                raw = _download_bytes(store, quote_parquets[key]["id"])
                frame = pd.read_parquet(io.BytesIO(raw))
                expected = int(manifest.get("expected_symbol_count") or 0)
                declared_rows = int(manifest.get("row_count") if manifest.get("row_count") is not None else len(frame))
                symbols = frame["symbol"].astype(str).str.strip().str.upper().str.replace(".NS", "", regex=False)
                if (
                    expected <= 0
                    or len(frame) > expected
                    or declared_rows != len(frame)
                    or symbols.eq("").any()
                    or symbols.duplicated().any()
                    or (slot_status == "OK" and len(frame) != expected)
                ):
                    raise ValueError(
                        f"quote rows are structurally inconsistent: rows={len(frame)} "
                        f"manifest_rows={declared_rows} unique_symbols={symbols.nunique()} expected={expected}"
                    )
                quote_frame = frame[[
                    "symbol", "quote_observed_at_utc", "actual_observed_price", "quote_status"
                ]].copy()
                quote_frame["symbol"] = symbols
                quote_frames.append(quote_frame)
                quote_slot_frames[key] = quote_frame
                quote_slot_manifests[key] = manifest
                loaded_quote_keys.add(key)
                slot_clean = quote_manifest_is_clean(manifest) and len(frame) == expected
                if slot_clean:
                    clean_quote_keys.add(key)
                    report["quote_slots_eligible"] += 1
                else:
                    report["quote_slots_partial_evaluated"] += 1
            except Exception as exc:
                report["parquet_errors"].append(f"quote/{current}/{key}: {type(exc).__name__}: {exc}")
                report["quote_slots_quarantined"] += 1
                report["excluded_reasons"]["quote_parquet_invalid"] = report["excluded_reasons"].get("quote_parquet_invalid", 0) + 1

        for key, mf in pred_manifests.items():
            report["prediction_slots_seen"] += 1
            try:
                manifest = _download_json(store, mf["id"])
            except Exception as exc:
                report["manifest_errors"].append(f"prediction/{current}/{key}: {type(exc).__name__}: {exc}")
                report["prediction_slots_quarantined"] += 1
                report["excluded_reasons"]["prediction_manifest_unreadable"] = report["excluded_reasons"].get("prediction_manifest_unreadable", 0) + 1
                continue

            slot_status = str(manifest.get("status") or "").upper()
            if (
                key not in pred_parquets
                or key not in loaded_quote_keys
                or slot_status not in {"OK", "PARTIAL"}
            ):
                report["prediction_slots_quarantined"] += 1
                reason = "prediction_or_contemporaneous_quote_snapshot_missing_or_not_persisted"
                report["excluded_reasons"][reason] = report["excluded_reasons"].get(reason, 0) + 1
                continue

            try:
                raw = _download_bytes(store, pred_parquets[key]["id"])
                frame = pd.read_parquet(io.BytesIO(raw))
                expected = int(manifest.get("expected_symbol_count") or 0)
                declared_rows = int(manifest.get("row_count") if manifest.get("row_count") is not None else len(frame))
                symbols = frame["symbol"].astype(str).str.strip().str.upper().str.replace(".NS", "", regex=False)
                if (
                    expected <= 0
                    or len(frame) > expected
                    or declared_rows != len(frame)
                    or symbols.eq("").any()
                    or symbols.duplicated().any()
                    or (slot_status == "OK" and len(frame) != expected)
                ):
                    raise ValueError(
                        f"prediction rows are structurally inconsistent: rows={len(frame)} "
                        f"manifest_rows={declared_rows} unique_symbols={symbols.nunique()} expected={expected}"
                    )
                report["prediction_rows_retained_for_diagnostics"] += len(frame)
                clean, row_gate = select_row_level_clean_predictions(
                    frame,
                    quote_slot_frames[key],
                    manifest,
                )
                manifest_model_version = str(manifest.get("model_version") or "LEGACY_UNKNOWN")
                for column in prediction_columns:
                    if column not in clean:
                        clean[column] = None
                if clean.empty:
                    clean["base_model_version"] = pd.Series(dtype=str)
                    clean["readiness_segment"] = pd.Series(dtype=str)
                    clean["model_version"] = pd.Series(dtype=str)
                else:
                    row_version = clean["model_version"].fillna("").astype(str)
                    if "base_model_version" in clean:
                        raw_base_version = clean["base_model_version"].fillna("").astype(str)
                        base_version = raw_base_version.where(raw_base_version.str.strip().ne(""), row_version)
                    else:
                        base_version = row_version
                    base_version = base_version.where(base_version.str.strip().ne(""), manifest_model_version)
                    clean["base_model_version"] = base_version
                    clean["readiness_segment"] = pd.to_numeric(
                        clean["data_readiness_score"], errors="coerce"
                    ).map(readiness_segment)
                    clean["model_version"] = [
                        calibration_model_version(base, score)
                        for base, score in zip(clean["base_model_version"], clean["data_readiness_score"])
                    ]
                clean["prediction_slot_status"] = slot_status
                quote_slot_status = str(quote_slot_manifests[key].get("status") or "UNKNOWN").upper()
                clean["quote_slot_status"] = quote_slot_status
                clean["slot_quality_gate_pass"] = bool(
                    prediction_manifest_is_clean(manifest)
                    and key in clean_quote_keys
                )
                prediction_frames.append(clean[prediction_columns].copy())
                report["clean_prediction_rows"] += len(clean)
                report["row_level_quality_rejections"] += int(row_gate.get("quarantined_rows", 0))
                for condition, count in (row_gate.get("condition_counts") or {}).items():
                    aggregate = report["row_gate_condition_counts"]
                    aggregate[condition] = int(aggregate.get(condition, 0)) + int(count)
                slot_clean = bool(
                    prediction_manifest_is_clean(manifest)
                    and key in clean_quote_keys
                    and row_gate.get("quarantined_rows", 0) == 0
                    and len(clean) == expected
                )
                if slot_clean:
                    report["prediction_slots_eligible"] += 1
                else:
                    report["prediction_slots_partial_evaluated"] += 1
                    report["prediction_rows_from_partial_slots"] += int(len(frame))
                    report["clean_prediction_rows_from_partial_slots"] += int(len(clean))
                if len(clean) == 0:
                    report["prediction_slots_row_quality_empty"] += 1
                if row_gate.get("quote_matched_rows", 0) < len(frame):
                    failed_quote_rows = len(frame) - int(row_gate.get("quote_matched_rows", 0))
                    report["excluded_reasons"]["prediction_rows_without_matching_same_slot_quote"] = (
                        report["excluded_reasons"].get("prediction_rows_without_matching_same_slot_quote", 0)
                        + failed_quote_rows
                    )
            except Exception as exc:
                report["parquet_errors"].append(f"prediction/{current}/{key}: {type(exc).__name__}: {exc}")
                report["prediction_slots_quarantined"] += 1
                report["excluded_reasons"]["prediction_parquet_invalid"] = report["excluded_reasons"].get("prediction_parquet_invalid", 0) + 1
        current += timedelta(days=1)

    predictions = (
        pd.concat(prediction_frames, ignore_index=True, copy=False)
        if prediction_frames else pd.DataFrame(columns=prediction_columns)
    )
    quotes = (
        pd.concat(quote_frames, ignore_index=True, copy=False)
        if quote_frames else pd.DataFrame(columns=["symbol", "quote_observed_at_utc", "actual_observed_price", "quote_status"])
    )
    # Drop references to individual Parquet fragments after concatenation so the
    # multi-week full-universe run does not retain both fragmented and combined data.
    prediction_frames.clear()
    quote_frames.clear()
    report["quote_rows_retained"] = len(quotes)
    if not predictions.empty:
        decision_ts = pd.to_datetime(predictions.get("actual_observed_at_utc"), errors="coerce", utc=True)
        target_ts = pd.to_datetime(predictions.get("forecast_target_timestamp_utc"), errors="coerce", utc=True)
        same_session = (
            decision_ts.notna()
            & target_ts.notna()
            & (decision_ts.dt.tz_convert(IST).dt.date == target_ts.dt.tz_convert(IST).dt.date)
        )
        predictions = predictions.loc[same_session].copy()
    return predictions, quotes, report


def run_learning(start_date: date, end_date: date, *, output_dir: str | Path = "diagnostics/all_nse_prediction_learning") -> dict[str, Any]:
    store = GoogleDriveStore()
    predictions, quotes, ingest = load_clean_session_data(store, start_date, end_date)
    # Keep the pre-match eligible prediction-row count for reporting before
    # releasing the large source frames to bound peak memory on runner.
    prediction_rows_for_outcome_summary = int(len(predictions))
    outcomes = match_matured_outcomes(predictions, quotes)
    del predictions, quotes
    if outcomes.empty:
        fit_frame = pd.DataFrame()
    else:
        # Old snapshots predate explicit model-version logging. Keep their scored
        # outcomes in the audit, but never mix them into a version-specific fit.
        fit_frame = outcomes.loc[
            outcomes["validation_status"].eq("VALIDATED")
            & ~outcomes["model_version"].astype(str).isin(["", "LEGACY_UNKNOWN", "None", "nan"])
            & pd.to_numeric(outcomes["probability_up"], errors="coerce").between(0, 1)
            & pd.to_numeric(outcomes["actual_return"], errors="coerce").notna()
        ].copy()
    fit_frame["target_timestamp_utc"] = fit_frame.get("target_timestamp_utc", pd.Series(dtype=object))
    report: dict[str, Any] = {
        "audit_name": "all-nse-production-outcome-learning",
        "schema_version": "all-nse-production-outcome-learning-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "scope": "ALL-NSE pooled production 30-minute classifier/regressor",
        "capture_interval_minutes": 5,
        "readiness_calibration_policy": {
            "segment_threshold": MIN_DATA_READINESS,
            "minimum_learning_readiness": MIN_LEARNING_READINESS,
            "segments": ["READINESS_LOW", "READINESS_READY", "READINESS_UNKNOWN"],
            "required_matured_sessions_for_activation": 15,
            "low_readiness_primary_signal_policy": "HOLD",
        },
        "partial_data_policy": "Preserve every slot's OK/PARTIAL status. Exclude missing/unreadable slots and quarantine rows with missing/stale/unmatched quotes, invalid probability/target, explicitly untrained membership, or near-empty readiness below 0.35. PASS and PASS_DATA_LIMITED rows are evaluated in readiness-scoped model versions; low-readiness calibration may update probability/return fields but the primary action remains HOLD. Pooled-model rows with unknown membership remain labelled UNKNOWN. Individually complete rows from a PARTIAL slot may be evaluated, but never mark that slot as OK.",
        "target_matching_policy": "Nearest clean same-symbol provider quote observation within 180 seconds of nominal +30-minute target; quote observations are not exchange-stamped 1-minute candle closes.",
        "ingestion": ingest,
        "outcome_summary": {
            "prediction_rows_in_clean_slots": prediction_rows_for_outcome_summary,
            "matched_validated_outcomes": int(outcomes["validation_status"].eq("VALIDATED").sum()) if not outcomes.empty else 0,
            "pending_or_unmatched_outcomes": int(outcomes["validation_status"].ne("VALIDATED").sum()) if not outcomes.empty else 0,
            "unique_symbols_scored": int(outcomes.loc[outcomes["validation_status"].eq("VALIDATED"), "symbol"].nunique()) if not outcomes.empty else 0,
            "model_versions_in_fit": sorted(fit_frame["model_version"].dropna().astype(str).unique().tolist()) if not fit_frame.empty else [],
            "fit_eligible_rows": int(len(fit_frame)),
            "fit_eligible_sessions": int(pd.to_datetime(fit_frame["target_timestamp_utc"], errors="coerce", utc=True).dt.tz_convert(IST).dt.strftime("%Y-%m-%d").nunique()) if not fit_frame.empty else 0,
        },
        "recommendation_update": {
            "status": "WAITING_FOR_CLEAN_MATURED_OUTCOMES",
            "base_classifier_regressor_weights_modified": False,
            "calibration_profile_auto_applied": False,
            "minimum_autopilot_rows": 600,
            "minimum_autopilot_sessions": 15,
            "activation_policy": "Readiness-scoped profiles enter SHADOW and must pass the existing minimum sample/session and holdout gates. Only profiles that improve live log loss without worsening Brier score can progress to ACTIVE. READINESS_LOW profiles may recalibrate probability/return outputs but the primary action remains HOLD. Base joblib weights remain unchanged.",
        },
        "production_model_changed": False,
        "orders_placed": False,
    }

    calibration_result = {
        "status": "NOT_RUN_INSUFFICIENT_CLEAN_VERSIONED_ROWS",
        "profiles_live": 0,
        "reason": "No validated outcomes were available for an explicitly versioned full-NSE model, or the versioned clean sample was empty.",
    }
    if not fit_frame.empty:
        # Run through the same persisted registry used by the runtime calibration
        # overlay. Version-specific entries keep this model separate from the
        # watchlist model even though both use the same lifecycle state file.
        full_fit_rows = int(len(fit_frame))
        runtime_frame = bounded_calibration_sample(fit_frame)
        runtime_frame["validation_status"] = "VALIDATED"
        runtime_frame["horizon_min"] = 30
        runtime_frame["probability_up"] = pd.to_numeric(runtime_frame["probability_up"], errors="coerce")
        runtime_frame["expected_return"] = pd.to_numeric(runtime_frame["expected_return"], errors="coerce")
        runtime_frame["input_age_seconds"] = pd.to_numeric(runtime_frame["input_age_seconds"], errors="coerce")
        autopilot_dir = Path(output_dir) / "calibration_autopilot"
        calibration_result = _run_calibration_autopilot(store, runtime_frame, autopilot_dir)
        from calibration_autopilot import profile_key
        full_nse_keys = {
            profile_key(30, version)
            for version in fit_frame["model_version"].dropna().astype(str).unique()
        }
        active_keys = sorted(set(calibration_result.get("active_profile_keys", [])) & full_nse_keys)
        shadow_keys = sorted(set(calibration_result.get("shadow_profile_keys", [])) & full_nse_keys)
        report["recommendation_update"]["autopilot_rows_full_clean_sample"] = full_fit_rows
        report["recommendation_update"]["autopilot_rows_used"] = int(len(runtime_frame))
        report["recommendation_update"]["autopilot_max_rows_per_session"] = 12000
        report["recommendation_update"]["autopilot_sampling"] = "deterministic hash sample stratified by 5-minute target slot; all complete clean outcomes remain in audit CSV"
        report["recommendation_update"]["status"] = calibration_result.get("status", "UNKNOWN")
        report["recommendation_update"]["calibration_profiles_live"] = len(active_keys) + len(shadow_keys)
        report["recommendation_update"]["calibration_profiles_active_for_full_nse"] = len(active_keys)
        report["recommendation_update"]["calibration_profiles_shadow_for_full_nse"] = len(shadow_keys)
        report["recommendation_update"]["active_profile_keys_for_full_nse"] = active_keys
        report["recommendation_update"]["shadow_profile_keys_for_full_nse"] = shadow_keys
        report["recommendation_update"]["base_classifier_regressor_weights_modified"] = False
        report["recommendation_update"]["calibration_profile_auto_applied"] = bool(active_keys)
    report["calibration_autopilot"] = calibration_result

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{start_date:%Y%m%d}_{end_date:%Y%m%d}"
    outcomes_path = out_dir / f"all_nse_outcomes_{stamp}.csv.gz"
    report_path = out_dir / f"all_nse_prediction_learning_report_{stamp}.json"
    outcomes.to_csv(outcomes_path, index=False, compression="gzip")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    remote_dir = f"{LEARNING_ROOT}/{end_date:%Y/%m/%d}"
    upload_results = {}
    for key, path, mime_type in [
        ("outcomes", outcomes_path, "application/gzip"),
        ("report", report_path, "application/json"),
    ]:
        result = store.upload_or_update(path, f"{remote_dir}/{path.name}", mime_type=mime_type)
        upload_results[key] = result
        if result.get("status") != "OK":
            raise RuntimeError(f"Drive upload failed for {key}: {result}")
    latest = out_dir / "latest_all_nse_learning_report.json"
    latest.write_text(json.dumps({
        "status": report["recommendation_update"]["status"],
        "generated_at_utc": report["generated_at_utc"],
        "date_range": report["date_range"],
        "report_path": f"{remote_dir}/{report_path.name}",
        "outcomes_path": f"{remote_dir}/{outcomes_path.name}",
        "upload_results": upload_results,
        "production_model_changed": False,
        "partial_data_quarantined": True,
    }, indent=2, sort_keys=True), encoding="utf-8")
    latest_result = store.upload_or_update(
        latest, f"{LEARNING_ROOT}/latest_all_nse_learning_report.json",
        mime_type="application/json",
    )
    if latest_result.get("status") != "OK":
        raise RuntimeError(f"Drive latest report upload failed: {latest_result}")
    report["google_drive_sync"] = {"status": "OK", "remote_directory": remote_dir, "uploads": upload_results, "latest_manifest": latest_result}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    final_report_upload = store.upload_or_update(report_path, f"{remote_dir}/{report_path.name}", mime_type="application/json")
    if final_report_upload.get("status") != "OK":
        raise RuntimeError(f"Drive final report update failed: {final_report_upload}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--out-dir", default="diagnostics/all_nse_prediction_learning")
    args = parser.parse_args()
    today = datetime.now(IST).date()
    end_date = date.fromisoformat(args.end_date) if args.end_date else today
    start_date = date.fromisoformat(args.start_date) if args.start_date else end_date - timedelta(days=29)
    if end_date < start_date:
        parser.error("--end-date must be on or after --start-date")
    result = run_learning(start_date, end_date, output_dir=args.out_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
