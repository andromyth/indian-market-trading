#!/usr/bin/env python3
"""Read-only deep diagnostics for the ALL-NSE Delivery decision policy.

Reads the persisted factor-weightage audit and the Drive-backed prediction /
validation ledgers. It does not write to Google Drive, alter model artifacts,
or update the live signal policy.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from persistent_store import GoogleDriveStore
from prediction_ledger import load_records, load_validation_records

ACTIVE_MODELS = {
    "2026-all-nse-delivery-multifactor-59-replacement-v1",
    "2026-all-nse-delivery-multifactor-59-candidate-v2",
    "2026-all-nse-delivery-multifactor-59-replacement-v2",
}
MIN_POLICY_SESSIONS = 20
MIN_HOLDOUT_SESSIONS = 5
ASSUMED_ROUND_TRIP_COST = 0.003


def _drive_read_json(store: GoogleDriveStore, remote_path: str) -> dict[str, Any]:
    from googleapiclient.http import MediaIoBaseDownload

    parts = [p for p in remote_path.strip("/").split("/") if p]
    if not parts:
        raise ValueError("Empty Drive path")
    parent = store.root_id
    for part in parts[:-1]:
        found = store._find_child(parent, part)
        if not found:
            raise FileNotFoundError(f"Drive folder not found: {part}")
        parent = found["id"]
    found = store._find_child(parent, parts[-1])
    if not found:
        raise FileNotFoundError(f"Drive file not found: {remote_path}")
    request = store.service.files().get_media(
        fileId=found["id"], supportsAllDrives=True
    )
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    parsed = json.loads(buffer.getvalue().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object in {remote_path}")
    return parsed


def _bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "ready"}
    try:
        return bool(int(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _date_key(value: Any) -> str | None:
    if value is None or str(value).strip() in {"", "nan", "NaT", "None"}:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except (TypeError, ValueError, OverflowError):
        return np.nan


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _counts(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(k): int(v) for k, v in series.fillna("<MISSING>").value_counts().items()}


def _load_latest_predictions(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty or "prediction_id" not in frame.columns:
        raise RuntimeError("Persisted prediction ledger is empty or has no prediction_id.")
    frame = frame[frame.get("mode", pd.Series("", index=frame.index)).astype(str)
                  .str.strip().str.lower().eq("delivery")].copy()
    required = {"symbol", "prediction_id", "prediction_for_date", "model_version"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Prediction ledger lacks required columns: {missing}")
    frame["_symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["_target_date"] = frame["prediction_for_date"].map(_date_key)
    frame["_model_version"] = frame["model_version"].astype(str)
    frame["_created_order"] = pd.to_datetime(
        frame.get("created_at", pd.Series(pd.NaT, index=frame.index)),
        errors="coerce", utc=True
    )
    frame["_revision_order"] = pd.to_numeric(
        frame.get("prediction_revision", pd.Series(0, index=frame.index)),
        errors="coerce"
    ).fillna(0)
    frame = frame.dropna(subset=["_symbol", "_target_date"])
    frame = frame.sort_values(
        ["_symbol", "_target_date", "_model_version", "_created_order",
         "_revision_order", "prediction_id"],
        kind="mergesort", na_position="first"
    )
    return frame.drop_duplicates(
        ["_symbol", "_target_date", "_model_version"], keep="last"
    ).reset_index(drop=True)


def _load_latest_validations(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty or "prediction_id" not in frame.columns:
        raise RuntimeError("Persisted validation ledger is empty or has no prediction_id.")
    frame["_validated_order"] = pd.to_datetime(
        frame.get("validated_at", pd.Series(pd.NaT, index=frame.index)),
        errors="coerce", utc=True
    )
    frame = frame.sort_values(
        ["_validated_order", "prediction_id"], kind="mergesort",
        na_position="first"
    )
    return frame.drop_duplicates("prediction_id", keep="last").drop(
        columns=["_validated_order"]
    )


def _join_matured(predictions: pd.DataFrame, validations: pd.DataFrame) -> pd.DataFrame:
    joined = predictions.merge(validations, on="prediction_id", how="inner",
                               suffixes=("", "_validation"))
    joined = joined[
        joined.get("validation_status", pd.Series("", index=joined.index))
        .astype(str).str.strip().str.upper().eq("VALIDATED")
    ].copy()
    joined["_actual_date"] = joined.get(
        "actual_d1_date", pd.Series(None, index=joined.index)
    ).map(_date_key)
    joined["_target_aligned"] = (
        joined["_actual_date"].notna()
        & joined["_target_date"].eq(joined["_actual_date"])
    )
    joined["_gate_pass"] = joined.get(
        "production_data_quality_gate", pd.Series(False, index=joined.index)
    ).map(_bool)
    for col in ("probability_up_next_session", "predicted_return_next_session",
                "actual_d1_return"):
        joined[col] = pd.to_numeric(
            joined.get(col, pd.Series(np.nan, index=joined.index)),
            errors="coerce"
        )
    joined["_actual_direction"] = joined.get(
        "actual_direction_d1", pd.Series("", index=joined.index)
    ).astype(str).str.strip().str.upper()
    joined["_model_version"] = joined.get(
        "model_version", pd.Series("", index=joined.index)
    ).astype(str)
    return joined


def _auc_row(group: pd.DataFrame, **labels: Any) -> dict[str, Any]:
    actual = group["_actual_direction"]
    auc_frame = group[actual.isin(["UP", "DOWN"])]
    up = int(auc_frame["_actual_direction"].eq("UP").sum())
    down = int(auc_frame["_actual_direction"].eq("DOWN").sum())
    result = {
        **labels,
        "validated_rows": int(len(group)),
        "scorable_up_down_rows": int(len(auc_frame)),
        "up_count": up,
        "down_count": down,
        "flat_count": int(actual.eq("FLAT").sum()),
        "roc_auc": (
            float(roc_auc_score(
                auc_frame["_actual_direction"].eq("UP").astype(int),
                auc_frame["probability_up_next_session"]
            ))
            if up > 0 and down > 0 else None
        ),
    }
    return result


def _current_batch_audit(report: dict[str, Any]) -> dict[str, Any]:
    rows = pd.DataFrame(report.get("per_symbol_predictions") or [])
    if rows.empty:
        raise RuntimeError("Current factor-weightage audit has no per_symbol_predictions.")
    policy = report.get("signal_policy") or {}
    buy = _safe_float(policy.get("buy_probability_threshold", 0.60))
    sell = _safe_float(policy.get("sell_probability_threshold", 0.40))
    min_return = _safe_float(policy.get("min_expected_return", 0.0015))
    p = pd.to_numeric(rows.get("probability_up"), errors="coerce")
    r_pct = pd.to_numeric(rows.get("predicted_return_pct"), errors="coerce")
    r = r_pct / 100.0
    gate = rows.get("quality_gate_pass", pd.Series(False, index=rows.index)).map(_bool)
    current = rows.loc[gate].copy()
    pc = p.loc[gate]
    rc = r.loc[gate]
    rc_pct = r_pct.loc[gate]

    # Explicit Step 1 two-condition cross-tab. Evaluate the probability and
    # predicted-return conditions independently, then cross-tab only rows
    # with both inputs so the four cells reconcile exactly to that population.
    probability_condition = pc.le(sell)
    return_cutoff_pct = -min_return * 100.0
    return_condition = rc_pct.lt(return_cutoff_pct)
    paired_inputs = pc.notna() & rc_pct.notna()

    def _distribution(series: pd.Series) -> dict[str, Any]:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return {"count": 0, "mean": None, "min": None, "p25": None,
                    "median": None, "p75": None, "max": None}
        return {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "min": float(values.min()),
            "p25": float(values.quantile(0.25)),
            "median": float(values.quantile(0.50)),
            "p75": float(values.quantile(0.75)),
            "max": float(values.max()),
        }

    sell_condition_cross_tab = {
        "probability_condition": f"probability_up <= {sell:.6f}",
        "predicted_return_condition": f"predicted_return_pct < {return_cutoff_pct:.6f}",
        "quality_approved_population": int(len(current)),
        "rows_with_both_inputs": int(paired_inputs.sum()),
        "rows_missing_probability": int(pc.isna().sum()),
        "rows_missing_predicted_return": int(rc_pct.isna().sum()),
        "probability_condition_count": int((probability_condition & pc.notna()).sum()),
        "return_condition_count": int((return_condition & rc_pct.notna()).sum()),
        "both_conditions": int((probability_condition & return_condition & paired_inputs).sum()),
        "probability_condition_only": int((probability_condition & ~return_condition & paired_inputs).sum()),
        "return_condition_only": int((~probability_condition & return_condition & paired_inputs).sum()),
        "neither_condition": int((~probability_condition & ~return_condition & paired_inputs).sum()),
    }
    sell_condition_cross_tab["four_way_total"] = sum(
        sell_condition_cross_tab[key] for key in (
            "both_conditions", "probability_condition_only",
            "return_condition_only", "neither_condition"
        )
    )

    quality_approved_distributions = {
        "probability_up": _distribution(pc),
        "predicted_return_pct": _distribution(rc_pct),
    }

    reason = pd.Series("MISSING_MODEL_INPUT", index=current.index, dtype="object")
    valid = pc.notna() & rc.notna()
    for idx in current.index[valid]:
        prob = float(p.loc[idx])
        ret = float(r.loc[idx])
        if ret > min_return:
            probability_pass = prob >= buy
            return_pass = True
        elif ret < -min_return:
            probability_pass = prob <= sell
            return_pass = True
        else:
            probability_pass = (prob >= buy) if prob >= 0.5 else (prob <= sell)
            return_pass = False
        if probability_pass and return_pass:
            reason.loc[idx] = "BOTH_PASS_DIRECTIONAL_ACTION"
        elif not probability_pass and return_pass:
            reason.loc[idx] = "PROBABILITY_ONLY_FAILURE"
        elif probability_pass and not return_pass:
            reason.loc[idx] = "RETURN_ONLY_FAILURE"
        else:
            reason.loc[idx] = "BOTH_FAIL"
    current["decision_condition_result"] = reason.loc[current.index]
    hold_mask = current.get(
        "final_signal", pd.Series("", index=current.index)
    ).astype(str).str.upper().eq("HOLD")
    final = current.get("final_signal", pd.Series("", index=current.index)).astype(str).str.upper()
    raw = current.get("raw_signal_before_gate", pd.Series("", index=current.index)).astype(str).str.upper()
    return {
        "snapshot_date": report.get("snapshot_date"),
        "prediction_for_date": report.get("prediction_for_date"),
        "model_version": report.get("model_version"),
        "symbols_in_audit": int(len(rows)),
        "quality_gate_passed": int(gate.sum()),
        "quality_gate_blocked": int((~gate).sum()),
        "final_signal_counts_all": _counts(rows.get("final_signal", pd.Series(dtype=object)).astype(str).str.upper()),
        "raw_signal_counts_all": _counts(rows.get("raw_signal_before_gate", pd.Series(dtype=object)).astype(str).str.upper()),
        "quality_approved_final_signal_counts": _counts(final),
        "quality_approved_raw_signal_counts": _counts(raw),
        "quality_approved_hold_rows": int(hold_mask.sum()),
        "probability_up_median_gate_pass": float(pc.median()) if pc.notna().any() else None,
        "probability_up_mean_gate_pass": float(pc.mean()) if pc.notna().any() else None,
        "probability_up_q10_gate_pass": float(pc.quantile(0.10)) if pc.notna().any() else None,
        "probability_up_q90_gate_pass": float(pc.quantile(0.90)) if pc.notna().any() else None,
        "predicted_return_pct_median_gate_pass": float(r_pct.loc[gate].median()) if r_pct.loc[gate].notna().any() else None,
        "sell_condition_cross_tab": sell_condition_cross_tab,
        "quality_approved_distributions": quality_approved_distributions,
        "decision_thresholds": {
            "buy_probability_threshold": buy,
            "sell_probability_threshold": sell,
            "minimum_expected_return_fraction": min_return,
            "minimum_expected_return_pct": min_return * 100.0,
        },
        "condition_failure_counts_all_gate_passed": _counts(reason),
        "condition_failure_counts_holds_only": _counts(reason.loc[hold_mask]),
        "condition_failure_counts_notes": (
            "For positive predicted returns, test the BUY probability threshold; "
            "for negative returns, test the SELL probability threshold. Within the "
            "minimum-return band, probability side selects the applicable direction. "
            "BOTH_PASS rows should appear as raw directional signals, not HOLD."
        ),
    }


def _historical_action_coverage(predictions: pd.DataFrame) -> dict[str, Any]:
    frame = predictions.copy()
    frame["_gate_pass"] = frame.get(
        "production_data_quality_gate", pd.Series(False, index=frame.index)
    ).map(_bool)
    frame["_signal"] = frame.get("signal", pd.Series("", index=frame.index)).astype(str).str.upper()
    frame["_raw_signal"] = frame.get(
        "raw_model_signal", frame.get("signal", pd.Series("", index=frame.index))
    ).astype(str).str.upper()
    frame["_final_action"] = frame["_signal"].isin(["BUY", "SELL"])
    frame["_raw_action"] = frame["_raw_signal"].isin(["BUY", "SELL"])
    output = []
    for (version, session), group in frame.groupby(
        ["_model_version", "_target_date"], dropna=False, sort=True
    ):
        approved = group[group["_gate_pass"]]
        output.append({
            "model_version": str(version),
            "prediction_for_date": str(session),
            "prediction_rows": int(len(group)),
            "gate_pass_rows": int(len(approved)),
            "gate_pass_pct": float(group["_gate_pass"].mean() * 100.0),
            "stored_final_buy": int(approved["_signal"].eq("BUY").sum()),
            "stored_final_sell": int(approved["_signal"].eq("SELL").sum()),
            "stored_final_hold": int(approved["_signal"].eq("HOLD").sum()),
            "data_insufficient_rows": int(group["_signal"].isin(
                ["DATA INSUFFICIENT", "DATA_INSUFFICIENT"]
            ).sum()),
            "stored_final_action_coverage_gate_pass_pct": (
                float(approved["_final_action"].mean() * 100.0)
                if len(approved) else None
            ),
            "raw_model_action_candidates_all_rows": int(group["_raw_action"].sum()),
            "raw_model_action_candidate_coverage_all_rows_pct": (
                float(group["_raw_action"].mean() * 100.0) if len(group) else None
            ),
        })
    by_version = []
    for version, group in frame.groupby("_model_version", dropna=False, sort=True):
        approved = group[group["_gate_pass"]]
        by_version.append({
            "model_version": str(version),
            "prediction_rows": int(len(group)),
            "sessions": int(group["_target_date"].nunique()),
            "gate_pass_rows": int(len(approved)),
            "final_buy": int(approved["_signal"].eq("BUY").sum()),
            "final_sell": int(approved["_signal"].eq("SELL").sum()),
            "final_hold": int(approved["_signal"].eq("HOLD").sum()),
            "action_coverage_gate_pass_pct": (
                float(approved["_final_action"].mean() * 100.0)
                if len(approved) else None
            ),
        })
    return {"by_model_version": by_version, "by_version_and_session": output}


def _ranking_backtest(joined: pd.DataFrame) -> dict[str, Any]:
    frame = joined[
        joined["_model_version"].isin(ACTIVE_MODELS)
        & joined["_gate_pass"]
        & joined["_target_aligned"]
        & joined["probability_up_next_session"].notna()
        & joined["predicted_return_next_session"].notna()
        & joined["actual_d1_return"].notna()
        & joined["_actual_direction"].isin(["UP", "DOWN", "FLAT"])
    ].copy()
    if frame.empty:
        return {"status": "NO_MATURED_QUALITY_APPROVED_ROWS"}
    # A symbol/session must be represented only once, even if two model
    # versions/revisions were validated for that target date.
    frame["_created_order"] = pd.to_datetime(
        frame.get("created_at", pd.Series(pd.NaT, index=frame.index)),
        errors="coerce", utc=True
    )
    frame = frame.sort_values(
        ["_target_date", "_symbol", "_created_order", "_model_version"],
        kind="mergesort", na_position="first"
    ).drop_duplicates(["_target_date", "_symbol"], keep="last")
    p = frame["probability_up_next_session"].astype(float)
    pred = frame["predicted_return_next_session"].astype(float)
    frame["_side"] = np.where(p >= 0.5, 1.0, -1.0)
    frame["_realised_signed_return"] = frame["actual_d1_return"] * frame["_side"]
    frame["_aligned_candidate"] = (pred * frame["_side"]) > 0
    frame["_rank_score"] = (2.0 * p - 1.0) * pred
    candidates = frame[frame["_aligned_candidate"]].copy()
    if candidates.empty:
        return {"status": "NO_DIRECTIONALLY_ALIGNED_CANDIDATES"}
    candidates["_net_return_after_cost"] = (
        candidates["_realised_signed_return"] - ASSUMED_ROUND_TRIP_COST
    )
    pool = {
        "rows": int(len(candidates)),
        "sessions": int(candidates["_target_date"].nunique()),
        "mean_gross_signed_return_pct": float(candidates["_realised_signed_return"].mean() * 100.0),
        "mean_net_signed_return_pct": float(candidates["_net_return_after_cost"].mean() * 100.0),
        "median_net_signed_return_pct": float(candidates["_net_return_after_cost"].median() * 100.0),
        "net_positive_rate_pct": float(candidates["_net_return_after_cost"].gt(0).mean() * 100.0),
    }
    rankings = []
    for fraction in (0.01, 0.02, 0.05, 0.10):
        selected_parts = []
        selected_per_session = []
        for session, group in candidates.groupby("_target_date", sort=True):
            group = group.sort_values(
                ["_rank_score", "probability_up_next_session", "_symbol"],
                ascending=[False, False, True], kind="mergesort"
            )
            take = max(1, int(math.ceil(len(group) * fraction)))
            selected = group.head(take)
            selected_parts.append(selected)
            selected_per_session.append({
                "prediction_for_date": str(session),
                "candidate_rows": int(len(group)),
                "selected_rows": int(len(selected)),
                "selected_net_return_pct": float(
                    selected["_net_return_after_cost"].mean() * 100.0
                ),
            })
        selected = pd.concat(selected_parts, ignore_index=True)
        rankings.append({
            "top_fraction_of_aligned_candidates_per_session_pct": fraction * 100.0,
            "selected_rows": int(len(selected)),
            "sessions": int(selected["_target_date"].nunique()),
            "mean_gross_signed_return_pct": float(
                selected["_realised_signed_return"].mean() * 100.0
            ),
            "mean_net_signed_return_pct_after_assumed_cost": float(
                selected["_net_return_after_cost"].mean() * 100.0
            ),
            "median_net_signed_return_pct_after_assumed_cost": float(
                selected["_net_return_after_cost"].median() * 100.0
            ),
            "net_positive_rate_pct": float(
                selected["_net_return_after_cost"].gt(0).mean() * 100.0
            ),
            "per_session": selected_per_session,
        })
    correlations = []
    for session, group in frame.groupby("_target_date", sort=True):
        if len(group) < 20 or group["_rank_score"].nunique() < 2:
            continue
        corr = group["_rank_score"].corr(
            group["_realised_signed_return"], method="spearman"
        )
        if pd.notna(corr):
            correlations.append({
                "prediction_for_date": str(session),
                "rows": int(len(group)),
                "spearman_rank_correlation": float(corr),
            })
    return {
        "status": "COMPUTED",
        "sample_definition": (
            "Matured, target-date-aligned, quality-approved latest prediction per "
            "symbol/session across active Delivery model versions; ranked only where "
            "the probability direction and predicted-return sign agree."
        ),
        "assumed_round_trip_cost_pct": ASSUMED_ROUND_TRIP_COST * 100.0,
        "short_side_caveat": (
            "SELL outcomes are modelled as a symmetric short return (-actual return). "
            "Borrow/short-sale availability, delivery restrictions, taxes, stamp duty, "
            "and slippage beyond the assumed 30 bps are not modelled."
        ),
        "aligned_candidate_pool": pool,
        "top_ranked_selection": rankings,
        "session_spearman_rank": {
            "sessions_with_correlation": int(len(correlations)),
            "mean_spearman": (
                float(np.mean([x["spearman_rank_correlation"] for x in correlations]))
                if correlations else None
            ),
            "positive_correlation_session_pct": (
                float(np.mean([
                    x["spearman_rank_correlation"] > 0 for x in correlations
                ]) * 100.0) if correlations else None
            ),
            "by_session": correlations,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", default="2026-11-02")
    parser.add_argument("--output", default="delivery_decision_deep_audit.json")
    args = parser.parse_args()

    store = GoogleDriveStore()
    remote_audit = (
        f"features/daily/DELIVERY_FACTOR_WEIGHTAGE_AUDIT_{args.snapshot_date}.json"
    )
    factor_audit = _drive_read_json(store, remote_audit)

    raw_predictions = load_records()
    raw_validations = load_validation_records()
    predictions = _load_latest_predictions(raw_predictions)
    validations = _load_latest_validations(raw_validations)
    joined = _join_matured(predictions, validations)
    current = _current_batch_audit(factor_audit)

    # Per-session ROC-AUC uses only matured, target-date-aligned, quality-approved
    # active-model rows; FLAT outcomes are excluded from the binary AUC.
    auc_source = joined[
        joined["_model_version"].isin(ACTIVE_MODELS)
        & joined["_gate_pass"]
        & joined["_target_aligned"]
        & joined["probability_up_next_session"].notna()
        & joined["_actual_direction"].isin(["UP", "DOWN", "FLAT"])
    ].copy()
    by_session = []
    for session, group in auc_source.groupby("_target_date", sort=True):
        by_session.append(_auc_row(group, prediction_for_date=str(session)))
    by_session_version = []
    for (version, session), group in auc_source.groupby(
        ["_model_version", "_target_date"], sort=True
    ):
        by_session_version.append(_auc_row(
            group, model_version=str(version), prediction_for_date=str(session)
        ))

    # Overall probability ranking on the newest active prediction per
    # symbol/session avoids double-counting overlapping v1/v2 revisions.
    latest_active = auc_source.copy()
    latest_active["_created_order"] = pd.to_datetime(
        latest_active.get("created_at", pd.Series(pd.NaT, index=latest_active.index)),
        errors="coerce", utc=True
    )
    latest_active = latest_active.sort_values(
        ["_target_date", "_symbol", "_created_order", "_model_version"],
        kind="mergesort", na_position="first"
    ).drop_duplicates(["_target_date", "_symbol"], keep="last")
    overall_auc = _auc_row(latest_active, scope="latest active version per symbol/session") if not latest_active.empty else {"validated_rows": 0, "roc_auc": None}

    coverage = _historical_action_coverage(predictions)
    ranking = _ranking_backtest(joined)

    # Reproduce delivery_policy_update.py::_join() on the RAW append-only ledgers.
    # The policy updater does not first collapse symbol/session/model revisions.
    # Count only active, VALIDATED, gate-approved rows with usable timestamps and
    # non-null model inputs, then deduplicate by prediction_id exactly as _join().
    raw_p = pd.DataFrame(raw_predictions)
    raw_v = pd.DataFrame(raw_validations)
    if raw_p.empty or raw_v.empty:
        policy_joined = pd.DataFrame()
    else:
        raw_p = raw_p[
            raw_p["mode"].astype(str).str.lower().eq("delivery")
            & raw_p["model_version"].astype(str).isin(ACTIVE_MODELS)
        ].copy()
        p_keep = [
            "prediction_id", "symbol", "decision_timestamp",
            "probability_up_next_session", "predicted_return_next_session",
            "production_data_quality_gate", "model_version",
        ]
        raw_p = raw_p[[c for c in p_keep if c in raw_p.columns]]
        v_keep = [
            "prediction_id", "actual_direction_d1", "validation_status",
            "actual_d1_return",
        ]
        raw_v = raw_v[[c for c in v_keep if c in raw_v.columns]]
        policy_joined = raw_p.merge(raw_v, on="prediction_id", how="inner")
        policy_joined = policy_joined[
            policy_joined["validation_status"].eq("VALIDATED")
        ].copy()
        policy_joined = policy_joined[
            policy_joined["production_data_quality_gate"].map(_bool)
        ].copy()
        policy_joined["decision_timestamp"] = pd.to_datetime(
            policy_joined["decision_timestamp"], errors="coerce", utc=True
        )
        for col in (
            "probability_up_next_session",
            "predicted_return_next_session",
            "actual_direction_d1",
        ):
            policy_joined = policy_joined.dropna(subset=[col])
        policy_joined = (
            policy_joined.sort_values("decision_timestamp")
            .drop_duplicates("prediction_id", keep="last")
        )
    policy_session_dates = sorted(
        policy_joined["decision_timestamp"].dt.tz_convert("Asia/Kolkata")
        .dt.strftime("%Y-%m-%d").dropna().unique().tolist()
    ) if not policy_joined.empty else []
    target_session_dates = sorted(
        joined.loc[
            joined["_model_version"].isin(ACTIVE_MODELS)
            & joined["_gate_pass"]
            & joined["_target_aligned"],
            "_target_date",
        ].dropna().unique().tolist()
    )
    policy_session_count = len(policy_session_dates)
    holdout_available = (
        max(0, policy_session_count - int(math.floor(policy_session_count * 0.70)))
        if policy_session_count >= MIN_POLICY_SESSIONS else 0
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "production_model_changed": False,
        "signal_policy_changed": False,
        "google_drive_written": False,
        "source_paths": {
            "current_factor_audit": remote_audit,
            "prediction_ledger": "production_data/predictions/prediction_ledger.jsonl",
            "validation_ledger": "production_data/predictions/prediction_validation.jsonl",
        },
        "ledger_inventory": {
            "raw_prediction_rows": int(len(raw_predictions)),
            "latest_symbol_target_version_prediction_rows": int(len(predictions)),
            "raw_validation_rows": int(len(raw_validations)),
            "latest_validation_rows_by_prediction_id": int(len(validations)),
            "matured_joined_rows": int(len(joined)),
            "target_date_aligned_validated_rows": int(joined["_target_aligned"].sum()),
            "active_model_versions_seen": sorted(set(
                predictions.loc[predictions["_model_version"].isin(ACTIVE_MODELS),
                                 "_model_version"].astype(str)
            )),
        },
        "step_1_current_symbol_decision_breakdown": current,
        "step_2_session_level_auc": {
            "scope": "Active ALL-NSE 59-feature Delivery models; gate-approved, matured, target-date-aligned validations.",
            "flat_outcomes_excluded_from_auc": True,
            "overall_latest_active_version_auc": overall_auc,
            "by_prediction_session": by_session,
            "by_model_version_and_session": by_session_version,
        },
        "step_3_historical_action_coverage": coverage,
        "step_4_session_history_gate": {
            "policy_updater_session_definition": "distinct Asia/Kolkata dates parsed from decision_timestamp, matching delivery_policy_update.py",
            "unique_sessions_counted_by_policy_updater": policy_session_count,
            "policy_session_dates": policy_session_dates,
            "unique_target_sessions_present_in_matured_validations": len(target_session_dates),
            "matured_target_session_dates": target_session_dates,
            "minimum_unique_sessions_required": MIN_POLICY_SESSIONS,
            "minimum_holdout_sessions_required": MIN_HOLDOUT_SESSIONS,
            "remaining_sessions_to_minimum": max(0, MIN_POLICY_SESSIONS - policy_session_count),
            "holdout_sessions_available_under_existing_guard": holdout_available,
            "activation_allowed": policy_session_count >= MIN_POLICY_SESSIONS and holdout_available >= MIN_HOLDOUT_SESSIONS,
            "interpretation": (
                "No policy activation is permitted until at least 20 distinct decision sessions "
                "and at least 5 chronological holdout sessions are available."
            ),
        },
        "step_5_within_session_ranking_and_cost_simulation": ranking,
        "important_limitations": [
            "ROC-AUC measures ranking of UP versus DOWN outcomes, not return magnitude or profitability.",
            "The 30 bps cost is an explicit simulation assumption, not a measured broker-specific cost.",
            "SELL-side returns assume symmetric short exposure; borrow availability and short-sale restrictions are not modelled.",
            "No threshold, weight, model artifact, prediction ledger, validation ledger, or Drive source was modified."
        ],
    }

    path = Path(args.output)
    path.write_text(json.dumps(_json_clean(report), indent=2, default=str), encoding="utf-8")
    summary = {
        "report_file": str(path),
        "generated_at_utc": report["generated_at_utc"],
        "current_quality_approved_hold_rows": current["quality_approved_hold_rows"],
        "current_condition_failure_counts_holds_only": current["condition_failure_counts_holds_only"],
        "per_session_auc_rows": len(by_session),
        "model_session_auc_rows": len(by_session_version),
        "model_version_count": len(coverage["by_model_version"]),
        "policy_sessions": policy_session_count,
        "sessions_remaining_to_20": max(0, MIN_POLICY_SESSIONS - policy_session_count),
        "ranking_status": ranking.get("status"),
        "ranking_selected_rows": {
            str(x["top_fraction_of_aligned_candidates_per_session_pct"]): x["selected_rows"]
            for x in ranking.get("top_ranked_selection", [])
        },
    }
    print("DELIVERY_DECISION_DEEP_AUDIT_SUMMARY")
    print(json.dumps(_json_clean(summary), indent=2))
    print(f"FULL_REPORT_PATH: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
