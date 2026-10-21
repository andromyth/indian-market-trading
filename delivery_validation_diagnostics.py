#!/usr/bin/env python3
"""Generate an automated OOS diagnostic for the Delivery prediction ledger."""
from __future__ import annotations
import json
from datetime import date, datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from persistent_store import GoogleDriveStore
from prediction_ledger import load_records, load_validation_records
from prediction_validation import ValidationConfig, validation_status_for_sample_count

REMOTE="features/daily/DELIVERY_VALIDATION_DIAGNOSTICS.json"
LOCAL=Path("/tmp/DELIVERY_VALIDATION_DIAGNOSTICS.json")

DIAGNOSTICS_SCHEMA_VERSION = "2.0"
MIN_TARGET_VALIDATION_SAMPLES = 80
MIN_TARGET_ACTIONS = 30
TARGET_ACTION_PRECISION = 0.80
ACTIVE_DELIVERY_MODEL_VERSIONS = {
    "2026-all-nse-delivery-multifactor-59-replacement-v1",
    "2026-all-nse-delivery-multifactor-59-candidate-v2",
    "2026-all-nse-delivery-multifactor-59-replacement-v2",
}

def _evaluate(x):
    x=x.copy()
    required={"probability_up_next_session","predicted_return_next_session","actual_direction_d1"}
    if x.empty or not required.issubset(x.columns):
        return {"samples":0}
    p=pd.to_numeric(x["probability_up_next_session"],errors="coerce")
    r=pd.to_numeric(x["predicted_return_next_session"],errors="coerce")
    actual=x["actual_direction_d1"].astype(str).str.upper()
    valid=actual.isin(["UP","DOWN"]) & p.notna() & r.notna()
    x=x[valid].copy()
    if x.empty:
        return {"samples":0}
    p=pd.to_numeric(x["probability_up_next_session"],errors="coerce")
    r=pd.to_numeric(x["predicted_return_next_session"],errors="coerce")
    actual=x["actual_direction_d1"].astype(str).str.upper()
    y=actual.map({"UP":1,"DOWN":0}).astype(float)
    signal=np.where((p>=0.60)&(r>0.0015),"BUY",np.where((p<=0.40)&(r<-0.0015),"SELL","HOLD"))
    action=np.isin(signal,["BUY","SELL"])
    correct=((signal=="BUY")&(y==1))|((signal=="SELL")&(y==0))
    out={
      "samples":int(len(x)),
      "roc_auc":float(roc_auc_score(y,p)) if y.nunique()==2 else None,
      "brier_score":float(((p-y)**2).mean()),
      "directional_accuracy":float((p.ge(0.5)==y.eq(1)).mean()),
      "action_precision":float(correct[action].mean()) if action.any() else None,
      "action_coverage":float(action.mean()),
      "buy_count":int((signal=="BUY").sum()),
      "sell_count":int((signal=="SELL").sum()),
      "hold_count":int((signal=="HOLD").sum()),
      "mean_actual_return":float(pd.to_numeric(x["actual_d1_return"],errors="coerce").mean()),
    }
    return out

def _gate_pass(value) -> bool:
    """Treat unknown data-quality state as unapproved."""
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "ready"}
    try:
        return bool(int(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _iso_date(value):
    """Parse a date-like field without assuming a missing value is current."""
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def build_report(
    preds: pd.DataFrame,
    vals: pd.DataFrame,
    *,
    latest_official_eod_date: str | date | None = None,
) -> dict:
    """Score completed outcomes, distinguishing un-matured forecasts from data gaps."""
    rolling_health_minimum = int(ValidationConfig().minimum_health_samples)
    minimum = MIN_TARGET_VALIDATION_SAMPLES
    eod_day = _iso_date(latest_official_eod_date)
    report = {
        "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "NO_DATA",
        "status_scope": "quality-approved predictions from the active ALL-NSE 59-feature Delivery model",
        "minimum_samples_required": minimum,
        "minimum_rolling_health_samples": rolling_health_minimum,
        "target_action_precision": TARGET_ACTION_PRECISION,
        "minimum_target_actions": MIN_TARGET_ACTIONS,
        "active_model_versions_expected": sorted(ACTIVE_DELIVERY_MODEL_VERSIONS),
        "latest_official_eod_date": eod_day,
        "active_model_prediction_count": 0,
        "active_model_target_date_min": None,
        "active_model_target_date_max": None,
        "active_model_targets_with_known_date": 0,
        "active_model_matured_prediction_count": None,
        "active_model_waiting_for_target_count": None,
        "active_model_scored_samples": 0,
        "quality_approved_active_samples": 0,
        "history_sufficient": False,
    }
    if preds.empty or "prediction_id" not in preds.columns:
        report["diagnosis"] = "No prediction rows with stable IDs are available."
        return report
    if "mode" not in preds.columns:
        report["diagnosis"] = "Prediction ledger has no mode field; refusing to score an ambiguous population."
        return report

    p_all = preds[preds["mode"].astype(str).str.lower().eq("delivery")].copy()
    if "model_version" in p_all.columns:
        p_all["model_version"] = p_all["model_version"].astype(str)
        active_prediction_mask = p_all["model_version"].isin(ACTIVE_DELIVERY_MODEL_VERSIONS)
    else:
        active_prediction_mask = pd.Series(False, index=p_all.index)
    active_predictions = p_all.loc[active_prediction_mask].copy()
    report["active_model_prediction_count"] = int(len(active_predictions))
    report["active_models_seen"] = (
        sorted(set(active_predictions["model_version"].dropna().astype(str)))
        if "model_version" in active_predictions.columns else []
    )

    target_dates = pd.Series(index=active_predictions.index, dtype="object")
    if "prediction_for_date" in active_predictions.columns:
        target_dates = active_predictions["prediction_for_date"].map(_iso_date)
    known_targets = target_dates.notna()
    report["active_model_targets_with_known_date"] = int(known_targets.sum())
    if bool(known_targets.any()):
        parsed_known = sorted(target_dates.loc[known_targets].tolist())
        report["active_model_target_date_min"] = parsed_known[0]
        report["active_model_target_date_max"] = parsed_known[-1]
    if eod_day is not None:
        report["active_model_matured_prediction_count"] = int(
            (target_dates.notna() & target_dates.le(eod_day)).sum()
        )
        report["active_model_waiting_for_target_count"] = int(
            (target_dates.notna() & target_dates.gt(eod_day)).sum()
        )

    if p_all.empty:
        report["diagnosis"] = "No Delivery prediction rows are available to score."
        return report

    cols = [
        "prediction_id", "symbol", "decision_timestamp", "prediction_for_date",
        "probability_up_next_session", "predicted_return_next_session",
        "production_data_quality_gate", "production_data_quality_score",
        "model_version", "engine_version", "decision_engine_version",
    ]
    p = p_all[[col for col in cols if col in p_all.columns]].copy()
    vcols = ["prediction_id", "actual_direction_d1", "actual_d1_return", "validation_status"]
    if vals.empty or "prediction_id" not in vals.columns:
        v = pd.DataFrame(columns=vcols)
    else:
        v = vals[[col for col in vcols if col in vals.columns]].copy()
        if "prediction_id" not in v.columns:
            v["prediction_id"] = pd.Series(dtype="object")
    x = p.merge(v, on="prediction_id", how="inner")
    if "validation_status" in x.columns:
        x = x[x["validation_status"].astype(str).str.upper().eq("VALIDATED")].copy()
    if "decision_timestamp" not in x.columns:
        report["diagnosis"] = "Joined prediction ledger lacks decision_timestamp."
        return report
    x["decision_timestamp"] = pd.to_datetime(x["decision_timestamp"], errors="coerce", utc=True)
    x = x.dropna(subset=["decision_timestamp"]).sort_values("decision_timestamp").drop_duplicates("prediction_id", keep="last")

    if "model_version" in x.columns:
        x["model_version"] = x["model_version"].astype(str)
        active_mask = x["model_version"].isin(ACTIVE_DELIVERY_MODEL_VERSIONS)
    else:
        active_mask = pd.Series(False, index=x.index)
    active = x.loc[active_mask].copy()
    legacy = x.loc[~active_mask].copy()
    if "production_data_quality_gate" in active.columns:
        quality_active = active.loc[active["production_data_quality_gate"].map(_gate_pass)].copy()
    else:
        quality_active = active.iloc[0:0].copy()

    active_metrics = _evaluate(active)
    quality_metrics = _evaluate(quality_active)
    all_metrics = _evaluate(x)
    legacy_metrics = _evaluate(legacy)
    active_count = int(active_metrics.get("samples", 0))
    quality_count = int(quality_metrics.get("samples", 0))

    if active_count == 0:
        matured_count = report["active_model_matured_prediction_count"]
        waiting_count = report["active_model_waiting_for_target_count"]
        known_count = report["active_model_targets_with_known_date"]
        active_prediction_count = report["active_model_prediction_count"]
        if (
            active_prediction_count > 0
            and eod_day is not None
            and known_count == active_prediction_count
            and matured_count == 0
            and waiting_count == active_prediction_count
        ):
            status = "ACTIVE_MODEL_AWAITING_MATURITY"
            diagnosis = (
                "Active-model forecasts are present, but every known target session is later "
                "than the latest official EOD date. No active-model accuracy score is available "
                "yet; these forecasts remain unscored until their official target bars arrive."
            )
        elif active_prediction_count > 0 and eod_day is not None and (matured_count or 0) > 0:
            status = "ACTIVE_MODEL_OUTCOME_GAP"
            diagnosis = (
                "Active-model forecasts have target sessions at or before the latest official "
                "EOD date, but no validated outcomes were joined for the active model. Repair "
                "the prediction-ID/validation persistence or target-bar coverage before claiming accuracy."
            )
        else:
            status = "NO_ACTIVE_MODEL_DATA"
            diagnosis = (
                "No completed outcomes from the active ALL-NSE 59-feature Delivery model were found, "
                "and the available target-date evidence is insufficient to classify this as a normal maturity wait."
            )
    elif quality_count == 0:
        status = "NO_QUALITY_APPROVED_DATA"
        diagnosis = "Active-model outcomes exist, but none passed the production data-quality gate. Do not claim accuracy; repair PIT/data-readiness gates first."
    elif quality_count < minimum:
        status = "INSUFFICIENT_HISTORY"
        diagnosis = f"Only {quality_count} quality-approved active-model outcomes are available; at least {minimum} are required for policy evaluation. Seven outcomes can support short rolling monitoring, but not a reliable 80-90% accuracy claim."
    else:
        status = "OOS_HISTORY_AVAILABLE_TARGET_UNPROVEN"
        diagnosis = "Sufficient sample history exists to evaluate the policy, but this diagnostic is not itself a sealed holdout. Require the OOS-gated policy updater's later chronological holdout before claiming the 80% target."

    quality_all = x.loc[x["production_data_quality_gate"].map(_gate_pass)] if "production_data_quality_gate" in x.columns else x.iloc[0:0]
    report.update({
        "status": status,
        "history_sufficient": quality_count >= minimum,
        "scored_samples": quality_count,
        "active_model_scored_samples": active_count,
        "quality_approved_active_samples": quality_count,
        "all_validated": quality_metrics,
        "recent_30": _evaluate(quality_active.tail(30)),
        "recent_60": _evaluate(quality_active.tail(60)),
        "recent_120": _evaluate(quality_active.tail(120)),
        "all_validated_all_versions_diagnostic_only": all_metrics,
        "legacy_model_metrics_diagnostic_only": legacy_metrics,
        "active_model_all_quality_states": active_metrics,
        "quality_approved_all_versions_diagnostic_only": _evaluate(quality_all),
        "action_precision_target_met_on_monitoring_sample": bool(
            quality_metrics.get("action_precision") is not None
            and (quality_metrics.get("buy_count", 0) + quality_metrics.get("sell_count", 0)) >= MIN_TARGET_ACTIONS
            and quality_metrics["action_precision"] >= TARGET_ACTION_PRECISION
        ),
        "target_interpretation": "Monitoring precision is descriptive only; target success must be demonstrated by a later chronological holdout with enough BUY/SELL actions.",
        "active_model_outcome_versions_seen": sorted(set(active["model_version"].dropna().astype(str))) if "model_version" in active.columns else [],
        "legacy_models_seen": sorted(set(legacy["model_version"].dropna().astype(str))) if "model_version" in legacy.columns else [],
        "decision_engines": sorted(set(x["decision_engine_version"].dropna().astype(str))) if "decision_engine_version" in x.columns else [],
        "symbols": int(active["symbol"].nunique()) if "symbol" in active.columns else 0,
        "diagnosis": diagnosis,
    })
    return report

def main():
    preds = pd.DataFrame(load_records())
    vals = pd.DataFrame(load_validation_records())
    report = build_report(preds, vals)
    LOCAL.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    g = GoogleDriveStore()
    g.upload_or_update(LOCAL, REMOTE, mime_type="application/json")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
