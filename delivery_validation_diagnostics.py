#!/usr/bin/env python3
"""Generate an automated OOS diagnostic for the Delivery prediction ledger."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from persistent_store import GoogleDriveStore
from prediction_ledger import load_records, load_validation_records

REMOTE="features/daily/DELIVERY_VALIDATION_DIAGNOSTICS.json"
LOCAL=Path("/tmp/DELIVERY_VALIDATION_DIAGNOSTICS.json")

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

def main():
    preds=pd.DataFrame(load_records())
    vals=pd.DataFrame(load_validation_records())
    report={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"status":"NO_DATA"}
    if not preds.empty and not vals.empty and "prediction_id" in preds and "prediction_id" in vals:
        p=preds[preds["mode"].astype(str).str.lower().eq("delivery")].copy()
        cols=["prediction_id","symbol","decision_timestamp","probability_up_next_session","predicted_return_next_session","production_data_quality_gate","production_data_quality_score","model_version","decision_engine_version"]
        p=p[[c for c in cols if c in p.columns]]
        v=vals[[c for c in ["prediction_id","actual_direction_d1","actual_d1_return","validation_status"] if c in vals.columns]]
        x=p.merge(v,on="prediction_id",how="inner")
        x=x[x["validation_status"].eq("VALIDATED")].copy()
        x["decision_timestamp"]=pd.to_datetime(x["decision_timestamp"],errors="coerce",utc=True)
        x=x.dropna(subset=["decision_timestamp"]).sort_values("decision_timestamp").drop_duplicates("prediction_id",keep="last")
        report.update({
          "status":"PASS" if len(x) else "NO_DATA",
          "all_validated":_evaluate(x),
          "recent_30":_evaluate(x.tail(30)),
          "recent_60":_evaluate(x.tail(60)),
          "recent_120":_evaluate(x.tail(120)),
          "quality_approved":_evaluate(x[x.get("production_data_quality_gate",pd.Series(True,index=x.index)).fillna(False).astype(bool)]) if "production_data_quality_gate" in x.columns else {},
          "symbols":int(x["symbol"].nunique()) if "symbol" in x.columns else 0,
          "models":sorted({str(z) for z in x["model_version"].dropna()}) if "model_version" in x.columns else [],
          "decision_engines":sorted({str(z) for z in x["decision_engine_version"].dropna()}) if "decision_engine_version" in x.columns else [],
        })
        if report["recent_30"].get("action_precision") is not None and report["recent_30"].get("action_precision") < 0.50:
            report["diagnosis"]="Recent action precision is below 50%; maintain conservative policy and wait for OOS-gated adaptation rather than force retraining."
        elif report["all_validated"].get("roc_auc") is not None and report["all_validated"]["roc_auc"] < 0.52:
            report["diagnosis"]="Directional probability separation is weak; do not claim high accuracy. Prioritize richer causal features and stronger walk-forward validation before changing model weights."
        else:
            report["diagnosis"]="Continue prospective OOS monitoring; only activate policy/model changes that beat a recent holdout."
    LOCAL.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    g=GoogleDriveStore()
    g.upload_or_update(LOCAL,REMOTE,mime_type="application/json")
    print(json.dumps(report,indent=2,default=str))

if __name__=="__main__":
    main()
