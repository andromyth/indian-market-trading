#!/usr/bin/env python3
"""OOS-gated adaptive Delivery decision-policy updater. Never changes model weights."""
from __future__ import annotations
import argparse, io, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from persistent_store import GoogleDriveStore
from prediction_ledger import load_records, load_validation_records

REMOTE_POLICY="models/all_nse_delivery_multifactor_replacement_2026/ACTIVE_SIGNAL_POLICY.json"
LOCAL_POLICY=Path("/tmp/ACTIVE_SIGNAL_POLICY.json")
DEFAULT={"buy_probability_threshold":0.60,"sell_probability_threshold":0.40,"min_expected_return":0.0015}
MIN_SAMPLES=80
MIN_HOLDOUT_ACTIONS=10

def _load_existing():
    try:
        g=GoogleDriveStore(); parts=REMOTE_POLICY.split("/"); p=g.root_id
        for part in parts[:-1]:
            hit=g._find_child(p,part)
            if not hit: return {"status":"DEFAULT","samples":0,**DEFAULT}
            p=hit["id"]
        hit=g._find_child(p,parts[-1])
        if not hit: return {"status":"DEFAULT","samples":0,**DEFAULT}
        import googleapiclient.http
        buf=io.BytesIO()
        dl=googleapiclient.http.MediaIoBaseDownload(buf,g.service.files().get_media(fileId=hit["id"]))
        done=False
        while not done: _,done=dl.next_chunk()
        return {**DEFAULT,**json.loads(buf.getvalue().decode("utf-8"))}
    except Exception:
        return {"status":"DEFAULT","samples":0,**DEFAULT}

def _precision(df,buy,sell,min_ret):
    p=pd.to_numeric(df["probability_up_next_session"],errors="coerce")
    r=pd.to_numeric(df["predicted_return_next_session"],errors="coerce")
    actual=df["actual_direction_d1"].astype(str).str.upper().to_numpy()
    signal=np.where((p>=buy)&(r>min_ret),"BUY",np.where((p<=sell)&(r<-min_ret),"SELL","HOLD"))
    action=np.isin(signal,["BUY","SELL"])
    if not action.any(): return {"precision":np.nan,"actions":0,"coverage":0.0}
    correct=((signal=="BUY")&(actual=="UP"))|((signal=="SELL")&(actual=="DOWN"))
    return {"precision":float(correct[action].mean()),"actions":int(action.sum()),"coverage":float(action.mean())}

def _join():
    p=pd.DataFrame(load_records()); v=pd.DataFrame(load_validation_records())
    if p.empty or v.empty or "prediction_id" not in p.columns or "prediction_id" not in v.columns: return pd.DataFrame()
    p=p[p["mode"].astype(str).str.lower().eq("delivery")].copy()
    keep=["prediction_id","symbol","decision_timestamp","probability_up_next_session","predicted_return_next_session","data_readiness_score","decision_engine_version"]
    p=p[[c for c in keep if c in p.columns]]
    keepv=["prediction_id","actual_direction_d1","validation_status","actual_d1_return"]
    v=v[[c for c in keepv if c in v.columns]]
    x=p.merge(v,on="prediction_id",how="inner")
    x=x[x["validation_status"].eq("VALIDATED")].copy()
    if "production_data_quality_gate" in x.columns:
        x=x[x["production_data_quality_gate"].fillna(False).astype(bool)].copy()
    x["decision_timestamp"]=pd.to_datetime(x["decision_timestamp"],errors="coerce",utc=True)
    return x.dropna(subset=["decision_timestamp","probability_up_next_session","predicted_return_next_session","actual_direction_d1"]).sort_values("decision_timestamp").drop_duplicates("prediction_id",keep="last")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--min-samples",type=int,default=MIN_SAMPLES); args=ap.parse_args()
    existing=_load_existing(); data=_join()
    if len(data)<args.min_samples:
        out={**existing,"status":"INSUFFICIENT_HISTORY","samples":int(len(data)),"updated_at":datetime.now(timezone.utc).isoformat(),"method":"walk-forward-policy-tuning"}
        LOCAL_POLICY.write_text(json.dumps(out,indent=2,default=str),encoding="utf-8")
        print(json.dumps(out,indent=2)); return
    split=max(1,int(len(data)*0.70)); tune=data.iloc[:split]; hold=data.iloc[split:]
    cb=float(existing.get("buy_probability_threshold",DEFAULT["buy_probability_threshold"]))
    cs=float(existing.get("sell_probability_threshold",DEFAULT["sell_probability_threshold"]))
    cr=float(existing.get("min_expected_return",DEFAULT["min_expected_return"]))
    baseline=_precision(hold,cb,cs,cr)
    candidates=[]
    for buy in np.arange(0.55,0.76,0.02):
        for sell in np.arange(0.25,0.46,0.02):
            if buy<=0.50 or sell>=0.50: continue
            t=_precision(tune,round(float(buy),2),round(float(sell),2),cr)
            if t["actions"]<15 or not np.isfinite(t["precision"]): continue
            candidates.append((t["precision"]+0.10*t["coverage"],round(float(buy),2),round(float(sell),2),t))
    chosen={**existing,"status":"UNCHANGED","samples":int(len(data)),"method":"walk-forward-policy-tuning"}
    if candidates:
        _,buy,sell,tuned=max(candidates,key=lambda z:(z[0],z[3]["actions"]))
        ch=_precision(hold,buy,sell,cr)
        improve=np.isfinite(ch["precision"]) and (not np.isfinite(baseline["precision"]) or ch["precision"]>=baseline["precision"]+0.05)
        keepcov=ch["coverage"]>=max(0.03,baseline["coverage"]*0.50)
        if ch["actions"]>=MIN_HOLDOUT_ACTIONS and improve and keepcov:
            chosen.update({"status":"ADAPTIVE","buy_probability_threshold":buy,"sell_probability_threshold":sell,"min_expected_return":cr,
                "selection_tuning_precision":tuned["precision"],"selection_tuning_actions":tuned["actions"],
                "holdout_precision":ch["precision"],"holdout_actions":ch["actions"],"holdout_coverage":ch["coverage"],
                "baseline_holdout_precision":baseline["precision"],"baseline_holdout_actions":baseline["actions"]})
    chosen["updated_at"]=datetime.now(timezone.utc).isoformat()
    g=GoogleDriveStore(); LOCAL_POLICY.write_text(json.dumps(chosen,indent=2,default=str),encoding="utf-8")
    g.upload_or_update(LOCAL_POLICY,REMOTE_POLICY,mime_type="application/json")
    print(json.dumps(chosen,indent=2,default=str))
if __name__=="__main__": main()
