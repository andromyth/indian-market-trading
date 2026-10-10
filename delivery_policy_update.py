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
MIN_TUNING_ACTIONS=20
MIN_HOLDOUT_ACTIONS=30
TARGET_HOLDOUT_PRECISION=0.80
MIN_UNIQUE_SESSIONS=20
MIN_HOLDOUT_SESSIONS=5

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

def _wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """Two-sided 95% Wilson lower bound for a binomial hit rate."""
    if trials <= 0:
        return float("nan")
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials)
    return float((center - margin) / denom)

def _precision(df,buy,sell,min_ret):
    p=pd.to_numeric(df["probability_up_next_session"],errors="coerce")
    r=pd.to_numeric(df["predicted_return_next_session"],errors="coerce")
    actual=df["actual_direction_d1"].astype(str).str.upper().to_numpy()
    signal=np.where((p>=buy)&(r>min_ret),"BUY",np.where((p<=sell)&(r<-min_ret),"SELL","HOLD"))
    action=np.isin(signal,["BUY","SELL"])
    if not action.any(): return {"precision":np.nan,"wilson_lower_95":np.nan,"actions":0,"coverage":0.0}
    correct=((signal=="BUY")&(actual=="UP"))|((signal=="SELL")&(actual=="DOWN"))
    successes=int(correct[action].sum()); actions=int(action.sum())
    return {"precision":float(successes/actions),"wilson_lower_95":_wilson_lower_bound(successes,actions),
            "actions":actions,"coverage":float(action.mean())}

def _gate_pass(value) -> bool:
    """Parse serialized quality-gate values safely; the string 'False' is false."""
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


def _date_block_split(data: pd.DataFrame, train_fraction: float = 0.70):
    """Split by NSE session date so one session cannot leak across tune/holdout."""
    if data.empty or "decision_timestamp" not in data.columns:
        return None, 0, 0
    x = data.copy()
    ts = pd.to_datetime(x["decision_timestamp"], errors="coerce", utc=True)
    valid = ts.notna()
    x = x.loc[valid].copy()
    ts = ts.loc[valid]
    if x.empty:
        return None, 0, 0
    x["_session_date"] = ts.dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d")
    dates = sorted(x["_session_date"].dropna().unique().tolist())
    hold_count = max(0, len(dates) - int(len(dates) * train_fraction))
    if len(dates) < MIN_UNIQUE_SESSIONS or hold_count < MIN_HOLDOUT_SESSIONS:
        return None, len(dates), hold_count
    split_idx = int(len(dates) * train_fraction)
    split_idx = min(max(1, split_idx), len(dates) - MIN_HOLDOUT_SESSIONS)
    tune_dates, hold_dates = set(dates[:split_idx]), set(dates[split_idx:])
    tune = x[x["_session_date"].isin(tune_dates)].drop(columns=["_session_date"])
    hold = x[x["_session_date"].isin(hold_dates)].drop(columns=["_session_date"])
    if tune.empty or hold.empty:
        return None, len(dates), len(hold_dates)
    # The assertion is deliberately explicit: all rows from a session belong
    # wholly to tuning OR holdout, never both.
    assert set(tune_dates).isdisjoint(hold_dates)
    return (tune, hold), len(dates), len(hold_dates)


def _join():
    p=pd.DataFrame(load_records()); v=pd.DataFrame(load_validation_records())
    if p.empty or v.empty or "prediction_id" not in p.columns or "prediction_id" not in v.columns: return pd.DataFrame()
    p=p[p["mode"].astype(str).str.lower().eq("delivery")].copy()
    keep=["prediction_id","symbol","decision_timestamp","probability_up_next_session","predicted_return_next_session","data_readiness_score","production_data_quality_gate","production_data_quality_score","decision_engine_version"]
    p=p[[c for c in keep if c in p.columns]]
    keepv=["prediction_id","actual_direction_d1","validation_status","actual_d1_return"]
    v=v[[c for c in keepv if c in v.columns]]
    x=p.merge(v,on="prediction_id",how="inner")
    x=x[x["validation_status"].eq("VALIDATED")].copy()
    if "production_data_quality_gate" in x.columns:
        x=x[x["production_data_quality_gate"].map(_gate_pass)].copy()
    x["decision_timestamp"]=pd.to_datetime(x["decision_timestamp"],errors="coerce",utc=True)
    return x.dropna(subset=["decision_timestamp","probability_up_next_session","predicted_return_next_session","actual_direction_d1"]).sort_values("decision_timestamp").drop_duplicates("prediction_id",keep="last")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--min-samples",type=int,default=MIN_SAMPLES); args=ap.parse_args()
    existing=_load_existing(); data=_join()
    if len(data)<args.min_samples:
        out={**existing,"status":"INSUFFICIENT_HISTORY","samples":int(len(data)),"updated_at":datetime.now(timezone.utc).isoformat(),"method":"date-blocked-walk-forward-policy-tuning"}
        LOCAL_POLICY.write_text(json.dumps(out,indent=2,default=str),encoding="utf-8")
        print(json.dumps(out,indent=2)); return
    split_frames, session_count, holdout_session_count = _date_block_split(data)
    if split_frames is None:
        out={**existing,"status":"INSUFFICIENT_SESSION_HISTORY","samples":int(len(data)),
             "unique_sessions":int(session_count),"holdout_sessions_available":int(holdout_session_count),
             "minimum_unique_sessions":MIN_UNIQUE_SESSIONS,"minimum_holdout_sessions":MIN_HOLDOUT_SESSIONS,
             "updated_at":datetime.now(timezone.utc).isoformat(),"method":"date-blocked-walk-forward-policy-tuning"}
        LOCAL_POLICY.write_text(json.dumps(out,indent=2,default=str),encoding="utf-8")
        print(json.dumps(out,indent=2)); return
    tune, hold = split_frames
    cb=float(existing.get("buy_probability_threshold",DEFAULT["buy_probability_threshold"]))
    cs=float(existing.get("sell_probability_threshold",DEFAULT["sell_probability_threshold"]))
    cr=float(existing.get("min_expected_return",DEFAULT["min_expected_return"]))
    baseline=_precision(hold,cb,cs,cr)
    candidates=[]
    for buy in np.arange(0.60,0.861,0.02):
        for sell in np.arange(0.14,0.401,0.02):
            if buy<=0.50 or sell>=0.50: continue
            t=_precision(tune,round(float(buy),2),round(float(sell),2),cr)
            if t["actions"]<MIN_TUNING_ACTIONS or not np.isfinite(t["precision"]): continue
            # Prefer a conservative confidence-bound estimate over raw precision.
            candidates.append((t["wilson_lower_95"],t["precision"],round(float(buy),2),round(float(sell),2),t))
    chosen={**existing,"status":"UNCHANGED","samples":int(len(data)),"method":"walk-forward-policy-tuning"}
    if candidates:
        _,_,buy,sell,tuned=max(candidates,key=lambda z:(z[0],z[1],z[4]["actions"]))
        ch=_precision(hold,buy,sell,cr)
        improve=np.isfinite(ch["precision"]) and (not np.isfinite(baseline["precision"]) or ch["precision"]>=baseline["precision"]+0.05)
        keepcov=ch["coverage"]>=max(0.02,baseline["coverage"]*0.25)
        # The final chronological holdout must itself meet the user's high-precision
        # objective. Never activate a merely better-than-baseline but weak policy.
        meets_target=np.isfinite(ch["precision"]) and ch["precision"]>=TARGET_HOLDOUT_PRECISION
        if ch["actions"]>=MIN_HOLDOUT_ACTIONS and meets_target and improve and keepcov:
            chosen.update({"status":"ADAPTIVE","buy_probability_threshold":buy,"sell_probability_threshold":sell,"min_expected_return":cr,
                "target_holdout_precision":TARGET_HOLDOUT_PRECISION,
                "selection_tuning_precision":tuned["precision"],"selection_tuning_wilson_lower_95":tuned["wilson_lower_95"],"selection_tuning_actions":tuned["actions"],
                "holdout_precision":ch["precision"],"holdout_wilson_lower_95":ch["wilson_lower_95"],"holdout_actions":ch["actions"],"holdout_coverage":ch["coverage"],
                "baseline_holdout_precision":baseline["precision"],"baseline_holdout_actions":baseline["actions"]})
        else:
            chosen.update({"status":"UNCHANGED_TARGET_NOT_MET","target_holdout_precision":TARGET_HOLDOUT_PRECISION,
                "candidate_holdout_precision":ch["precision"],"candidate_holdout_wilson_lower_95":ch["wilson_lower_95"],
                "candidate_holdout_actions":ch["actions"],"baseline_holdout_precision":baseline["precision"],
                "baseline_holdout_actions":baseline["actions"],
                "reason":"Keep current policy unless the later chronological holdout has >=80% action precision, enough actions, and a meaningful improvement."})
    chosen["updated_at"]=datetime.now(timezone.utc).isoformat()
    g=GoogleDriveStore(); LOCAL_POLICY.write_text(json.dumps(chosen,indent=2,default=str),encoding="utf-8")
    g.upload_or_update(LOCAL_POLICY,REMOTE_POLICY,mime_type="application/json")
    print(json.dumps(chosen,indent=2,default=str))
if __name__=="__main__": main()
