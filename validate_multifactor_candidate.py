#!/usr/bin/env python3
"""Validate an ALL-NSE multi-factor candidate bundle."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib
from multifactor_data_layer_v2 import FEATURES_59

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model-dir",default="/content/drive/MyDrive/Indian_Market_Research/models/all_nse_delivery_multifactor_candidate_2026"); a=ap.parse_args()
    root=Path(a.model_dir); mp=root/"ALL_NSE_MULTIFACTOR_59_MODEL_MANIFEST_2026.json"
    if not mp.exists(): raise SystemExit(f"Missing manifest: {mp}")
    m=json.loads(mp.read_text(encoding="utf-8"))
    if m.get("promotion_status")!="CANDIDATE_ONLY": raise SystemExit("Refusing non-candidate bundle")
    if m.get("features")!=FEATURES_59: raise SystemExit("59-feature contract mismatch")
    report={"engine_version":m.get("engine_version"),"promotion_status":m.get("promotion_status"),"universe":m.get("universe"),"artifacts":{}}
    for name,item in m.get("artifacts",{}).items():
        report["artifacts"][name]=item
        if not isinstance(item,dict) or item.get("status")!="CANDIDATE_READY": continue
        for kind in ("classifier","regressor"):
            p=Path(item[kind])
            if not p.exists(): raise SystemExit(f"Missing {kind}: {p}")
            model=joblib.load(p); steps=getattr(model,"named_steps",{}) ; imp=steps.get("imputer") if hasattr(steps,"get") else None
            learned=int(getattr(imp,"n_features_in_",0)) if imp is not None else 0
            if not learned and imp is not None and getattr(imp,"statistics_",None) is not None: learned=int(len(imp.statistics_))
            if learned!=59: raise SystemExit(f"{p}: expected 59 original features, found {learned}")
            report["artifacts"].setdefault(name,{})[kind+"_sha256"]=sha256(p)
    out=root/"CANDIDATE_VALIDATION_REPORT_2026.json"; out.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print(json.dumps(report,indent=2,default=str))

if __name__=="__main__": main()
