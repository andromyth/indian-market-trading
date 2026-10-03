#!/usr/bin/env python3
"""Build the pooled ALL-NSE Intraday 30-minute replacement model in Colab.

The trainer consumes persistent NSE 1-minute partitions from Google Drive,
optionally adds timestamp-aligned NIFTY 50 context and point-in-time daily F&O
features, and writes small deployment artifacts plus a manifest to Google Drive.
It never places orders.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

from all_nse_intraday_model import FEATURES

IST="Asia/Kolkata"
UNIVERSE_FILE_CANDIDATES=("production_data/nse_official/intraday_1m", "production_data/intraday_1m")

def _norm(q: pd.DataFrame) -> pd.DataFrame:
    if q is None or q.empty: return pd.DataFrame()
    x=q.copy(); x.columns=[str(c) for c in x.columns]
    aliases={"open":"Open","high":"High","low":"Low","close":"Close","adj close":"Close","ltp":"Close","volume":"Volume","timestamp":"Timestamp","datetime":"Timestamp","date":"Timestamp","time":"Timestamp"}
    x=x.rename(columns={c:aliases.get(str(c).strip().lower(),c) for c in x.columns})
    if "Timestamp" in x.columns:
        idx=pd.to_datetime(x.pop("Timestamp"),errors="coerce",utc=True).dt.tz_convert(IST); x.index=idx
    else:
        x.index=pd.to_datetime(x.index,errors="coerce",utc=True).tz_convert(IST)
    for c in ("Open","High","Low","Close","Volume"):
        if c not in x: x[c]=np.nan
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x[~x.index.isna()].sort_index()
    return x[~x.index.duplicated(keep="last")]

def rsi_series(c,n=14):
    d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False,min_periods=n).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))

def make_features(g: pd.DataFrame, market: pd.DataFrame, fno: pd.DataFrame) -> pd.DataFrame:
    x=_norm(g)
    c,h,l,o,v=x["Close"],x["High"],x["Low"],x["Open"],x["Volume"]
    out=pd.DataFrame(index=x.index)
    for n in (1,3,5,10,15,30): out[f"ret_{n}"]=c.pct_change(n)
    out["rsi14"]=rsi_series(c)
    pc=c.shift(1); tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    out["atr14_pct"]=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/c
    for n in (5,10,20,50): out[f"ma{n}_ratio"]=c/c.rolling(n,min_periods=n).mean()-1
    out["range_pct"]=(h-l)/c; out["body_pct"]=(c-o)/o
    vm=v.rolling(20,min_periods=20).mean(); vs=v.rolling(20,min_periods=20).std()
    out["volume_ratio"]=v/vm; out["volume_z20"]=(v-vm)/vs
    mins=x.index.hour*60+x.index.minute; out["tod_sin"]=np.sin(2*np.pi*mins/390.0); out["tod_cos"]=np.cos(2*np.pi*mins/390.0)
    if market is not None and not market.empty:
        m=_norm(market)
        mc=m["Close"].reindex(x.index,method="ffill",tolerance=pd.Timedelta(minutes=2))
        for n in (1,3,5,10,15,30): out[f"market_ret_{n}"]=mc.pct_change(n)
        out["market_rsi14"]=rsi_series(mc)
    else:
        for n in (1,3,5,10,15,30): out[f"market_ret_{n}"]=np.nan
        out["market_rsi14"]=np.nan
    out["relative_ret_1"]=out["ret_1"]-out["market_ret_1"]
    out["relative_ret_5"]=out["ret_5"]-out["market_ret_5"]
    out["relative_ret_15"]=out["ret_15"]-out["market_ret_15"]

    for k in ("fno_pcr","fno_oi_change","fno_fut_return","fno_fut_oi","fno_basis"): out[k]=np.nan
    out["fno_available"]=0.0
    if fno is not None and not fno.empty:
        z=fno.copy()
        cols={str(c).lower():c for c in z.columns}
        symcol=next((cols[k] for k in ("symbol","ticker","underlying") if k in cols),None)
        datecol=next((cols[k] for k in ("date","trade_date","timestamp") if k in cols),None)
        if symcol and datecol:
            z=z.copy(); z["_sym"]=z[symcol].astype(str).str.upper().str.replace(".NS","",regex=False)
            z["_date"]=pd.to_datetime(z[datecol],errors="coerce").dt.normalize()
            for c in z.columns:
                z[c]=pd.to_numeric(z[c],errors="ignore")
            aliases={
                "fno_pcr": next((cols[k] for k in ("fno_pcr","pcr") if k in cols),None),
                "fno_oi_change": next((cols[k] for k in ("fno_oi_change","oi_change") if k in cols),None),
                "fno_fut_return": next((cols[k] for k in ("fno_fut_return","fut_return") if k in cols),None),
                "fno_fut_oi": next((cols[k] for k in ("fno_fut_oi","fut_oi") if k in cols),None),
                "fno_basis": next((cols[k] for k in ("fno_basis","fut_basis") if k in cols),None),
            }
            # point-in-time: only the last available F&O row strictly before the intraday session
            target_date=pd.Series(x.index.tz_convert(IST).normalize().date,index=x.index)
            z=z.sort_values("_date")
            for outcol,src in aliases.items():
                if src:
                    mp=z.set_index(["_sym","_date"])[src]
                    # symbol-specific daily mapping is applied below after concatenation.
            # caller sets symbol-specific mapping externally
    future=x["Close"].shift(-30)
    out["future_return_30m"]=future/c-1
    # Strictly require an actual timestamp +30m, not simply 30 rows, by timestamp lookup.
    right=pd.DataFrame({"future_ts":x.index,"future_close":c.to_numpy()}).sort_values("future_ts")
    left=pd.DataFrame({"decision_ts":x.index,"future_ts":x.index+pd.Timedelta(minutes=30)}).sort_values("future_ts")
    merged=pd.merge_asof(left,right,on="future_ts",direction="nearest",tolerance=pd.Timedelta(minutes=1))
    f=merged.set_index("decision_ts")["future_close"]
    out["future_return_30m"]=f.reindex(out.index)/c-1
    out["target_30m"]=np.where(out["future_return_30m"].notna(),(out["future_return_30m"]>0).astype(float),np.nan)
    return out

def load_symbol(root: Path, sym: str) -> pd.DataFrame:
    base=next((root/p for p in UNIVERSE_FILE_CANDIDATES if (root/p).exists()),None)
    if base is None: return pd.DataFrame()
    parts=[]
    for p in sorted(base.rglob(f"{sym}.parquet")):
        try:
            z=pd.read_parquet(p)
            if not z.empty: parts.append(_norm(z))
        except Exception: pass
    if not parts:return pd.DataFrame()
    q=pd.concat(parts).sort_index()
    return q[~q.index.duplicated(keep="last")]

def load_market(root: Path)->pd.DataFrame:
    cands=[root/"raw/2018_2025/NIFTY50-INDEX.parquet",root/"raw/2026/NIFTY50-INDEX.parquet"]
    parts=[]
    for p in cands:
        if p.exists():
            try: parts.append(_norm(pd.read_parquet(p)))
            except Exception: pass
    return pd.concat(parts).sort_index() if parts else pd.DataFrame()

def load_fno(root: Path)->pd.DataFrame:
    p=root/"production_data/nse_official/fno/fno_daily_features_2026.parquet"
    try:return pd.read_parquet(p) if p.exists() else pd.DataFrame()
    except Exception:return pd.DataFrame()

def fno_join_features(feature_frame: pd.DataFrame, fno: pd.DataFrame, sym: str)->pd.DataFrame:
    if fno.empty:return feature_frame
    z=fno.copy(); z.columns=[str(c) for c in z.columns]
    lower={c.lower():c for c in z.columns}
    datecol=next((lower[k] for k in ("date","trade_date","timestamp") if k in lower),None)
    symcol=next((lower[k] for k in ("symbol","ticker","underlying") if k in lower),None)
    if not datecol or not symcol:return feature_frame
    z["_date"]=pd.to_datetime(z[datecol],errors="coerce").dt.normalize()
    z=z[z[symcol].astype(str).str.upper().str.replace(".NS","",regex=False).eq(sym.upper())].sort_values("_date")
    if z.empty:return feature_frame
    for outc, aliases in {
        "fno_pcr":("fno_pcr","pcr"),"fno_oi_change":("fno_oi_change","oi_change"),
        "fno_fut_return":("fno_fut_return","fut_return"),"fno_fut_oi":("fno_fut_oi","fut_oi"),
        "fno_basis":("fno_basis","fut_basis","fno_fut_basis")
    }.items():
        src=next((lower[a] for a in aliases if a in lower),None)
        if src:
            d=z[["_date",src]].dropna(subset=["_date"]).copy()
            d["value"]=pd.to_numeric(d[src],errors="coerce")
            session_dates=pd.Series(feature_frame.index.tz_convert(IST).normalize().date,index=feature_frame.index)
            d["decision_date"]=d["_date"].dt.date
            mapping={row.decision_date:row.value for row in d.itertuples()}
            # use strictly prior trading-day record
            out=[]
            dates=sorted(mapping)
            for dt in session_dates:
                prior=[q for q in dates if q < dt]
                out.append(mapping[prior[-1]] if prior else np.nan)
            feature_frame[outc]=out
    feature_frame["fno_available"]=feature_frame[["fno_pcr","fno_oi_change","fno_fut_return","fno_fut_oi","fno_basis"]].notna().any(axis=1).astype(float)
    return feature_frame

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--drive-root",default="/content/drive/MyDrive/Indian_Market_Research")
    ap.add_argument("--out",default="/content/drive/MyDrive/Indian_Market_Research/models/all_nse_intraday_multifactor_replacement_2026")
    ap.add_argument("--min-rows-symbol",type=int,default=300)
    args=ap.parse_args()
    root=Path(args.drive_root); out=Path(args.out)
    # Universe comes from the current official NSE roster if available; fall back to symbols present in the archive.
    try:
        from delivery_training_universe import official_nse_symbols
        universe=sorted(official_nse_symbols(force_refresh=True))
    except Exception:
        base=next((root/p for p in UNIVERSE_FILE_CANDIDATES if (root/p).exists()),None)
        universe=sorted({p.stem.upper() for p in base.rglob("*.parquet")}) if base else []
    market=load_market(root); fno=load_fno(root)
    frames=[]; coverage=[]
    for sym in universe:
        raw=load_symbol(root,sym)
        if len(raw)<args.min_rows_symbol:
            coverage.append((sym,len(raw),"EXCLUDED_INSUFFICIENT_HISTORY")); continue
        feat=make_features(raw,market,fno)
        feat=fno_join_features(feat,fno,sym)
        feat["_symbol"]=sym
        frames.append(feat)
        coverage.append((sym,len(raw),"TRAINABLE"))
    if not frames: raise RuntimeError("No ALL-NSE intraday symbols have sufficient stored 1-minute history.")
    data=pd.concat(frames,axis=0)
    data=data.dropna(subset=["target_30m"])
    data=data.replace([np.inf,-np.inf],np.nan)
    cut=data.index.quantile(0.80)
    train=data[data.index<cut]; test=data[data.index>=cut]
    ytr=train["target_30m"].astype(int); yte=test["target_30m"].astype(int)
    if ytr.nunique()!=2 or yte.nunique()!=2: raise RuntimeError("Target split does not contain both classes")
    clf=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(max_iter=3000,class_weight="balanced",random_state=20261003))])
    reg=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=1.0))])
    clf.fit(train[FEATURES],ytr); reg.fit(train[FEATURES],train["future_return_30m"].astype(float))
    p=clf.predict_proba(test[FEATURES])[:,1]; pred=(p>=0.5).astype(int)
    pred_ret=reg.predict(test[FEATURES])
    train_symbols=sorted(data["_symbol"].unique().tolist())
    universe_size=len(universe); trainable=len(train_symbols)
    manifest={
        "status":"BRANCH_REPLACEMENT_ONLY",
        "model_family":"logistic_regression+ridge",
        "target":"target_30m",
        "feature_count":len(FEATURES),
        "features":FEATURES,
        "universe_size":universe_size,
        "training_symbols":train_symbols,
        "training_symbol_count":trainable,
        "coverage_pct":round(100*trainable/universe_size,4) if universe_size else 0.0,
        "minimum_rows_per_symbol":args.min_rows_symbol,
        "training_rows":len(train),"holdout_rows":len(test),
        "trained_until":str(train.index.max()),"holdout_start":str(test.index.min()),
        "validation":{"accuracy":float(accuracy_score(yte,pred)),"roc_auc":float(roc_auc_score(yte,p)),"mean_abs_return_error":float(mean_absolute_error(test["future_return_30m"],pred_ret))},
        "classifier_sha256":"","regressor_sha256":"",
        "no_auto_orders":True,"data_source":"NSE_OFFICIAL_INTRADAY_1M",
        "point_in_time_fno":True,"volume_policy":"Use reported volume only; missing volume stays NaN."
    }
    out.mkdir(parents=True,exist_ok=True)
    cp=out/"deployment_all_nse_intraday_multifactor_classifier_2026.joblib"; rp=out/"deployment_all_nse_intraday_multifactor_regressor_2026.joblib"
    joblib.dump(clf,cp); joblib.dump(reg,rp)
    manifest["classifier_sha256"]=hashlib.sha256(cp.read_bytes()).hexdigest()
    manifest["regressor_sha256"]=hashlib.sha256(rp.read_bytes()).hexdigest()
    (out/"ALL_NSE_INTRADAY_MULTIFACTOR_2026_MODEL_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    pd.DataFrame(coverage,columns=["symbol","stored_rows","status"]).to_csv(out/"INTRADAY_TRAINING_COVERAGE.csv",index=False)
    print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()
