#!/usr/bin/env python3
"""Build the ALL-NSE 5-minute Intraday model.

This is a separate 5-minute model. It does not overwrite or retrain the
existing certified 30-minute production artifact.

Outputs:
- LogisticRegression classifier for target_5m
- Ridge return regressor for future_return_5m
- numeric model-estimated price at +5 minutes
- holdout metrics and provenance manifest

Training uses the same 1-minute OHLC/market/F&O feature family as the existing
ALL-NSE Intraday multifactor candidate. The target is strictly aligned to
decision_timestamp + 5 minutes, within a 1-minute tolerance.

No automatic orders.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from all_nse_intraday_5m_model import FEATURES

IST = "Asia/Kolkata"
INTRADAY_REL = Path("production_data/nse_official/intraday_1m")
SEED_REL = Path("production_data/nse_official/intraday_seed/olo_minute_archive/symbols")


def _norm(q: pd.DataFrame) -> pd.DataFrame:
    if q is None or q.empty:
        return pd.DataFrame()
    x = q.copy()
    x.columns = [str(c).strip() for c in x.columns]
    aliases = {
        "open":"Open","high":"High","low":"Low","close":"Close",
        "adj close":"Close","ltp":"Close","volume":"Volume",
        "timestamp":"Timestamp","datetime":"Timestamp","date":"Timestamp","time":"Timestamp",
    }
    x = x.rename(columns={c: aliases.get(str(c).strip().lower(), c) for c in x.columns})
    if "Timestamp" in x.columns:
        idx = pd.to_datetime(x.pop("Timestamp"), errors="coerce")
    else:
        idx = pd.to_datetime(x.index, errors="coerce")
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    else:
        idx = idx.tz_convert(IST)
    x.index = idx
    for col in ("Open","High","Low","Close","Volume"):
        if col not in x.columns:
            x[col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x[~x.index.isna()].sort_index()
    return x[~x.index.duplicated(keep="last")]


def _returns(c: pd.Series, n: int) -> pd.Series:
    return c.pct_change(n)


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    return 100 - 100/(1 + up/dn.replace(0, np.nan))


def make_features(g: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    x = _norm(g)
    if x.empty:
        return pd.DataFrame()
    c,h,l,o,v = x["Close"],x["High"],x["Low"],x["Open"],x["Volume"]
    out = pd.DataFrame(index=x.index)
    for n in (1,3,5,10,15,30):
        out[f"ret_{n}"] = c.pct_change(n)
    out["rsi14"] = _rsi(c)
    pc = c.shift(1)
    tr = pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    out["atr14_pct"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()/c
    for n in (5,10,20,50):
        out[f"ma{n}_ratio"] = c/c.rolling(n,min_periods=n).mean()-1
    out["range_pct"]=(h-l)/c
    out["body_pct"]=(c-o)/o
    vm=v.rolling(20,min_periods=20).mean()
    vs=v.rolling(20,min_periods=20).std()
    out["volume_ratio"]=v/vm
    out["volume_z20"]=(v-vm)/vs
    mins=x.index.hour*60+x.index.minute-(9*60+15)
    mins=np.clip(mins,0,389)
    out["tod_sin"]=np.sin(2*np.pi*mins/390.0)
    out["tod_cos"]=np.cos(2*np.pi*mins/390.0)

    if market is not None and not market.empty:
        m=market if isinstance(market.index,pd.DatetimeIndex) else _norm(market)
        mc=m["Close"].reindex(x.index,method="ffill",tolerance=pd.Timedelta(minutes=2))
        for n in (1,3,5,10,15,30):
            out[f"market_ret_{n}"]=mc.pct_change(n,fill_method=None)
        out["market_rsi14"]=_rsi(mc)
    else:
        for n in (1,3,5,10,15,30):
            out[f"market_ret_{n}"]=np.nan
        out["market_rsi14"]=np.nan

    out["relative_ret_1"]=out["ret_1"]-out["market_ret_1"]
    out["relative_ret_5"]=out["ret_5"]-out["market_ret_5"]
    out["relative_ret_15"]=out["ret_15"]-out["market_ret_15"]

    right=pd.DataFrame({"future_ts":x.index,"future_close":c.to_numpy()}).sort_values("future_ts")
    left=pd.DataFrame({"decision_ts":x.index,"future_ts":x.index+pd.Timedelta(minutes=5)}).sort_values("future_ts")
    merged=pd.merge_asof(left,right,on="future_ts",direction="nearest",tolerance=pd.Timedelta(minutes=1))
    future_close=merged.set_index("decision_ts")["future_close"].reindex(out.index)
    out["future_return_5m"]=future_close/c-1
    out["target_5m"]=np.where(out["future_return_5m"].notna(),(out["future_return_5m"]>0).astype(float),np.nan)
    return out


def load_symbol(root: Path, sym: str) -> pd.DataFrame:
    parts=[]
    seed_dir=root/SEED_REL/f"symbol={sym}"
    if seed_dir.exists():
        for p in sorted(seed_dir.glob("*.parquet")):
            try:
                z=pd.read_parquet(p)
                if not z.empty: parts.append(_norm(z))
            except Exception: pass
    official_dir=root/INTRADAY_REL
    if official_dir.exists():
        for p in sorted(official_dir.rglob(f"{sym}.parquet")):
            try:
                z=pd.read_parquet(p)
                if not z.empty: parts.append(_norm(z))
            except Exception: pass
    parts=[p for p in parts if p is not None and not p.empty]
    if not parts: return pd.DataFrame()
    q=pd.concat(parts).sort_index()
    return q[~q.index.duplicated(keep="last")]


def load_market(root: Path) -> pd.DataFrame:
    parts=[]
    for p in [root/"raw/2018_2025/NIFTY50-INDEX.parquet",root/"raw/2026/NIFTY50-INDEX.parquet"]:
        if p.exists():
            try:
                z=pd.read_parquet(p)
                if not z.empty: parts.append(_norm(z))
            except Exception: pass
    if not parts: return pd.DataFrame()
    q=pd.concat(parts).sort_index()
    return q[~q.index.duplicated(keep="last")]


def load_fno(root: Path) -> pd.DataFrame:
    p=root/"production_data/nse_official/fno/fno_daily_features_2026.parquet"
    try: return pd.read_parquet(p) if p.exists() else pd.DataFrame()
    except Exception: return pd.DataFrame()


def fno_join_features(feature_frame: pd.DataFrame, fno: pd.DataFrame, sym: str) -> pd.DataFrame:
    cols=["fno_pcr","fno_oi_change","fno_fut_return","fno_fut_oi","fno_basis"]
    for c in cols: feature_frame[c]=np.nan
    feature_frame["fno_available"]=0.0
    if fno is None or fno.empty: return feature_frame
    z=fno.copy(); z.columns=[str(c) for c in z.columns]; lower={c.lower():c for c in z.columns}
    dc=next((lower[k] for k in ("date","trade_date","timestamp") if k in lower),None)
    sc=next((lower[k] for k in ("symbol","ticker","underlying") if k in lower),None)
    if not dc or not sc: return feature_frame
    z["_date"]=pd.to_datetime(z[dc],errors="coerce").dt.normalize()
    z=z[z[sc].astype(str).str.upper().str.replace(".NS","",regex=False).eq(sym.upper())].sort_values("_date")
    if z.empty: return feature_frame
    amap={
        "fno_pcr":("fno_pcr","pcr"),
        "fno_oi_change":("fno_oi_change","oi_change"),
        "fno_fut_return":("fno_fut_return","fut_return"),
        "fno_fut_oi":("fno_fut_oi","fut_oi"),
        "fno_basis":("fno_basis","fut_basis","fno_fut_basis"),
    }
    idx=pd.DatetimeIndex(feature_frame.index)
    if idx.tz is None: idx=idx.tz_localize(IST)
    else: idx=idx.tz_convert(IST)
    left=pd.DataFrame({"row_ts":feature_frame.index,"session_date":idx.normalize().tz_localize(None)}).sort_values("session_date")
    for outc,aliases in amap.items():
        src=next((lower[a] for a in aliases if a in lower),None)
        if not src: continue
        d=z[["_date",src]].copy(); d["value"]=pd.to_numeric(d[src],errors="coerce")
        d=d.dropna(subset=["_date","value"]).drop_duplicates("_date",keep="last").sort_values("_date")
        if d.empty: continue
        m=pd.merge_asof(left,d,left_on="session_date",right_on="_date",direction="backward",allow_exact_matches=False)
        feature_frame[outc]=m.set_index("row_ts")["value"].reindex(feature_frame.index).to_numpy()
    feature_frame["fno_available"]=feature_frame[cols].notna().any(axis=1).astype(float)
    return feature_frame


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--drive-root",default="/content/drive/MyDrive/Indian_Market_Research")
    ap.add_argument("--out",default="/content/drive/MyDrive/Indian_Market_Research/models/all_nse_intraday_5m_multifactor_replacement_2026")
    ap.add_argument("--min-rows-symbol",type=int,default=300)
    ap.add_argument("--max-train-rows-per-symbol",type=int,default=500)
    ap.add_argument("--global-retained-row-cap",type=int,default=750000)
    args=ap.parse_args()

    root=Path(args.drive_root)
    out=Path(args.out)
    from delivery_training_universe import official_nse_symbols
    universe=sorted(set(official_nse_symbols(force_refresh=True)))
    if not universe: raise RuntimeError("Official NSE universe is empty")

    market=load_market(root)
    fno=load_fno(root)
    frames=[]; coverage=[]; retained=0

    for i,sym in enumerate(universe,1):
        raw=load_symbol(root,sym)
        if len(raw)<args.min_rows_symbol:
            coverage.append((sym,len(raw),"EXCLUDED_INSUFFICIENT_HISTORY")); continue
        feat=make_features(raw,market)
        if feat.empty:
            coverage.append((sym,len(raw),"EXCLUDED_FEATURE_BUILD")); continue
        feat=fno_join_features(feat,fno,sym)
        feat=feat.replace([np.inf,-np.inf],np.nan).dropna(subset=["target_5m","future_return_5m"])
        if feat.empty:
            coverage.append((sym,len(raw),"EXCLUDED_NO_LABELS")); continue
        # Keep chronological coverage across the entire history rather than
        # only the most recent samples.
        if len(feat)>args.max_train_rows_per_symbol:
            positions=np.linspace(0,len(feat)-1,args.max_train_rows_per_symbol,dtype=int)
            feat=feat.iloc[np.unique(positions)]
        if retained + len(feat) > args.global_retained_row_cap:
            remaining = args.global_retained_row_cap - retained
            if remaining <= 0:
                coverage.append((sym,len(raw),"EXCLUDED_GLOBAL_ROW_CAP"))
                continue
            positions=np.linspace(0,len(feat)-1,remaining,dtype=int)
            feat=feat.iloc[np.unique(positions)]
        feat["_symbol"]=sym
        feat["decision_ts"]=feat.index
        frames.append(feat)
        retained += len(feat)
        coverage.append((sym,len(raw),"TRAINABLE"))
        del raw,feat
        if i%100==0:
            import gc; gc.collect()
            print(f"prepared {i}/{len(universe)} symbols; retained_rows={retained}",flush=True)

    if not frames: raise RuntimeError("No trainable ALL-NSE 5-minute symbols found")
    data=pd.concat(frames,ignore_index=False).sort_values("decision_ts")
    del frames
    train_cut=data["decision_ts"].quantile(0.80)
    train=data[data["decision_ts"]<train_cut].copy()
    test=data[data["decision_ts"]>=train_cut].copy()
    ytr=train["target_5m"].astype(int); yte=test["target_5m"].astype(int)
    if ytr.nunique()!=2 or yte.nunique()!=2: raise RuntimeError("Chronological split missing one target class")

    clf=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("model",LogisticRegression(max_iter=3000,class_weight="balanced",random_state=20261004)),
    ])
    reg=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("model",Ridge(alpha=1.0)),
    ])
    clf.fit(train[FEATURES],ytr)
    reg.fit(train[FEATURES],train["future_return_5m"].astype(float))
    p=clf.predict_proba(test[FEATURES])[:,1]
    pred=(p>=0.5).astype(int)
    pred_ret=reg.predict(test[FEATURES])
    base_price=test["Close"].astype(float).to_numpy()
    pred_price=base_price*(1.0+pred_ret)
    actual_price=base_price*(1.0+test["future_return_5m"].astype(float).to_numpy())
    baseline_price_mae=float(mean_absolute_error(actual_price,base_price))
    model_price_mae=float(mean_absolute_error(actual_price,pred_price))
    direction_hit_rate=float(accuracy_score(yte,pred))
    manifest={
        "status":"CANDIDATE",
        "model_family":"logistic_regression+ridge",
        "target":"target_5m",
        "feature_count":len(FEATURES),
        "features":FEATURES,
        "universe_size":int(len(universe)),
        "training_symbol_count":int(data["_symbol"].nunique()),
        "training_symbols":sorted(data["_symbol"].unique().tolist()),
        "training_rows":int(len(train)),
        "holdout_rows":int(len(test)),
        "minimum_rows_per_symbol":int(args.min_rows_symbol),
        "max_train_rows_per_symbol":int(args.max_train_rows_per_symbol),
        "global_retained_row_cap":int(args.global_retained_row_cap),
        "trained_until":str(train["decision_ts"].max()),
        "holdout_start":str(test["decision_ts"].min()),
        "target_definition":"future close at decision timestamp + 5 minutes; nearest observation within 1 minute",
        "validation":{
            "accuracy":direction_hit_rate,
            "roc_auc":float(roc_auc_score(yte,p)),
            "brier":float(brier_score_loss(yte,p)),
            "return_mae":float(mean_absolute_error(test["future_return_5m"],pred_ret)),
            "price_mae":model_price_mae,
            "baseline_current_price_mae":baseline_price_mae,
            "price_mae_improvement_pct":(
                float(100.0*(baseline_price_mae-model_price_mae)/baseline_price_mae)
                if baseline_price_mae > 0 else None
            ),
            "model_beats_current_price_baseline":bool(model_price_mae < baseline_price_mae),
            "direction_hit_rate":direction_hit_rate,
        },
        "no_auto_orders":True,
        "point_in_time_fno":True,
        "volume_policy":"Source-reported volume only; missing volume stays NaN.",
    }
    out.mkdir(parents=True,exist_ok=True)
    cp=out/"deployment_all_nse_intraday_5m_multifactor_classifier_2026.joblib"
    rp=out/"deployment_all_nse_intraday_5m_price_regressor_2026.joblib"
    joblib.dump(clf,cp); joblib.dump(reg,rp)
    manifest["classifier_sha256"]=hashlib.sha256(cp.read_bytes()).hexdigest()
    manifest["regressor_sha256"]=hashlib.sha256(rp.read_bytes()).hexdigest()
    # Training creates a candidate only. Explicit verification/promotion must
    # happen after reviewing the holdout metrics; the existing 30-minute
    # production model remains untouched in the meantime.
    manifest["status"]="CANDIDATE"
    manifest["production_ready"]=False
    (out/"ALL_NSE_INTRADAY_5M_MULTIFACTOR_2026_MODEL_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    pd.DataFrame(coverage,columns=["symbol","stored_rows","status"]).to_csv(out/"INTRADAY_5M_TRAINING_COVERAGE.csv",index=False)
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
