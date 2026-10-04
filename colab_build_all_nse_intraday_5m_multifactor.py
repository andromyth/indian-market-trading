#!/usr/bin/env python3
"""Train the ALL-NSE 5-minute-step multi-horizon Intraday candidate.

Creates direct models for +5/+10/+15/+20/+25/+30 minutes. Every horizon has
its own classifier and price-return regressor so future prices are not
recursively generated from earlier predictions.

This never touches the certified 30-minute production model.
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

from all_nse_intraday_5m_model import FEATURES, HORIZONS

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
    x=x.rename(columns={c:aliases.get(str(c).lower(),c) for c in x.columns})
    if "Timestamp" in x.columns:
        idx=pd.to_datetime(x.pop("Timestamp"),errors="coerce")
    else:
        idx=pd.to_datetime(x.index,errors="coerce")
    idx=pd.DatetimeIndex(idx)
    if idx.tz is None: idx=idx.tz_localize(IST)
    else: idx=idx.tz_convert(IST)
    x.index=idx
    for c in ("Open","High","Low","Close","Volume"):
        if c not in x: x[c]=np.nan
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x[~x.index.isna()].sort_index()
    return x[~x.index.duplicated(keep="last")]


def _rsi(c: pd.Series, n:int=14) -> pd.Series:
    d=c.diff()
    up=d.clip(lower=0).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))


def make_features(g: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    x=_norm(g)
    if x.empty: return pd.DataFrame()
    c,h,l,o,v=x["Close"],x["High"],x["Low"],x["Open"],x["Volume"]
    out=pd.DataFrame(index=x.index)
    for n in (1,3,5,10,15,30): out[f"ret_{n}"]=c.pct_change(n)
    out["rsi14"]=_rsi(c)
    pc=c.shift(1)
    tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    out["atr14_pct"]=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/c
    for n in (5,10,20,50): out[f"ma{n}_ratio"]=c/c.rolling(n,min_periods=n).mean()-1
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
        for n in (1,3,5,10,15,30): out[f"market_ret_{n}"]=mc.pct_change(n,fill_method=None)
        out["market_rsi14"]=_rsi(mc)
    else:
        for n in (1,3,5,10,15,30): out[f"market_ret_{n}"]=np.nan
        out["market_rsi14"]=np.nan
    out["relative_ret_1"]=out["ret_1"]-out["market_ret_1"]
    out["relative_ret_5"]=out["ret_5"]-out["market_ret_5"]
    out["relative_ret_15"]=out["ret_15"]-out["market_ret_15"]
    return out


def load_symbol(root:Path,sym:str)->pd.DataFrame:
    parts=[]
    seed=root/SEED_REL/f"symbol={sym}"
    if seed.exists():
        for p in sorted(seed.glob("*.parquet")):
            try:
                z=pd.read_parquet(p)
                if not z.empty: parts.append(_norm(z))
            except Exception: pass
    official=root/INTRADAY_REL
    if official.exists():
        for p in sorted(official.rglob(f"{sym}.parquet")):
            try:
                z=pd.read_parquet(p)
                if not z.empty: parts.append(_norm(z))
            except Exception: pass
    parts=[x for x in parts if x is not None and not x.empty]
    if not parts: return pd.DataFrame()
    q=pd.concat(parts).sort_index()
    return q[~q.index.duplicated(keep="last")]


def load_market(root:Path)->pd.DataFrame:
    parts=[]
    for p in (root/"raw/2018_2025/NIFTY50-INDEX.parquet",root/"raw/2026/NIFTY50-INDEX.parquet"):
        if p.exists():
            try:
                z=pd.read_parquet(p)
                if not z.empty: parts.append(_norm(z))
            except Exception: pass
    if not parts: return pd.DataFrame()
    q=pd.concat(parts).sort_index()
    return q[~q.index.duplicated(keep="last")]


def load_fno(root:Path)->pd.DataFrame:
    p=root/"production_data/nse_official/fno/fno_daily_features_2026.parquet"
    try: return pd.read_parquet(p) if p.exists() else pd.DataFrame()
    except Exception: return pd.DataFrame()


def join_fno(x:pd.DataFrame,fno:pd.DataFrame,sym:str)->pd.DataFrame:
    cols=["fno_pcr","fno_oi_change","fno_fut_return","fno_fut_oi","fno_basis"]
    for c in cols: x[c]=np.nan
    x["fno_available"]=0.0
    if fno is None or fno.empty: return x
    z=fno.copy(); z.columns=[str(c) for c in z.columns]; lower={c.lower():c for c in z.columns}
    dc=next((lower[k] for k in ("date","trade_date","timestamp") if k in lower),None)
    sc=next((lower[k] for k in ("symbol","ticker","underlying") if k in lower),None)
    if not dc or not sc: return x
    z["_date"]=pd.to_datetime(z[dc],errors="coerce").dt.normalize()
    z=z[z[sc].astype(str).str.upper().str.replace(".NS","",regex=False).eq(sym.upper())].sort_values("_date")
    if z.empty: return x
    amap={
        "fno_pcr":("fno_pcr","pcr"),
        "fno_oi_change":("fno_oi_change","oi_change"),
        "fno_fut_return":("fno_fut_return","fut_return"),
        "fno_fut_oi":("fno_fut_oi","fut_oi"),
        "fno_basis":("fno_basis","fut_basis","fno_fut_basis"),
    }
    idx=pd.DatetimeIndex(x.index)
    if idx.tz is None: idx=idx.tz_localize(IST)
    else: idx=idx.tz_convert(IST)
    left=pd.DataFrame({"row_ts":x.index,"session_date":idx.normalize().tz_localize(None)}).sort_values("session_date")
    for outc,aliases in amap.items():
        src=next((lower[a] for a in aliases if a in lower),None)
        if not src: continue
        d=z[["_date",src]].copy()
        d["value"]=pd.to_numeric(d[src],errors="coerce")
        d=d.dropna(subset=["_date","value"]).drop_duplicates("_date",keep="last").sort_values("_date")
        if d.empty: continue
        m=pd.merge_asof(left,d,left_on="session_date",right_on="_date",direction="backward",allow_exact_matches=False)
        x[outc]=m.set_index("row_ts")["value"].reindex(x.index).to_numpy()
    x["fno_available"]=x[cols].notna().any(axis=1).astype(float)
    return x


def build_label_table(raw:pd.DataFrame,market:pd.DataFrame,fno:pd.DataFrame,sym:str)->pd.DataFrame:
    feat=make_features(raw,market)
    if feat.empty: return feat
    feat=join_fno(feat,fno,sym)
    c=_norm(raw)["Close"]
    right=pd.DataFrame({"future_ts":c.index,"future_close":c.to_numpy()}).sort_values("future_ts")
    out=[]
    for h in HORIZONS:
        left=pd.DataFrame({"decision_ts":feat.index,"future_ts":feat.index+pd.Timedelta(minutes=h)}).sort_values("future_ts")
        merged=pd.merge_asof(left,right,on="future_ts",direction="nearest",tolerance=pd.Timedelta(minutes=1))
        future=merged.set_index("decision_ts")["future_close"].reindex(feat.index)
        z=feat.copy()
        z["future_return"]=future/c-1
        z["target"] = np.where(z["future_return"].notna(),(z["future_return"]>0).astype(float),np.nan)
        z["horizon"]=h
        z["symbol"]=sym
        z["decision_ts"]=z.index
        z=z.dropna(subset=["future_return","target"])
        out.append(z)
    return pd.concat(out,axis=0) if out else pd.DataFrame()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--drive-root",default="/content/drive/MyDrive/Indian_Market_Research")
    ap.add_argument("--out",default="/content/drive/MyDrive/Indian_Market_Research/models/all_nse_intraday_5m_multifactor_replacement_2026")
    ap.add_argument("--min-rows-symbol",type=int,default=300)
    ap.add_argument("--max-train-rows-per-symbol",type=int,default=500)
    ap.add_argument("--global-retained-row-cap",type=int,default=750000)
    args=ap.parse_args()
    root=Path(args.drive_root); out=Path(args.out)
    from delivery_training_universe import official_nse_symbols
    universe=sorted(set(official_nse_symbols(force_refresh=True)))
    if not universe: raise RuntimeError("Official NSE universe is empty")
    market=load_market(root); fno=load_fno(root)
    frames=[]; coverage=[]; retained=0
    for i,sym in enumerate(universe,1):
        raw=load_symbol(root,sym)
        if len(raw)<args.min_rows_symbol:
            coverage.append((sym,len(raw),"EXCLUDED_INSUFFICIENT_HISTORY")); continue
        table=build_label_table(raw,market,fno,sym)
        if table.empty:
            coverage.append((sym,len(raw),"EXCLUDED_NO_LABELS")); continue
        if retained + len(table) > args.global_retained_row_cap:
            allowed=max(0,args.global_retained_row_cap-retained)
            if allowed<=0:
                coverage.append((sym,len(raw),"EXCLUDED_GLOBAL_ROW_CAP")); continue
            positions=np.linspace(0,len(table)-1,allowed,dtype=int)
            table=table.iloc[np.unique(positions)]
        # cap per symbol per horizon to avoid overweighting liquid names
        pieces=[]
        for h,z in table.groupby("horizon"):
            if len(z)>args.max_train_rows_per_symbol:
                pos=np.linspace(0,len(z)-1,args.max_train_rows_per_symbol,dtype=int)
                z=z.iloc[np.unique(pos)]
            pieces.append(z)
        table=pd.concat(pieces,axis=0)
        frames.append(table)
        retained += len(table)
        coverage.append((sym,len(raw),"TRAINABLE"))
        del raw,table
        if i%100==0:
            import gc; gc.collect()
            print(f"prepared {i}/{len(universe)} symbols; retained_rows={retained}",flush=True)
    if not frames: raise RuntimeError("No trainable 5-minute-step symbols found")
    data=pd.concat(frames,ignore_index=False).sort_values("decision_ts")
    del frames

    bundle={}; validation={}; training_rows_by_horizon={}; trainable_symbols=sorted(data["symbol"].unique().tolist())
    for h in HORIZONS:
        z=data[data["horizon"]==h].copy()
        cut=z["decision_ts"].quantile(0.80)
        train=z[z["decision_ts"]<cut].copy()
        test=z[z["decision_ts"]>=cut].copy()
        if len(train)<500 or len(test)<200:
            raise RuntimeError(f"Insufficient split for +{h}m: train={len(train)} test={len(test)}")
        ytr=train["target"].astype(int); yte=test["target"].astype(int)
        if ytr.nunique()!=2 or yte.nunique()!=2:
            raise RuntimeError(f"Target split missing class for +{h}m")
        clf=Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("model",LogisticRegression(max_iter=3000,class_weight="balanced",random_state=20261004+h)),
        ])
        reg=Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("model",Ridge(alpha=1.0)),
        ])
        clf.fit(train[FEATURES],ytr)
        reg.fit(train[FEATURES],train["future_return"].astype(float))
        p=clf.predict_proba(test[FEATURES])[:,1]
        pred=(p>=0.5).astype(int)
        pred_ret=reg.predict(test[FEATURES])
        base_price=test["Close"].astype(float).to_numpy()
        pred_price=base_price*(1.0+pred_ret)
        actual_price=base_price*(1.0+test["future_return"].astype(float).to_numpy())
        model_mae=float(mean_absolute_error(actual_price,pred_price))
        base_mae=float(mean_absolute_error(actual_price,base_price))
        auc=float(roc_auc_score(yte,p))
        acc=float(accuracy_score(yte,pred))
        validation[str(h)]={
            "accuracy":acc,
            "roc_auc":auc,
            "brier":float(brier_score_loss(yte,p)),
            "return_mae":float(mean_absolute_error(test["future_return"],pred_ret)),
            "price_mae":model_mae,
            "baseline_current_price_mae":base_mae,
            "price_mae_improvement_pct":float(100*(base_mae-model_mae)/base_mae) if base_mae>0 else None,
            "model_beats_current_price_baseline":bool(model_mae<base_mae),
            "train_rows":int(len(train)),
            "holdout_rows":int(len(test)),
            "trained_until":str(train["decision_ts"].max()),
            "holdout_start":str(test["decision_ts"].min()),
        }
        training_rows_by_horizon[str(h)] = int(len(train))
        bundle[str(h)]={"classifier":clf,"regressor":reg}
        print(f"+{h}m AUC={auc:.4f} acc={acc:.4f} price_MAE={model_mae:.6f} baseline_MAE={base_mae:.6f}",flush=True)

    verified = all(
        validation[str(h)]["model_beats_current_price_baseline"]
        and validation[str(h)]["roc_auc"] > 0.50
        and validation[str(h)]["accuracy"] > 0.50
        for h in HORIZONS
    )

    out.mkdir(parents=True,exist_ok=True)
    bundle_path=out/"deployment_all_nse_intraday_5m_multihorizon_bundle_2026.joblib"
    joblib.dump(bundle,bundle_path)
    sha=hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    manifest={
        "status":"VERIFIED" if verified else "CANDIDATE",
        "production_ready":bool(verified),
        "model_family":"logistic_regression+ridge_per_horizon",
        "horizons_minutes":list(HORIZONS),
        "target_definitions":{str(h):f"future close at decision timestamp + {h} minutes; nearest observation within 1 minute" for h in HORIZONS},
        "feature_count":len(FEATURES),
        "features":FEATURES,
        "universe_size":int(len(universe)),
        "training_symbol_count":int(len(trainable_symbols)),
        "training_symbols":trainable_symbols,
        "training_rows_by_horizon":training_rows_by_horizon,
        "validation":validation,
        "verified_horizons":[int(h) for h in HORIZONS] if verified else [],
        "trained_until":max(validation[str(h)]["trained_until"] for h in HORIZONS),
        "bundle_sha256":sha,
        "no_auto_orders":True,
        "point_in_time_fno":True,
        "volume_policy":"Source-reported volume only; missing volume stays NaN.",
        "promotion_rule":"Every horizon must beat current-price price-MAE baseline and exceed 0.50 ROC-AUC and 0.50 direction accuracy.",
    }
    (out/"ALL_NSE_INTRADAY_5M_MULTIHORIZON_2026_MODEL_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    pd.DataFrame(coverage,columns=["symbol","stored_rows","status"]).to_csv(out/"INTRADAY_5M_TRAINING_COVERAGE.csv",index=False)
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
