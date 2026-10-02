#!/usr/bin/env python3
"""Build ALL-NSE 59-feature multi-factor Delivery candidates in Colab."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from delivery_training_universe import benchmark_for_symbol, training_roster
from nse_data_layer import fetch_index_history
from multifactor_data_layer_v2 import FEATURES_59, attach_readiness_flags, load_optional_layer, merge_pit_layer, normalize_symbol, readiness_report

DEFAULT_BULK="/content/drive/MyDrive/Indian_Market_Research/production_data/nse_official/bulk_eod"
DEFAULT_OUT="/content/drive/MyDrive/Indian_Market_Research/models/all_nse_delivery_multifactor_candidate_2026"
BASE=Path("/content/drive/MyDrive/Indian_Market_Research")

def sha256(path: Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def load_bulk(root: Path)->pd.DataFrame:
    files=sorted(root.rglob("*.parquet"))+sorted(root.rglob("*.csv"))
    if not files: raise FileNotFoundError(f"No EOD bulk files under {root}")
    parts=[]
    for p in files:
        try: x=pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_csv(p)
        except Exception: continue
        if x.empty: continue
        ren={}
        for c in x.columns:
            lc=str(c).strip().lower()
            mp={"ticker":"symbol","tckrsymb":"symbol","trade_date":"date","trad_dt":"date",
                "open_price":"open","opnpric":"open","high_price":"high","hghpric":"high",
                "low_price":"low","lwpric":"low","close_price":"close","clspric":"close",
                "traded_volume":"volume","ttltradgvol":"volume","dly_pct":"delivery_pct",
                "dly_qty":"delivery_qty"}
            if lc in ("symbol","date","open","high","low","close","volume","delivery_pct","delivery_qty"): ren[c]=lc
            elif lc in mp: ren[c]=mp[lc]
        x=x.rename(columns=ren)
        req={"symbol","date","open","high","low","close","volume"}
        if not req.issubset(x.columns): continue
        x["symbol"]=x["symbol"].map(normalize_symbol)
        x["date"]=pd.to_datetime(x["date"],errors="coerce").dt.normalize()
        for c in ("open","high","low","close","volume","delivery_pct","delivery_qty"):
            if c in x.columns: x[c]=pd.to_numeric(x[c],errors="coerce")
        keep=list(req|set(["delivery_pct","delivery_qty"])&set(x.columns))
        parts.append(x[keep])
    if not parts: raise RuntimeError("No readable EOD bulk dataset")
    return pd.concat(parts,ignore_index=True).dropna(subset=["symbol","date","open","high","low","close"]).sort_values(["symbol","date"]).drop_duplicates(["symbol","date"],keep="last").reset_index(drop=True)

def rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0).rolling(n,min_periods=n).mean(); l=(-d.clip(upper=0)).rolling(n,min_periods=n).mean()
    return (100-100/(1+g/l.replace(0,np.nan))).where(l.ne(0),100.0)

def add_features(x):
    z=x.sort_values("date").copy(); c=z["close"]; v=z["volume"]
    for n in (1,3,6,12,24,48): z[f"ret_{n}"]=c.pct_change(n)
    z["rsi14"]=rsi(c,14)
    prev=c.shift(1); tr=pd.concat([(z["high"]-z["low"]),(z["high"]-prev).abs(),(z["low"]-prev).abs()],axis=1).max(axis=1)
    z["atr14"]=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    ma20=c.rolling(20).mean(); ma50=c.rolling(50).mean(); vm=v.rolling(20).mean(); vs=v.rolling(20).std()
    z["ma20_ratio"]=c/ma20-1; z["ma50_ratio"]=c/ma50-1; z["volume_ratio"]=v/vm; z["volume_z20"]=(v-vm)/vs
    z["range_pct"]=(z["high"]-z["low"])/c; z["body_pct"]=(c-z["open"])/z["open"]; z["up_bar"]=(c>z["open"]).astype(float)
    for c0 in ("delivery_pct","delivery_qty"):
        if c0 not in z.columns: z[c0]=np.nan
    dp=z["delivery_pct"]; dq=z["delivery_qty"]
    z["delivery_pct_change_1"]=dp.pct_change(fill_method=None); z["delivery_pct_ratio20"]=dp/dp.rolling(20).mean()
    z["delivery_pct_z20"]=(dp-dp.rolling(20).mean())/dp.rolling(20).std()
    z["delivery_qty_ratio20"]=dq/dq.rolling(20).mean(); z["delivery_qty_z20"]=(dq-dq.rolling(20).mean())/dq.rolling(20).std()
    return z

def _official_market_context(index_symbol: str, start_date, end_date) -> pd.DataFrame:
    """Load official NSE benchmark history from Drive first, API second."""
    frame = fetch_index_history(
        index_symbol,
        pd.Timestamp(start_date).date(),
        pd.Timestamp(end_date).date(),
        sync_drive=True,
    )
    if frame is None or frame.empty:
        raise RuntimeError(
            f"Official NSE benchmark history unavailable for {index_symbol}: "
            f"{start_date}..{end_date}"
        )

    x = frame.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x = (
        x.dropna(subset=["date", "close"])
         .sort_values("date")
         .drop_duplicates("date", keep="last")
         .reset_index(drop=True)
    )

    c0 = x["close"]
    d = c0.diff()
    gain = d.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-d.clip(upper=0)).rolling(14, min_periods=14).mean()
    rsi14 = (
        100.0 - 100.0 / (1.0 + gain / loss.replace(0, np.nan))
    ).where(loss.ne(0), 100.0)

    out = pd.DataFrame({"date": x["date"]})
    for n in (1, 3, 6, 12):
        out[f"market_ret_{n}"] = c0.pct_change(n)
    out["market_rsi14"] = rsi14
    out["market_mood_proxy"] = out["market_ret_1"]
    out["market_stress_proxy"] = c0.pct_change().rolling(20, min_periods=20).std()
    return out


def market_context_tables(bulk) -> dict[str, pd.DataFrame]:
    start_date = pd.Timestamp(bulk["date"].min()).normalize()
    end_date = pd.Timestamp(bulk["date"].max()).normalize()
    return {
        "^NSEI": _official_market_context("^NSEI", start_date, end_date),
        "^NSEBANK": _official_market_context("^NSEBANK", start_date, end_date),
    }


def make_panel(bulk, symbols):
    ctxs = market_context_tables(bulk)
    parts = []
    for sym in symbols:
        z = add_features(bulk[bulk["symbol"].eq(sym)].copy())
        benchmark = benchmark_for_symbol(sym)
        ctx = ctxs.get(benchmark)
        if ctx is None or ctx.empty:
            raise RuntimeError(f"Missing official NSE benchmark context for {sym}: {benchmark}")
        z = z.merge(ctx, on="date", how="left")
        for n in (1, 6, 12):
            z[f"relative_ret_{n}"] = z[f"ret_{n}"] - z[f"market_ret_{n}"]
        z["decision_timestamp"] = z["date"] + pd.Timedelta(hours=15, minutes=30)
        parts.append(z)
    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "decision_timestamp"]).reset_index(drop=True)

def merge_optional(panel):
    out=panel.copy(); audit={}
    sources=[
      ("news",BASE/"processed/pit_replay/NEWS_ENRICHED_PIT_REPLAY_2026.parquet",["news_available"]),
      ("events",BASE/"processed/pit_replay/EVENTS_FUNDAMENTALS_PIT_REPLAY_2026.parquet",["event_available","event_high_impact"]),
      ("fno",BASE/"processed/pit_replay/FNO_PIT_REPLAY_2026.parquet",["fno_pcr","fno_oi_change","fno_iv_change","fno_available","fno_fut_close","fno_fut_return","fno_fut_oi"]),
    ]
    for kind,path,cols in sources:
        layer=load_optional_layer(path)
        if layer.empty: audit[kind]={"exists":False,"rows":0}; continue
        out=merge_pit_layer(out,layer,cols,kind)
        audit[kind]={"exists":True,"rows":int(len(layer)),"symbols":int(layer["symbol"].nunique())}
    fp=BASE/"corporate/fundamentals_v2/PIT_FUNDAMENTALS_STRICT_2025_2026.parquet"
    if fp.exists():
        f=pd.read_parquet(fp); f["symbol"]=f["symbol"].map(normalize_symbol); f["fundamental_available_at"]=pd.to_datetime(f["fundamental_available_at"],errors="coerce",utc=True)
        f=f.dropna(subset=["symbol","fundamental_available_at"]).sort_values(["symbol","fundamental_available_at"])
        vals=[c for c in FEATURES_59 if c.startswith("fundamental_") or c.startswith("fund_")]
        pieces=[]
        for sym,lp in out.groupby("symbol",sort=False):
            rp=f[f["symbol"].eq(sym)]
            if rp.empty:
                pieces.append(lp)
                continue
            cols=["fundamental_available_at"]+[c for c in vals if c in rp.columns]
            m=pd.merge_asof(
                lp.sort_values("decision_timestamp"),
                rp[cols].sort_values("fundamental_available_at"),
                left_on="decision_timestamp",
                right_on="fundamental_available_at",
                direction="backward",
                allow_exact_matches=True,
            )
            pieces.append(m)
        if pieces: out=pd.concat(pieces,ignore_index=True)
        audit["fundamentals"]={"exists":True,"rows":int(len(f)),"symbols":int(f["symbol"].nunique())}
    else:
        audit["fundamentals"]={"exists":False,"rows":0}
    return attach_readiness_flags(out),audit

def add_targets(panel,steps,name):
    x=panel.sort_values(["symbol","decision_timestamp"]).copy()
    x[f"target_return_{name}"]=x.groupby("symbol")["close"].shift(-steps)/x["close"]-1.0
    x[f"target_up_{name}"]=(x[f"target_return_{name}"]>0).astype(int)
    return x.dropna(subset=[f"target_return_{name}"])

def fit_models(frame,horizon):
    cutoff=frame["decision_timestamp"].max()-pd.Timedelta(days=60); tr=frame[frame["decision_timestamp"]<cutoff]; te=frame[frame["decision_timestamp"]>=cutoff]
    if tr.empty or te.empty or tr[f"target_up_{horizon}"].nunique()<2: raise RuntimeError(f"{horizon}: insufficient chronological train/holdout data")
    Xtr=tr[FEATURES_59].apply(pd.to_numeric,errors="coerce"); Xte=te[FEATURES_59].apply(pd.to_numeric,errors="coerce")
    clf=Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),("scale",StandardScaler(with_mean=False)),("model",LogisticRegression(C=.5,class_weight="balanced",solver="lbfgs",max_iter=2500,random_state=42))])
    reg=Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),("model",HistGradientBoostingRegressor(max_iter=180,learning_rate=.04,max_leaf_nodes=19,l2_regularization=2.0,random_state=42))])
    clf.fit(Xtr,tr[f"target_up_{horizon}"].astype(int)); reg.fit(Xtr,tr[f"target_return_{horizon}"].astype(float))
    p=clf.predict_proba(Xte)[:,1]; pr=reg.predict(Xte); y=te[f"target_up_{horizon}"].to_numpy()
    return clf,reg,{"train_rows":int(len(tr)),"holdout_rows":int(len(te)),"holdout_accuracy":float(np.mean((p>=.5)==y)),"mean_abs_return_error":float(np.mean(np.abs(pr-te[f"target_return_{horizon}"].to_numpy()))),"holdout_mean_readiness_pct":float(te["data_readiness_score"].mean()*100)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bulk-root",default=DEFAULT_BULK); ap.add_argument("--out",default=DEFAULT_OUT); ap.add_argument("--min-history",type=int,default=60); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); bulk=load_bulk(Path(args.bulk_root))
    roster=training_roster(bulk,min_history=args.min_history)
    trainable=list(roster["trainable_symbols"])
    universe={"universe_source":roster.get("universe_source"),"universe_as_of":roster.get("universe_as_of"),"universe_symbol_count":int(roster["universe_symbol_count"]),"symbols_with_bulk_history":int(roster["symbols_with_bulk_history"]),"trainable_symbol_count":int(roster["trainable_symbol_count"]),"excluded_symbol_count":int(roster["excluded_symbol_count"]),"minimum_history_sessions":int(roster["minimum_history_sessions"])}
    panel,audit=merge_optional(make_panel(bulk,trainable))
    (out/"DATA_READINESS_REPORT_2026.json").write_text(json.dumps({"universe":universe,"readiness":readiness_report(panel),"optional_sources":audit},indent=2,default=str),encoding="utf-8")
    panel.to_parquet(out/"ALL_NSE_MULTIFACTOR_59_FEATURE_PIT_PANEL_2026.parquet",index=False)
    artifacts={}
    for horizon,steps in (("next_session",1),("3d",3),("5d",5),("7d",7)):
        try:
            d=add_targets(panel,steps,horizon); clf,reg,metrics=fit_models(d,horizon)
            cp=out/f"deployment_all_nse_{horizon}_multifactor59_classifier_2026.joblib"; rp=out/f"deployment_all_nse_{horizon}_multifactor59_regressor_2026.joblib"
            joblib.dump(clf,cp); joblib.dump(reg,rp)
            artifacts[horizon]={"status":"CANDIDATE_READY","classifier":str(cp),"regressor":str(rp),"classifier_sha256":sha256(cp),"regressor_sha256":sha256(rp),"metrics":metrics}
        except Exception as exc: artifacts[horizon]={"status":"BLOCKED","reason":f"{type(exc).__name__}: {exc}"}
    manifest={"engine_version":"2026-all-nse-delivery-multifactor-59-candidate","promotion_status":"CANDIDATE_ONLY","universe":universe,"feature_count":59,"features":FEATURES_59,"artifacts":artifacts,"data_readiness_path":str(out/"DATA_READINESS_REPORT_2026.json"),"pit_policy":"Only observations at or before decision_timestamp are eligible. Missing information is not fabricated.","no_auto_orders":True}
    (out/"ALL_NSE_MULTIFACTOR_59_MODEL_MANIFEST_2026.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
    print(json.dumps(manifest,indent=2,default=str))

if __name__=="__main__": main()
