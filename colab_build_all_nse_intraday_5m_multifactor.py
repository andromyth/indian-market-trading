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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from all_nse_intraday_5m_model import FEATURES, HORIZONS

IST = "Asia/Kolkata"
INTRADAY_REL = Path("production_data/nse_official/intraday_1m")
SEED_REL = Path("production_data/nse_official/intraday_seed/olo_minute_archive/symbols")
CHECKPOINT_SCHEMA_VERSION = 3


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

    for n in (1,3,5,10,15,30):
        out[f"ret_{n}"]=c.pct_change(n)
    out["rsi7"]=_rsi(c,7)
    out["rsi14"]=_rsi(c,14)

    pc=c.shift(1)
    tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr14=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    out["atr14_pct"]=atr14/c

    for span in (8,21):
        ema=c.ewm(span=span,adjust=False,min_periods=span).mean()
        out[f"ema{span}_ratio"]=c/ema-1
    for n in (5,10,20,50):
        out[f"ma{n}_ratio"]=c/c.rolling(n,min_periods=n).mean()-1

    one_min_ret=c.pct_change()
    out["volatility_5"]=one_min_ret.rolling(5,min_periods=5).std()
    out["volatility_15"]=one_min_ret.rolling(15,min_periods=15).std()

    session_key=pd.Series(x.index.normalize(),index=x.index)
    session_open=o.groupby(session_key).transform("first")
    session_high=h.groupby(session_key).cummax()
    session_low=l.groupby(session_key).cummin()
    out["session_return"]=c/session_open-1
    out["day_high_ratio"]=c/session_high-1
    out["day_low_ratio"]=c/session_low-1

    pv=((h+l+c)/3.0)*v
    cum_pv=pv.groupby(session_key).cumsum()
    cum_v=v.groupby(session_key).cumsum()
    session_vwap=cum_pv/cum_v
    out["vwap_ratio"]=c/session_vwap-1

    out["range_pct"]=(h-l)/c
    out["body_pct"]=(c-o)/o
    out["volume_ratio"]=v/v.rolling(20,min_periods=20).mean()
    out["volume_z20"]=(v-v.rolling(20,min_periods=20).mean())/v.rolling(20,min_periods=20).std()
    out["volume_trend"]=v.rolling(5,min_periods=5).mean()/v.rolling(20,min_periods=20).mean()-1
    out["ret_accel_1_5"]=out["ret_1"]-(out["ret_5"]/5.0)

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
        market_ret_1=mc.pct_change()
        out["market_volatility_15"]=market_ret_1.rolling(15,min_periods=15).std()
        market_ma10=mc.rolling(10,min_periods=10).mean()
        out["market_ma10_ratio"]=mc/market_ma10-1
    else:
        for n in (1,3,5,10,15,30): out[f"market_ret_{n}"]=np.nan
        out["market_rsi14"]=np.nan
        out["market_volatility_15"]=np.nan
        out["market_ma10_ratio"]=np.nan

    out["relative_ret_1"]=out["ret_1"]-out["market_ret_1"]
    out["relative_ret_5"]=out["ret_5"]-out["market_ret_5"]
    out["relative_ret_15"]=out["ret_15"]-out["market_ret_15"]
    out["relative_ret_30"]=out["ret_30"]-out["market_ret_30"]
    out["log_price"]=np.log1p(c.clip(lower=0))
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
        # Preserve the decision-time/reference price for price-baseline and
        # predicted-price validation. It is not a model feature.
        z["reference_close"]=c.to_numpy()
        z["future_close"]=future.to_numpy()
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

    # Drive-backed per-symbol checkpoints make the long ALL-NSE preparation
    # resumable. A failure after symbol N only requires reprocessing the
    # current symbol, not rebuilding the entire prepared dataset.
    checkpoint_base = out / "training_checkpoints"
    signature_payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "trainer": "all_nse_intraday_5m_multifactor",
        "horizons": list(HORIZONS),
        "features": list(FEATURES),
        "universe_sha256": hashlib.sha256(
            json.dumps(universe, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest(),
        "min_rows_symbol": int(args.min_rows_symbol),
        "max_train_rows_per_symbol": int(args.max_train_rows_per_symbol),
        "global_retained_row_cap": int(args.global_retained_row_cap),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    checkpoint_dir = checkpoint_base / f"run_{signature}"
    symbols_dir = checkpoint_dir / "symbols"
    symbols_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / "progress.json"

    def atomic_write_json(path: Path, obj: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        tmp.replace(path)

    state = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "signature": signature,
        "signature_payload": signature_payload,
        "completed_symbols": [],
        "coverage": [],
    }
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                loaded.get("signature") == signature
                and loaded.get("checkpoint_schema_version") == CHECKPOINT_SCHEMA_VERSION
            ):
                state.update(loaded)
                print(
                    f"RESUME_CHECKPOINT: {len(state.get('completed_symbols', []))}/{len(universe)} "
                    f"symbols already checkpointed in {checkpoint_dir}",
                    flush=True,
                )
            else:
                print("CHECKPOINT_SIGNATURE_MISMATCH: starting a new isolated checkpoint run.", flush=True)
        except Exception as exc:
            print(f"CHECKPOINT_READ_WARNING: {exc}; starting from available symbol checkpoints.", flush=True)

    completed = set(state.get("completed_symbols", []))
    coverage_map = {
        str(row[0]): [row[1], row[2]]
        for row in state.get("coverage", [])
        if isinstance(row, list) and len(row) >= 3
    }

    # Allocate the global training budget across the full live NSE universe
    # and all six horizons so early symbols cannot consume the whole pool.
    per_symbol_horizon_cap = max(
        25,
        min(
            int(args.max_train_rows_per_symbol),
            int(args.global_retained_row_cap // max(1, len(universe) * len(HORIZONS))),
        ),
    )

    # First pass: create/load a Drive checkpoint for every symbol. Trainable
    # symbol tables are stored individually as parquet, so a mid-run crash
    # cannot invalidate the work already prepared.
    for i,sym in enumerate(universe,1):
        checkpoint_path = symbols_dir / f"{sym}.parquet"
        known_status = coverage_map.get(sym, [None, None])[1]
        already = sym in completed and (known_status != "TRAINABLE" or checkpoint_path.exists())
        if already:
            continue

        raw=load_symbol(root,sym)
        stored_rows=len(raw)
        if stored_rows < args.min_rows_symbol:
            status="EXCLUDED_INSUFFICIENT_HISTORY"
            coverage_map[sym]=[stored_rows,status]
            completed.add(sym)
            state["completed_symbols"]=sorted(completed, key=lambda x: universe.index(x))
            state["coverage"]=[[k,v[0],v[1]] for k,v in coverage_map.items()]
            atomic_write_json(state_path,state)
            del raw
            continue

        table=build_label_table(raw,market,fno,sym)
        if table.empty:
            status="EXCLUDED_NO_LABELS"
            coverage_map[sym]=[stored_rows,status]
            completed.add(sym)
            state["completed_symbols"]=sorted(completed, key=lambda x: universe.index(x))
            state["coverage"]=[[k,v[0],v[1]] for k,v in coverage_map.items()]
            atomic_write_json(state_path,state)
            del raw,table
            continue

        # Bound each symbol's contribution separately for every horizon.
        pieces=[]
        for h,z in table.groupby("horizon"):
            if len(z)>per_symbol_horizon_cap:
                pos=np.linspace(0,len(z)-1,per_symbol_horizon_cap,dtype=int)
                z=z.iloc[np.unique(pos)]
            pieces.append(z)
        table=pd.concat(pieces,axis=0)

        # Persist the prepared table BEFORE marking the symbol complete.
        # This ordering guarantees a crash cannot leave a false-complete
        # checkpoint without its data.
        tmp_checkpoint = checkpoint_path.with_suffix(".parquet.tmp")
        table.to_parquet(tmp_checkpoint,index=False)
        tmp_checkpoint.replace(checkpoint_path)

        coverage_map[sym]=[stored_rows,"TRAINABLE"]
        completed.add(sym)
        state["completed_symbols"]=sorted(completed, key=lambda x: universe.index(x))
        state["coverage"]=[[k,v[0],v[1]] for k,v in coverage_map.items()]
        state["last_completed_symbol"]=sym
        state["last_completed_index"]=i
        atomic_write_json(state_path,state)

        del raw,table
        if i%25==0:
            import gc; gc.collect()
            trainable_done=sum(1 for v in coverage_map.values() if v[1]=="TRAINABLE")
            print(
                f"checkpointed {i}/{len(universe)} symbols; "
                f"trainable_checkpoints={trainable_done}",
                flush=True,
            )

    # Reassemble the prepared table from Drive checkpoints in deterministic
    # universe order. Apply the global retained-row cap here so a resumed run
    # produces the same bounded training pool as a clean run.
    frames=[]; coverage=[]; retained=0
    for sym in universe:
        stored_rows,status = coverage_map.get(sym, [0,"EXCLUDED_NOT_PROCESSED"])
        if status != "TRAINABLE":
            coverage.append((sym,int(stored_rows or 0),status))
            continue
        checkpoint_path = symbols_dir / f"{sym}.parquet"
        if not checkpoint_path.exists():
            raise RuntimeError(f"TRAINABLE checkpoint missing for {sym}: {checkpoint_path}")
        table=pd.read_parquet(checkpoint_path)
        if table.empty:
            coverage.append((sym,int(stored_rows or 0),"EXCLUDED_EMPTY_CHECKPOINT"))
            continue
        if retained + len(table) > args.global_retained_row_cap:
            allowed=max(0,args.global_retained_row_cap-retained)
            if allowed<=0:
                coverage.append((sym,int(stored_rows or 0),"EXCLUDED_GLOBAL_ROW_CAP")); continue
            pos=np.linspace(0,len(table)-1,allowed,dtype=int)
            table=table.iloc[np.unique(pos)]
        frames.append(table)
        retained += len(table)
        coverage.append((sym,int(stored_rows or 0),"TRAINABLE"))
        del table

    if not frames: raise RuntimeError("No trainable 5-minute-step symbols found")
    data=pd.concat(frames,ignore_index=False).sort_values("decision_ts")
    del frames

    required_training_columns = list(FEATURES) + [
        "reference_close", "future_close", "future_return",
        "target", "horizon", "symbol", "decision_ts",
    ]
    missing_training_columns = [
        col for col in required_training_columns if col not in data.columns
    ]
    if missing_training_columns:
        raise RuntimeError(
            "Prepared training table is missing required columns: "
            + ", ".join(missing_training_columns)
        )
    if data["reference_close"].isna().all() or data["future_close"].isna().all():
        raise RuntimeError(
            "Prepared training table has no usable reference/future prices. "
            "Refusing to train a price-regression model."
        )
    print(
        f"prepared_training_table: rows={len(data):,} "
        f"symbols={data['symbol'].nunique():,} "
        f"horizons={sorted(data['horizon'].dropna().astype(int).unique().tolist())}",
        flush=True,
    )

    bundle={}; validation={}; training_rows_by_horizon={}; trainable_symbols=sorted(data["symbol"].unique().tolist())
    for h in HORIZONS:
        z=data[data["horizon"]==h].copy().sort_values("decision_ts")
        cut_fit=z["decision_ts"].quantile(0.70)
        cut_cal=z["decision_ts"].quantile(0.85)
        fit=z[z["decision_ts"]<cut_fit].copy()
        cal=z[(z["decision_ts"]>=cut_fit)&(z["decision_ts"]<cut_cal)].copy()
        test=z[z["decision_ts"]>=cut_cal].copy()

        if len(fit)<500 or len(cal)<200 or len(test)<200:
            raise RuntimeError(
                f"Insufficient chronological split for +{h}m: "
                f"fit={len(fit)} calibration={len(cal)} holdout={len(test)}"
            )
        for name,frame in (("fit",fit),("cal",cal),("test",test)):
            if frame["target"].nunique()!=2:
                raise RuntimeError(f"+{h}m {name} split missing one target class")

        Xfit=fit[FEATURES]
        Xcal=cal[FEATURES]
        Xtest=test[FEATURES]
        yfit=fit["target"].astype(int)
        ycal=cal["target"].astype(int)
        ytest=test["target"].astype(int)

        # Nonlinear tree models capture threshold/interactions that the first
        # linear candidate could not represent. Logistic/Ridge are retained as
        # stable ensemble members rather than discarded.
        clf_hgb=Pipeline([
            ("imputer",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),
            ("model",HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=20261004+h,
            )),
        ])
        clf_log=Pipeline([
            ("imputer",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),
            ("scale",StandardScaler()),
            ("model",LogisticRegression(max_iter=3000,class_weight="balanced",random_state=20261004+h)),
        ])
        reg_hgb=Pipeline([
            ("imputer",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),
            ("model",HistGradientBoostingRegressor(
                loss="absolute_error",
                max_iter=180,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=50,
                l2_regularization=1.0,
                random_state=20261004+h,
            )),
        ])
        reg_ridge=Pipeline([
            ("imputer",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),
            ("scale",StandardScaler()),
            ("model",Ridge(alpha=2.0)),
        ])

        clf_hgb.fit(Xfit,yfit)
        clf_log.fit(Xfit,yfit)

        # Weight the tree regressor by reference-price level so absolute
        # price-error performance is represented during fitting.
        price_weights=fit["reference_close"].astype(float).to_numpy()
        price_weights=price_weights/np.nanmedian(price_weights)
        price_weights=np.clip(price_weights,0.25,4.0)
        reg_hgb.fit(Xfit,fit["future_return"].astype(float),model__sample_weight=price_weights)
        reg_ridge.fit(Xfit,fit["future_return"].astype(float))

        p_hgb_cal=clf_hgb.predict_proba(Xcal)[:,1]
        p_log_cal=clf_log.predict_proba(Xcal)[:,1]
        best_classifier_weight=0.0
        best_brier=float("inf")
        for w in np.linspace(0.0,1.0,9):
            p=w*p_hgb_cal+(1.0-w)*p_log_cal
            b=float(brier_score_loss(ycal,p))
            if b<best_brier:
                best_brier=b
                best_classifier_weight=float(w)

        p_cal=best_classifier_weight*p_hgb_cal+(1.0-best_classifier_weight)*p_log_cal
        r_hgb_cal=reg_hgb.predict(Xcal)
        r_ridge_cal=reg_ridge.predict(Xcal)
        abs_move=float(np.nanmedian(np.abs(fit["future_return"].astype(float).to_numpy())))
        if not np.isfinite(abs_move) or abs_move<=0:
            abs_move=float(np.nanmean(np.abs(fit["future_return"].astype(float).to_numpy())))
        if not np.isfinite(abs_move) or abs_move<=0:
            abs_move=0.001
        direction_ret=(2.0*p_cal-1.0)*abs_move

        best=None
        best_mae=float("inf")
        cal_base=cal["reference_close"].astype(float).to_numpy()
        cal_actual=cal["future_close"].astype(float).to_numpy()
        for rw in np.linspace(0.0,1.0,5):
            raw_ret=rw*r_hgb_cal+(1.0-rw)*r_ridge_cal
            for alpha in (0.0,0.25,0.5,0.75,1.0):
                for beta in (0.0,0.5,1.0):
                    pred_ret=alpha*raw_ret+beta*direction_ret
                    pred_price=cal_base*(1.0+pred_ret)
                    mae=float(mean_absolute_error(cal_actual,pred_price))
                    if mae<best_mae:
                        best_mae=mae
                        best=(float(rw),float(alpha),float(beta))

        reg_weight_hgb,return_alpha,return_beta=best

        p_hgb=clf_hgb.predict_proba(Xtest)[:,1]
        p_log=clf_log.predict_proba(Xtest)[:,1]
        p=best_classifier_weight*p_hgb+(1.0-best_classifier_weight)*p_log
        pred_hgb=reg_hgb.predict(Xtest)
        pred_ridge=reg_ridge.predict(Xtest)
        raw_ret=reg_weight_hgb*pred_hgb+(1.0-reg_weight_hgb)*pred_ridge
        expected_ret=return_alpha*raw_ret+return_beta*((2.0*p-1.0)*abs_move)

        pred=(p>=0.5).astype(int)
        base_price=test["reference_close"].astype(float).to_numpy()
        pred_price=base_price*(1.0+expected_ret)
        actual_price=test["future_close"].astype(float).to_numpy()
        model_mae=float(mean_absolute_error(actual_price,pred_price))
        base_mae=float(mean_absolute_error(actual_price,base_price))
        auc=float(roc_auc_score(ytest,p))
        acc=float(accuracy_score(ytest,pred))
        validation[str(h)]={
            "accuracy":acc,
            "roc_auc":auc,
            "brier":float(brier_score_loss(ytest,p)),
            "return_mae":float(mean_absolute_error(test["future_return"],expected_ret)),
            "price_mae":model_mae,
            "baseline_current_price_mae":base_mae,
            "price_mae_improvement_pct":float(100*(base_mae-model_mae)/base_mae) if base_mae>0 else None,
            "model_beats_current_price_baseline":bool(model_mae<base_mae),
            "calibration_price_mae":float(best_mae),
            "calibration_baseline_price_mae":float(mean_absolute_error(cal_actual,cal_base)),
            "classifier_hgb_weight":float(best_classifier_weight),
            "regressor_hgb_weight":float(reg_weight_hgb),
            "return_alpha":float(return_alpha),
            "return_beta_direction":float(return_beta),
            "direction_move_scale":float(abs_move),
            "fit_rows":int(len(fit)),
            "calibration_rows":int(len(cal)),
            "train_rows":int(len(fit)),
            "holdout_rows":int(len(test)),
            "trained_until":str(fit["decision_ts"].max()),
            "calibration_end":str(cal["decision_ts"].max()),
            "holdout_start":str(test["decision_ts"].min()),
        }
        training_rows_by_horizon[str(h)] = int(len(fit))
        bundle[str(h)]={
            "classifier_hgb":clf_hgb,
            "classifier_logistic":clf_log,
            "regressor_hgb":reg_hgb,
            "regressor_ridge":reg_ridge,
            "blend":{
                "classifier_hgb_weight":float(best_classifier_weight),
                "regressor_hgb_weight":float(reg_weight_hgb),
                "return_alpha":float(return_alpha),
                "return_beta_direction":float(return_beta),
                "direction_move_scale":float(abs_move),
            },
        }
        print(
            f"+{h}m AUC={auc:.4f} acc={acc:.4f} "
            f"price_MAE={model_mae:.6f} baseline_MAE={base_mae:.6f} "
            f"cal_MAE={best_mae:.6f}",
            flush=True,
        )


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
        "model_family":"hgb_classifier+logistic_ensemble+hgb_absolute_error_ridge_regressor_per_horizon_v2",
        "horizons_minutes":list(HORIZONS),
        "target_definitions":{str(h):f"future close at decision timestamp + {h} minutes; nearest observation within 1 minute; reference price is decision-time close" for h in HORIZONS},
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
        "model_version":"2026-5m-step-multihorizon-v2",
        "validation_protocol":"70% chronological fit / 15% calibration / 15% final holdout per horizon.",
        "promotion_rule":"Every horizon must beat the current-price price-MAE baseline on the final untouched chronological holdout and exceed 0.50 ROC-AUC and 0.50 direction accuracy; calibration data is used only for ensemble/blend selection.",
    }
    (out/"ALL_NSE_INTRADAY_5M_MULTIHORIZON_2026_MODEL_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    pd.DataFrame(coverage,columns=["symbol","stored_rows","status"]).to_csv(out/"INTRADAY_5M_TRAINING_COVERAGE.csv",index=False)
    atomic_write_json(
        checkpoint_dir/"ASSEMBLY_STATE.json",
        {
            "status":"TRAINING_COMPLETED",
            "symbols":int(len(universe)),
            "trainable_symbols":int(len(trainable_symbols)),
            "retained_rows":int(len(data)),
            "signature":signature,
            "production_ready":bool(verified),
            "bundle":str(bundle_path),
        },
    )
    state["status"]="TRAINING_COMPLETED"
    state["completed_at"]=pd.Timestamp.now(tz=IST).isoformat()
    atomic_write_json(state_path,state)
    print(f"CHECKPOINTS_RETAINED: {checkpoint_dir}",flush=True)
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
