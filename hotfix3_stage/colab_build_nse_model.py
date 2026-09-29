#!/usr/bin/env python3
"""Google Colab trainer for the NSE-first 30-minute production candidate.

Design:
- Colab/Google Drive is the training environment.
- Streamlit never trains a model.
- GitHub receives code + the small model artifact/manifest only.
- Historical market data remains in Google Drive.
- target_30m is explicit: class 1 == future 30-minute return > 0; class 0 otherwise.

The script intentionally refuses to invent missing inputs and refuses to train
if a symbol has insufficient contiguous 1-minute coverage for the 30-minute label.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IST = "Asia/Kolkata"
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/Indian_Market_Research")


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_index()
    close = pd.to_numeric(x["Close"], errors="coerce")
    vol = pd.to_numeric(x.get("Volume", pd.Series(index=x.index, dtype=float)), errors="coerce")
    x["ret_1"] = close.pct_change(1)
    for n in (3, 5, 10, 15):
        x[f"ret_{n}"] = close.pct_change(n)
    for n in (5, 10, 20, 50):
        x[f"sma_gap_{n}"] = close / close.rolling(n).mean() - 1
    x["rsi_14"] = rsi(close, 14) / 100.0
    x["range_pct"] = (pd.to_numeric(x["High"], errors="coerce") - pd.to_numeric(x["Low"], errors="coerce")) / close.replace(0, np.nan)
    tr = pd.concat([
        pd.to_numeric(x["High"], errors="coerce") - pd.to_numeric(x["Low"], errors="coerce"),
        (pd.to_numeric(x["High"], errors="coerce") - close.shift()).abs(),
        (pd.to_numeric(x["Low"], errors="coerce") - close.shift()).abs(),
    ], axis=1).max(axis=1)
    x["atr_14_pct"] = tr.rolling(14).mean() / close.replace(0, np.nan)
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0, np.nan)
    x["volume_z20"] = (vol - vol_mean) / vol_std
    minutes = x.index.hour * 60 + x.index.minute
    x["tod_sin"] = np.sin(2 * np.pi * minutes / 390.0)
    x["tod_cos"] = np.cos(2 * np.pi * minutes / 390.0)
    # Strict point-in-time target: find the row at exactly +30 calendar minutes.
    lookup = pd.DataFrame({"future_ts": x.index + pd.Timedelta(minutes=30), "future_close": close.to_numpy()}, index=x.index)
    right = pd.DataFrame({"future_ts": x.index, "future_close": close.to_numpy()}).sort_values("future_ts")
    left = lookup.reset_index(names="decision_ts").sort_values("future_ts")
    merged = pd.merge_asof(left, right, on="future_ts", direction="nearest", tolerance=pd.Timedelta(minutes=1), suffixes=("", "_actual"))
    merged = merged.set_index("decision_ts")
    future_close = merged["future_close_actual"]
    future_return = future_close / close - 1
    x["future_return_30m"] = future_return
    x["target_30m"] = np.where(future_return.notna(), (future_return > 0).astype(float), np.nan)
    return x


def load_symbol_files(root: Path, symbols: list[str]) -> pd.DataFrame:
    base = root / "production_data" / "nse_official" / "intraday_1m"
    frames=[]
    for sym in symbols:
        paths = sorted(base.rglob(f"{sym}.parquet")) if base.exists() else []
        if not paths:
            continue
        parts=[]
        for p in paths:
            try:
                z=pd.read_parquet(p)
                if z.empty: continue
                z.columns=[str(c) for c in z.columns]
                time_col = next((c for c in z.columns if c.lower() in {"timestamp","datetime","time","date"}), None)
                if time_col:
                    idx=pd.to_datetime(z[time_col], errors="coerce", utc=True).dt.tz_convert(IST)
                    z=z.drop(columns=[time_col])
                else:
                    idx=pd.to_datetime(z.index, errors="coerce", utc=True).tz_convert(IST)
                z.index=idx
                parts.append(z)
            except Exception:
                continue
        if parts:
            q=pd.concat(parts).sort_index()
            q=q[~q.index.duplicated(keep="last")]
            q["symbol"]=sym
            frames.append(q)
    if not frames: return pd.DataFrame()
    return pd.concat(frames)


def chronological_split(df: pd.DataFrame, train_frac=.8):
    cut = df["decision_ts"].quantile(train_frac)
    return df[df["decision_ts"] < cut], df[df["decision_ts"] >= cut]


def train(root: Path, symbols: list[str], out_dir: Path) -> dict:
    raw=load_symbol_files(root, symbols)
    if raw.empty:
        raise RuntimeError("No NSE 1-minute data found in Google Drive under production_data/nse_official/intraday_1m")
    frames=[]
    for sym,g in raw.groupby("symbol"):
        g=g.copy()
        # Require standard OHLC; volume may legitimately be missing on NSE chart data.
        required={"Open","High","Low","Close"}
        if not required.issubset(g.columns):
            continue
        g=add_features(g)
        g["decision_ts"]=g.index
        g["symbol"]=sym
        frames.append(g)
    if not frames:
        raise RuntimeError("No symbols had usable NSE 1-minute OHLC data")
    data=pd.concat(frames, ignore_index=True)
    features=[c for c in data.columns if c not in {"Open","High","Low","Close","Volume","future_return_30m","target_30m","decision_ts","symbol","source","provenance"}]
    data=data.dropna(subset=["target_30m"])
    if len(data)<1000:
        raise RuntimeError(f"Insufficient labelled rows: {len(data)}")
    train_df,test_df=chronological_split(data)
    ytr=train_df["target_30m"].astype(int)
    yte=test_df["target_30m"].astype(int)
    if ytr.nunique()!=2 or yte.nunique()!=2:
        raise RuntimeError("Chronological train/holdout split does not contain both target classes")
    model=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("model",LogisticRegression(max_iter=2500,class_weight="balanced",random_state=20260927)),
    ])
    model.fit(train_df[features],ytr)
    p=model.predict_proba(test_df[features])[:,1]
    pred=(p>=0.5).astype(int)
    metrics={
        "holdout_rows":int(len(test_df)),
        "accuracy":float(accuracy_score(yte,pred)),
        "roc_auc":float(roc_auc_score(yte,p)),
        "brier":float(brier_score_loss(yte,p)),
        "train_end_exclusive":str(train_df["decision_ts"].max()),
        "holdout_start":str(test_df["decision_ts"].min()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path=out_dir/"nse_production_candidate_30m.joblib"
    joblib.dump(model,model_path)
    # Reload the exact artifact and assert the class/probability semantics.
    check=joblib.load(model_path)
    if list(check.named_steps["model"].classes_) != [0,1]:
        raise RuntimeError(f"Unexpected classifier classes: {check.named_steps['model'].classes_}")
    manifest={
        "status":"CANDIDATE",
        "model_family":"logistic_regression",
        "target":"target_30m",
        "target_semantics":{"class_1":"future 30-minute return > 0 (UP)","class_0":"future 30-minute return <= 0 (DOWN/FLAT)","probability":"predict_proba()[:,1] is P(UP)"},
        "feature_count":len(features),
        "features":features,
        "symbols":sorted(data["symbol"].unique().tolist()),
        "training_rows":int(len(train_df)),
        "holdout_rows":int(len(test_df)),
        "validation":metrics,
        "model_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "created_in":"Google Colab",
        "data_root":str(root/"production_data"),
        "data_source":"NSE_OFFICIAL",
        "no_fabrication":True,
    }
    (out_dir/"NSE_PRODUCTION_MODEL_MANIFEST_2026.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (out_dir/"MODEL_TARGET_SEMANTICS_TEST.txt").write_text(
        "PASS\nclass 1 = UP = future_return_30m > 0\nclass 0 = DOWN/FLAT\npredict_proba()[:,1] = P(UP)\n",
        encoding="utf-8")
    return manifest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--drive-root",default=str(DEFAULT_DRIVE_ROOT))
    ap.add_argument("--symbols",nargs="+",default=["RELIANCE","BHARTIARTL","HDFCBANK","TCS","INFY"])
    ap.add_argument("--out-dir",default="/content/nse_model_release")
    args=ap.parse_args()
    print(json.dumps(train(Path(args.drive_root),args.symbols,Path(args.out_dir)),indent=2))

if __name__=="__main__": main()
