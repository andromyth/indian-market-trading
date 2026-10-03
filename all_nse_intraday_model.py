"""Live ALL-NSE Intraday 30-minute model adapter.

The model is pooled across the NSE equity universe. Training artifacts live in
Google Drive; GitHub contains only the adapter. The adapter never places orders.
Missing F&O/context inputs remain NaN and are surfaced through readiness fields.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

FEATURES = [
    "ret_1","ret_3","ret_5","ret_10","ret_15","ret_30",
    "rsi14","atr14_pct","ma5_ratio","ma10_ratio","ma20_ratio","ma50_ratio",
    "range_pct","body_pct","volume_ratio","volume_z20","tod_sin","tod_cos",
    "market_ret_1","market_ret_3","market_ret_5","market_ret_10","market_ret_15","market_ret_30",
    "market_rsi14","relative_ret_1","relative_ret_5","relative_ret_15",
    "fno_pcr","fno_oi_change","fno_fut_return","fno_fut_oi","fno_basis","fno_available",
]

DRIVE_MODEL_ROOT = "models/all_nse_intraday_multifactor_replacement_2026"
REMOTE_MANIFEST = f"{DRIVE_MODEL_ROOT}/ALL_NSE_INTRADAY_MULTIFACTOR_2026_MODEL_MANIFEST.json"
REMOTE_CLASSIFIER = f"{DRIVE_MODEL_ROOT}/deployment_all_nse_intraday_multifactor_classifier_2026.joblib"
REMOTE_REGRESSOR = f"{DRIVE_MODEL_ROOT}/deployment_all_nse_intraday_multifactor_regressor_2026.joblib"

CACHE_ROOT = Path(__file__).resolve().parent / "data" / "all_nse_intraday_replacement"
LOCAL_MANIFEST = CACHE_ROOT / "ALL_NSE_INTRADAY_MULTIFACTOR_2026_MODEL_MANIFEST.json"
LOCAL_CLASSIFIER = CACHE_ROOT / "deployment_all_nse_intraday_multifactor_classifier_2026.joblib"
LOCAL_REGRESSOR = CACHE_ROOT / "deployment_all_nse_intraday_multifactor_regressor_2026.joblib"

BUY_P = 0.60
SELL_P = 0.40


def _safe(v: Any) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _norm(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [str(v).strip() for v in x.columns.get_level_values(-1)]
    else:
        x.columns = [str(c).strip() for c in x.columns]
    aliases = {"open":"Open","high":"High","low":"Low","close":"Close","adj close":"Close","ltp":"Close",
               "volume":"Volume","vol":"Volume","timestamp":"Timestamp","datetime":"Timestamp","date":"Timestamp","time":"Timestamp"}
    x = x.rename(columns={c: aliases.get(str(c).strip().lower(), c) for c in x.columns})
    if "Timestamp" in x.columns:
        idx = pd.to_datetime(x.pop("Timestamp"), errors="coerce")
    else:
        idx = pd.to_datetime(x.index, errors="coerce")
    # The historical seed source provides no timezone field and documents its
    # timestamps as source-local; the project convention treats these as IST.
    # Official NSE live data may already be timezone-aware, so preserve an
    # existing timezone and only localize naive timestamps.
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize("Asia/Kolkata")
    else:
        idx = idx.tz_convert("Asia/Kolkata")
    x.index = idx
    for c in ("Open","High","Low","Close","Volume"):
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x[~x.index.isna()].sort_index()
    return x[~x.index.duplicated(keep="last")]


def _returns(c: pd.Series, n: int) -> float:
    x = pd.to_numeric(c, errors="coerce").dropna()
    if len(x) <= n:
        return np.nan
    return _safe(x.iloc[-1] / x.iloc[-1-n] - 1)


def _rsi(c: pd.Series, n: int=14) -> float:
    x = pd.to_numeric(c, errors="coerce")
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    if len(x) < n or pd.isna(up.iloc[-1]) or pd.isna(dn.iloc[-1]):
        return np.nan
    if dn.iloc[-1] == 0:
        return 100.0
    return _safe(100 - 100/(1 + up.iloc[-1]/dn.iloc[-1]))


def _atr_pct(x: pd.DataFrame, n: int=14) -> float:
    h,l,c = x["High"],x["Low"],x["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean().iloc[-1]
    close = c.iloc[-1]
    return _safe(atr/close) if np.isfinite(close) and close else np.nan


def _market_series(market_df: pd.DataFrame | None) -> pd.Series:
    m = _norm(market_df)
    return pd.to_numeric(m["Close"], errors="coerce") if not m.empty else pd.Series(dtype=float)


def _latest_features(price_df: pd.DataFrame, market_df: pd.DataFrame | None = None,
                     fno: dict[str, Any] | None = None,
                     decision_timestamp: Any = None) -> dict[str, float]:
    x = _norm(price_df)
    if x.empty or x["Close"].dropna().empty:
        raise ValueError("No usable intraday OHLC data")
    c,h,l,o,v = x["Close"],x["High"],x["Low"],x["Open"],x["Volume"]
    close = _safe(c.iloc[-1]); open_ = _safe(o.iloc[-1])
    vals = {
        "ret_1":_returns(c,1),"ret_3":_returns(c,3),"ret_5":_returns(c,5),
        "ret_10":_returns(c,10),"ret_15":_returns(c,15),"ret_30":_returns(c,30),
        "rsi14":_rsi(c),"atr14_pct":_atr_pct(x),
        "ma5_ratio":_safe(close/c.rolling(5,min_periods=5).mean().iloc[-1]-1) if np.isfinite(close) and c.rolling(5,min_periods=5).mean().iloc[-1] else np.nan,
        "ma10_ratio":_safe(close/c.rolling(10,min_periods=10).mean().iloc[-1]-1) if np.isfinite(close) and c.rolling(10,min_periods=10).mean().iloc[-1] else np.nan,
        "ma20_ratio":_safe(close/c.rolling(20,min_periods=20).mean().iloc[-1]-1) if np.isfinite(close) and c.rolling(20,min_periods=20).mean().iloc[-1] else np.nan,
        "ma50_ratio":_safe(close/c.rolling(50,min_periods=50).mean().iloc[-1]-1) if np.isfinite(close) and c.rolling(50,min_periods=50).mean().iloc[-1] else np.nan,
        "range_pct":_safe((h.iloc[-1]-l.iloc[-1])/close) if np.isfinite(close) and close else np.nan,
        "body_pct":_safe((close-open_)/open_) if np.isfinite(close) and np.isfinite(open_) and open_ else np.nan,
    }
    vma = v.rolling(20,min_periods=20).mean().iloc[-1]
    vsd = v.rolling(20,min_periods=20).std().iloc[-1]
    vals["volume_ratio"] = _safe(v.iloc[-1]/vma) if np.isfinite(v.iloc[-1]) and np.isfinite(vma) and vma else np.nan
    vals["volume_z20"] = _safe((v.iloc[-1]-vma)/vsd) if np.isfinite(v.iloc[-1]) and np.isfinite(vma) and np.isfinite(vsd) and vsd else np.nan
    ts = x.index[-1]
    if decision_timestamp is not None:
        q = pd.Timestamp(decision_timestamp)
        if q.tzinfo is None: q=q.tz_localize("Asia/Kolkata")
        ts=q.tz_convert("Asia/Kolkata")
    minutes = ts.hour*60+ts.minute
    vals["tod_sin"]=float(np.sin(2*np.pi*minutes/390.0))
    vals["tod_cos"]=float(np.cos(2*np.pi*minutes/390.0))

    mc = _market_series(market_df)
    vals["market_ret_1"]=_returns(mc,1)
    vals["market_ret_3"]=_returns(mc,3)
    vals["market_ret_5"]=_returns(mc,5)
    vals["market_ret_10"]=_returns(mc,10)
    vals["market_ret_15"]=_returns(mc,15)
    vals["market_ret_30"]=_returns(mc,30)
    vals["market_rsi14"]=_rsi(mc)
    vals["relative_ret_1"] = _safe(vals["ret_1"]-vals["market_ret_1"]) if np.isfinite(vals["ret_1"]) and np.isfinite(vals["market_ret_1"]) else np.nan
    vals["relative_ret_5"] = _safe(vals["ret_5"]-vals["market_ret_5"]) if np.isfinite(vals["ret_5"]) and np.isfinite(vals["market_ret_5"]) else np.nan
    vals["relative_ret_15"] = _safe(vals["ret_15"]-vals["market_ret_15"]) if np.isfinite(vals["ret_15"]) and np.isfinite(vals["market_ret_15"]) else np.nan

    fno = fno or {}
    vals["fno_pcr"]=_safe(fno.get("pcr", fno.get("fno_pcr")))
    vals["fno_oi_change"]=_safe(fno.get("oi_change", fno.get("fno_oi_change")))
    vals["fno_fut_return"]=_safe(fno.get("fut_return", fno.get("fno_fut_return")))
    vals["fno_fut_oi"]=_safe(fno.get("fut_oi", fno.get("fno_fut_oi")))
    vals["fno_basis"]=_safe(fno.get("basis", fno.get("fno_basis")))
    vals["fno_available"]=1.0 if any(pd.notna(vals[k]) for k in ["fno_pcr","fno_oi_change","fno_fut_return","fno_fut_oi","fno_basis"]) else 0.0
    return {k:_safe(vals.get(k)) for k in FEATURES}


def _drive_download(remote: str, local: Path) -> None:
    from persistent_store import GoogleDriveStore
    from googleapiclient.http import MediaIoBaseDownload
    store = GoogleDriveStore()
    parent = store.root_id
    parts = remote.strip("/").split("/")
    for part in parts[:-1]:
        found = store._find_child(parent, part)
        if not found: raise FileNotFoundError(f"Google Drive path component not found: {part}")
        parent = found["id"]
    found = store._find_child(parent, parts[-1])
    if not found: raise FileNotFoundError(f"Google Drive artifact not found: {remote}")
    buf=io.BytesIO(); request=store.service.files().get_media(fileId=found["id"]); dl=MediaIoBaseDownload(buf,request)
    done=False
    while not done: _,done=dl.next_chunk()
    local.parent.mkdir(parents=True,exist_ok=True); local.write_bytes(buf.getvalue())


def _ensure_artifacts() -> None:
    CACHE_ROOT.mkdir(parents=True,exist_ok=True)
    for remote,local in [(REMOTE_MANIFEST,LOCAL_MANIFEST),(REMOTE_CLASSIFIER,LOCAL_CLASSIFIER),(REMOTE_REGRESSOR,LOCAL_REGRESSOR)]:
        if not local.exists() or local.stat().st_size==0:
            _drive_download(remote,local)


def _manifest() -> dict[str,Any]:
    _ensure_artifacts()
    return json.loads(LOCAL_MANIFEST.read_text(encoding="utf-8"))


_MODEL_CACHE = {}


def model_target_contract() -> dict[str, Any]:
    return {
        "target": "target_30m",
        "positive_class": 1,
        "positive_class_meaning": "UP: future 30-minute return > 0",
        "negative_class": 0,
        "negative_class_meaning": "DOWN/FLAT: future 30-minute return <= 0",
        "probability_column": "probability_up_30m",
        "signal_policy": "BUY if p>=0.60, SELL if p<=0.40, otherwise HOLD",
        "feature_count": len(FEATURES),
        "features": FEATURES,
    }


def model_status() -> dict[str,Any]:
    try:
        m=_manifest()
    except Exception as exc:
        return {"status":"MODEL_ARTIFACT_UNAVAILABLE","error_type":type(exc).__name__,"error":str(exc)}
    return {
        "status":m.get("status","REPLACEMENT"),
        "model_family":m.get("model_family"),
        "target":m.get("target"),
        "feature_count":len(FEATURES),
        "training_symbols":m.get("training_symbols",[]),
        "training_symbol_count":m.get("training_symbol_count"),
        "universe_size":m.get("universe_size"),
        "coverage_pct":m.get("coverage_pct"),
        "trained_until":m.get("trained_until"),
        "manifest_sha256_match": all([
            LOCAL_CLASSIFIER.exists() and _sha256(LOCAL_CLASSIFIER)==m.get("classifier_sha256"),
            LOCAL_REGRESSOR.exists() and _sha256(LOCAL_REGRESSOR)==m.get("regressor_sha256"),
        ]),
        "model_path_classifier":str(LOCAL_CLASSIFIER),
        "model_path_regressor":str(LOCAL_REGRESSOR),
    }


def _sha256(path: Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def _models():
    if _MODEL_CACHE: return _MODEL_CACHE["classifier"],_MODEL_CACHE["regressor"]
    s=model_status()
    if not s.get("manifest_sha256_match"): raise RuntimeError("ALL-NSE Intraday model hashes do not match manifest")
    clf=joblib.load(LOCAL_CLASSIFIER); reg=joblib.load(LOCAL_REGRESSOR)
    _MODEL_CACHE["classifier"],_MODEL_CACHE["regressor"]=clf,reg
    return clf,reg


def predict_symbol(symbol: str, price_df: pd.DataFrame, market_df: pd.DataFrame | None=None,
                   fno: dict[str,Any] | None=None, decision_timestamp: Any=None) -> dict[str,Any]:
    clf,reg=_models()
    row=_latest_features(price_df,market_df,fno,decision_timestamp)
    frame=pd.DataFrame([row],columns=FEATURES)
    p=float(np.clip(clf.predict_proba(frame)[:,1][0],0,1))
    expected=_safe(reg.predict(frame)[0])
    signal="BUY" if p>=BUY_P else ("SELL" if p<=SELL_P else "HOLD")
    readiness=float(frame.notna().mean(axis=1).iloc[0])
    return {
        "prediction_status":"PASS",
        "symbol":str(symbol),
        "decision_timestamp":str(pd.Timestamp(decision_timestamp) if decision_timestamp is not None else pd.Timestamp.now(tz="Asia/Kolkata")),
        "probability_up_30m":p,
        "signal":signal,
        "expected_return_30m":expected,
        "confidence":"HIGH" if abs(p-.5)>=.25 else ("MEDIUM" if abs(p-.5)>=.12 else "LOW"),
        "feature_count":len(FEATURES),
        "non_missing_feature_count":int(frame.notna().sum(axis=1).iloc[0]),
        "data_readiness_score":readiness,
        "model_family":"logistic_regression+ridge",
        "target":"target_30m",
        "positive_class":1,
        "positive_class_meaning":"UP: future 30-minute return > 0",
        "model_coverage_symbols":int(model_status().get("training_symbol_count") or 0),
        "data_limited":readiness < 0.70,
    }
