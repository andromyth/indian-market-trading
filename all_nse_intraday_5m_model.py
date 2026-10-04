"""ALL-NSE multi-horizon 5-minute-step Intraday model adapter.

This is a separate 5-minute-step model family and does not replace or modify the
certified 30-minute production artifact.

For every live 1-minute update it can estimate direct prices at +5, +10, +15,
+20, +25 and +30 minutes. Each horizon has its own classifier and price-return
regressor; horizons are not recursively compounded.
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

HORIZONS = (5, 10, 15, 20, 25, 30)
FEATURES = [
    "ret_1","ret_3","ret_5","ret_10","ret_15","ret_30",
    "rsi7","rsi14","atr14_pct","ema8_ratio","ema21_ratio",
    "ma5_ratio","ma10_ratio","ma20_ratio","ma50_ratio",
    "volatility_5","volatility_15","session_return","day_high_ratio","day_low_ratio",
    "vwap_ratio","range_pct","body_pct","volume_ratio","volume_z20","volume_trend",
    "ret_accel_1_5","tod_sin","tod_cos",
    "market_ret_1","market_ret_3","market_ret_5","market_ret_10","market_ret_15","market_ret_30",
    "market_rsi14","market_volatility_15","market_ma10_ratio",
    "relative_ret_1","relative_ret_5","relative_ret_15","relative_ret_30",
    "log_price",
    "fno_pcr","fno_oi_change","fno_fut_return","fno_fut_oi","fno_basis","fno_available",
]

DRIVE_MODEL_ROOT = "models/all_nse_intraday_5m_multifactor_replacement_2026"
REMOTE_MANIFEST = f"{DRIVE_MODEL_ROOT}/ALL_NSE_INTRADAY_5M_MULTIHORIZON_2026_MODEL_MANIFEST.json"
REMOTE_BUNDLE = f"{DRIVE_MODEL_ROOT}/deployment_all_nse_intraday_5m_multihorizon_bundle_2026.joblib"
CACHE_ROOT = Path(__file__).resolve().parent / "data" / "all_nse_intraday_5m_multihorizon_v2"
LOCAL_MANIFEST = CACHE_ROOT / "ALL_NSE_INTRADAY_5M_MULTIHORIZON_2026_MODEL_MANIFEST.json"
LOCAL_BUNDLE = CACHE_ROOT / "deployment_all_nse_intraday_5m_multihorizon_bundle_2026.joblib"

BUY_P = 0.60
SELL_P = 0.40
IST = "Asia/Kolkata"


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
    aliases = {
        "open":"Open","high":"High","low":"Low","close":"Close",
        "adj close":"Close","ltp":"Close","volume":"Volume","vol":"Volume",
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


def _returns(c: pd.Series, n: int) -> float:
    x = pd.to_numeric(c, errors="coerce").dropna()
    if len(x) <= n:
        return np.nan
    return _safe(x.iloc[-1] / x.iloc[-1-n] - 1.0)


def _rsi(c: pd.Series, n: int = 14) -> float:
    x = pd.to_numeric(c, errors="coerce")
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    if len(x) < n or pd.isna(up.iloc[-1]) or pd.isna(dn.iloc[-1]):
        return np.nan
    if dn.iloc[-1] == 0:
        return 100.0
    return _safe(100 - 100/(1 + up.iloc[-1]/dn.iloc[-1]))


def _regular_session(x: pd.DataFrame) -> pd.DataFrame:
    """Keep regular NSE cash-market session bars only."""
    if x is None or x.empty:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(x.index)
    mask = (
        (idx.time >= pd.Timestamp("09:15").time())
        & (idx.time < pd.Timestamp("15:30").time())
    )
    return x.loc[mask].copy()


def _feature_lookback_ok(index: pd.DatetimeIndex, window: int = 50) -> bool:
    """Require 1-minute continuity inside the recent feature window."""
    idx = pd.DatetimeIndex(index)
    if len(idx) < window:
        return False
    recent = idx[-window:]
    diffs = pd.Series(recent).diff()
    day_change = pd.Series(recent.normalize()).diff().ne(pd.Timedelta(0))
    link_ok = diffs.eq(pd.Timedelta(minutes=1)) | day_change
    link_ok.iloc[0] = False
    return bool(link_ok.iloc[1:].all())


def _atr_pct(x: pd.DataFrame, n: int = 14) -> float:
    h, l, c = x["High"], x["Low"], x["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean().iloc[-1]
    close = c.iloc[-1]
    return _safe(atr/close) if np.isfinite(close) and close else np.nan


def _market_series(market_df: pd.DataFrame | None) -> pd.Series:
    m = _norm(market_df)
    return pd.to_numeric(m["Close"], errors="coerce") if not m.empty else pd.Series(dtype=float)


def _latest_features(price_df: pd.DataFrame, market_df: pd.DataFrame | None = None,
                     fno: dict[str, Any] | None = None,
                     decision_timestamp: Any = None) -> dict[str, float]:
    x = _regular_session(_norm(price_df))
    if x.empty or x["Close"].dropna().empty:
        raise ValueError("No usable regular-session NSE intraday OHLC data")
    if not _feature_lookback_ok(x.index, window=50):
        raise ValueError(
            "5-minute prediction blocked: recent 1-minute NSE data contains "
            "an intraday timestamp gap or fewer than 50 contiguous observations"
        )

    c,h,l,o,v=x["Close"],x["High"],x["Low"],x["Open"],x["Volume"]
    close=_safe(c.iloc[-1]); open_=_safe(o.iloc[-1])
    vals: dict[str,float]={}

    for n in (1,3,5,10,15,30):
        vals[f"ret_{n}"]=_returns(c,n)
    vals["rsi7"]=_rsi(c,7)
    vals["rsi14"]=_rsi(c,14)
    vals["atr14_pct"]=_atr_pct(x)

    for span in (8,21):
        ema=c.ewm(span=span,adjust=False,min_periods=span).mean().iloc[-1]
        vals[f"ema{span}_ratio"]=_safe(close/ema-1) if np.isfinite(close) and np.isfinite(ema) and ema else np.nan
    for n in (5,10,20,50):
        ma=c.rolling(n,min_periods=n).mean().iloc[-1]
        vals[f"ma{n}_ratio"]=_safe(close/ma-1) if np.isfinite(close) and np.isfinite(ma) and ma else np.nan

    one_min_ret=c.pct_change(fill_method=None)
    vals["volatility_5"]=_safe(one_min_ret.rolling(5,min_periods=5).std().iloc[-1])
    vals["volatility_15"]=_safe(one_min_ret.rolling(15,min_periods=15).std().iloc[-1])

    session_key=pd.Series(x.index.normalize(),index=x.index)
    session_open=o.groupby(session_key).transform("first").iloc[-1]
    session_high=h.groupby(session_key).cummax().iloc[-1]
    session_low=l.groupby(session_key).cummin().iloc[-1]
    vals["session_return"]=_safe(close/session_open-1) if np.isfinite(close) and np.isfinite(session_open) and session_open else np.nan
    vals["day_high_ratio"]=_safe(close/session_high-1) if np.isfinite(close) and np.isfinite(session_high) and session_high else np.nan
    vals["day_low_ratio"]=_safe(close/session_low-1) if np.isfinite(close) and np.isfinite(session_low) and session_low else np.nan

    pv=((h+l+c)/3.0)*v
    cum_pv=pv.groupby(session_key).cumsum().iloc[-1]
    cum_v=v.groupby(session_key).cumsum().iloc[-1]
    if np.isfinite(cum_pv) and np.isfinite(cum_v) and cum_v:
        vals["vwap_ratio"]=_safe(close/(cum_pv/cum_v)-1)
    else:
        vals["vwap_ratio"]=np.nan

    vm20=v.rolling(20,min_periods=20).mean().iloc[-1]
    vs20=v.rolling(20,min_periods=20).std().iloc[-1]
    vm5=v.rolling(5,min_periods=5).mean().iloc[-1]
    vals["range_pct"]=_safe((h.iloc[-1]-l.iloc[-1])/close) if np.isfinite(close) and close else np.nan
    vals["body_pct"]=_safe((close-open_)/open_) if np.isfinite(close) and np.isfinite(open_) and open_ else np.nan
    vals["volume_ratio"]=_safe(v.iloc[-1]/vm20) if np.isfinite(v.iloc[-1]) and np.isfinite(vm20) and vm20 else np.nan
    vals["volume_z20"]=_safe((v.iloc[-1]-vm20)/vs20) if np.isfinite(v.iloc[-1]) and np.isfinite(vm20) and np.isfinite(vs20) and vs20 else np.nan
    vals["volume_trend"]=_safe(vm5/vm20-1) if np.isfinite(vm5) and np.isfinite(vm20) and vm20 else np.nan
    vals["ret_accel_1_5"]=_safe(vals["ret_1"]-vals["ret_5"]/5.0) if np.isfinite(vals["ret_1"]) and np.isfinite(vals["ret_5"]) else np.nan

    ts=pd.Timestamp(decision_timestamp) if decision_timestamp is not None else x.index[-1]
    if ts.tzinfo is None: ts=ts.tz_localize(IST)
    else: ts=ts.tz_convert(IST)
    session_minutes=max(0,min(389,ts.hour*60+ts.minute-(9*60+15)))
    vals["tod_sin"]=float(np.sin(2*np.pi*session_minutes/390.0))
    vals["tod_cos"]=float(np.cos(2*np.pi*session_minutes/390.0))

    mc=_market_series(market_df)
    if len(mc):
        mc=mc[mc.index<=ts]
        aligned_mc=mc.reindex(x.index,method="ffill",tolerance=pd.Timedelta(minutes=2))
    else:
        aligned_mc=pd.Series(dtype=float,index=x.index)
    for n in (1,3,5,10,15,30):
        vals[f"market_ret_{n}"]=_returns(aligned_mc,n)
    vals["market_rsi14"]=_rsi(aligned_mc)
    market_ret_1=aligned_mc.pct_change(fill_method=None)
    vals["market_volatility_15"]=_safe(market_ret_1.rolling(15,min_periods=15).std().iloc[-1]) if len(aligned_mc) else np.nan
    market_ma10=aligned_mc.rolling(10,min_periods=10).mean().iloc[-1] if len(aligned_mc) else np.nan
    vals["market_ma10_ratio"]=_safe(aligned_mc.iloc[-1]/market_ma10-1) if len(aligned_mc) and np.isfinite(aligned_mc.iloc[-1]) and np.isfinite(market_ma10) and market_ma10 else np.nan
    vals["relative_ret_1"]=_safe(vals["ret_1"]-vals["market_ret_1"]) if np.isfinite(vals["ret_1"]) and np.isfinite(vals["market_ret_1"]) else np.nan
    vals["relative_ret_5"]=_safe(vals["ret_5"]-vals["market_ret_5"]) if np.isfinite(vals["ret_5"]) and np.isfinite(vals["market_ret_5"]) else np.nan
    vals["relative_ret_15"]=_safe(vals["ret_15"]-vals["market_ret_15"]) if np.isfinite(vals["ret_15"]) and np.isfinite(vals["market_ret_15"]) else np.nan
    vals["relative_ret_30"]=_safe(vals["ret_30"]-vals["market_ret_30"]) if np.isfinite(vals["ret_30"]) and np.isfinite(vals["market_ret_30"]) else np.nan
    vals["log_price"]=_safe(np.log1p(max(close,0.0))) if np.isfinite(close) else np.nan

    fo=fno or {}
    vals["fno_pcr"]=_safe(fo.get("pcr",fo.get("fno_pcr")))
    vals["fno_oi_change"]=_safe(fo.get("oi_change",fo.get("fno_oi_change")))
    vals["fno_fut_return"]=_safe(fo.get("fut_return",fo.get("fno_fut_return")))
    vals["fno_fut_oi"]=_safe(fo.get("fut_oi",fo.get("fno_fut_oi")))
    vals["fno_basis"]=_safe(fo.get("basis",fo.get("fno_basis")))
    vals["fno_available"]=1.0 if any(pd.notna(vals[k]) for k in ("fno_pcr","fno_oi_change","fno_fut_return","fno_fut_oi","fno_basis")) else 0.0
    return {k:_safe(vals.get(k)) for k in FEATURES}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b):
            h.update(b)
    return h.hexdigest()


def _drive_download(remote: str, local: Path) -> None:
    from persistent_store import GoogleDriveStore
    from googleapiclient.http import MediaIoBaseDownload
    store = GoogleDriveStore()
    parent = store.root_id
    parts = remote.strip("/").split("/")
    for part in parts[:-1]:
        found = store._find_child(parent, part)
        if not found:
            raise FileNotFoundError(f"Google Drive path component not found: {part}")
        parent = found["id"]
    found = store._find_child(parent, parts[-1])
    if not found:
        raise FileNotFoundError(f"Google Drive artifact not found: {remote}")
    buf = io.BytesIO()
    request = store.service.files().get_media(fileId=found["id"])
    dl = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = dl.next_chunk()
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(buf.getvalue())


REMOTE_FNO = "production_data/nse_official/fno/fno_daily_features_2026.parquet"
LOCAL_FNO = CACHE_ROOT / "fno_daily_features_2026.parquet"
_FNO_DAILY: pd.DataFrame | None = None


def _load_pit_fno(symbol: str, decision_ts: pd.Timestamp) -> dict[str, float]:
    """Use the previous-session daily F&O snapshot, matching training semantics."""
    global _FNO_DAILY
    if _FNO_DAILY is None:
        try:
            if not LOCAL_FNO.exists() or LOCAL_FNO.stat().st_size == 0:
                _drive_download(REMOTE_FNO, LOCAL_FNO)
            z = pd.read_parquet(LOCAL_FNO)
            z.columns = [str(col).strip() for col in z.columns]
            lower = {str(col).strip().lower(): col for col in z.columns}
            dc = next((lower[k] for k in ("date","trade_date","timestamp") if k in lower), None)
            sc = next((lower[k] for k in ("symbol","ticker","underlying") if k in lower), None)
            if dc is None or sc is None:
                _FNO_DAILY = pd.DataFrame()
            else:
                z["_date"] = pd.to_datetime(z[dc], errors="coerce").dt.normalize()
                z["_symbol"] = z[sc].astype(str).str.upper().str.replace(".NS","",regex=False)
                _FNO_DAILY = z.dropna(subset=["_date"]).copy()
        except Exception:
            _FNO_DAILY = pd.DataFrame()

    z = _FNO_DAILY
    if z is None or z.empty:
        return {}
    root = str(symbol).upper().replace(".NS","").replace(".BO","")
    cutoff = pd.Timestamp(decision_ts).tz_convert(IST).normalize().tz_localize(None)
    z = z[(z["_symbol"] == root) & (z["_date"] < cutoff)].sort_values("_date")
    if z.empty:
        return {}
    row = z.iloc[-1]
    lower = {str(col).strip().lower(): col for col in z.columns}
    aliases = {
        "pcr": ("fno_pcr","pcr"),
        "oi_change": ("fno_oi_change","oi_change"),
        "fut_return": ("fno_fut_return","fut_return"),
        "fut_oi": ("fno_fut_oi","fut_oi"),
        "basis": ("fno_basis","fut_basis","fno_fut_basis"),
    }
    out = {}
    for target, names in aliases.items():
        col = next((lower[n] for n in names if n in lower), None)
        if col is not None:
            out[target] = _safe(row.get(col))
    return out


def _ensure_artifact() -> dict[str, Any]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for remote, local in ((REMOTE_MANIFEST, LOCAL_MANIFEST), (REMOTE_BUNDLE, LOCAL_BUNDLE)):
        if not local.exists() or local.stat().st_size == 0:
            _drive_download(remote, local)
    return json.loads(LOCAL_MANIFEST.read_text(encoding="utf-8"))


_BUNDLE: dict[str, Any] | None = None
_MANIFEST: dict[str, Any] | None = None


def model_status() -> dict[str, Any]:
    try:
        manifest = _ensure_artifact()
        bundle_hash_ok = LOCAL_BUNDLE.exists() and _sha256(LOCAL_BUNDLE) == manifest.get("bundle_sha256")
        verified_horizons = manifest.get("verified_horizons") or []
        manifest_features=list(manifest.get("features") or [])
        contract_ok=(int(manifest.get("feature_count",0) or 0)==len(FEATURES) and manifest_features==FEATURES)
        return {
            "status": "READY" if manifest.get("status") == "VERIFIED" and bundle_hash_ok and contract_ok and set(verified_horizons) == set(HORIZONS) else "UNVERIFIED",
            "target_horizons": list(HORIZONS),
            "feature_count": len(FEATURES),
            "feature_contract_ok": bool(contract_ok),
            "training_symbol_count": int(manifest.get("training_symbol_count", 0) or 0),
            "universe_size": int(manifest.get("universe_size", 0) or 0),
            "training_rows_by_horizon": manifest.get("training_rows_by_horizon", {}),
            "validation": manifest.get("validation", {}),
            "bundle_sha256_match": bool(bundle_hash_ok),
            "trained_until": manifest.get("trained_until"),
        }
    except Exception as exc:
        return {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}


def _bundle_models() -> dict[str, Any]:
    global _BUNDLE, _MANIFEST
    if _BUNDLE is None:
        manifest = _ensure_artifact()
        status = model_status()
        if status.get("status") != "READY":
            raise RuntimeError("Verified 5-minute-step Intraday model bundle is not available")
        _BUNDLE = joblib.load(LOCAL_BUNDLE)
        _MANIFEST = manifest
    return _BUNDLE


def predict_multihorizon(symbol: str, price_df: pd.DataFrame, market_df: pd.DataFrame | None = None,
                         fno: dict[str, Any] | None = None, decision_timestamp: Any = None) -> dict[str, Any]:
    ts = pd.Timestamp(decision_timestamp) if decision_timestamp is not None else pd.Timestamp.now(tz=IST)
    if ts.tzinfo is None: ts = ts.tz_localize(IST)
    else: ts = ts.tz_convert(IST)
    last_regular_target = pd.Timestamp("15:25").time()
    if ts.time() > last_regular_target:
        raise RuntimeError("5-minute-step prediction unavailable after 15:25 IST")

    bundle = _bundle_models()
    pit_fno = _load_pit_fno(symbol, ts)
    row = _latest_features(price_df, market_df, pit_fno, ts)
    frame = pd.DataFrame([row], columns=FEATURES)
    current_price = _safe(pd.to_numeric(price_df["Close"], errors="coerce").dropna().iloc[-1])
    predictions=[]

    for horizon in HORIZONS:
        target_ts=ts+pd.Timedelta(minutes=int(horizon))
        if target_ts.time()>pd.Timestamp("15:30").time():
            continue
        h=str(horizon)
        model_set=bundle[h]

        if "classifier_hgb" in model_set:
            blend=model_set.get("blend",{})
            clf_w=float(blend.get("classifier_hgb_weight",0.5))
            reg_w=float(blend.get("regressor_hgb_weight",0.5))
            alpha=float(blend.get("return_alpha",1.0))
            beta=float(blend.get("return_beta_direction",0.0))
            p_hgb=float(model_set["classifier_hgb"].predict_proba(frame)[:,1][0])
            p_log=float(model_set["classifier_logistic"].predict_proba(frame)[:,1][0])
            p=float(np.clip(clf_w*p_hgb+(1.0-clf_w)*p_log,0.0,1.0))
            r_hgb=_safe(model_set["regressor_hgb"].predict(frame)[0])
            r_ridge=_safe(model_set["regressor_ridge"].predict(frame)[0])
            raw_return=reg_w*r_hgb+(1.0-reg_w)*r_ridge
            up_move=_safe(blend.get("direction_up_return",0.001))
            down_move=_safe(blend.get("direction_down_return",-0.001))
            if np.isfinite(up_move) and np.isfinite(down_move):
                direction_return=p*up_move+(1.0-p)*down_move
            else:
                direction_return=(2.0*p-1.0)*0.001
            expected_return=_safe(alpha*raw_return+beta*direction_return)
            model_family="hgb+logistic / hgb(abs)+ridge calibrated ensemble"
        else:
            clf=model_set["classifier"]
            reg=model_set["regressor"]
            p=float(np.clip(clf.predict_proba(frame)[:,1][0],0.0,1.0))
            expected_return=_safe(reg.predict(frame)[0])
            model_family="legacy 5m logistic+ridge"

        predicted_price=current_price*(1.0+expected_return) if np.isfinite(current_price) and np.isfinite(expected_return) else np.nan
        signal="BUY" if p>=BUY_P and expected_return>0 else ("SELL" if p<=SELL_P and expected_return<0 else "HOLD")
        predictions.append({
            "horizon_minutes":int(horizon),
            "target_timestamp":str(target_ts),
            "probability_up":p,
            "probability_down":float(1.0-p),
            "signal":signal,
            "expected_return":expected_return,
            "current_price":current_price,
            "predicted_price":predicted_price,
            "predicted_change":(
                predicted_price-current_price
                if np.isfinite(predicted_price) and np.isfinite(current_price)
                else np.nan
            ),
            "model_family":model_family,
        })

    return {
        "prediction_status": "PASS",
        "symbol": str(symbol),
        "decision_timestamp": str(ts),
        "fno_data_mode": "previous_session_daily_pit" if pit_fno else "unavailable",
        "base_price": current_price,
        "horizons": predictions,
        "feature_count": len(FEATURES),
        "model_status": model_status(),
        "model_version": "2026-5m-step-multihorizon-v2",
        "ensemble": True,
    }
