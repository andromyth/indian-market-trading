"""ALL-NSE multi-horizon 5-minute-step Intraday model adapter.

This is a separate candidate model family and does not replace or modify the
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
    "rsi14","atr14_pct","ma5_ratio","ma10_ratio","ma20_ratio","ma50_ratio",
    "range_pct","body_pct","volume_ratio","volume_z20","tod_sin","tod_cos",
    "market_ret_1","market_ret_3","market_ret_5","market_ret_10","market_ret_15","market_ret_30",
    "market_rsi14","relative_ret_1","relative_ret_5","relative_ret_15",
    "fno_pcr","fno_oi_change","fno_fut_return","fno_fut_oi","fno_basis","fno_available",
]

DRIVE_MODEL_ROOT = "models/all_nse_intraday_5m_multifactor_replacement_2026"
REMOTE_MANIFEST = f"{DRIVE_MODEL_ROOT}/ALL_NSE_INTRADAY_5M_MULTIHORIZON_2026_MODEL_MANIFEST.json"
REMOTE_BUNDLE = f"{DRIVE_MODEL_ROOT}/deployment_all_nse_intraday_5m_multihorizon_bundle_2026.joblib"
CACHE_ROOT = Path(__file__).resolve().parent / "data" / "all_nse_intraday_5m_multihorizon"
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
    x = _norm(price_df)
    if x.empty or x["Close"].dropna().empty:
        raise ValueError("No usable intraday OHLC data")
    c,h,l,o,v = x["Close"],x["High"],x["Low"],x["Open"],x["Volume"]
    close = _safe(c.iloc[-1])
    open_ = _safe(o.iloc[-1])
    ma = {n: c.rolling(n, min_periods=n).mean().iloc[-1] for n in (5,10,20,50)}
    vma = v.rolling(20, min_periods=20).mean().iloc[-1]
    vsd = v.rolling(20, min_periods=20).std().iloc[-1]
    vals = {
        "ret_1":_returns(c,1),"ret_3":_returns(c,3),"ret_5":_returns(c,5),
        "ret_10":_returns(c,10),"ret_15":_returns(c,15),"ret_30":_returns(c,30),
        "rsi14":_rsi(c),"atr14_pct":_atr_pct(x),
        "ma5_ratio":_safe(close/ma[5]-1) if np.isfinite(close) and pd.notna(ma[5]) and ma[5] else np.nan,
        "ma10_ratio":_safe(close/ma[10]-1) if np.isfinite(close) and pd.notna(ma[10]) and ma[10] else np.nan,
        "ma20_ratio":_safe(close/ma[20]-1) if np.isfinite(close) and pd.notna(ma[20]) and ma[20] else np.nan,
        "ma50_ratio":_safe(close/ma[50]-1) if np.isfinite(close) and pd.notna(ma[50]) and ma[50] else np.nan,
        "range_pct":_safe((h.iloc[-1]-l.iloc[-1])/close) if np.isfinite(close) and close else np.nan,
        "body_pct":_safe((close-open_)/open_) if np.isfinite(close) and np.isfinite(open_) and open_ else np.nan,
        "volume_ratio":_safe(v.iloc[-1]/vma) if np.isfinite(v.iloc[-1]) and np.isfinite(vma) and vma else np.nan,
        "volume_z20":_safe((v.iloc[-1]-vma)/vsd) if np.isfinite(v.iloc[-1]) and np.isfinite(vma) and np.isfinite(vsd) and vsd else np.nan,
    }
    ts = pd.Timestamp(decision_timestamp) if decision_timestamp is not None else x.index[-1]
    if ts.tzinfo is None: ts = ts.tz_localize(IST)
    else: ts = ts.tz_convert(IST)
    session_minutes = max(0, min(389, ts.hour*60+ts.minute-(9*60+15)))
    vals["tod_sin"] = float(np.sin(2*np.pi*session_minutes/390.0))
    vals["tod_cos"] = float(np.cos(2*np.pi*session_minutes/390.0))

    mc = _market_series(market_df)
    for n in (1,3,5,10,15,30):
        vals[f"market_ret_{n}"] = _returns(mc,n)
    vals["market_rsi14"] = _rsi(mc)
    vals["relative_ret_1"] = _safe(vals["ret_1"]-vals["market_ret_1"]) if np.isfinite(vals["ret_1"]) and np.isfinite(vals["market_ret_1"]) else np.nan
    vals["relative_ret_5"] = _safe(vals["ret_5"]-vals["market_ret_5"]) if np.isfinite(vals["ret_5"]) and np.isfinite(vals["market_ret_5"]) else np.nan
    vals["relative_ret_15"] = _safe(vals["ret_15"]-vals["market_ret_15"]) if np.isfinite(vals["ret_15"]) and np.isfinite(vals["market_ret_15"]) else np.nan

    fo = fno or {}
    vals["fno_pcr"] = _safe(fo.get("pcr", fo.get("fno_pcr")))
    vals["fno_oi_change"] = _safe(fo.get("oi_change", fo.get("fno_oi_change")))
    vals["fno_fut_return"] = _safe(fo.get("fut_return", fo.get("fno_fut_return")))
    vals["fno_fut_oi"] = _safe(fo.get("fut_oi", fo.get("fno_fut_oi")))
    vals["fno_basis"] = _safe(fo.get("basis", fo.get("fno_basis")))
    vals["fno_available"] = 1.0 if any(pd.notna(vals[k]) for k in ("fno_pcr","fno_oi_change","fno_fut_return","fno_fut_oi","fno_basis")) else 0.0
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
        return {
            "status": "READY" if manifest.get("status") == "VERIFIED" and bundle_hash_ok and set(verified_horizons) == set(HORIZONS) else "UNVERIFIED",
            "target_horizons": list(HORIZONS),
            "feature_count": len(FEATURES),
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
    row = _latest_features(price_df, market_df, fno, ts)
    frame = pd.DataFrame([row], columns=FEATURES)
    current_price = _safe(pd.to_numeric(price_df["Close"], errors="coerce").dropna().iloc[-1])
    predictions = []
    for horizon in HORIZONS:
        target_ts = ts + pd.Timedelta(minutes=int(horizon))
        # Only emit horizons whose target remains inside the NSE regular
        # session. Near the close, the table therefore naturally shortens.
        if target_ts.time() > pd.Timestamp("15:30").time():
            continue
        h = str(horizon)
        clf = bundle[h]["classifier"]
        reg = bundle[h]["regressor"]
        p = float(np.clip(clf.predict_proba(frame)[:,1][0], 0.0, 1.0))
        expected_return = _safe(reg.predict(frame)[0])
        predicted_price = (
            current_price * (1.0 + expected_return)
            if np.isfinite(current_price) and np.isfinite(expected_return)
            else np.nan
        )
        predictions.append({
            "horizon_minutes": int(horizon),
            "target_timestamp": str(target_ts),
            "probability_up": p,
            "probability_down": float(1.0-p),
            "signal": "BUY" if p >= BUY_P else ("SELL" if p <= SELL_P else "HOLD"),
            "expected_return": expected_return,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_change": (
                predicted_price-current_price
                if np.isfinite(predicted_price) and np.isfinite(current_price)
                else np.nan
            ),
        })
    return {
        "prediction_status": "PASS",
        "symbol": str(symbol),
        "decision_timestamp": str(ts),
        "base_price": current_price,
        "horizons": predictions,
        "feature_count": len(FEATURES),
        "model_status": model_status(),
        "model_version": "2026-5m-step-multihorizon",
    }
