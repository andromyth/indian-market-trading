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

from snapshot_5m_features import (
    FEATURES, FEATURE_SCHEMA_VERSION, SNAPSHOT_CADENCE_MINUTES,
    latest_snapshot_features,
)

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


def _session_aware_returns(c: pd.Series, n: int) -> float:
    """Calculate returns within a session; use prior-session bars during warm-up."""
    if c is None or c.empty:
        return np.nan
    values = pd.to_numeric(c, errors="coerce")
    index = pd.DatetimeIndex(values.index)
    index = index.tz_localize("Asia/Kolkata") if index.tz is None else index.tz_convert("Asia/Kolkata")
    session_dates = pd.Index(index.date)
    current_date = session_dates[-1]
    current = values.loc[session_dates == current_date].dropna()
    if len(current) > n:
        return _returns(current, n)
    previous_dates = sorted({day for day in session_dates if day < current_date})
    if previous_dates:
        previous = values.loc[session_dates == previous_dates[-1]].dropna()
        if len(previous) > n:
            return _returns(previous, n)
    return np.nan


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


def _latest_features(
    price_df: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
    fno: dict[str, Any] | None = None,
    decision_timestamp: Any = None,
) -> dict[str, float]:
    """Build features from five-minute snapshots; no one-minute history feed is used."""
    del market_df, decision_timestamp
    return latest_snapshot_features(price_df, fno=fno)


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


def _download_artifact_atomically(remote: str, local: Path) -> None:
    """Refresh one cached artifact without leaving a half-written model file."""
    local.parent.mkdir(parents=True, exist_ok=True)
    staging = local.with_name(local.name + ".download")
    try:
        staging.unlink(missing_ok=True)
        _drive_download(remote, staging)
        if not staging.is_file() or staging.stat().st_size == 0:
            raise RuntimeError(f"Downloaded model artifact is empty: {remote}")
        staging.replace(local)
    finally:
        staging.unlink(missing_ok=True)


def _artifact_hashes_match(manifest: dict[str, Any]) -> bool:
    expected_classifier = str(manifest.get("classifier_sha256") or "")
    expected_regressor = str(manifest.get("regressor_sha256") or "")
    return bool(
        expected_classifier
        and expected_regressor
        and LOCAL_CLASSIFIER.is_file()
        and LOCAL_REGRESSOR.is_file()
        and _sha256(LOCAL_CLASSIFIER) == expected_classifier
        and _sha256(LOCAL_REGRESSOR) == expected_regressor
    )


def _ensure_artifacts(force_refresh: bool = False) -> None:
    """Download missing artifacts, or atomically refresh all when hashes disagree."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    artifacts = [
        (REMOTE_MANIFEST, LOCAL_MANIFEST),
        (REMOTE_CLASSIFIER, LOCAL_CLASSIFIER),
        (REMOTE_REGRESSOR, LOCAL_REGRESSOR),
    ]
    for remote, local in artifacts:
        if force_refresh or not local.exists() or local.stat().st_size == 0:
            _download_artifact_atomically(remote, local)


def _manifest() -> dict[str, Any]:
    _ensure_artifacts()
    manifest = json.loads(LOCAL_MANIFEST.read_text(encoding="utf-8"))
    # A container image can contain older local model files while Drive already
    # has a newer manifest. Previously these cached files were never refreshed,
    # so valid forecasts could be blocked indefinitely by a hash mismatch.
    # Refresh the manifest and both estimators once; never bypass the hash gate.
    if not _artifact_hashes_match(manifest):
        _ensure_artifacts(force_refresh=True)
        manifest = json.loads(LOCAL_MANIFEST.read_text(encoding="utf-8"))
    return manifest

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
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "input_cadence_minutes": SNAPSHOT_CADENCE_MINUTES,
    }


def model_status() -> dict[str,Any]:
    try:
        m=_manifest()
    except Exception as exc:
        return {"status":"MODEL_ARTIFACT_UNAVAILABLE","error_type":type(exc).__name__,"error":str(exc)}
    classifier_sha = str(m.get("classifier_sha256") or "")
    regressor_sha = str(m.get("regressor_sha256") or "")
    # Profile identity must change when either deployed estimator changes, even
    # when two builds share the same training cutoff timestamp.
    artifact_version = (
        f"{m.get('model_version') or m.get('trained_until') or 'unknown'}__{classifier_sha[:12]}_{regressor_sha[:12]}"
        if classifier_sha and regressor_sha else (
            str(m.get("model_version") or m.get("trained_until") or "unversioned")
        )
    )
    return {
        "status":m.get("status","REPLACEMENT"),
        "model_family":m.get("model_family"),
        "target":m.get("target"),
        "model_version":str(artifact_version or "unversioned"),
        "feature_count":len(FEATURES),
        "feature_schema_version":m.get("feature_schema_version"),
        "input_cadence_minutes":m.get("input_cadence_minutes"),
        "training_symbols":m.get("training_symbols",[]),
        "training_symbol_count":m.get("training_symbol_count"),
        "universe_size":m.get("universe_size"),
        "coverage_pct":m.get("coverage_pct"),
        "trained_until":m.get("trained_until"),
        "classifier_sha256":classifier_sha or None,
        "regressor_sha256":regressor_sha or None,
        "artifact_hashes_match": all([
            LOCAL_CLASSIFIER.exists() and _sha256(LOCAL_CLASSIFIER)==m.get("classifier_sha256"),
            LOCAL_REGRESSOR.exists() and _sha256(LOCAL_REGRESSOR)==m.get("regressor_sha256"),
        ]),
        "feature_schema_match": (
            m.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
            and int(m.get("input_cadence_minutes") or 0) == SNAPSHOT_CADENCE_MINUTES
            and list(m.get("features") or []) == FEATURES
        ),
        "manifest_sha256_match": bool(
            all([
                LOCAL_CLASSIFIER.exists() and _sha256(LOCAL_CLASSIFIER)==m.get("classifier_sha256"),
                LOCAL_REGRESSOR.exists() and _sha256(LOCAL_REGRESSOR)==m.get("regressor_sha256"),
            ])
            and m.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
            and int(m.get("input_cadence_minutes") or 0) == SNAPSHOT_CADENCE_MINUTES
            and list(m.get("features") or []) == FEATURES
        ),
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
    s = model_status()
    if not s.get("artifact_hashes_match"):
        raise RuntimeError("ALL-NSE Intraday classifier/regressor hashes do not match the manifest")
    if not s.get("feature_schema_match"):
        raise RuntimeError(
            "ALL-NSE Intraday model is not trained for the five-minute snapshot feature schema "
            f"{FEATURE_SCHEMA_VERSION} at cadence={SNAPSHOT_CADENCE_MINUTES}m; retraining is required"
        )
    if not s.get("manifest_sha256_match"):
        raise RuntimeError("ALL-NSE Intraday model manifest integrity validation failed")
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
