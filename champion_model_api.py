"""Drop-in model API for all_nse_minute_prediction_capture.py, backed by the nightly champion.

Place this file in the repo root next to all_nse_intraday_model.py. It exposes exactly the hooks the capture
uses today: model_status(), _models(), _latest_features(), active_feature_names(), calibrate_probability(),
signal_thresholds(), plus FEATURES and _drive_download. See INTEGRATION.md for the capture changes.

Rules enforced here
  * fail closed: missing, unreadable or hash-mismatched champion => model_status() reports unavailable
  * serving mode HOLD (stale model, gate not active, live circuit breaker) => thresholds that can never fire
  * the champion's isotonic calibration is the ONLY calibration: calibrate_probability() applies it once;
    the legacy calibration overlay must be bypassed for this model
  * features come from features_5m, the same code as training; the last bar must be a CLOSED, bar-close-stamped
    5-minute candle on the 5-minute grid, otherwise ValueError (the capture then marks the row data-limited)
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pandas as pd

from all_nse_intraday_model import HORIZON_NAME, _drive_download, model_target_contract  # noqa: F401 (re-exported)
from probability_calibration import apply_probability_calibration
import daily_cycle as D
import pretrain_pipeline as P
from features_5m import DEFAULT_WARMUP, FEATURES_5M, FNO_COLS, make_features_5m, market_features_5m

FEATURES = list(FEATURES_5M)
_LOCK = threading.Lock()
_CFG: dict[str, Any] = {"store": None, "session_dates": lambda: [], "holidays": frozenset(), "now": None, "ledger": lambda: pd.DataFrame(columns=D.LEDGER_COLS),
                        "max_lag": 1, "recheck_seconds": 120.0}
_STATE: dict[str, Any] = {"stamp": None, "champion": None, "bundle": None, "error": "not configured", "checked": 0.0, "served_stamp": None}
_NEVER = (1.01, -0.01)           # thresholds that cannot fire: BUY needs p >= 1.01, SELL needs p <= -0.01


def configure(store, session_dates=None, ledger=None, max_lag_sessions: int = 1, recheck_seconds: float = 120.0,
              holidays=(), now=None) -> None:
    """store: daily_cycle.LocalStore or DriveStore. session_dates() -> optional extra stored session dates.
    holidays: NSE holiday dates (YYYY-MM-DD) so a market holiday is not counted as a missed nightly run.
    now: optional callable returning the current IST Timestamp (tests). ledger() -> daily_cycle.read_ledger(store)."""
    _CFG.update(store=store, max_lag=max_lag_sessions, recheck_seconds=recheck_seconds,
                holidays=frozenset(str(h) for h in holidays), now=now)
    if session_dates is not None:
        _CFG["session_dates"] = session_dates
    _CFG["ledger"] = ledger if ledger is not None else (lambda: D.read_ledger(store))
    with _LOCK:
        _STATE.update(stamp=None, champion=None, bundle=None, error="not loaded", checked=0.0)


def _refresh(force: bool = False) -> None:
    store = _CFG["store"]
    if store is None:
        _STATE.update(champion=None, bundle=None, error="not configured")
        return
    with _LOCK:
        now = time.monotonic()
        if not force and _STATE["bundle"] is not None and now - _STATE["checked"] < _CFG["recheck_seconds"]:
            return
        _STATE["checked"] = now
        stamp = store.stamp("champion.json")
        if not force and stamp == _STATE["stamp"] and _STATE["bundle"] is not None:
            return                                           # nothing changed: keep the verified bundle in memory
        champ, bundle, err = D.load_champion_bundle(store)
        _STATE.update(stamp=stamp, champion=champ, bundle=bundle, error=err)


def champion_changed() -> bool:
    """True when a newer champion has been published since the capture last bound its models.
    The capture should rebind (call _ensure_model again) at the start of the next decision bar, never mid-batch."""
    _refresh()
    return _STATE["stamp"] != _STATE["served_stamp"]


def _target():
    _refresh()
    if _STATE["bundle"] is None:
        raise RuntimeError(f"Champion model unavailable: {_STATE['error']}")
    return _STATE["champion"], _STATE["bundle"]["close"]


def expected_sessions(trained_through: str) -> list[str]:
    """Weekdays after `trained_through` and strictly before today (IST), minus holidays: the sessions whose data
    the model should already contain. Uses the wall clock, so a silently failed nightly job still ages the model."""
    now = _CFG["now"]() if _CFG["now"] else pd.Timestamp.now(tz="Asia/Kolkata")
    days = pd.bdate_range(pd.Timestamp(trained_through) + pd.Timedelta(days=1), now.tz_localize(None).normalize() - pd.Timedelta(days=1))
    return [d.strftime("%Y-%m-%d") for d in days if d.strftime("%Y-%m-%d") not in _CFG["holidays"]]


def serving_policy() -> dict:
    _refresh()
    if _STATE["champion"] is None:
        return D._hold_policy(_STATE["error"] or "unavailable")
    champ = _STATE["champion"]
    dates = sorted(set(expected_sessions(champ["trained_through"])) | {str(d) for d in _CFG["session_dates"]()})
    return D.serving_policy(champ, dates, _CFG["ledger"](), _CFG["max_lag"])


def serving_mode() -> str:
    return serving_policy()["horizons"]["close"]["mode"]


def active_feature_names() -> list[str]:
    return list(FEATURES_5M)


def model_status() -> dict[str, Any]:
    _refresh()
    if _STATE["champion"] is None:
        return {"status": "MODEL_ARTIFACT_UNAVAILABLE", "error": _STATE["error"], "manifest_sha256_match": False}
    champ, ms = _target()
    pol = serving_policy()["horizons"]["close"]
    contract = model_target_contract()
    return {"status": "CHAMPION", "model_family": "hgb_logistic_ensemble", "target": contract["target"],
            "artifact_target": champ.get("target"),
            "artifact_feature_module": champ.get("feature_module"),
            "artifact_feature_count": champ.get("feature_count"),
            "artifact_features": list(ms.get("features") or []),
            "prediction_horizon": HORIZON_NAME, "model_version": f"{champ['version']}__{champ['bundle_sha256'][:12]}",
            "trained_through": champ["trained_through"], "feature_count": len(FEATURES_5M), "manifest_feature_count": len(FEATURES_5M),
            "features": list(FEATURES_5M), "feature_contract_match": True, "target_contract_match": True,
            "manifest_sha256_match": True, "training_symbols": ms.get("training_symbols", []),
            "probability_calibration_method": (ms.get("calibration") or {}).get("method"),
            "serving_mode": pol["mode"], "serving_reason": pol["reason"]}


class _EnsembleClassifier:
    def __init__(self, ms): self._ms = ms

    def predict_proba(self, X):
        p, _ = P.predict_raw(self._ms, X[FEATURES_5M].astype("float32"))      # RAW: calibrate_probability() is applied once, by the capture
        return np.column_stack([1.0 - p, p])


class _EnsembleRegressor:
    def __init__(self, ms): self._ms = ms

    def predict(self, X):
        return P.predict_raw(self._ms, X[FEATURES_5M].astype("float32"))[1]


def _models():
    champ, ms = _target()
    with _LOCK:
        _STATE["served_stamp"] = _STATE["stamp"]
    return _EnsembleClassifier(ms), _EnsembleRegressor(ms)


def calibrate_probability(probability: float) -> float:
    try:
        _, ms = _target()
        return float(apply_probability_calibration([probability], ms.get("calibration"))[0])
    except Exception:                                        # noqa: BLE001 - never take the capture down
        return float(probability)


def signal_thresholds() -> tuple[float, float]:
    """(buy, sell). HOLD mode returns thresholds that cannot fire. PAPER mode returns the real ones: the
    capture must label PAPER rows as not tradable (use serving_mode())."""
    try:
        pol = serving_policy()["horizons"]["close"]
        if pol["mode"] == "HOLD" or not pol.get("thresholds") or pol["thresholds"].get("buy_p") is None:
            return _NEVER
        return float(pol["thresholds"]["buy_p"]), float(pol["thresholds"]["sell_p"])
    except Exception:                                        # noqa: BLE001
        return _NEVER


def signal_min_edge() -> float | None:
    """Expected-return margin (fraction) a signal must also clear, or None for probability-only signals.
    Only honoured if the capture's _signal is patched (INTEGRATION.md); otherwise train with --no-edge-rule."""
    try:
        return serving_policy()["horizons"]["close"]["thresholds"].get("min_edge")
    except Exception:                                        # noqa: BLE001
        return None


def _norm(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]
    alias = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "ltp": "Close", "volume": "Volume", "vol": "Volume",
             "timestamp": "Timestamp", "datetime": "Timestamp", "date": "Timestamp", "time": "Timestamp"}
    x = x.rename(columns={c: alias.get(c.lower(), c) for c in x.columns})
    if "Timestamp" in x.columns:
        x = x.set_index("Timestamp")
    idx = pd.DatetimeIndex(pd.to_datetime(x.index))
    x.index = idx.tz_localize("Asia/Kolkata") if idx.tz is None else idx.tz_convert("Asia/Kolkata")
    if "Volume" not in x.columns:
        x["Volume"] = np.nan
    return x[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()


def _latest_features(price_df: pd.DataFrame, market_df: pd.DataFrame | None = None, fno: dict[str, Any] | None = None,
                     decision_timestamp: Any = None, feature_names: list[str] | None = None) -> dict[str, float]:
    x = _norm(price_df)
    if x.empty:
        raise ValueError("No usable intraday OHLC data")
    ts = x.index[-1] if decision_timestamp is None else pd.Timestamp(decision_timestamp)
    ts = ts.tz_localize("Asia/Kolkata") if ts.tzinfo is None else ts.tz_convert("Asia/Kolkata")
    x = x.loc[x.index <= ts]
    _, ms = _target()
    mkt = market_features_5m(_norm(market_df)) if market_df is not None and len(market_df) else None
    f = make_features_5m(x, mkt, None, int(ms.get("feature_warmup", DEFAULT_WARMUP)))
    if ts not in f.index:
        raise ValueError("Feature row unavailable: the decision bar must be a closed 5-minute candle on the grid, with "
                         "enough contiguous history, an on-time session start, and a stamp <= 15:25")
    row = f.loc[ts].copy()
    if fno:                                                  # capture supplies the point-in-time daily F&O context as a dict
        for name, keys in (("fno_pcr", ("pcr", "fno_pcr")), ("fno_oi_change", ("oi_change", "fno_oi_change")),
                           ("fno_fut_return", ("fut_return", "fno_fut_return")), ("fno_fut_oi", ("fut_oi", "fno_fut_oi")),
                           ("fno_basis", ("basis", "fno_basis"))):
            v = next((fno[k] for k in keys if k in fno and fno[k] is not None), np.nan)
            row[name] = pd.to_numeric(v, errors="coerce")
        row["fno_available"] = float(any(np.isfinite(row[c]) for c in FNO_COLS))
    names = list(feature_names) if feature_names else list(FEATURES_5M)
    return {n: float(row[n]) for n in names}
