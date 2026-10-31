"""Legacy 53-feature Delivery compatibility adapter.

Live Delivery predictions must use all_nse_delivery_model.py, which enforces
current official NSE-universe membership and sufficient clean history. The
53-feature artifacts and feature builder below are retained only for historical
reproducibility; they are not the active prediction route.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "production_delivery_model"
MANIFEST_PATH = MODEL_DIR / "PREDICTION_ENGINE_MANIFEST_V2_2026.json"
CLASSIFIER_PATH = MODEL_DIR / "deployment_next_session_logistic_classifier_2026.joblib"
REGRESSOR_PATH = MODEL_DIR / "deployment_next_session_hgb_return_regressor_2026.joblib"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


MANIFEST = _load_json(MANIFEST_PATH)

_CLASSIFIER = None
_REGRESSOR = None
_FEATURES: list[str] | None = None


def feature_names() -> list[str]:
    global _FEATURES
    if _FEATURES is None:
        # The manifest does not include the actual feature list, so read it from
        # the certified sklearn artifact itself. This prevents schema drift.
        probe = _load_classifier()
        names = getattr(probe, "feature_names_in_", None)
        if names is None:
            names = getattr(probe.named_steps.get("imputer"), "feature_names_in_", None)
        if names is None:
            raise RuntimeError("Delivery classifier does not expose feature_names_in_")
        _FEATURES = [str(x) for x in names]
    return list(_FEATURES)


def _verify(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Delivery model artifact missing: {path}")


def _patch_sklearn_compat(model):
    # Artifacts were trained with sklearn 1.6.1. Newer sklearn renamed an
    # internal SimpleImputer attribute used during transform. Keep this narrow
    # compatibility shim; the learned parameters are not changed.
    try:
        imp = model.named_steps.get("imputer")
        if imp is not None and not hasattr(imp, "_fill_dtype") and hasattr(imp, "_fit_dtype"):
            imp._fill_dtype = imp._fit_dtype
    except Exception:
        pass
    return model


def _load_classifier():
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _verify(CLASSIFIER_PATH)
        _CLASSIFIER = _patch_sklearn_compat(joblib.load(CLASSIFIER_PATH))
        classes = list(getattr(_CLASSIFIER.named_steps.get("model"), "classes_", []))
        if classes != [0, 1]:
            raise RuntimeError(f"Unexpected delivery classifier classes: {classes}; expected [0, 1]")
    return _CLASSIFIER


def _load_regressor():
    global _REGRESSOR
    if _REGRESSOR is None:
        _verify(REGRESSOR_PATH)
        _REGRESSOR = _patch_sklearn_compat(joblib.load(REGRESSOR_PATH))
    return _REGRESSOR


def model_status() -> dict[str, Any]:
    from all_nse_delivery_model import model_status as active_model_status
    return {**active_model_status(), "legacy_53_feature_adapter": "INACTIVE"}


def supported_symbol(symbol: str) -> bool:
    """Use the active official-NSE eligibility policy, never the legacy pilot list."""
    try:
        from all_nse_delivery_model import supported_symbol as active_supported_symbol
        return bool(active_supported_symbol(symbol))
    except Exception:
        return False


def _safe(v: Any) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _returns(s: pd.Series, n: int) -> float:
    x = s.dropna()
    if len(x) <= n:
        return np.nan
    return _safe(x.iloc[-1] / x.iloc[-1 - n] - 1.0)


def _rsi(s: pd.Series, n: int = 14) -> float:
    x = s.dropna()
    if len(x) < n + 1:
        return np.nan
    d = x.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    gl = _safe(gain.iloc[-1]); ll = _safe(loss.iloc[-1])
    if not np.isfinite(gl) or not np.isfinite(ll):
        return np.nan
    if ll == 0:
        return 100.0
    rs = gl / ll
    return float(100.0 - 100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    h = _series(df, "High")
    l = _series(df, "Low")
    c = _series(df, "Close")
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    v = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return _safe(v.iloc[-1]) if len(v) else np.nan


def _feature_row(
    price_df: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
    news: Mapping[str, Any] | None = None,
    event: Mapping[str, Any] | None = None,
    fno: Mapping[str, Any] | None = None,
    fundamentals: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    if price_df is None or price_df.empty:
        raise ValueError("Delivery prediction requires non-empty price history")

    close = _series(price_df, "Close")
    high = _series(price_df, "High")
    low = _series(price_df, "Low")
    open_ = _series(price_df, "Open")
    volume = _series(price_df, "Volume")
    last_close = _safe(close.iloc[-1])
    ma20 = _safe(close.rolling(20).mean().iloc[-1])
    ma50 = _safe(close.rolling(50).mean().iloc[-1])
    vma20 = _safe(volume.rolling(20).mean().iloc[-1])
    vstdev = _safe(volume.rolling(20).std().iloc[-1])

    row: dict[str, float] = {
        "ret_1": _returns(close, 1), "ret_3": _returns(close, 3),
        "ret_6": _returns(close, 6), "ret_12": _returns(close, 12),
        "ret_24": _returns(close, 24), "ret_48": _returns(close, 48),
        "rsi14": _rsi(close, 14), "atr14": _atr(price_df, 14),
        "ma20_ratio": last_close / ma20 - 1 if np.isfinite(last_close) and np.isfinite(ma20) and ma20 else np.nan,
        "ma50_ratio": last_close / ma50 - 1 if np.isfinite(last_close) and np.isfinite(ma50) and ma50 else np.nan,
        "volume_ratio": _safe(volume.iloc[-1]) / vma20 if np.isfinite(_safe(volume.iloc[-1])) and np.isfinite(vma20) and vma20 else np.nan,
        "volume_z20": (_safe(volume.iloc[-1]) - vma20) / vstdev if np.isfinite(_safe(volume.iloc[-1])) and np.isfinite(vma20) and np.isfinite(vstdev) and vstdev else np.nan,
        "range_pct": (_safe(high.iloc[-1]) - _safe(low.iloc[-1])) / last_close if np.isfinite(last_close) and last_close else np.nan,
        "body_pct": (last_close - _safe(open_.iloc[-1])) / _safe(open_.iloc[-1]) if np.isfinite(last_close) and np.isfinite(_safe(open_.iloc[-1])) and _safe(open_.iloc[-1]) else np.nan,
        "up_bar": 1.0 if np.isfinite(last_close) and np.isfinite(_safe(open_.iloc[-1])) and last_close > _safe(open_.iloc[-1]) else 0.0 if np.isfinite(last_close) and np.isfinite(_safe(open_.iloc[-1])) else np.nan,
    }

    market = market_df if market_df is not None else pd.DataFrame()
    mc = _series(market, "Close")
    row.update({
        "market_ret_1": _returns(mc, 1), "market_ret_3": _returns(mc, 3),
        "market_ret_6": _returns(mc, 6), "market_ret_12": _returns(mc, 12),
        "market_rsi14": _rsi(mc, 14),
    })
    row["relative_ret_1"] = row["ret_1"] - row["market_ret_1"] if np.isfinite(row["ret_1"]) and np.isfinite(row["market_ret_1"]) else np.nan
    row["relative_ret_6"] = row["ret_6"] - row["market_ret_6"] if np.isfinite(row["ret_6"]) and np.isfinite(row["market_ret_6"]) else np.nan
    row["relative_ret_12"] = row["ret_12"] - row["market_ret_12"] if np.isfinite(row["ret_12"]) and np.isfinite(row["market_ret_12"]) else np.nan

    # These two proxies are kept deterministic and point-in-time. They are not
    # presented as independent predictive signals.
    mret = row.get("market_ret_1", np.nan)
    mvol = mc.pct_change().rolling(20).std().iloc[-1] if len(mc) else np.nan
    row["market_mood_proxy"] = _safe(mret)
    row["market_stress_proxy"] = _safe(mvol)

    news = news or {}
    event = event or {}
    fno = fno or {}
    fundamentals = fundamentals or {}
    row.update({
        "news_available": _safe(news.get("available", 0.0)),
        "event_available": _safe(event.get("available", 0.0)),
        "event_high_impact": _safe(event.get("high_impact", 0.0)),
        "fno_pcr": _safe(fno.get("pcr")),
        "fno_oi_change": _safe(fno.get("oi_change")),
        "fno_iv_change": _safe(fno.get("iv_change")),
        "fno_available": _safe(fno.get("available", 0.0)),
        "fno_fut_close": _safe(fno.get("fut_close")),
        "fno_fut_return": _safe(fno.get("fut_return")),
        "fno_fut_oi": _safe(fno.get("fut_oi")),
        "fundamental_revenue": _safe(fundamentals.get("revenue")),
        "fundamental_pat": _safe(fundamentals.get("pat")),
        "fundamental_eps": _safe(fundamentals.get("eps")),
        "fundamental_ebitda": _safe(fundamentals.get("ebitda")),
        "fundamental_finance_cost": _safe(fundamentals.get("finance_cost")),
        "fundamental_total_equity": _safe(fundamentals.get("total_equity")),
        "fundamental_total_debt": _safe(fundamentals.get("total_debt")),
        "fundamental_cash": _safe(fundamentals.get("cash")),
        "fund_revenue_growth": _safe(fundamentals.get("revenue_growth")),
        "fund_pat_growth": _safe(fundamentals.get("pat_growth")),
        "fund_eps_growth": _safe(fundamentals.get("eps_growth")),
        "fund_ebitda_growth": _safe(fundamentals.get("ebitda_growth")),
        "fund_pat_margin": _safe(fundamentals.get("pat_margin")),
        "fund_debt_to_equity": _safe(fundamentals.get("debt_to_equity")),
        "fund_cash_to_debt": _safe(fundamentals.get("cash_to_debt")),
        "fundamental_age_days": _safe(fundamentals.get("age_days")),
        "fundamental_stale_90d": _safe(fundamentals.get("stale_90d")),
        "fundamental_stale_180d": _safe(fundamentals.get("stale_180d")),
    })

    return row


def build_feature_row(
    price_df: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    row = _feature_row(price_df, market_df=market_df, **kwargs)
    frame = pd.DataFrame([row])
    for col in feature_names():
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[feature_names()]


def predict(feature_df: pd.DataFrame, symbol: str, decision_timestamp: Any = None) -> dict[str, Any]:
    """Compatibility signature; live scoring delegates to the active ALL-NSE model."""
    from all_nse_delivery_model import predict_symbol

    result = predict_symbol(symbol, decision_timestamp=decision_timestamp)
    result["legacy_53_feature_adapter"] = "INACTIVE"
    return result
