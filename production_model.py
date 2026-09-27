"""Verified production-model adapter for the Streamlit dashboard.

The bundled model is the exact production LogisticRegression pipeline whose
manifest declares target_30m and 137 features.  This adapter constructs the
same named feature surface from the dashboard's live inputs.  Missing optional
information is represented explicitly with NaN/missing flags so the model's
trained imputer handles it; no values are fabricated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

MODEL_DIR = Path(__file__).resolve().parent / "production_model"
MANIFEST_PATH = MODEL_DIR / "DEPLOYMENT_MODEL_MANIFEST_2026.json"
MODEL_PATH = MODEL_DIR / "deployment_classifier_logistic_regression_2026.joblib"


def _load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


_MANIFEST = _load_manifest()
FEATURES = list(_MANIFEST["features"])


def model_status() -> dict[str, Any]:
    sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() if MODEL_PATH.exists() else None
    return {
        "model_family": _MANIFEST.get("model_family"),
        "target": _MANIFEST.get("target"),
        "feature_count": len(FEATURES),
        "manifest_sha256_match": MODEL_PATH.exists() and sha == _MANIFEST.get("model_sha256"),
        "model_path": str(MODEL_PATH),
        "training_rows": _MANIFEST.get("training_rows"),
        "training_end_exclusive": _MANIFEST.get("training_end_exclusive"),
    }


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Production model missing: {MODEL_PATH}")
        status = model_status()
        if not status["manifest_sha256_match"]:
            raise RuntimeError("Production model SHA-256 does not match the certified manifest")
        _MODEL = joblib.load(MODEL_PATH)
        # The certified artifact was trained with scikit-learn 1.6.1.
        # Newer sklearn releases renamed an internal SimpleImputer attribute;
        # keep a compatibility fallback without altering the trained model.
        try:
            imp = _MODEL.named_steps.get("imputer")
            if imp is not None and not hasattr(imp, "_fill_dtype") and hasattr(imp, "_fit_dtype"):
                imp._fill_dtype = imp._fit_dtype
        except Exception:
            pass
    return _MODEL


def _safe_float(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _rsi(s: pd.Series, period: int = 14) -> float:
    x = pd.to_numeric(s, errors="coerce")
    d = x.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    if len(x) == 0 or pd.isna(gain.iloc[-1]) or pd.isna(loss.iloc[-1]):
        return np.nan
    if loss.iloc[-1] == 0:
        return 100.0
    rs = gain.iloc[-1] / loss.iloc[-1]
    return float(100 - 100 / (1 + rs))


def _returns(s: pd.Series, n: int) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) <= n:
        return np.nan
    return _safe_float(x.iloc[-1] / x.iloc[-1 - n] - 1.0)


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h = pd.to_numeric(df.get("High"), errors="coerce")
    l = pd.to_numeric(df.get("Low"), errors="coerce")
    c = pd.to_numeric(df.get("Close"), errors="coerce")
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    v = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return _safe_float(v.iloc[-1]) if len(v) else np.nan


def _price_features(df: pd.DataFrame) -> dict[str, float]:
    c = pd.to_numeric(df["Close"], errors="coerce")
    h = pd.to_numeric(df["High"], errors="coerce")
    l = pd.to_numeric(df["Low"], errors="coerce")
    o = pd.to_numeric(df["Open"], errors="coerce")
    v = pd.to_numeric(df["Volume"], errors="coerce")
    close = _safe_float(c.iloc[-1]); open_ = _safe_float(o.iloc[-1])
    high = _safe_float(h.iloc[-1]); low = _safe_float(l.iloc[-1]); vol = _safe_float(v.iloc[-1])
    ma20 = _safe_float(c.rolling(20, min_periods=20).mean().iloc[-1])
    ma50 = _safe_float(c.rolling(50, min_periods=50).mean().iloc[-1])
    vma20 = _safe_float(v.rolling(20, min_periods=20).mean().iloc[-1])
    return {
        "open": open_, "high": high, "low": low, "close": close, "volume": vol,
        "ret_1": _returns(c, 1), "ret_3": _returns(c, 3), "ret_6": _returns(c, 6),
        "ret_12": _returns(c, 12), "ret_24": _returns(c, 24), "rsi14": _rsi(c),
        "atr14": _atr(df), "ma20_ratio": close / ma20 - 1 if np.isfinite(close) and np.isfinite(ma20) and ma20 else np.nan,
        "ma50_ratio": close / ma50 - 1 if np.isfinite(close) and np.isfinite(ma50) and ma50 else np.nan,
        "volume_ratio": vol / vma20 if np.isfinite(vol) and np.isfinite(vma20) and vma20 else np.nan,
        "range_pct": (high - low) / close if np.isfinite(high) and np.isfinite(low) and np.isfinite(close) and close else np.nan,
        "body_pct": (close - open_) / open_ if np.isfinite(close) and np.isfinite(open_) and open_ else np.nan,
    }


def _market_features(market_df: pd.DataFrame | None) -> dict[str, float]:
    if market_df is None or market_df.empty:
        return {k: np.nan for k in ["nifty_close","market_ret_1","market_ret_3","market_ret_6","market_ret_12","market_rsi14"]}
    c = pd.to_numeric(market_df["Close"], errors="coerce")
    return {
        "nifty_close": _safe_float(c.iloc[-1]), "market_ret_1": _returns(c, 1),
        "market_ret_3": _returns(c, 3), "market_ret_6": _returns(c, 6),
        "market_ret_12": _returns(c, 12), "market_rsi14": _rsi(c),
    }


def _cross_sectional(rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    for source, out in [("ret_1", "xs_ret_1_rank"), ("ret_6", "xs_ret_6_rank"),
                        ("ret_12", "xs_ret_12_rank"), ("volume_ratio", "xs_volume_ratio_rank")]:
        if source in frame:
            frame[out] = frame[source].rank(pct=True)
    valid = pd.to_numeric(frame.get("ret_1"), errors="coerce")
    breadth = float((valid > 0).mean()) if valid.notna().any() else np.nan
    for i, row in enumerate(rows):
        for col in ["xs_ret_1_rank","xs_ret_6_rank","xs_ret_12_rank","xs_volume_ratio_rank"]:
            rows[i][col] = _safe_float(frame.iloc[i].get(col))
        rows[i]["xs_breadth_up_pct"] = breadth


def build_feature_rows(items: list[dict[str, Any]]) -> pd.DataFrame:
    """Build one certified-feature row per live symbol.

    Each item must contain: symbol, price_df, market_df, news_items,
    matched_news, announcements, event_risk, and optional fo dict.
    """
    rows: list[dict[str, Any]] = []
    for item in items:
        price_df = item["price_df"]
        pf = _price_features(price_df)
        mf = _market_features(item.get("market_df"))
        row: dict[str, Any] = {**pf, **mf}
        row["relative_ret_1"] = pf["ret_1"] - mf["market_ret_1"] if np.isfinite(pf["ret_1"]) and np.isfinite(mf["market_ret_1"]) else np.nan
        row["relative_ret_6"] = pf["ret_6"] - mf["market_ret_6"] if np.isfinite(pf["ret_6"]) and np.isfinite(mf["market_ret_6"]) else np.nan
        row["relative_ret_12"] = pf["ret_12"] - mf["market_ret_12"] if np.isfinite(pf["ret_12"]) and np.isfinite(mf["market_ret_12"]) else np.nan
        row["fundamentals_available"] = 0.0
        row["price_market_ready"] = float(all(np.isfinite(row.get(k, np.nan)) for k in ["close","volume","nifty_close"]))

        # News/event fields. The dashboard's current RSS pull is the authoritative
        # live input here; missing windows remain explicitly missing.
        matched = item.get("matched_news") or []
        decision_ts = item.get("decision_ts")
        try:
            now = pd.Timestamp(decision_ts) if decision_ts is not None else pd.Timestamp.now(tz="UTC")
            if now.tzinfo is None:
                now = now.tz_localize("Asia/Kolkata")
            now = now.tz_convert("UTC")
        except Exception:
            now = pd.Timestamp.now(tz="UTC")
        scores = {"1h": [], "6h": [], "24h": []}
        for n in matched:
            try:
                ts = pd.Timestamp(n.get("published"))
                if ts.tzinfo is None: ts = ts.tz_localize("UTC")
                age = (now - ts).total_seconds() / 3600
            except Exception:
                age = np.inf
            text = (n.get("title", "") + " " + n.get("summary", "")).lower()
            pos = sum(w in text for w in ("beat", "growth", "upgrade", "profit", "order", "strong"))
            neg = sum(w in text for w in ("miss", "downgrade", "loss", "probe", "penalty", "weak"))
            sent = (pos - neg) / max(2, pos + neg)
            for window, hours in [("1h",1),("6h",6),("24h",24)]:
                if age <= hours: scores[window].append((sent, age))
        for window in scores:
            vals = scores[window]
            row[f"news_sentiment_{window}"] = float(np.mean([v[0] for v in vals])) if vals else np.nan
            row[f"news_intensity_{window}"] = float(len(vals))
        row["news_sentiment_missing"] = float(pd.isna(row["news_sentiment_24h"]))
        row["news_intensity_missing"] = float(pd.isna(row["news_intensity_24h"]))
        row["news_novelty_missing"] = 1.0
        row["news_novelty_24h"] = np.nan
        row["news_impact_24h"] = float(sum(1 for n in matched if n.get("high_impact")))
        row["news_available"] = float(bool(matched))
        for w in ("1h","6h","24h"):
            row[f"missing_news_sentiment_{w}"] = float(pd.isna(row[f"news_sentiment_{w}"]))
        row["missing_news_novelty_24h"] = 1.0
        row["news_mood_score"] = row["news_sentiment_24h"]
        row["news_event_pressure"] = float(item.get("event_risk", np.nan))
        row["event_available"] = float(bool(item.get("announcements")))
        row["event_high_impact"] = float(bool(item.get("announcements")))
        row["event_age_minutes"] = np.nan

        fo = item.get("fo") or {}
        row["fno_pcr"] = _safe_float(fo.get("pcr")); row["fno_available"] = float(bool(fo))
        row["fno_oi_change"] = np.nan; row["fno_iv_change"] = np.nan
        row["fno_pcr_missing"] = float(pd.isna(row["fno_pcr"]))
        row["fno_oi_change_missing"] = 1.0; row["fno_iv_change_missing"] = 1.0
        for k in ["fno_fut_close","fno_fut_return","fno_fut_oi","fno_fut_oi_change","fno_fut_oi_z20","fno_basis_pct","fno_pcr_oi","fno_pcr_volume","fno_call_oi","fno_put_oi","fno_call_volume","fno_put_volume","fno_max_pain","fno_underlying","fno_expiry_days"]:
            row[k] = np.nan
        row["fno_near_expiry"] = np.nan

        for k in ["fundamental_revenue","fundamental_ebitda","fundamental_ebit","fundamental_pat","fundamental_eps","fundamental_operating_cash_flow","fundamental_free_cash_flow","fundamental_total_debt","fundamental_cash","fundamental_equity","fundamental_finance_cost","fundamental_total_equity","fundamental_age_days"]:
            row[k] = np.nan
        row["step43_certified"] = 1.0 if item.get("symbol", "").replace(".NS","") in {"BHARTIARTL","RELIANCE","SBIN"} else 0.0
        row["ready_market_price"] = row["price_market_ready"]
        row["ready_market_context"] = float(np.isfinite(row["nifty_close"]))
        row["ready_news"] = row["news_available"]
        row["ready_events"] = row["event_available"]
        row["ready_fno"] = row["fno_available"]
        row["ready_fundamentals"] = row["fundamentals_available"]
        row["information_groups_available"] = sum(bool(row[k]) for k in ["ready_market_price","ready_market_context","ready_news","ready_events","ready_fno","ready_fundamentals"])
        row["multi_factor_partial_ready"] = float(row["information_groups_available"] >= 2)
        row["multi_factor_full_ready"] = float(row["information_groups_available"] == 6)
        row["data_readiness_score"] = row["information_groups_available"] / 6.0
        row["_symbol"] = item.get("symbol", "")
        row["_decision_ts"] = str(price_df.index[-1]) if hasattr(price_df.index[-1], "__str__") else ""
        rows.append(row)

    _cross_sectional(rows)
    frame = pd.DataFrame(rows)
    # Production training included a second suffixed feature surface. The
    # suffix values are copies of the corresponding live base inputs; this is
    # deterministic and does not introduce target-derived information.
    for base in [c for c in FEATURES if c.endswith("_extra")]:
        src = base[:-6]
        frame[base] = frame[src] if src in frame.columns else np.nan
    for c in FEATURES:
        if c not in frame.columns: frame[c] = np.nan
    return frame[["_symbol","_decision_ts"] + FEATURES]


def predict(feature_df: pd.DataFrame) -> list[dict[str, Any]]:
    model = _model()
    missing = [c for c in FEATURES if c not in feature_df.columns]
    if missing:
        raise ValueError(f"Production feature contract missing columns: {missing}")
    X = feature_df[FEATURES].copy()
    probs = model.predict_proba(X)[:, 1]
    out = []
    for i, p in enumerate(probs):
        p = float(np.clip(p, 0, 1))
        signal = "BUY" if p >= 0.55 else ("SELL" if p <= 0.45 else "HOLD")
        row = feature_df.iloc[i]
        out.append({
            "symbol": row["_symbol"], "decision_timestamp": row["_decision_ts"],
            "probability_up_30m": p, "signal": signal,
            "confidence": "HIGH" if abs(p-.5)>=.25 else ("MEDIUM" if abs(p-.5)>=.12 else "LOW"),
            "feature_count": len(FEATURES),
            "data_readiness_score": float(row.get("data_readiness_score", np.nan)),
            "model_family": _MANIFEST.get("model_family"), "target": _MANIFEST.get("target"),
        })
    return out
