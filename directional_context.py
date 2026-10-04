"""Causal multi-source directional context for Delivery predictions.

This module does NOT modify the locked 59-feature ALL-NSE production model.
It provides a transparent evidence layer that can be validated alongside the
model before any future retraining/promotion decision.

Inputs are restricted to information available at or before the supplied
as-of timestamp:
- stock price/technical evidence
- stock-specific news sentiment
- official NSE corporate announcements
- broad-market/benchmark regime

The output is deliberately separate from the production model probability.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from signals import combined_signal, market_regime_score, news_metrics, technical_score
from triggers import announcement_to_news, event_risk_score, tag_news_items
from historical_replay import historical_announcements, historical_news


def _label(score: float) -> str:
    if not np.isfinite(score):
        return "UNKNOWN"
    if score >= 0.25:
        return "UP"
    if score <= -0.25:
        return "DOWN"
    return "NEUTRAL"


def _safe_float(value: Any) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _normalize_market(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    x = frame.copy()
    x.columns = [str(c).strip() for c in x.columns]
    rename = {}
    for c in x.columns:
        k = str(c).strip().lower()
        if k in {"open", "o"}:
            rename[c] = "Open"
        elif k in {"high", "h"}:
            rename[c] = "High"
        elif k in {"low", "l"}:
            rename[c] = "Low"
        elif k in {"close", "c", "ltp"}:
            rename[c] = "Close"
        elif k in {"volume", "vol", "v"}:
            rename[c] = "Volume"
    x = x.rename(columns=rename)
    if "Close" not in x.columns:
        return pd.DataFrame()
    if "Open" not in x.columns:
        x["Open"] = x["Close"]
    if "High" not in x.columns:
        x["High"] = x["Close"]
    if "Low" not in x.columns:
        x["Low"] = x["Close"]
    if "Volume" not in x.columns:
        x["Volume"] = np.nan
    return x[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])


def analyze_directional_context(
    symbol: str,
    as_of: pd.Timestamp,
    stock_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
    model_p_up: float | None = None,
    model_expected_return: float | None = None,
) -> dict[str, Any]:
    """Analyze independent directional evidence using only point-in-time data."""
    as_of = pd.Timestamp(as_of)

    stock = stock_history.copy() if stock_history is not None else pd.DataFrame()
    if not stock.empty and "date" in stock.columns:
        stock["date"] = pd.to_datetime(stock["date"], errors="coerce").dt.normalize()
        stock = stock[stock["date"] <= as_of.normalize()].copy()

    market = _normalize_market(benchmark_history)
    if not market.empty:
        idx = pd.to_datetime(market.index, errors="coerce")
        market.index = idx
        market = market[market.index <= as_of]

    # Historical news is best-effort and is never allowed to use future items.
    raw_news = historical_news(symbol, as_of)
    raw_announcements = historical_announcements(symbol, as_of)
    tagged_news = tag_news_items(raw_news)
    official_news = announcement_to_news(raw_announcements)
    all_news = tagged_news + official_news

    news = news_metrics(all_news, now=as_of.to_pydatetime())
    event_risk = event_risk_score(tagged_news, raw_announcements)

    technical = 0.0
    if not stock.empty:
        tech_frame = stock.rename(columns={
            "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
        })
        try:
            from signals import add_indicators
            tech_frame = add_indicators(tech_frame)
            technical = _safe_float(technical_score(tech_frame))
        except Exception:
            technical = 0.0

    market_score = 0.0
    if not market.empty:
        try:
            market_score = _safe_float(market_regime_score(market))
        except Exception:
            market_score = 0.0

    context_label, context_score = combined_signal(
        technical_score_value=0.0 if not np.isfinite(technical) else technical,
        news_score=0.0 if not np.isfinite(news["score"]) else news["score"],
        market_score=0.0 if not np.isfinite(market_score) else market_score,
        event_risk=event_risk,
        tech_weight=0.45,
        news_weight=0.30,
        market_weight=0.25,
    )

    model_score = np.nan
    if model_p_up is not None:
        p = _safe_float(model_p_up)
        if np.isfinite(p):
            model_score = 2.0 * p - 1.0

    # Transparent ensemble of the locked ML probability and independent context
    # evidence. This is an evidence diagnostic, not a replacement for the
    # production classifier and does not feed back into its 59 features.
    fused_score = np.nan
    if np.isfinite(model_score) and np.isfinite(context_score):
        fused_score = 0.70 * model_score + 0.30 * float(context_score)

    source_names = []
    for item in tagged_news:
        source = str(item.get("source", "")).strip()
        if source:
            source_names.append(source)
    source_names = sorted(set(source_names))

    return {
        "as_of": as_of.isoformat(),
        "stock_news_count": len(tagged_news),
        "official_nse_announcement_count": len(raw_announcements),
        "news_source_count": len(source_names),
        "news_sources": source_names,
        "news_score": _safe_float(news.get("score")),
        "news_impact": _safe_float(news.get("impact")),
        "high_impact_news_count": int(news.get("high_impact", 0) or 0),
        "event_risk": _safe_float(event_risk),
        "technical_score": _safe_float(technical),
        "market_regime_score": _safe_float(market_score),
        "market_regime": _label(market_score),
        "context_score": _safe_float(context_score),
        "context_direction": _label(context_score),
        "model_p_up": _safe_float(model_p_up),
        "model_expected_return": _safe_float(model_expected_return),
        "model_direction": _label(model_score),
        "fused_score": _safe_float(fused_score),
        "fused_direction": _label(fused_score),
        "model_context_agreement": (
            _label(model_score) == _label(context_score)
            if np.isfinite(model_score) and np.isfinite(context_score)
            else None
        ),
        "status": "OK",
    }
