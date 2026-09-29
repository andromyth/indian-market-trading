"""Evidence-weighted stock signal engine for Indian cash-equity use.

The engine remains rule-based. It does not claim to forecast the future with
certainty. It combines:

* price/trend/momentum/volatility/volume evidence
* market-regime context (NIFTY/BANKNIFTY)
* stock-specific news sentiment and high-impact event tags
* trigger/event risk flags

The live signal is deliberately separated from the backtest. Backtests use
only information available on or before the decision bar and enter on the
next bar, avoiding the old same-close look-ahead bias.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math

import pandas as pd
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _analyzer = SentimentIntensityAnalyzer()
except Exception:
    _analyzer = None

_FALLBACK_POSITIVE = {
    "beat", "beats", "strong", "growth", "profit", "profits", "surge",
    "upgrade", "raised", "record", "order win", "contract win", "buyback",
    "bonus", "dividend", "acquisition", "positive", "recovery",
}
_FALLBACK_NEGATIVE = {
    "miss", "misses", "weak", "loss", "losses", "decline", "fall", "falls",
    "downgrade", "cut", "penalty", "probe", "fraud", "resign", "warning",
    "default", "litigation", "negative",
}

BULLISH_TRIGGER_WORDS = {
    "order win", "contract win", "guidance raised", "upgrade", "buyback",
    "bonus issue", "dividend", "acquisition", "merger", "stake increase",
    "profit rises", "profit growth", "record profit", "strong results",
    "beats estimates", "beats expectations",
}
BEARISH_TRIGGER_WORDS = {
    "guidance cut", "downgrade", "penalty", "fraud", "probe", "raid",
    "show cause", "resignation", "steps down", "profit warning",
    "profit falls", "loss widens", "stake sale", "default", "litigation",
    "regulatory action", "weak results", "misses estimates", "misses expectations",
}


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance/NSE/Dhan OHLCV responses to flat Series columns."""
    x = df.copy()
    if x.empty:
        return x
    wanted = {"open":"Open", "high":"High", "low":"Low", "close":"Close", "volume":"Volume", "adj close":"Adj Close"}
    if isinstance(x.columns, pd.MultiIndex):
        data = {}
        used = set()
        for target, canonical in wanted.items():
            chosen = None
            for col in x.columns:
                parts = [str(v).strip().lower() for v in (col if isinstance(col, tuple) else (col,))]
                if target in parts:
                    chosen = col
                    if str(col[0]).strip().lower() == target:
                        break
            if chosen is not None and canonical not in used:
                data[canonical] = x[chosen]
                used.add(canonical)
        x = pd.DataFrame(data, index=x.index)
    else:
        data = {}
        for col in x.columns:
            key = str(col).strip().lower().replace("_", " ")
            canonical = wanted.get(key)
            if canonical and canonical not in data:
                data[canonical] = x[col]
        if data:
            x = pd.DataFrame(data, index=x.index)
    if "Close" not in x.columns and "Adj Close" in x.columns:
        x["Close"] = x["Adj Close"]
    return x


def add_indicators(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    rsi_period: int = 14,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Add causal indicators. Current-row indicators use current/past data only."""
    df = _normalize_ohlcv_columns(df)
    if df.empty:
        return df
    missing = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c not in df.columns]
    if missing:
        raise ValueError(f"History is missing required OHLCV columns: {missing}")

    df[f"SMA{fast}"] = df["Close"].rolling(fast, min_periods=fast).mean()
    df[f"SMA{slow}"] = df["Close"].rolling(slow, min_periods=slow).mean()
    df[f"EMA{fast}"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df[f"EMA{slow}"] = df["Close"].ewm(span=slow, adjust=False).mean()
    df[f"RSI{rsi_period}"] = _rsi(df["Close"], rsi_period)

    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df[f"ATR{atr_period}"] = tr.ewm(alpha=1 / atr_period, adjust=False, min_periods=atr_period).mean()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    vol_window = max(20, fast)
    df["VolAvg"] = df["Volume"].rolling(vol_window, min_periods=vol_window).mean()
    df["VolumeRatio"] = df["Volume"] / df["VolAvg"].replace(0, float("nan"))

    breakout_window = max(slow, fast * 2)
    df["RollingHigh"] = df["High"].rolling(breakout_window, min_periods=breakout_window).max().shift(1)
    df["RollingLow"] = df["Low"].rolling(breakout_window, min_periods=breakout_window).min().shift(1)
    df["Return1"] = df["Close"].pct_change()
    df["Return5"] = df["Close"].pct_change(5)
    df["TrendSlope"] = df[f"EMA{fast}"].pct_change(5)
    return df


def market_regime_score(market_df: pd.DataFrame, fast: int = 20, slow: int = 50) -> float:
    """Score broad-market regime from -1 to +1, using the latest completed bar."""
    if market_df is None or market_df.empty:
        return 0.0
    m = add_indicators(market_df, fast=fast, slow=slow)
    last = m.iloc[-1]
    needed = [f"SMA{slow}", f"RSI14", "MACD_hist"]
    if any(pd.isna(last.get(c)) for c in needed):
        return 0.0

    score = 0.0
    if last["Close"] > last[f"SMA{slow}"]:
        score += 0.35
    else:
        score -= 0.35
    if last[f"SMA{fast}"] > last[f"SMA{slow}"]:
        score += 0.30
    else:
        score -= 0.30
    if last["MACD_hist"] > 0:
        score += 0.20
    else:
        score -= 0.20
    rsi = float(last["RSI14"])
    if rsi > 55:
        score += 0.15
    elif rsi < 45:
        score -= 0.15
    return max(-1.0, min(1.0, score))


def technical_components(df: pd.DataFrame, fast: int = 20, slow: int = 50, rsi_period: int = 14) -> dict:
    """Return interpretable technical components, each scaled roughly -1..+1."""
    if df is None or df.empty:
        return {"trend": 0.0, "momentum": 0.0, "volume": 0.0, "breakout": 0.0, "volatility": 0.0}
    needed = [f"SMA{fast}", f"SMA{slow}", f"RSI{rsi_period}", "VolumeRatio", "MACD_hist", "ATR14", "RollingHigh", "RollingLow"]
    if any(c not in df.columns for c in needed):
        return {"trend": 0.0, "momentum": 0.0, "volume": 0.0, "breakout": 0.0, "volatility": 0.0}
    last = df.iloc[-1]
    if pd.isna(last[f"SMA{slow}"]) or pd.isna(last[f"RSI{rsi_period}"]):
        return {"trend": 0.0, "momentum": 0.0, "volume": 0.0, "breakout": 0.0, "volatility": 0.0}

    trend = 0.5 if last[f"SMA{fast}"] > last[f"SMA{slow}"] else -0.5
    if last["MACD_hist"] > 0:
        trend += 0.25
    else:
        trend -= 0.25
    if last["TrendSlope"] > 0:
        trend += 0.25
    else:
        trend -= 0.25

    rsi = float(last[f"RSI{rsi_period}"])
    momentum = max(-1.0, min(1.0, (rsi - 50.0) / 20.0))
    # Avoid treating extreme overbought as a blind sell; just reduce long bias.
    if rsi > 72:
        momentum *= 0.35
    elif rsi < 28:
        momentum *= 0.35

    vr = float(last["VolumeRatio"]) if pd.notna(last["VolumeRatio"]) else 1.0
    volume = 0.0
    if vr >= 1.8:
        volume = 0.8 if momentum >= 0 else -0.8
    elif vr >= 1.3:
        volume = 0.4 if momentum >= 0 else -0.4

    breakout = 0.0
    if pd.notna(last["RollingHigh"]) and last["Close"] > last["RollingHigh"]:
        breakout = 1.0
    elif pd.notna(last["RollingLow"]) and last["Close"] < last["RollingLow"]:
        breakout = -1.0

    atr = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0.0
    volatility = 0.0
    if atr > 0 and pd.notna(last.get("Return1")):
        # Very large bars are risk warnings rather than automatically bullish/bearish.
        bar_atr = abs(float(last["Return1"])) * float(last["Close"]) / atr
        if bar_atr >= 2.0:
            volatility = -0.5

    return {
        "trend": max(-1.0, min(1.0, trend)),
        "momentum": momentum,
        "volume": volume,
        "breakout": breakout,
        "volatility": volatility,
    }


def technical_score(df: pd.DataFrame, fast: int = 20, slow: int = 50, rsi_period: int = 14) -> float:
    c = technical_components(df, fast, slow, rsi_period)
    score = 0.40 * c["trend"] + 0.25 * c["momentum"] + 0.15 * c["volume"] + 0.15 * c["breakout"] + 0.05 * c["volatility"]
    return round(max(-1.0, min(1.0, score)), 4)


def _parse_published(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.to_pydatetime()
    except Exception:
        return None


def _keyword_bias(text: str) -> float:
    t = text.lower()
    bull = sum(1 for k in BULLISH_TRIGGER_WORDS if k in t)
    bear = sum(1 for k in BEARISH_TRIGGER_WORDS if k in t)
    if bull == bear == 0:
        return 0.0
    return max(-1.0, min(1.0, (bull - bear) / max(1, bull + bear)))


def news_metrics(news_items: list[dict], now: datetime | None = None) -> dict:
    """Recency/impact-weighted news score plus trigger and freshness metrics."""
    if not news_items:
        return {"score": 0.0, "impact": 0.0, "count": 0, "high_impact": 0, "fresh_hours": None}

    now = now or datetime.now(timezone.utc)
    weighted_sum = 0.0
    weights = 0.0
    impact_sum = 0.0
    freshest = None
    high_impact = 0

    for item in news_items:
        title = item.get("title", "")
        summary = item.get("summary", "")
        text = f"{title} {summary}".strip()
        if _analyzer is not None:
            sent = _analyzer.polarity_scores(title)["compound"]
        else:
            words = set(title.lower().replace(",", " ").replace(".", " ").split())
            pos = sum(1 for w in _FALLBACK_POSITIVE if w in title.lower() or w in words)
            neg = sum(1 for w in _FALLBACK_NEGATIVE if w in title.lower() or w in words)
            sent = max(-1.0, min(1.0, (pos - neg) / max(2.0, pos + neg)))
        kw = _keyword_bias(text)
        base = 0.65 * sent + 0.35 * kw

        tags = item.get("tags", [])
        impact = 1.75 if item.get("high_impact") else 1.0
        if tags and len(tags) >= 2:
            impact = 2.0
        published = _parse_published(item.get("published"))
        age_hours = 48.0
        if published is not None:
            age_hours = max(0.0, (now - published).total_seconds() / 3600.0)
            if freshest is None or age_hours < freshest:
                freshest = age_hours
        recency = math.exp(-age_hours / 24.0)
        w = impact * max(0.12, recency)

        weighted_sum += base * w
        weights += w
        impact_sum += 1.0 if item.get("high_impact") else 0.0
        high_impact += 1 if item.get("high_impact") else 0

    score = weighted_sum / weights if weights else 0.0
    impact_score = min(1.0, impact_sum / max(3.0, len(news_items) * 0.5))
    return {
        "score": round(max(-1.0, min(1.0, score)), 4),
        "impact": round(impact_score, 4),
        "count": len(news_items),
        "high_impact": high_impact,
        "fresh_hours": round(freshest, 1) if freshest is not None else None,
    }


def sentiment_score(news_items: list[dict]) -> float:
    return news_metrics(news_items)["score"]


def combined_signal(
    tech_score_value: float,
    news_score: float,
    market_score: float = 0.0,
    event_risk: float = 0.0,
    tech_weight: float = 0.55,
    news_weight: float = 0.25,
    market_weight: float = 0.20,
) -> tuple[str, float]:
    """Blend independent evidence streams into a directional flag.

    event_risk reduces confidence; it is not automatically directional.
    """
    denom = max(1e-9, tech_weight + news_weight + market_weight)
    combined = (
        tech_weight * tech_score_value
        + news_weight * news_score
        + market_weight * market_score
    ) / denom
    # High event risk dampens conviction rather than inventing a direction.
    combined *= (1.0 - 0.35 * max(0.0, min(1.0, event_risk)))
    if combined >= 0.25:
        label = "BUY"
    elif combined <= -0.25:
        label = "SELL"
    else:
        label = "HOLD"
    return label, round(max(-1.0, min(1.0, combined)), 4)
