"""Scored, point-in-time news features.

Replaces the single `news_available` flag with sentiment, intensity, novelty and
impact features. The scorer is pluggable: the default is VADER plus an Indian
market tag lexicon (a weak baseline); on Colab GPU swap in a finance transformer
with `set_scorer(fn)` where fn(list[str]) -> list[float] in [-1, 1].

Archive honesty: historical headline feeds are not recoverable for free, so the
news archive only starts when the daily job starts. `archive_start` marks that date;
decisions before it get NaN (unknown), decisions after it with no news get 0 (known none).
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

POS = {
    r"\border (win|wins|inflow)|bags? (an )?order|wins? (an )?order": 0.35,
    r"record (profit|revenue|sales)|profit (rises|jumps|surges)|net profit (up|rises)": 0.35,
    r"\bupgrade[sd]?\b|target price (raised|hiked)": 0.30,
    r"\bbuyback\b|\bbonus\b|special dividend|\bdividend\b": 0.20,
    r"usfda approval|regulatory approval|capacity expansion|stake (buy|acquisition)": 0.25,
    r"debt[- ]free|debt reduction|rating (upgrade|upgraded)": 0.25,
}
NEG = {
    r"\bfraud\b|\bprobe\b|\braid[s]?\b|\bcbi\b|enforcement directorate|\bed\b raid": -0.45,
    r"sebi (order|show[- ]cause|penalty)|\bpenalt(y|ies)\b": -0.35,
    r"\bdowngrade[sd]?\b|target price (cut|slashed)|rating (cut|downgrade)": -0.30,
    r"net loss|loss widens|profit (falls|plunges|slumps|declines)": -0.35,
    r"\bdefault(s|ed)?\b|insolvency|\bnclt\b|going concern|promoter pledge": -0.45,
    r"resign(s|ed|ation)|warning letter|usfda observation|plant (shutdown|fire)": -0.30,
}
HIGH_IMPACT = re.compile(
    r"quarterly results|financial results|block deal|bulk deal|merger|acquisition|demerger|"
    r"open offer|delisting|fund ?raise|qip|rights issue|guidance|outcome of board meeting", re.I)

_POS = [(re.compile(k, re.I), v) for k, v in POS.items()]
_NEG = [(re.compile(k, re.I), v) for k, v in NEG.items()]
_vader = None
_custom: Optional[Callable[[List[str]], List[float]]] = None


def set_scorer(fn: Optional[Callable[[List[str]], List[float]]]) -> None:
    global _custom
    _custom = fn


def _vader_scores(texts: List[str]) -> List[float]:
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except Exception:
            _vader = False
    if _vader is False:
        return [0.0] * len(texts)
    return [_vader.polarity_scores(t)["compound"] for t in texts]


def score_texts(texts: List[str]) -> np.ndarray:
    texts = [str(t or "") for t in texts]
    base = np.array(_custom(texts) if _custom else _vader_scores(texts), dtype=float)
    tag = np.zeros(len(texts))
    for i, t in enumerate(texts):
        s = sum(w for rx, w in _POS if rx.search(t)) + sum(w for rx, w in _NEG if rx.search(t))
        tag[i] = s
    mix = 0.5 * base + tag if _custom is None else 0.7 * base + 0.5 * tag
    return np.clip(mix, -1.0, 1.0)


def _norm_hash(t: str) -> str:
    t = re.sub(r"[^a-z0-9 ]", "", str(t).lower())
    t = re.sub(r"\s+", " ", t).strip()[:160]
    return hashlib.md5(t.encode()).hexdigest()


def score_news_frame(news: pd.DataFrame) -> pd.DataFrame:
    """news columns: symbol, published_at, text. Returns with score, impact, hash."""
    n = news.copy()
    n["published_at"] = pd.to_datetime(n["published_at"], utc=True, errors="coerce")
    n = n.dropna(subset=["symbol", "published_at"]).reset_index(drop=True)
    n["score"] = score_texts(n["text"].tolist()) if len(n) else []
    n["impact"] = n["text"].astype(str).map(lambda t: 1.0 if HIGH_IMPACT.search(t) else 0.0)
    n["h"] = n["text"].map(_norm_hash)
    return n


def build_news_features(scored: pd.DataFrame, decisions: pd.DataFrame,
                        archive_start: Optional[str] = None,
                        halflife_sessions: float = 3.0) -> pd.DataFrame:
    """Join decayed news features onto `decisions` [symbol, decision_timestamp].

    An item is visible to the first decision whose timestamp >= published_at.
    Rows from MARKET (symbol == 'MARKET') become market-wide features.
    """
    d = decisions[["symbol", "decision_timestamp"]].copy()
    d["decision_timestamp"] = pd.to_datetime(
        d["decision_timestamp"], errors="coerce", utc=True
    ).astype("datetime64[ns, UTC]")
    cal = pd.DatetimeIndex(
        sorted(d["decision_timestamp"].dropna().unique())
    ).astype("datetime64[ns, UTC]").tz_convert("UTC")
    out = d.copy()
    feat_cols = ["news_cnt_1", "news_sent_1", "news_sent_ewm", "news_cnt_ewm",
                 "news_impact_ewm", "news_novelty_1", "news_available"]
    market_feat_cols = ["mkt_news_sent_ewm", "mkt_news_cnt_ewm"]
    if scored is None or scored.empty or len(cal) == 0:
        for c in feat_cols + market_feat_cols:
            out[c] = np.nan
        return out

    s = scored.copy()
    s["pub"] = pd.to_datetime(s["published_at"], utc=True)
    s = s.sort_values("pub")
    # novelty: first time this exact normalized headline is seen for the symbol
    s["new"] = (~s.duplicated(["symbol", "h"], keep="first")).astype(float)
    pos = cal.searchsorted(pd.DatetimeIndex(s["pub"]), side="left")
    s["bin"] = pos
    s = s[s["bin"] < len(cal)]  # published after the last decision: invisible

    # A fresh headline archive can contain only items published after the most
    # recent PIT decision (e.g. Saturday news after Friday's close). These rows
    # are correctly invisible to that snapshot. Preserve unknown history as NaN
    # rather than crashing on pd.concat([]) or inventing zero-news observations.
    if s.empty:
        for c in feat_cols + market_feat_cols:
            out[c] = np.nan
        return out

    def grid(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=[
                "symbol", "decision_timestamp", "news_cnt_1", "news_sent_1",
                "news_sent_ewm", "news_cnt_ewm", "news_impact_ewm",
                "news_novelty_1", "news_recent5",
            ])
        g = frame.groupby(["symbol", "bin"]).agg(
            cnt=("score", "size"), ssum=("score", "sum"),
            imp=("impact", "sum"), nov=("new", "mean")).reset_index()
        syms = g["symbol"].unique()
        idx = pd.MultiIndex.from_product([syms, np.arange(len(cal))], names=["symbol", "bin"])
        g = g.set_index(["symbol", "bin"]).reindex(idx)
        g[["cnt", "ssum", "imp"]] = g[["cnt", "ssum", "imp"]].fillna(0.0)
        parts = []
        for sym, gg in g.groupby(level=0, sort=False):
            gg = gg.copy()
            hl = halflife_sessions
            num = gg["ssum"].ewm(halflife=hl, adjust=False).mean()
            den = gg["cnt"].ewm(halflife=hl, adjust=False).mean()
            gg["news_sent_ewm"] = (num / den.replace(0, np.nan)).fillna(0.0)
            gg["news_cnt_ewm"] = den
            gg["news_impact_ewm"] = gg["imp"].ewm(halflife=hl, adjust=False).mean()
            gg["news_cnt_1"] = gg["cnt"]
            gg["news_sent_1"] = (gg["ssum"] / gg["cnt"].replace(0, np.nan))
            gg["news_novelty_1"] = gg["nov"]
            gg["news_recent5"] = gg["cnt"].rolling(5, min_periods=1).sum()
            parts.append(gg)
        r = pd.concat(parts).reset_index()
        r["decision_timestamp"] = cal[r["bin"].to_numpy()]
        return r

    stock = grid(s[s["symbol"] != "MARKET"])
    keep = ["symbol", "decision_timestamp", "news_cnt_1", "news_sent_1", "news_sent_ewm",
            "news_cnt_ewm", "news_impact_ewm", "news_novelty_1", "news_recent5"]
    out = out.merge(stock[keep], on=["symbol", "decision_timestamp"], how="left")
    # symbols with no news at all in the archive are "known none" after archive start
    out["news_available"] = (out["news_recent5"].fillna(0) > 0).astype(float)
    for c in ("news_cnt_1", "news_cnt_ewm", "news_impact_ewm", "news_sent_ewm"):
        out[c] = out[c].fillna(0.0)
    out = out.drop(columns=["news_recent5"])

    if (s["symbol"] == "MARKET").any():
        m = grid(s[s["symbol"] == "MARKET"])[["decision_timestamp", "news_sent_ewm", "news_cnt_ewm"]]
        m = m.rename(columns={"news_sent_ewm": "mkt_news_sent_ewm", "news_cnt_ewm": "mkt_news_cnt_ewm"})
        out = out.merge(m, on="decision_timestamp", how="left")

    if archive_start is not None:
        t0 = pd.Timestamp(archive_start, tz="UTC")
        before = out["decision_timestamp"] < t0
        for c in out.columns:
            if c.startswith(("news_", "mkt_news_")):
                out.loc[before, c] = np.nan
    return out


def coverage(features: pd.DataFrame, col: str = "news_sent_ewm") -> float:
    """Share of decision rows where the news group is known (not NaN)."""
    return float(features[col].notna().mean()) if len(features) else 0.0
