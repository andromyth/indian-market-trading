"""Point-in-time historical prediction replay for the Indian-market dashboard.

The replay engine reconstructs a historical information set using only data at or
before the selected timestamp. It then compares the prediction with subsequent
actual market data. It deliberately labels unavailable historical information
(news/F&O/corporate data) rather than substituting today's values.
"""
from __future__ import annotations

import io
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from data_fetcher import (
    NSE_BASE,
    _nse_get,
    benchmark_for_symbol,
    stock_news_keywords,
)
from free_data import (
    EQUITY_2018_2025,
    EQUITY_2026,
    load_local_equity,
    normalize_root_symbol,
    resample_daily,
    resample_intraday,
    INDEX_RELEASE_URL,
)
from signals import combined_signal, market_regime_score, news_metrics, technical_score
from triggers import announcement_to_news, event_risk_score

REPLAY_CACHE = Path(__file__).resolve().parent / "data" / "replay_cache"
REPLAY_CACHE.mkdir(parents=True, exist_ok=True)

EQUITY_REPLAY_URLS = {
    "2018_2025": "https://raw.githubusercontent.com/ganeshbiyer/Nse_Historical_Data/main/{symbol}.parquet",
    "2026": "https://raw.githubusercontent.com/ganeshbiyer/Nse_Historical_Data_2026/main/{symbol}.parquet",
}


def _date_only(ts: pd.Timestamp | datetime | date) -> pd.Timestamp:
    x = pd.Timestamp(ts)
    if x.tzinfo is not None:
        x = x.tz_convert("Asia/Kolkata").tz_localize(None)
    return x.normalize()


def _ensure_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    idx = pd.to_datetime(x.index, errors="coerce", utc=False)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    x.index = pd.DatetimeIndex(idx)
    return x[~x.index.isna()].sort_index()


def _remote_equity(symbol: str, year_source: str) -> pd.DataFrame:
    root = normalize_root_symbol(symbol)
    path = REPLAY_CACHE / f"{year_source}_{root}.parquet"
    if not path.exists() or path.stat().st_size == 0:
        url = EQUITY_REPLAY_URLS[year_source].format(symbol=root)
        r = requests.get(url, timeout=(15, 180), headers={"User-Agent": "IndianStockSignalDashboard/1.0"})
        r.raise_for_status()
        path.write_bytes(r.content)
    return _ensure_naive_index(pd.read_parquet(path))


def historical_benchmark(symbol: str, end: pd.Timestamp) -> pd.DataFrame:
    """Load historical broad-market context without using future bars."""
    if symbol == "^NSEI":
        path = REPLAY_CACHE / "NIFTY_5min_5yr_2021_2026.csv"
        if not path.exists() or path.stat().st_size == 0:
            # Public index archive is optional. A failed download must not abort
            # the entire replay; the production model can explicitly handle
            # missing market-context features through its trained imputer.
            try:
                r = requests.get(
                    INDEX_RELEASE_URL,
                    timeout=(15, 180),
                    headers={"User-Agent": "IndianStockSignalDashboard/1.0"},
                )
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".part")
                tmp.write_bytes(r.content)
                tmp.replace(path)
            except requests.RequestException:
                pass
        try:
            x = pd.read_csv(path)
            rename = {}
            for c in x.columns:
                lc = str(c).strip().lower().replace(" ", "_")
                if lc in {"timestamp","datetime","date","time","dt"}: rename[c] = "_time"
                elif lc in {"open","o"}: rename[c] = "Open"
                elif lc in {"high","h"}: rename[c] = "High"
                elif lc in {"low","l"}: rename[c] = "Low"
                elif lc in {"close","c","ltp"}: rename[c] = "Close"
                elif lc in {"volume","vol","v"}: rename[c] = "Volume"
            x = x.rename(columns=rename)
            if "_time" in x.columns:
                idx = pd.to_datetime(x.pop("_time"), errors="coerce", utc=False)
            else:
                idx = pd.to_datetime(x.index, errors="coerce", utc=False)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
            for c in ["Open","High","Low","Close"]:
                if c not in x.columns: x[c] = np.nan
                x[c] = pd.to_numeric(x[c], errors="coerce")
            if "Volume" not in x.columns: x["Volume"] = 0.0
            x.index = pd.DatetimeIndex(idx)
            x = x[["Open","High","Low","Close","Volume"]].dropna(subset=["Open","High","Low","Close"])
            x = x[~x.index.isna()].sort_index()
            return x[x.index <= pd.Timestamp(end)]
        except Exception:
            pass
    try:
        from data_fetcher import get_historical
        return get_historical(symbol, period="5y", interval="5m")
    except Exception:
        return pd.DataFrame()


def historical_price(symbol: str, start: pd.Timestamp, end: pd.Timestamp, interval: str = "1m") -> pd.DataFrame:
    """Local-first historical loader, falling back to public 1-minute archives.

    For dates in 2026 the current-year archive is used; earlier dates use the
    2018-2025 archive. The loader only returns rows through `end`.
    """
    root = normalize_root_symbol(symbol)
    start = _date_only(start)
    end = pd.Timestamp(end)
    try:
        local = load_local_equity(root, start=start, end=end, interval="1m")
    except Exception:
        local = pd.DataFrame()
    if not local.empty:
        x = local.copy()
    else:
        years = []
        if start.year <= 2025 <= end.year:
            years.append("2018_2025")
        if start.year <= 2026 <= end.year or end.year >= 2026:
            years.append("2026")
        if not years:
            years = ["2018_2025"]
        frames = []
        for src in years:
            try:
                frames.append(_remote_equity(root, src))
            except Exception:
                continue
        x = pd.concat(frames).sort_index() if frames else pd.DataFrame()
        if not x.empty:
            x = x[~x.index.duplicated(keep="last")]
    if x.empty:
        return x
    x = x[(x.index >= start) & (x.index <= end)]
    if interval in {"1m", "1min"}:
        return x
    if interval in {"5m", "5min"}:
        return resample_intraday(x, "5min")
    if interval in {"1d", "D", "day", "daily"}:
        return resample_daily(x)
    raise ValueError(f"Unsupported replay interval: {interval}")


def next_trading_row(df: pd.DataFrame, ts: pd.Timestamp, rows_forward: int = 1) -> pd.Series | None:
    x = df[df.index > ts]
    if x.empty:
        return None
    idx = min(max(0, rows_forward - 1), len(x) - 1)
    return x.iloc[idx]


def _google_news_rss(url: str) -> list[dict]:
    try:
        r = requests.get(url, timeout=(10, 25), headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception:
        return []
    try:
        import feedparser
        feed = feedparser.parse(r.content)
    except Exception:
        return []
    rows = []
    for e in feed.entries[:80]:
        rows.append({
            "title": str(e.get("title", "")).strip(),
            "summary": str(e.get("summary", "")),
            "link": str(e.get("link", "")),
            "published": str(e.get("published", "") or e.get("updated", "")),
            "source": str(feed.feed.get("title", "Google News RSS")),
        })
    return rows


def historical_news(symbol: str, when: pd.Timestamp) -> list[dict]:
    """Best-effort point-in-time headline retrieval from Google News RSS.

    This is not an exchange-authoritative historical news archive; the UI labels
    it accordingly and the replay never substitutes today's news.
    """
    root = normalize_root_symbol(symbol)
    day = _date_only(when)
    day2 = day + pd.Timedelta(days=1)
    query_terms = stock_news_keywords(root)
    q = ' OR '.join(f'"{x}"' for x in query_terms[:5])
    q = f"({q}) after:{day.date().isoformat()} before:{day2.date().isoformat()}"
    from urllib.parse import quote_plus
    url = (
        "https://news.google.com/rss/search?q=" + quote_plus(q)
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )
    rows = _google_news_rss(url)
    cutoff = pd.Timestamp(when)
    out = []
    for n in rows:
        try:
            ts = pd.Timestamp(n.get("published"))
            if ts.tzinfo is not None:
                ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
            if ts <= cutoff:
                n["published_ts"] = ts
                out.append(n)
        except Exception:
            # Keep headlines with no parseable timestamp only for Delivery/EOD
            # replay, where the exact publication time cannot be reconstructed.
            if pd.Timestamp(when).time().hour >= 15:
                out.append(n)
    return out


def historical_announcements(symbol: str, when: pd.Timestamp) -> list[dict]:
    """Best-effort filter of NSE announcement data when the endpoint exposes the date."""
    try:
        from data_fetcher import get_corporate_announcements
        raw = get_corporate_announcements(normalize_root_symbol(symbol))
    except Exception:
        return []
    cutoff_date = _date_only(when).date()
    out = []
    for item in raw or []:
        text_date = item.get("an_dt") or item.get("date") or item.get("sort_date") or ""
        try:
            d = pd.Timestamp(text_date).date()
        except Exception:
            continue
        if d <= cutoff_date:
            out.append({
                "subject": item.get("subject") or item.get("desc") or item.get("attchmntText") or "Corporate announcement",
                "date": text_date,
                "link": item.get("attchmntFile") or item.get("link") or "",
            })
    return out


def _entry_exit_metrics(df: pd.DataFrame, decision_ts: pd.Timestamp, mode: str, horizon: int) -> dict:
    """Calculate reference/target/stop from information available at decision time."""
    hist = df[df.index <= decision_ts]
    if hist.empty:
        return {}
    row = hist.iloc[-1]
    entry = float(row["Close"])
    h = pd.to_numeric(hist["High"], errors="coerce")
    l = pd.to_numeric(hist["Low"], errors="coerce")
    c = pd.to_numeric(hist["Close"], errors="coerce")
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().iloc[-1]) if len(hist) >= 14 else np.nan
    if not np.isfinite(atr) or atr <= 0:
        atr = entry * (0.006 if mode == "Intraday" else 0.02)
    if mode == "Intraday":
        stop_dist, target_dist = 1.2 * atr, 2.0 * atr
    else:
        stop_dist, target_dist = 1.8 * atr, 3.0 * atr
    return {
        "entry_reference": entry,
        "atr": atr,
        "buy_zone_low": entry - 0.25 * atr,
        "buy_zone_high": entry + 0.25 * atr,
        "target_distance": target_dist,
        "stop_distance": stop_dist,
        "horizon": horizon,
    }


def actual_outcome(price_df: pd.DataFrame, decision_ts: pd.Timestamp, signal: str, mode: str, horizon: int, entry_info: dict) -> dict:
    entry = float(entry_info.get("entry_reference", np.nan))
    if mode == "Intraday":
        future = price_df[price_df.index > decision_ts]
        # Approximate 30 one-minute bars; require a timestamp at least 30 minutes after the decision.
        target_ts = decision_ts + pd.Timedelta(minutes=horizon)
        exact = price_df[price_df.index >= target_ts]
        if exact.empty:
            return {"status": "INSUFFICIENT_FUTURE_DATA"}
        future30 = exact.iloc[0]
        actual_price = float(future30["Close"])
        move = actual_price / entry - 1 if entry else np.nan
        direction = "UP" if move > 0 else ("DOWN" if move < 0 else "FLAT")
        target = entry + entry_info["target_distance"] if signal == "BUY" else entry - entry_info["target_distance"]
        stop = entry - entry_info["stop_distance"] if signal == "BUY" else entry + entry_info["stop_distance"]
        target_hit = False; stop_hit = False; first_touch = None
        if signal in {"BUY", "SELL"}:
            for idx, r in future[future.index <= target_ts].iterrows():
                hi, lo = float(r["High"]), float(r["Low"])
                if signal == "BUY":
                    if hi >= target: target_hit = True
                    if lo <= stop: stop_hit = True
                else:
                    if lo <= target: target_hit = True
                    if hi >= stop: stop_hit = True
                if target_hit or stop_hit:
                    touches = []
                    if target_hit: touches.append((idx, "TARGET"))
                    if stop_hit: touches.append((idx, "STOP"))
                    first_touch = sorted(touches, key=lambda x: x[0])[0][1]
                    break
        correct = None
        if signal == "BUY": correct = move > 0
        elif signal == "SELL": correct = move < 0
        return {
            "status": "OK", "actual_price": actual_price, "actual_move_pct": move * 100,
            "actual_direction": direction, "target_hit": target_hit, "stop_hit": stop_hit,
            "first_touch": first_touch, "correct_direction": correct,
            "horizon_label": f"{horizon}m",
        }

    daily = resample_daily(price_df)
    base = daily[daily.index.normalize() <= _date_only(decision_ts)]
    future = daily[daily.index > _date_only(decision_ts)]
    if base.empty or len(future) < horizon:
        return {"status": "INSUFFICIENT_FUTURE_DATA"}
    first = future.iloc[0]
    actual = future.iloc[horizon - 1]
    entry_price = float(first["Open"])
    actual_close = float(actual["Close"])
    move = actual_close / entry_price - 1 if entry_price else np.nan
    direction = "UP" if move > 0 else ("DOWN" if move < 0 else "FLAT")
    target = entry_price + entry_info["target_distance"] if signal == "BUY" else entry_price - entry_info["target_distance"]
    stop = entry_price - entry_info["stop_distance"] if signal == "BUY" else entry_price + entry_info["stop_distance"]
    target_hit = False; stop_hit = False; first_touch = None
    for idx, r in future.iloc[:horizon].iterrows():
        hi, lo = float(r["High"]), float(r["Low"])
        if signal == "BUY":
            if hi >= target: target_hit = True
            if lo <= stop: stop_hit = True
        elif signal == "SELL":
            if lo <= target: target_hit = True
            if hi >= stop: stop_hit = True
        if target_hit or stop_hit:
            choices = []
            if target_hit: choices.append((idx, "TARGET"))
            if stop_hit: choices.append((idx, "STOP"))
            first_touch = sorted(choices, key=lambda x: x[0])[0][1]
            break
    correct = None
    if signal == "BUY": correct = move > 0
    elif signal == "SELL": correct = move < 0
    return {
        "status": "OK", "actual_entry": entry_price, "actual_price": actual_close,
        "actual_move_pct": move * 100, "actual_direction": direction,
        "target_hit": target_hit, "stop_hit": stop_hit, "first_touch": first_touch,
        "correct_direction": correct, "horizon_label": f"{horizon} trading day(s)",
    }


def replay_rule_delivery(
    symbol: str,
    price_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    decision_ts: pd.Timestamp,
    tech_weight: float = 0.60,
    news_weight: float = 0.20,
    market_weight: float = 0.20,
    horizon_days: int = 5,
) -> dict:
    hist_1m = price_df[price_df.index <= decision_ts].copy()
    hist = resample_daily(hist_1m)
    mkt = benchmark_df[benchmark_df.index <= decision_ts].copy()
    if hist.empty or len(hist) < 55:
        raise ValueError("Not enough historical daily bars before the selected date.")
    t_score = technical_score(hist, fast=20, slow=50)
    m_score = market_regime_score(mkt, fast=20, slow=50) if not mkt.empty else 0.0
    # Historical news is fetched separately; when unavailable, the news component is explicitly zero/not-ready.
    return {"t_score": t_score, "m_score": m_score, "news_score": 0.0,
            "news_ready": False, "horizon_days": horizon_days}
