"""Market/news data adapters used by the dashboard.

NSE is the primary data layer for NSE-listed equities. Dhan may be used for
execution-grade live fields only when configured, while public providers are
reference-only fallbacks and are never silently labelled as NSE data.

No order execution is included.
"""
from __future__ import annotations

import os
import datetime as dt
from zoneinfo import ZoneInfo

try:
    from nse_data_layer import fetch_equity_history, fetch_intraday_1m_current, fetch_quote
except Exception:
    fetch_equity_history = fetch_intraday_1m_current = fetch_quote = None

import feedparser
import requests
import pandas as pd

try:
    from dhan_client import DhanConfig, historical_daily, historical_intraday, resolve_equity_symbol, get_access_token
except Exception:
    DhanConfig = None
    historical_daily = historical_intraday = resolve_equity_symbol = get_access_token = None

NEWS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.moneycontrol.com/rss/business.xml",
]

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)
BENCHMARK_TICKER = "^NSEI"

INDEX_TICKER_MAP = {
    "^NSEI": "NIFTY",
    "^NSEBANK": "BANKNIFTY",
}

# Generic aliases help stock-news matching when publishers use the company name
# instead of the exchange ticker. Extend this map as needed.
SYMBOL_ALIASES = {
    "BHARTIARTL": ["bharti airtel", "airtel", "bharti airtel ltd", "bharti airtel limited"],
    "RELIANCE": ["reliance industries", "reliance industries ltd", "ril"],
    "TCS": ["tata consultancy services", "tcs ltd", "tcs"],
    "INFY": ["infosys", "infosys ltd"],
    "HDFCBANK": ["hdfc bank", "hdfc bank ltd"],
    "ICICIBANK": ["icici bank", "icici bank ltd"],
    "SBIN": ["state bank of india", "sbi", "sbi bank"],
    "ITC": ["itc ltd", "itc limited", "itc"],
}

NSE_BASE = "https://www.nseindia.com"
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/option-chain",
}
_nse_session: requests.Session | None = None


def market_is_open(now: dt.datetime | None = None) -> bool:
    """Rough NSE cash-market hours check in IST; holidays are not checked."""
    local = now.astimezone(IST) if now and now.tzinfo else (now or dt.datetime.now(IST))
    if local.weekday() >= 5:
        return False
    return MARKET_OPEN <= local.time() <= MARKET_CLOSE


def _dhan_configured() -> bool:
    try:
        cfg = DhanConfig.from_env()
        return bool(cfg.client_id and get_access_token(cfg))
    except Exception:
        return False


def _period_days(period: str) -> int:
    p = str(period).lower().strip()
    if p.endswith("y"):
        return int(float(p[:-1]) * 365)
    if p.endswith("mo"):
        return int(float(p[:-2]) * 30)
    if p.endswith("d"):
        return int(float(p[:-1]))
    return 365


def _dhan_history(symbol: str, period: str, interval: str):
    cfg = DhanConfig.from_env()
    info = resolve_equity_symbol(symbol)
    end = dt.datetime.now(IST)
    start = end - dt.timedelta(days=_period_days(period))
    if interval in {"1d", "1D", "day", "D"}:
        df = historical_daily(cfg, info["security_id"], start.date().isoformat(), (end.date() + dt.timedelta(days=1)).isoformat())
    else:
        # Dashboard intraday uses 1m/5m; a normal intraday request is limited by Dhan to a 90-day window.
        minutes = {"1m": 1, "5m": 5, "15m": 15, "25m": 25, "60m": 60}
        iv = minutes.get(str(interval).lower(), 5)
        try:
            from dhan_client import historical_intraday_range
            df = historical_intraday_range(cfg, info["security_id"], iv, start, end)
        except Exception:
            df = historical_intraday(cfg, info["security_id"], iv, start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"))
    return df


def _normalize_historical_frame(df):
    """Normalize provider-specific OHLC(V) shapes for all dashboard callers.

    Defensive boundary: provider frames may be flat, MultiIndex, or contain
    lower-case/alternate field names. This function must never depend on an
    external variable named ``x`` and must return either a valid canonical
    frame or an empty frame.
    """
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()

    frame = df.copy()

    # Flatten Yahoo-style MultiIndex columns deterministically.  Some Yahoo
    # responses use (field, ticker), others (ticker, field).
    try:
        columns = frame.columns
        if isinstance(columns, pd.MultiIndex):
            field_names = {
                "open", "high", "low", "close", "adj close",
                "adj_close", "volume", "ltp", "vol"
            }
            chosen_level = None
            for level in range(columns.nlevels):
                values = {
                    str(v).strip().lower()
                    for v in columns.get_level_values(level)
                }
                if values & field_names:
                    chosen_level = level
                    break

            if chosen_level is not None:
                frame.columns = [
                    str(v).strip()
                    for v in columns.get_level_values(chosen_level)
                ]
            else:
                frame.columns = [
                    "_".join(
                        str(v).strip()
                        for v in tup
                        if str(v).strip()
                    )
                    for tup in columns.to_list()
                ]
        else:
            frame.columns = [str(c).strip() for c in columns]
    except Exception:
        # Final conservative fallback: stringified columns. Never let a
        # provider column-shape problem crash the dashboard.
        frame.columns = [str(c).strip() for c in frame.columns]

    aliases = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Close",
        "adj_close": "Close",
        "ltp": "Close",
        "last price": "Close",
        "lastprice": "Close",
        "volume": "Volume",
        "vol": "Volume",
    }

    rename = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        if key in aliases:
            rename[column] = aliases[key]
    frame = frame.rename(columns=rename)

    # Canonicalize required numeric columns.  With duplicate canonical column
    # names pandas returns a DataFrame; take the first column in that case.
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        if field not in frame.columns:
            continue
        value = frame[field]
        if isinstance(value, pd.DataFrame):
            value = value.iloc[:, 0]
        frame[field] = pd.to_numeric(value, errors="coerce")

    if "Close" not in frame.columns:
        # Handle any remaining case where Adj Close survived under its original
        # spelling after an unusual provider response.
        for candidate in list(frame.columns):
            if str(candidate).strip().lower() in {"adj close", "adj_close"}:
                value = frame[candidate]
                if isinstance(value, pd.DataFrame):
                    value = value.iloc[:, 0]
                frame["Close"] = pd.to_numeric(value, errors="coerce")
                break

    if "Volume" not in frame.columns:
        frame["Volume"] = float("nan")

    required = ["Open", "High", "Low", "Close"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()

    try:
        index = pd.to_datetime(frame.index, errors="coerce")
        frame.index = index
        frame = frame[~frame.index.isna()]
    except Exception:
        # Keep the frame if the provider uses a non-datetime index.
        pass

    return frame.sort_index()


def _local_persistent_history(symbol: str, period: str, interval: str):
    """Best-effort local/Drive-backed history before public-provider fallback."""
    try:
        from free_data import local_history_provider
        df = local_history_provider(symbol, period=period, interval=interval)
        if df is not None and not df.empty:
            return _normalize_historical_frame(df)
    except Exception:
        pass
    return pd.DataFrame()


def _yahoo_period_bounds(period: str, interval: str):
    """Return epoch bounds for Yahoo's public chart endpoint."""
    end = dt.datetime.now(dt.timezone.utc)
    days = _period_days(period)
    # Yahoo limits 1m intraday history; keep the requested period but cap it
    # conservatively so the endpoint responds quickly instead of retrying.
    if str(interval).lower() in {"1m", "1min"}:
        days = min(days, 7)
    elif str(interval).lower() in {"2m", "5m", "15m", "30m", "60m", "90m", "1h"}:
        days = min(days, 60)
    start = end - dt.timedelta(days=days)
    return int(start.timestamp()), int(end.timestamp())


def _yahoo_chart_history(symbol: str, period: str, interval: str):
    """Fetch Yahoo history through the public chart JSON API.

    This deliberately avoids yfinance's internal cookie/session machinery,
    which can fail on hosted runtimes with repeated `Connection already opened`
    OperationalErrors. A fresh HTTP request is used for each fetch.
    """
    period1, period2 = _yahoo_period_bounds(period, interval)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": str(interval).lower().replace("min", "m"),
        "events": "history",
        "includeAdjustedClose": "true",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=(5, 15))
        response.raise_for_status()
        payload = response.json()
        result = (payload.get("chart") or {}).get("result") or []
        if not result:
            return pd.DataFrame()
        result0 = result[0]
        timestamps = result0.get("timestamp") or []
        indicators = result0.get("indicators") or {}
        quote_list = indicators.get("quote") or []
        if not timestamps or not quote_list:
            return pd.DataFrame()
        quote = quote_list[0]
        adj_list = indicators.get("adjclose") or []
        adj = adj_list[0].get("adjclose", []) if adj_list else []
        frame = pd.DataFrame({
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Volume": quote.get("volume", []),
        }, index=pd.to_datetime(timestamps, unit="s", utc=True))
        if len(adj) == len(frame):
            frame["Adj Close"] = adj
        frame.index.name = "Datetime"
        return _normalize_historical_frame(frame)
    except Exception:
        return pd.DataFrame()


def _yahoo_history(symbol: str, period: str, interval: str):
    """Yahoo reference fallback without yfinance."""
    return _yahoo_chart_history(symbol, period, interval)

def get_historical(symbol: str, period: str = "1y", interval: str = "1d", force_refresh: bool = False):
    """NSE-first historical provider with explicit local/remote fallbacks."""
    is_nse = str(symbol).upper().endswith(".NS")
    root = str(symbol).upper().replace(".NS", "").replace(".BO", "")

    # 1) Persistent/local history is cheapest and avoids unnecessary remote calls.
    if not force_refresh:
        local = _local_persistent_history(symbol, period, interval)
        if not local.empty:
            return local

    # 2) NSE official daily history for NSE equities.
    if is_nse and fetch_equity_history is not None and interval.lower() in {"1d", "d", "day", "daily"}:
        try:
            days = _period_days(period)
            end = dt.datetime.now(IST).date()
            start = end - dt.timedelta(days=days)
            # Interactive serving must not upload chunks to Google Drive.
            # Persistence is handled by the ingestion/scheduled data layer;
            # blocking the Streamlit request on Drive sync can make a simple
            # stock search appear to hang. NSE remains the first remote source.
            nse_df = fetch_equity_history(root, start, end, sync_drive=False)
            if not nse_df.empty:
                nse_df = nse_df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
                if "date" in nse_df.columns:
                    nse_df = nse_df.set_index("date")
                normalized = _normalize_historical_frame(nse_df)
                if not normalized.empty:
                    return normalized
        except Exception:
            pass

    # 3) NSE quote-chart 1m for current-session intraday.
    if is_nse and interval.lower() in {"1m", "1min"} and fetch_intraday_1m_current is not None:
        try:
            nse_1m = fetch_intraday_1m_current(root, sync_drive=True)
            if not nse_1m.empty:
                normalized = _normalize_historical_frame(nse_1m)
                if not normalized.empty:
                    return normalized
        except Exception:
            pass

    # 4) Dhan when configured.
    if _dhan_configured() and is_nse:
        try:
            df = _dhan_history(symbol, period, interval)
            if not df.empty:
                normalized = _normalize_historical_frame(df)
                if not normalized.empty:
                    return normalized
        except Exception:
            pass

    # 5) Yahoo reference fallback.
    return _yahoo_history(symbol, period, interval)


def get_intraday(symbol: str, interval: str = "1m"):
    if str(symbol).upper().endswith(".NS") and str(interval).lower() in {"1m", "1min"} and fetch_intraday_1m_current is not None:
        try:
            nse_1m = fetch_intraday_1m_current(str(symbol).upper().replace(".NS", ""), sync_drive=True)
            if not nse_1m.empty:
                return _normalize_historical_frame(nse_1m)
        except Exception:
            pass
    if _dhan_configured() and str(symbol).upper().endswith(".NS"):
        try:
            df = _dhan_history(symbol, "5d", interval)
            if not df.empty:
                return _normalize_historical_frame(df)
        except Exception:
            pass
    return _yahoo_chart_history(symbol, "5d", interval)


def get_latest_price(symbol: str) -> float | None:
    if str(symbol).upper().endswith(".NS") and fetch_quote is not None:
        try:
            quote = fetch_quote(str(symbol).upper().replace(".NS", ""))
            value = quote.get("priceInfo", {}).get("lastPrice")
            if value is not None:
                return float(value)
        except Exception:
            pass
    if _dhan_configured() and str(symbol).upper().endswith(".NS"):
        try:
            cfg = DhanConfig.from_env()
            info = resolve_equity_symbol(symbol)
            from dhan_client import ltp
            vals = ltp(cfg, [info["security_id"]])
            value = vals.get(info["security_id"])
            if value is not None:
                return float(value)
        except Exception:
            pass
    df = _yahoo_chart_history(symbol, "1d", "1m")
    if df.empty or "Close" not in df.columns:
        return None
    value = pd.to_numeric(df["Close"], errors="coerce").dropna()
    return float(value.iloc[-1]) if not value.empty else None


def benchmark_for_symbol(symbol: str) -> str:
    """Use NIFTY as the broad benchmark; Bank Nifty for obvious banking stocks."""
    root = symbol.upper().replace(".NS", "").replace(".BO", "")
    banking = {"HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BANKBARODA", "PNB"}
    return "^NSEBANK" if root in banking else BENCHMARK_TICKER


def stock_news_keywords(symbol: str, company_name: str | None = None) -> list[str]:
    root = symbol.upper().replace(".NS", "").replace(".BO", "")
    vals = [root.lower()]
    vals.extend(x.lower() for x in SYMBOL_ALIASES.get(root, []))
    if company_name:
        vals.append(company_name.lower())
    # Deduplicate while preserving order.
    return list(dict.fromkeys(v for v in vals if v.strip()))


# -------------------------------------------------------------------- NSE ---
def _get_nse_session() -> requests.Session:
    global _nse_session
    s = requests.Session()
    s.headers.update(_NSE_HEADERS)
    try:
        s.get(NSE_BASE, timeout=6)
        s.get(f"{NSE_BASE}/option-chain", timeout=6)
    except requests.RequestException:
        pass
    _nse_session = s
    return s


def _nse_get(path: str, params: dict | None = None, retries: int = 1):
    global _nse_session
    if _nse_session is None:
        _get_nse_session()
    url = f"{NSE_BASE}{path}"
    for attempt in range(retries + 1):
        try:
            resp = _nse_session.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    return None
        except requests.RequestException:
            pass
        if attempt < retries:
            _get_nse_session()
    return None


def get_option_chain(symbol: str, is_index: bool = False):
    path = "/api/option-chain-indices" if is_index else "/api/option-chain-equities"
    return _nse_get(path, params={"symbol": symbol})


def get_corporate_announcements(symbol: str) -> list[dict]:
    data = _nse_get("/api/corporate-announcements", params={"index": "equities", "symbol": symbol})
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        payload = data.get("data", [])
        return payload if isinstance(payload, list) else []
    return []


# ------------------------------------------------------------------ news ---
def get_news(max_items: int = 80) -> list[dict]:
    items = []
    for url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                items.append(
                    {
                        "title": entry.get("title", "").strip(),
                        "summary": entry.get("summary", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", "") or entry.get("updated", ""),
                        "source": feed.feed.get("title", url),
                    }
                )
        except Exception:
            continue

    # Deduplicate identical headlines collected from multiple feeds.
    seen = set()
    out = []
    for item in items:
        key = item["title"].strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out[:max_items]


def news_for_symbol(company_keywords: list[str], news_items: list[dict]) -> list[dict]:
    keywords = [k.lower() for k in company_keywords if k]
    matches = []
    for item in news_items:
        text = (item.get("title", "") + " " + item.get("summary", "")).lower()
        if any(kw in text for kw in keywords):
            matches.append(item)
    return matches
