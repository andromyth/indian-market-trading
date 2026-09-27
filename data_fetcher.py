"""Market/news data adapters used by the dashboard.

Free mode:
* yfinance for historical/intraday OHLCV (unofficial Yahoo endpoints)
* NSE website JSON endpoints for F&O and corporate announcements (undocumented)
* public RSS feeds for current news

No order execution is included.
"""
from __future__ import annotations

import os
import datetime as dt
from zoneinfo import ZoneInfo

import feedparser
import requests
import yfinance as yf

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


def get_historical(symbol: str, period: str = "1y", interval: str = "1d"):
    if _dhan_configured() and str(symbol).upper().endswith(".NS"):
        try:
            df = _dhan_history(symbol, period, interval)
            if not df.empty:
                return df
        except Exception:
            pass
    return yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)


def get_intraday(symbol: str, interval: str = "5m"):
    if _dhan_configured() and str(symbol).upper().endswith(".NS"):
        try:
            df = _dhan_history(symbol, "5d", interval)
            if not df.empty:
                return df
        except Exception:
            pass
    return yf.Ticker(symbol).history(period="5d", interval=interval, auto_adjust=False)


def get_latest_price(symbol: str) -> float | None:
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
    df = yf.Ticker(symbol).history(period="1d", interval="1m", auto_adjust=False)
    if df.empty:
        return None
    value = df["Close"].dropna()
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
