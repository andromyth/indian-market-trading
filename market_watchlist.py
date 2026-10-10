"""Internet-sourced daily NSE stock watchlist for the Streamlit market dashboard.

This module identifies stocks mentioned in recent Indian market coverage. It does
not treat a media mention as a BUY recommendation: model signals are added by
the dashboard through the project's own Delivery and Intraday predictors.
"""
from __future__ import annotations

import calendar
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from html import unescape
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import feedparser
import pandas as pd
import requests


IST = ZoneInfo("Asia/Kolkata")
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
MAX_ARTICLES_PER_SOURCE = 12
DEFAULT_MAX_STOCKS = 5
MAX_ARTICLE_AGE_HOURS = 96

# Search each publisher independently so the watchlist is traceable to several
# outlets instead of being a single site's editorial list.
WATCHLIST_SOURCES = (
    ("Moneycontrol", "moneycontrol.com", '("stocks to watch" OR "stocks in focus" OR "stocks in news")'),
    ("Economic Times", "economictimes.indiatimes.com", '("stocks to watch" OR "stocks in focus" OR "stocks in news")'),
    ("Financial Express", "financialexpress.com", '("stocks to watch" OR "stocks in focus" OR "stocks in news")'),
    ("Mint", "livemint.com", '("stocks to watch" OR "stocks in focus" OR "stocks in news")'),
    ("CNBC-TV18", "cnbctv18.com", '("stocks to watch" OR "stocks in focus" OR "stocks in news")'),
)

# Alias-only coverage for widely used company names and exchange symbols. Every
# entry is accepted only if its symbol exists in the latest NSE universe.
COMMON_ALIASES = {
    "TCS": ("TCS", "TATA CONSULTANCY SERVICES"),
    "INFY": ("INFOSYS",),
    "HCLTECH": ("HCLTECH", "HCL TECHNOLOGIES"),
    "SBIN": ("SBI", "STATE BANK OF INDIA"),
    "RELIANCE": ("RELIANCE INDUSTRIES",),
    "HDFCBANK": ("HDFC BANK",),
    "ICICIBANK": ("ICICI BANK",),
    "ITC": ("ITC",),
    "LT": ("LARSEN AND TOUBRO", "LARSEN TOUBRO"),
    "M&M": ("MAHINDRA AND MAHINDRA", "MAHINDRA MAHINDRA"),
    "BAJFINANCE": ("BAJAJ FINANCE",),
    "BAJAJFINSV": ("BAJAJ FINSERV",),
    "TATAMOTORS": ("TATA MOTORS",),
    "BHARTIARTL": ("BHARTI AIRTEL",),
    "AXISBANK": ("AXIS BANK",),
    "KOTAKBANK": ("KOTAK MAHINDRA BANK",),
    "ONGC": ("ONGC",),
    "POWERGRID": ("POWER GRID",),
    "TATASTEEL": ("TATA STEEL",),
    "ADANIENT": ("ADANI ENTERPRISES",),
    "ADANIPORTS": ("ADANI PORTS",),
    "MARUTI": ("MARUTI SUZUKI",),
    "ULTRACEMCO": ("ULTRATECH CEMENT", "ULTRATECH"),
    "TATAPOWER": ("TATA POWER",),
    "COALINDIA": ("COAL INDIA",),
    "TECHM": ("TECH MAHINDRA",),
    "DRREDDY": ("DR REDDYS", "DR REDDY"),
    "ASIANPAINT": ("ASIAN PAINTS",),
    "TITAN": ("TITAN COMPANY",),
    "NESTLEIND": ("NESTLE INDIA",),
    "INDIGO": ("INTERGLOBE AVIATION",),
}

LEGAL_SUFFIXES = {
    "LIMITED", "LTD", "LTD.", "PRIVATE", "PVT", "PVT.", "COMPANY", "CO",
    "CO.", "CORPORATION", "CORP", "CORP.", "PLC",
}


def normalize_text(value: object) -> str:
    """Normalize text for safe phrase matching without relying on regex input."""
    text = unescape(str(value or "")).upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return " ".join(text.split())


def _find_column(frame: pd.DataFrame, options: set[str]) -> str | None:
    return next(
        (c for c in frame.columns if str(c).strip().upper() in options),
        None,
    )


def build_alias_map(universe: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Create unambiguous phrase-to-symbol aliases from the current NSE universe."""
    if universe is None or universe.empty:
        return {}

    frame = universe.copy()
    symbol_col = _find_column(frame, {"SYMBOL", "CH_SYMBOL"})
    company_col = _find_column(
        frame, {"NAME OF COMPANY", "COMPANY NAME", "NAME", "COMPANYNAME"}
    )
    if symbol_col is None:
        return {}

    valid_symbols = set(
        frame[symbol_col].astype(str).str.strip().str.upper()
        .loc[lambda s: s.ne("") & ~s.isin({"NAN", "NONE", "SYMBOL"})]
    )
    alias_symbols: dict[str, set[str]] = {}

    def add(alias: object, symbol: str, *, allow_short: bool = False) -> None:
        normalized = normalize_text(alias)
        if not normalized or (len(normalized) < 4 and not allow_short):
            return
        # Very generic single tokens create false positives. Keep short aliases
        # only for the explicitly curated high-coverage mappings above.
        if len(normalized.split()) == 1 and len(normalized) < 5 and not allow_short:
            return
        alias_symbols.setdefault(normalized, set()).add(symbol)

    for _, row in frame.iterrows():
        symbol = str(row.get(symbol_col, "")).strip().upper()
        if symbol not in valid_symbols:
            continue
        add(symbol, symbol)
        if company_col is not None:
            company = normalize_text(row.get(company_col, ""))
            if company:
                add(company, symbol)
                tokens = company.split()
                while tokens and tokens[-1] in LEGAL_SUFFIXES:
                    tokens.pop()
                core = " ".join(tokens)
                if len(core) >= 6 and len(core.split()) >= 2:
                    add(core, symbol)

    for symbol, aliases in COMMON_ALIASES.items():
        if symbol not in valid_symbols:
            # Allow alternate company symbol spellings only where the official
            # universe confirms the actual security identifier.
            continue
        for alias in aliases:
            add(alias, symbol, allow_short=(len(normalize_text(alias)) >= 3))

    # If a company alias maps to multiple securities, discard it instead of
    # arbitrarily allocating a news article to one listing.
    return {
        alias: {"symbol": next(iter(symbols)), "alias": alias}
        for alias, symbols in alias_symbols.items()
        if len(symbols) == 1
    }


def extract_mentions(text: str, aliases: dict[str, dict[str, str]]) -> list[str]:
    """Return unique NSE symbols explicitly named in one article's text."""
    normalized = f" {normalize_text(text)} "
    found = set()
    for alias, info in aliases.items():
        if f" {alias} " in normalized:
            found.add(info["symbol"])
    return sorted(found)


def _published_utc(entry: dict) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)


def _fetch_source(source: tuple[str, str, str], now_utc: datetime) -> dict:
    publisher, domain, terms = source
    query = f"site:{domain} {terms} India when:4d"
    url = GOOGLE_NEWS_RSS + "?" + urlencode({
        "q": query,
        "hl": "en-IN",
        "gl": "IN",
        "ceid": "IN:en",
    })
    output = {"publisher": publisher, "domain": domain, "status": "ERROR", "articles": [], "error": None}
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "IndianMarketTradingDashboard/1.0 (news source attribution)"},
            timeout=(3, 6),
        )
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        if getattr(feed, "bozo", False) and not feed.entries:
            raise ValueError("RSS feed could not be parsed")
        for entry in feed.entries[:MAX_ARTICLES_PER_SOURCE]:
            title = unescape(str(entry.get("title", "")).strip())
            link = str(entry.get("link", "")).strip()
            published = _published_utc(entry)
            if not title or not link or published is None:
                continue
            if published > now_utc + timedelta(hours=2):
                continue
            if now_utc - published > timedelta(hours=MAX_ARTICLE_AGE_HOURS):
                continue
            summary = unescape(re.sub(r"<[^>]+>", " ", str(entry.get("summary", ""))))
            output["articles"].append({
                "title": title,
                "url": link,
                "published_utc": published.isoformat(),
                "published_at": published,
                "publisher": publisher,
                "domain": domain,
                "text": f"{title} {summary}",
            })
        output["status"] = "PASS" if output["articles"] else "NO_RECENT_ARTICLES"
    except Exception as exc:
        output["error"] = f"{type(exc).__name__}: {exc}"
    return output


def fetch_watchlist_articles(
    now: datetime | None = None,
    max_age_hours: int = MAX_ARTICLE_AGE_HOURS,
) -> dict:
    """Query several publishers in parallel and return only recent article rows."""
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    sources = []
    with ThreadPoolExecutor(max_workers=len(WATCHLIST_SOURCES)) as pool:
        futures = {
            pool.submit(_fetch_source, source, now_utc): source
            for source in WATCHLIST_SOURCES
        }
        for future in as_completed(futures):
            try:
                sources.append(future.result())
            except Exception as exc:
                publisher, domain, _ = futures[future]
                sources.append({
                    "publisher": publisher, "domain": domain, "status": "ERROR",
                    "articles": [], "error": f"{type(exc).__name__}: {exc}",
                })

    # Preserve configured source order in the status panel.
    source_order = {name: i for i, (name, _, _) in enumerate(WATCHLIST_SOURCES)}
    sources.sort(key=lambda item: source_order.get(item["publisher"], 999))
    cutoff = now_utc - timedelta(hours=max_age_hours)
    articles = [
        article
        for source in sources
        for article in source["articles"]
        if cutoff <= article["published_at"] <= now_utc + timedelta(hours=2)
    ]
    articles.sort(key=lambda item: item["published_at"], reverse=True)
    return {
        "checked_at_utc": now_utc.isoformat(),
        "sources": [
            {key: source.get(key) for key in ("publisher", "domain", "status", "error")}
            | {"article_count": len(source["articles"])}
            for source in sources
        ],
        "articles": articles,
    }


def build_stock_watchlist(
    universe: pd.DataFrame,
    source_result: dict,
    max_stocks: int = DEFAULT_MAX_STOCKS,
) -> dict:
    """Rank source-mentioned stocks by independent publisher support and recency."""
    aliases = build_alias_map(universe)
    articles = source_result.get("articles", []) or []
    by_symbol: dict[str, dict] = {}

    for article in articles:
        text = f"{article.get('title', '')} {article.get('text', '')}"
        mentioned = extract_mentions(text, aliases)
        for symbol in mentioned:
            item = by_symbol.setdefault(symbol, {
                "symbol": symbol,
                "mention_count": 0,
                "publishers": set(),
                "headlines": [],
                "latest_article_utc": None,
            })
            item["mention_count"] += 1
            item["publishers"].add(article.get("publisher") or article.get("domain") or "Unknown")
            article_row = {
                "title": str(article.get("title", "")),
                "url": str(article.get("url", "")),
                "publisher": str(article.get("publisher", "Unknown")),
                "domain": str(article.get("domain", "")),
                "published_utc": str(article.get("published_utc", "")),
            }
            if article_row["url"] and article_row["title"] and not any(
                old["url"] == article_row["url"] for old in item["headlines"]
            ):
                item["headlines"].append(article_row)
            published = article.get("published_at")
            if isinstance(published, datetime):
                previous = item["latest_article_utc"]
                if previous is None or published > previous:
                    item["latest_article_utc"] = published

    if not by_symbol:
        return {
            "status": "NO_SOURCE_MATCHES",
            "stocks": [],
            "article_count": len(articles),
            "sources": source_result.get("sources", []),
            "checked_at_utc": source_result.get("checked_at_utc"),
            "message": (
                "No recent source headlines could be matched confidently to NSE symbols. "
                "No fallback stocks were invented."
            ),
        }

    rows = list(by_symbol.values())
    rows.sort(
        key=lambda item: (
            len(item["publishers"]),
            item["mention_count"],
            item["latest_article_utc"] or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    stocks = []
    for item in rows[:max(1, int(max_stocks))]:
        latest = item["latest_article_utc"]
        stocks.append({
            "symbol": item["symbol"],
            "mention_count": int(item["mention_count"]),
            "source_count": int(len(item["publishers"])),
            "publishers": sorted(item["publishers"]),
            "latest_article_utc": latest.isoformat() if latest else None,
            "headlines": sorted(
                item["headlines"],
                key=lambda article: article["published_utc"],
                reverse=True,
            )[:3],
        })

    return {
        "status": "PASS",
        "stocks": stocks,
        "article_count": len(articles),
        "sources": source_result.get("sources", []),
        "checked_at_utc": source_result.get("checked_at_utc"),
        "message": (
            "Media mentions determine watchlist inclusion, not the model signal. "
            "BUY/SELL/HOLD/ABSTAIN comes only from the configured prediction engine."
        ),
    }


def target_session_date(now: datetime | None = None, holidays: set | None = None):
    """Return today's session before/during market hours, otherwise next NSE weekday."""
    current = (now or datetime.now(IST)).astimezone(IST)
    holiday_set = holidays or set()
    today = current.date()
    market_close = current.replace(hour=15, minute=30, second=0, microsecond=0)
    if today.weekday() < 5 and today not in holiday_set and current <= market_close:
        # Before today's open, watch today's session; during the session, intraday
        # applies now while Delivery may target the next session in its own output.
        return today
    candidate = today + timedelta(days=1)
    while candidate.weekday() >= 5 or candidate in holiday_set:
        candidate += timedelta(days=1)
    return candidate
