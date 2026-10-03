"""Daily headline collector (RSS) that archives forward in time.

`published_at` stored for PIT use is max(feed timestamp, our first-seen time), so an
item can never appear to be known before we actually fetched it. Symbols are matched by
company-name tokens from the NSE equity list; unmatched market-wide items go to MARKET.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.moneycontrol.com/rss/business.xml",
    "https://www.livemint.com/rss/markets",
    "https://www.business-standard.com/rss/markets-106.rss",
]  # verify each URL in your environment; dead feeds are logged, not fatal

_STOP = re.compile(r"\b(limited|ltd|india|industries|corporation|corp|company|co|the|of|and)\b\.?", re.I)


def build_alias_map(equity_list: pd.DataFrame) -> Dict[str, str]:
    """equity_list: columns SYMBOL, 'NAME OF COMPANY' (NSE EQUITY_L.csv). Returns alias -> symbol."""
    m: Dict[str, str] = {}
    eq = equity_list.rename(columns=lambda c: str(c).strip().upper())
    name_col = next((c for c in eq.columns if c in ("NAME OF COMPANY", "NAME_OF_COMPANY", "COMPANY NAME")), None)
    for _, r in eq.iterrows():
        sym = str(r["SYMBOL"]).strip().upper()
        core = re.sub(r"\s+", " ", _STOP.sub(" ", str(r[name_col] if name_col else "").lower())).strip()
        if len(core) >= 5:
            m[core] = sym
        if len(sym) >= 5:
            m[sym.lower()] = sym
    return m


def match_symbols(text: str, alias_map: Dict[str, str]) -> List[str]:
    t = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
    t = re.sub(r"\s+", " ", t)
    return sorted({sym for a, sym in alias_map.items() if f" {a} " in t})


def collect(alias_map: Dict[str, str], store: Path, feeds: List[str] = FEEDS) -> Dict:
    import feedparser
    now = pd.Timestamp(datetime.now(timezone.utc))
    rows, dead = [], []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            if not d.entries:
                dead.append(url)
                continue
            for e in d.entries:
                title = f"{e.get('title','')}. {e.get('summary','')}"[:600]
                ts = pd.to_datetime(e.get("published", None), utc=True, errors="coerce")
                # knowledge time = when WE fetched it (or a later feed time if the feed clock is ahead)
                pit = now if (pd.isna(ts) or ts <= now) else ts
                syms = match_symbols(title, alias_map) or ["MARKET"]
                for s in syms[:5]:
                    rows.append({"symbol": s, "published_at": pit, "text": title, "source": url})
        except Exception:
            dead.append(url)
    new = pd.DataFrame(rows)
    store.parent.mkdir(parents=True, exist_ok=True)
    if store.exists() and not new.empty:
        old = pd.read_parquet(store)
        new = pd.concat([old, new]).drop_duplicates(["symbol", "text"], keep="first")
    if not new.empty:
        new.to_parquet(store, index=False)
    return {"rows_total": int(len(new)), "dead_feeds": dead, "fetched_at": str(now)}
