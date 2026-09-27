"""News and event trigger classification."""
from __future__ import annotations

from data_fetcher import get_corporate_announcements

HIGH_IMPACT_KEYWORDS = {
    "results": ["quarterly results", "q1 results", "q2 results", "q3 results", "q4 results", "annual results", "earnings", "profit", "revenue"],
    "rating action": ["upgrade", "downgrade", "rating action", "outlook revised"],
    "corporate action": ["stock split", "bonus issue", "buyback", "rights issue", "dividend"],
    "ownership": ["stake sale", "block deal", "bulk deal", "promoter pledge", "acquisition", "merger", "open offer", "stake increase"],
    "leadership": ["resign", "resignation", "steps down", "appoint", "appointed"],
    "regulatory/legal": ["sebi order", "probe", "raid", "fraud", "penalty", "show cause", "litigation", "regulatory action"],
    "business update": ["order win", "contract win", "guidance cut", "guidance raised", "profit warning", "guidance"],
    "capital/credit": ["default", "credit downgrade", "debt restructuring", "fund raise", "rights issue"],
}


def tag_headline(text: str) -> list[str]:
    text = (text or "").lower()
    tags = []
    for category, keywords in HIGH_IMPACT_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            tags.append(category)
    return tags


def tag_news_items(news_items: list[dict]) -> list[dict]:
    tagged = []
    for item in news_items:
        tags = tag_headline(item.get("title", "") + " " + item.get("summary", ""))
        tagged.append({**item, "tags": tags, "high_impact": bool(tags)})
    return tagged


def event_risk_score(news_items: list[dict], announcements: list[dict] | None = None) -> float:
    """0..1 risk/context score. It intentionally does not force direction."""
    news_hi = sum(1 for n in (news_items or []) if n.get("high_impact"))
    ann = announcements or []
    # One recent official/corporate event is meaningful; many concurrent events raise attention.
    return min(1.0, (news_hi * 0.18) + (len(ann) * 0.20))


def announcement_to_news(announcements: list[dict]) -> list[dict]:
    out = []
    for a in announcements or []:
        text = f"{a.get('subject', '')} {a.get('date', '')}".strip()
        tags = tag_headline(text)
        out.append({
            "title": a.get("subject", "Corporate announcement"),
            "summary": "NSE corporate announcement",
            "link": a.get("link", ""),
            "published": a.get("date", ""),
            "tags": tags or ["corporate event"],
            "high_impact": True,
        })
    return out


def upcoming_announcements(symbol: str, limit: int = 5) -> list[dict]:
    raw = get_corporate_announcements(symbol)
    out = []
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "subject": item.get("subject") or item.get("desc") or item.get("attchmntText") or "(no subject field found)",
                "date": item.get("an_dt") or item.get("date") or item.get("sort_date") or "",
                "link": item.get("attchmntFile") or item.get("link") or "",
            }
        )
    return out
