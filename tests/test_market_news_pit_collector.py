from datetime import datetime, timedelta, timezone

import pandas as pd

from market_news_pit_collector import (
    build_pit_layer,
    collect_sources,
    merge_article_archive,
)


class _FakeAnalyzer:
    def polarity_scores(self, _text):
        return {"compound": 0.42}


def _article(now):
    published = now - timedelta(minutes=20)
    return {
        "article_id": "article-1",
        "title": "State Bank of India announces quarterly performance",
        "summary": "The lender reported business updates.",
        "url": "https://example.com/sbin-story",
        "publisher": "Economic Times",
        "source_id": "economic_times_stocks",
        "source_domain": "economictimes.indiatimes.com",
        "source_method": "publisher_rss",
        "source_weight": 0.95,
        "published_at_utc": published.isoformat(),
        "publication_time_quality": "FEED_TIMESTAMP",
    }


def test_news_availability_is_not_backdated_before_first_seen():
    now = datetime(2026, 10, 16, 4, 0, tzinfo=timezone.utc)
    article = _article(now)
    archive = merge_article_archive(
        pd.DataFrame(), [article], now, analyzer=_FakeAnalyzer()
    )

    published = pd.Timestamp(article["published_at_utc"])
    available = pd.to_datetime(archive.iloc[0]["available_at_utc"], utc=True)
    assert published < available
    assert available == pd.Timestamp(now)
    assert archive.iloc[0]["sentiment_compound"] == 0.42


def test_pit_layer_matches_verified_nse_symbol_and_keeps_market_context():
    now = datetime(2026, 10, 16, 4, 0, tzinfo=timezone.utc)
    article = _article(now)
    archive = merge_article_archive(
        pd.DataFrame(), [article], now, analyzer=_FakeAnalyzer()
    )
    aliases = {
        "STATE BANK OF INDIA": {
            "symbol": "SBIN",
            "alias": "STATE BANK OF INDIA",
        },
        "SBIN": {"symbol": "SBIN", "alias": "SBIN"},
    }

    pit = build_pit_layer(archive, aliases, now_utc=now)
    row = pit.loc[pit["symbol"].eq("SBIN")].iloc[0]

    assert row["news_available"] == 1.0
    assert row["news_sentiment"] == 0.42
    assert pd.Timestamp(row["decision_timestamp"]) == pd.Timestamp(now)
    assert set(pit["symbol"]) == {"SBIN"}


def test_collect_sources_reports_each_feed_without_failing_the_batch():
    now = datetime(2026, 10, 16, 4, 0, tzinfo=timezone.utc)
    configured = (
        {
            "source_id": "good", "publisher": "Good Publisher",
            "domain": "good.example", "url": "https://good.example/rss",
            "method": "publisher_rss", "source_weight": 0.9, "required": True,
        },
        {
            "source_id": "bad", "publisher": "Bad Publisher",
            "domain": "bad.example", "url": "https://bad.example/rss",
            "method": "publisher_rss", "source_weight": 0.8, "required": False,
        },
    )

    def fake_fetch(source, _now):
        if source["source_id"] == "bad":
            raise RuntimeError("source unavailable")
        return {
            "source_id": source["source_id"],
            "publisher": source["publisher"],
            "domain": source["domain"],
            "url": source["url"],
            "method": source["method"],
            "required": source["required"],
            "status": "PASS",
            "article_count": 1,
            "error": None,
            "articles": [{"article_id": "good-1", "title": "headline"}],
        }

    articles, status = collect_sources(configured, now_utc=now, fetcher=fake_fetch)

    assert len(articles) == 1
    assert status[0]["source_id"] == "good"
    assert status[0]["status"] == "PASS"
    assert status[1]["source_id"] == "bad"
    assert status[1]["status"] == "ERROR"


def test_financial_news_classifier_uses_finance_context_and_impact():
    from market_news_pit_collector import classify_financial_news

    positive = classify_financial_news(
        "SBIN wins order contract and beats estimates in strong results",
        sentiment_score=0.0,
        source_weight=0.95,
    )
    negative = classify_financial_news(
        "Regulator imposes penalty after accounting fraud investigation",
        sentiment_score=0.0,
        source_weight=0.95,
    )

    assert positive["news_signal"] == "NEWS_BUY_BIAS"
    assert positive["news_event_category"] in {"ORDER_CONTRACT", "EARNINGS_RESULTS"}
    assert positive["news_direction_score"] > 0
    assert negative["news_signal"] == "NEWS_SELL_BIAS"
    assert negative["news_event_category"] == "FRAUD_REGULATORY"
    assert negative["event_impact_score"] >= 0.95


def test_financial_news_classifier_does_not_treat_cleared_fraud_allegation_as_bearish():
    from market_news_pit_collector import classify_financial_news

    result = classify_financial_news(
        "Company cleared of fraud allegations after regulator closes probe",
        sentiment_score=0.0,
        source_weight=0.9,
    )

    assert result["news_signal"] != "NEWS_SELL_BIAS"
    assert "negation_or_exoneration_detected" in result["news_signal_reason"]
