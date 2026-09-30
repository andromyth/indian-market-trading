#!/usr/bin/env python3
"""Autonomous end-of-day Delivery prediction runner.

Uses the already-deployed next-session Delivery model for the certified
deployment universe. It creates one idempotent prediction per symbol/day,
without placing orders or retraining the model.
"""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from data_fetcher import (
    benchmark_for_symbol,
    get_corporate_announcements,
    get_historical,
    get_news,
    get_option_chain,
    news_for_symbol,
    stock_news_keywords,
)
from delivery_model import supported_symbol, MANIFEST
from autonomous_engine import delivery_prediction
from fo_analysis import summarize as fo_summarize
from nse_data_layer import fetch_nse_bulk_symbol_history
from nse_bulk_eod_ingest import NSE_EQUITY_HOLIDAYS_BY_YEAR
from triggers import announcement_to_news, event_risk_score, tag_news_items

IST = ZoneInfo("Asia/Kolkata")
MARKET_EOD_READY_TIME = time(16, 0)


def _is_nse_equity_trading_day(d) -> bool:
    return d.weekday() < 5 and d not in NSE_EQUITY_HOLIDAYS_BY_YEAR.get(
        d.year, frozenset()
    )


def _nse_daily(symbol: str) -> pd.DataFrame:
    end = pd.Timestamp.now(tz=IST).date()
    start = end - pd.Timedelta(days=140)
    # Use the existing official CM-UDiFF bulk EOD archive path. This avoids
    # the security-wise historical endpoint, which may return no response in
    # GitHub Actions even when the official daily archives are available.
    df = fetch_nse_bulk_symbol_history(
        symbol,
        start,
        end,
        sync_drive=True,
        download_missing=True,
    )
    if df.empty:
        return pd.DataFrame()
    x = df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    ).copy()
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
    x = x.dropna(subset=["Date"]).set_index("Date").sort_index()
    return x


def run_symbol(symbol: str, all_news: list[dict]) -> dict:
    root = symbol.replace(".NS", "").replace(".BO", "").upper()
    now = datetime.now(IST)
    expected_date = now.date()
    if not _is_nse_equity_trading_day(expected_date):
        return {
            "symbol": root,
            "status": "BLOCKED",
            "reason": "non_trading_day_no_new_delivery_prediction",
            "expected_date": expected_date.isoformat(),
        }
    price_df = _nse_daily(root)
    if price_df.empty or len(price_df) < 65:
        return {"symbol": root, "status": "BLOCKED", "reason": "insufficient_official_nse_daily_history"}
    latest_date = price_df.index[-1].date()
    if now.time() < MARKET_EOD_READY_TIME or latest_date != expected_date:
        return {
            "symbol": root,
            "status": "BLOCKED",
            "reason": "official_nse_eod_not_ready_for_current_session",
            "expected_date": expected_date.isoformat(),
            "latest_official_date": latest_date.isoformat(),
            "eod_ready_after_ist": MARKET_EOD_READY_TIME.strftime("%H:%M"),
        }

    benchmark = benchmark_for_symbol(root)
    market_df = get_historical(benchmark, period="140d", interval="1d")
    aliases = stock_news_keywords(root)
    matched = news_for_symbol(aliases, all_news)

    raw_ann = get_corporate_announcements(root)
    announcements = []
    for item in raw_ann[:10]:
        if not isinstance(item, dict):
            continue
        announcements.append(
            {
                "subject": item.get("subject") or item.get("desc") or item.get("attchmntText") or "Corporate announcement",
                "date": item.get("an_dt") or item.get("date") or item.get("sort_date") or "",
                "link": item.get("attchmntFile") or item.get("link") or "",
            }
        )

    raw_chain = get_option_chain(root, is_index=False)
    fo = fo_summarize(raw_chain) if raw_chain else {}

    item = {
        "symbol": root,
        "price_df": price_df,
        "market_df": market_df,
        "news": {"available": 1.0 if matched else 0.0},
        "event": {
            "available": 1.0 if announcements else 0.0,
            "high_impact": 1.0 if announcements else 0.0,
        },
        "fno": {**fo, "available": 1.0 if fo else 0.0},
        "decision_timestamp": str(
            datetime.combine(price_df.index[-1].date(), time(15, 30), tzinfo=IST)
        ),
    }
    result = delivery_prediction(item, sync_drive=True)
    return {
        "symbol": root,
        "status": result.get("prediction_status", "ERROR"),
        "signal": result.get("signal"),
        "confidence": result.get("confidence"),
        "probability_up": result.get("probability_up_next_session"),
        "predicted_return": result.get("predicted_return_next_session"),
        "decision_timestamp": result.get("decision_timestamp"),
        "prediction_id": result.get("prediction_id"),
        "data_readiness_score": result.get("data_readiness_score"),
        "fno_available": bool(fo),
        "news_available": bool(matched),
        "announcements_available": bool(announcements),
        "reason": result.get("reason"),
    }


def main() -> int:
    trained = [str(x).upper() for x in MANIFEST.get("trained_symbols", [])]
    trained = [x for x in trained if supported_symbol(x)]
    if not trained:
        raise SystemExit("No supported trained Delivery symbols in production manifest.")

    all_news = tag_news_items(get_news(max_items=100))
    rows = []
    for symbol in trained:
        try:
            rows.append(run_symbol(symbol, all_news))
        except Exception as exc:
            rows.append(
                {
                    "symbol": symbol,
                    "status": "ERROR",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    df = pd.DataFrame(rows)
    print("AUTONOMOUS DELIVERY PREDICTIONS")
    print(df.to_string(index=False))
    hard_errors = int((df["status"] == "ERROR").sum()) if not df.empty else len(trained)
    blocked = int((df["status"] == "BLOCKED").sum()) if not df.empty else 0
    print(
        "SUMMARY:",
        {
            "symbols": len(trained),
            "passed": int((df["status"] == "PASS").sum()) if not df.empty else 0,
            "blocked": blocked,
            "errors": hard_errors,
        },
    )
    return 1 if hard_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
