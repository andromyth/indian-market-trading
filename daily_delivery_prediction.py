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
from all_nse_delivery_model import (
    supported_symbol as all_nse_supported_symbol,
    _load_manifest as load_all_nse_manifest,
)
from autonomous_engine import delivery_prediction
from prediction_ledger import LOCAL_LEDGER, REMOTE_LEDGER
from fo_analysis import summarize as fo_summarize
from nse_bulk_eod_ingest import NSE_EQUITY_HOLIDAYS_BY_YEAR
from delivery_history_cache import ensure_delivery_history
from triggers import announcement_to_news, event_risk_score, tag_news_items

IST = ZoneInfo("Asia/Kolkata")
MARKET_EOD_READY_TIME = time(16, 0)


def _is_nse_equity_trading_day(d) -> bool:
    return d.weekday() < 5 and d not in NSE_EQUITY_HOLIDAYS_BY_YEAR.get(
        d.year, frozenset()
    )


def _nse_daily(symbol: str, histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = histories.get(symbol.upper(), pd.DataFrame())
    if df.empty:
        return pd.DataFrame()
    x = df.copy()
    x["Date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    x = x.dropna(subset=["Date"]).set_index("Date").sort_index()
    return x


def run_symbol(
    symbol: str,
    all_news: list[dict],
    histories: dict[str, pd.DataFrame],
) -> dict:
    """Run one ALL-NSE next-session prediction from the persisted PIT layer.

    The scheduled job deliberately avoids per-symbol remote history, news and
    option-chain fan-out across 2,372 stocks. Those sources are attached by the
    interactive engine when available; the scheduled base prediction uses the
    same persisted PIT feature snapshot as the live model.
    """
    root = (
        symbol
        .replace(".NS", "")
        .replace(".BO", "")
        .upper()
    )

    now = datetime.now(IST)
    expected_date = now.date()

    if not _is_nse_equity_trading_day(
        expected_date
    ):
        return {
            "symbol": root,
            "status": "BLOCKED",
            "reason": "non_trading_day_no_new_delivery_prediction",
            "expected_date": expected_date.isoformat(),
        }

    if now.time() < MARKET_EOD_READY_TIME:
        return {
            "symbol": root,
            "status": "BLOCKED",
            "reason": "official_nse_eod_not_ready_for_current_session",
            "expected_date": expected_date.isoformat(),
            "eod_ready_after_ist": MARKET_EOD_READY_TIME.strftime("%H:%M"),
        }

    decision_timestamp = datetime.combine(
        expected_date,
        time(15, 30),
        tzinfo=IST,
    )

    result = delivery_prediction(
        {
            "symbol": root,
            "decision_timestamp": decision_timestamp,
        },
        # Batch Drive persistence once after all 2,372 symbols are processed.
        sync_drive=False,
    )

    return {
        "symbol": root,
        "status": result.get(
            "prediction_status",
            "ERROR",
        ),
        "signal": result.get(
            "signal"
        ),
        "confidence": result.get(
            "confidence"
        ),
        "probability_up": result.get(
            "probability_up_next_session"
        ),
        "predicted_return": result.get(
            "predicted_return_next_session"
        ),
        "decision_timestamp": result.get(
            "decision_timestamp"
        ),
        "prediction_id": result.get(
            "prediction_id"
        ),
        "prediction_ledger_status": result.get(
            "prediction_ledger_status"
        ),
        "prediction_ledger_synced": result.get(
            "prediction_ledger_synced"
        ),
        "data_readiness_score": result.get(
            "data_readiness_score"
        ),
        "auxiliary_model_consensus_p_up": result.get(
            "auxiliary_model_consensus_p_up"
        ),
        "auxiliary_model_consensus_direction": result.get(
            "auxiliary_model_consensus_direction"
        ),
        "reason": result.get(
            "reason"
        ),
    }


def main() -> int:
    # Scheduled Delivery uses the current ALL-NSE trainable roster,
    # not the retired 3-stock Delivery manifest.
    manifest = load_all_nse_manifest()
    trained = [
        str(x).upper()
        for x in manifest.get(
            "trainable_symbols",
            [],
        )
    ]
    trained = [
        x
        for x in trained
        if all_nse_supported_symbol(x)
    ]

    if not trained:
        raise SystemExit(
            "No supported symbols in the ALL-NSE Delivery manifest."
        )

    all_news = tag_news_items(get_news(max_items=100))
    # The ALL-NSE adapter prefers its persisted PIT snapshot. Do not
    # pre-download 65+ daily bars for 2,372 symbols in the scheduled runner.
    histories = {}
    rows = []
    for symbol in trained:
        try:
            rows.append(run_symbol(symbol, all_news, histories))
        except Exception as exc:
            rows.append(
                {
                    "symbol": symbol,
                    "status": "ERROR",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    # Sync the completed all-NSE ledger once rather than issuing a
    # Google Drive upload for every prediction row.
    ledger_sync = "NOT_ATTEMPTED"

    try:
        from persistent_store import GoogleDriveStore

        store = GoogleDriveStore()

        sync_result = store.upload_or_update(
            LOCAL_LEDGER,
            REMOTE_LEDGER,
            mime_type="application/x-ndjson",
        )

        ledger_sync = (
            "OK"
            if sync_result.get("status") == "OK"
            else str(sync_result.get("status", "ERROR"))
        )

    except Exception as exc:
        ledger_sync = (
            f"{type(exc).__name__}: {exc}"
        )

    for row in rows:
        row["prediction_ledger_synced"] = (
            ledger_sync == "OK"
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
