#!/usr/bin/env python3
"""Diagnostic-only audit of live Delivery inputs and model differentiation.

This script does not retrain, change thresholds, place orders, or write the
prediction ledger. It builds the exact current 53-feature Delivery input for
the certified deployment universe and reports:
- source availability/row counts;
- missing feature names by symbol;
- feature values that differ across symbols;
- whether the model receives distinct inputs but produces identical outputs.

It is intended to identify data-pipeline gaps before any model change.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo
import hashlib
import json

import numpy as np
import pandas as pd

from data_fetcher import (
    benchmark_for_symbol,
    get_corporate_announcements,
    get_news,
    get_option_chain,
    news_for_symbol,
    stock_news_keywords,
)
from delivery_model import MANIFEST, build_feature_row, feature_names, predict
from delivery_history_cache import ensure_delivery_history
from nse_data_layer import fetch_index_history
from fo_analysis import summarize as fo_summarize
from nse_bulk_eod_ingest import NSE_EQUITY_HOLIDAYS_BY_YEAR
from triggers import tag_news_items

IST = ZoneInfo("Asia/Kolkata")
EOD_READY = time(16, 0)


def _is_trading_day(d) -> bool:
    return d.weekday() < 5 and d not in NSE_EQUITY_HOLIDAYS_BY_YEAR.get(
        d.year, frozenset()
    )


def _latest_completed_session(now: datetime | None = None) -> date:
    """Return the most recent calendar-valid NSE session whose EOD may be complete."""
    current = now or datetime.now(IST)
    session = current.date()
    if not (_is_trading_day(session) and current.time() >= EOD_READY):
        session -= timedelta(days=1)
    while not _is_trading_day(session):
        session -= timedelta(days=1)
    return session


def _nse_market_history(
    benchmark: str,
    as_of: date,
    calendar_days: int,
) -> pd.DataFrame:
    raw = fetch_index_history(
        benchmark,
        as_of - timedelta(days=calendar_days),
        as_of,
        sync_drive=True,
    )
    if raw is None or raw.empty or "date" not in raw.columns:
        return pd.DataFrame()
    x = raw.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"]).set_index("date")
    x = x.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return x.sort_index()


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
    return x.dropna(subset=["Date"]).set_index("Date").sort_index()


def _fingerprint(frame: pd.DataFrame) -> str:
    row = frame.iloc[0].replace({np.nan: None}).to_dict()
    payload = json.dumps(
        {k: row.get(k) for k in feature_names()},
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _value_repr(v):
    try:
        if pd.isna(v):
            return "MISSING"
    except Exception:
        pass
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.10g}"
    return str(v)


def main() -> int:
    symbols = [str(x).upper() for x in MANIFEST.get("trained_symbols", [])]
    now = datetime.now(IST)
    as_of = _latest_completed_session(now)

    print(
        "DELIVERY INPUT DIAGNOSTICS",
        {
            "symbols": symbols,
            "as_of": as_of.isoformat(),
            "run_at_ist": now.isoformat(),
            "eod_gate_ist": EOD_READY.isoformat(),
        },
        flush=True,
    )

    histories = ensure_delivery_history(symbols, as_of, min_rows=65)
    all_news = tag_news_items(get_news(max_items=100))

    frames: dict[str, pd.DataFrame] = {}
    outputs: dict[str, dict] = {}
    source_rows: list[dict] = []

    for symbol in symbols:
        price_df = _nse_daily(symbol, histories)
        benchmark = benchmark_for_symbol(symbol)
        market_error = None
        try:
            market_df = _nse_market_history(benchmark, as_of, calendar_days=140)
        except Exception as exc:
            market_df = pd.DataFrame()
            market_error = f"{type(exc).__name__}: {exc}"

        # Diagnostic-only longer context request, anchored to the same
        # completed NSE session as the stock history.
        market_df_long = pd.DataFrame()
        if len(market_df) < 21:
            try:
                market_df_long = _nse_market_history(
                    benchmark, as_of, calendar_days=260
                )
            except Exception as exc:
                market_error = market_error or f"{type(exc).__name__}: {exc}"

        aliases = stock_news_keywords(symbol)
        matched = news_for_symbol(aliases, all_news)

        raw_ann = get_corporate_announcements(symbol)
        announcements = [x for x in raw_ann[:10] if isinstance(x, dict)]

        raw_chain = get_option_chain(symbol, is_index=False)
        fo = fo_summarize(raw_chain) if raw_chain else {}

        decision_ts = str(
            datetime.combine(price_df.index[-1].date(), time(15, 30), tzinfo=IST)
        ) if not price_df.empty else None

        frame = build_feature_row(
            price_df,
            market_df,
            news={"available": 1.0 if matched else 0.0},
            event={
                "available": 1.0 if announcements else 0.0,
                "high_impact": 1.0 if announcements else 0.0,
            },
            fno={**fo, "available": 1.0 if fo else 0.0},
        )
        frames[symbol] = frame

        result = predict(frame, symbol, decision_ts)
        outputs[symbol] = result

        present = [
            c for c in feature_names()
            if pd.notna(frame.iloc[0].get(c))
        ]
        missing = [c for c in feature_names() if c not in present]

        source_rows.append(
            {
                "symbol": symbol,
                "price_rows": len(price_df),
                "price_latest": str(price_df.index[-1].date()) if not price_df.empty else None,
                "eod_session_match": bool(
                    not price_df.empty and price_df.index[-1].date() == as_of
                ),
                "benchmark": benchmark,
                "market_rows_140d": len(market_df),
                "market_latest_140d": str(market_df.index[-1]) if not market_df.empty else None,
                "market_data_error": market_error,
                "market_rows_260d_diag": len(market_df_long),
                "news_matches": len(matched),
                "announcements": len(announcements),
                "fno_available": bool(fo),
                "fno_keys": sorted(str(k) for k in fo.keys()),
                "ready_features": len(present),
                "missing_features": len(missing),
                "missing_list": missing,
                "input_fingerprint": _fingerprint(frame),
                "probability_up": result.get("probability_up_next_session"),
                "predicted_return": result.get("predicted_return_next_session"),
                "signal": result.get("signal"),
            }
        )

    print("SOURCE_AND_READINESS")
    print(pd.DataFrame(source_rows).to_string(index=False), flush=True)
    stale_symbols = [
        row["symbol"] for row in source_rows if not row["eod_session_match"]
    ]
    print(
        "EOD_SESSION_READINESS:",
        {
            "as_of": as_of.isoformat(),
            "symbols_checked": len(source_rows),
            "fresh_symbols": len(source_rows) - len(stale_symbols),
            "stale_or_missing_symbols": stale_symbols,
            "status": "PASS" if not stale_symbols else "PARTIAL",
        },
        flush=True,
    )

    print("MISSING_FEATURES_BY_SYMBOL")
    for row in source_rows:
        print(
            row["symbol"],
            ":",
            ", ".join(row["missing_list"]) if row["missing_list"] else "NONE",
            flush=True,
        )

    # Across the three live rows, identify which features actually vary.
    matrix = pd.concat(
        [
            frames[s].assign(_symbol=s).set_index("_symbol")
            for s in symbols
        ],
        axis=0,
    )
    varying = []
    constant = []
    for col in feature_names():
        vals = pd.to_numeric(matrix[col], errors="coerce")
        non_missing = vals.dropna()
        unique = int(non_missing.nunique())
        if unique > 1:
            varying.append((col, unique))
        else:
            constant.append(
                (
                    col,
                    _value_repr(non_missing.iloc[0]) if len(non_missing) else "MISSING",
                )
            )

    print("VARYING_FEATURES_ACROSS_SYMBOLS:", varying, flush=True)
    print("CONSTANT_OR_ALL_MISSING_FEATURES:", constant, flush=True)

    output_fingerprints = [outputs[s].get("prediction_id") for s in symbols]
    out_pairs = {
        s: (
            outputs[s].get("probability_up_next_session"),
            outputs[s].get("predicted_return_next_session"),
            outputs[s].get("signal"),
        )
        for s in symbols
    }

    print("MODEL_OUTPUTS:", out_pairs, flush=True)
    print(
        "DIAGNOSTIC_CONCLUSION:",
        {
            "distinct_inputs": len({r["input_fingerprint"] for r in source_rows}),
            "distinct_model_outputs": len(set(out_pairs.values())),
            "same_output_all_symbols": len(set(out_pairs.values())) == 1,
            "feature_count": len(feature_names()),
            "prediction_ids_present": output_fingerprints,
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
