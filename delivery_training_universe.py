"""Training-universe policy for the next-session Delivery model.

The training universe is the current official NSE equity universe, not a hardcoded
watchlist. Historical observations still come from the persistent official NSE
CM-UDiFF bulk EOD archive. A listed symbol is included in the training roster,
but it is only trainable when enough clean historical sessions are present.

This module does not alter the currently deployed production model. It defines
the universe that future candidate training runs must use.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


EQUITY_SERIES = {"EQ", "BE", "BZ", "SM", "ST", "SZ", "GB", "RR"}


def normalize_symbol(value: Any) -> str:
    s = str(value or "").strip().upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        s = s.rsplit(".", 1)[0]
    return s


def load_official_nse_universe(force_refresh: bool = False) -> pd.DataFrame:
    """Load the official current NSE securities-available-for-trading roster."""
    from nse_data_layer import fetch_nse_equity_universe

    frame = fetch_nse_equity_universe(force_refresh=force_refresh).copy()
    column = next(
        (
            c
            for c in frame.columns
            if str(c).strip().upper() in {"SYMBOL", "CH_SYMBOL", "SYMBOL_NAME"}
        ),
        None,
    )
    if column is None:
        raise RuntimeError(
            f"NSE universe response has no symbol column: {list(frame.columns)}"
        )
    frame["_symbol"] = frame[column].map(normalize_symbol)
    frame = frame[frame["_symbol"].ne("")].copy()

    series_col = next(
        (
            c
            for c in frame.columns
            if str(c).strip().upper() in {"SERIES", "CH_SERIES"}
        ),
        None,
    )
    if series_col is not None:
        frame["_series"] = frame[series_col].astype(str).str.strip().str.upper()
        frame = frame[
            frame["_series"].isin(EQUITY_SERIES) | frame["_series"].eq("")
        ].copy()

    return frame.drop_duplicates("_symbol", keep="last").reset_index(drop=True)


def official_nse_symbols(force_refresh: bool = False) -> list[str]:
    return sorted(load_official_nse_universe(force_refresh=force_refresh)["_symbol"].tolist())


def training_roster(bulk_frame: pd.DataFrame, min_history: int = 60) -> dict[str, Any]:
    """Return the full official roster plus trainable/excluded symbols.

    The bulk_frame is expected to be the already-normalized official NSE bulk
    EOD dataset. No public fallback data is substituted here.
    """
    universe = set(official_nse_symbols(force_refresh=False))
    bulk = bulk_frame.copy()
    bulk["_symbol"] = bulk["symbol"].map(normalize_symbol)
    counts = bulk.groupby("_symbol")["date"].nunique()

    available = sorted(universe & set(counts.index))
    trainable = sorted(s for s in available if int(counts.get(s, 0)) >= min_history)
    excluded_set = set(universe) - set(trainable)
    excluded = sorted(excluded_set)

    return {
        "universe_source": "NSE_OFFICIAL_EQUITY_L",
        "universe_as_of": datetime.now().astimezone().isoformat(),
        "universe_symbol_count": len(universe),
        "symbols_with_bulk_history": len(available),
        "trainable_symbol_count": len(trainable),
        "trainable_symbols": trainable,
        "excluded_symbol_count": len(excluded),
        "excluded_symbols": excluded,
        "minimum_history_sessions": int(min_history),
    }


def benchmark_for_symbol(symbol: str) -> str:
    root = normalize_symbol(symbol)
    banking = {
        "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK",
        "INDUSINDBK", "BANKBARODA", "PNB",
    }
    return "^NSEBANK" if root in banking else "^NSEI"
