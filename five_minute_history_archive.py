"""Helpers for the durable archive of genuine five-minute OHLCV observations."""
from __future__ import annotations

import pandas as pd

IST = "Asia/Kolkata"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
ARCHIVE_COLUMNS = ["symbol", "Timestamp", *OHLCV, "source"]


def build_history_archive_frame(
    frames: dict[str, pd.DataFrame],
    sources: dict[str, str] | None = None,
    cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Convert symbol-indexed close-timestamped OHLCV frames into a slot archive."""
    sources = sources or {}
    parts = []
    for symbol, frame in (frames or {}).items():
        if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
            continue
        x = frame.copy()
        idx = pd.DatetimeIndex(x.index)
        idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
        x.index = idx
        for column in OHLCV:
            if column not in x:
                x[column] = pd.NA
        x = x[OHLCV].apply(pd.to_numeric, errors="coerce").sort_index()
        x = x.loc[~x.index.duplicated(keep="last")]
        x = x.loc[x[["Open", "High", "Low", "Close"]].notna().all(axis=1)]
        if cutoff is not None:
            limit = pd.Timestamp(cutoff)
            limit = limit.tz_localize(IST) if limit.tzinfo is None else limit.tz_convert(IST)
            x = x.loc[x.index <= limit]
        if x.empty:
            continue
        part = x.reset_index(names="Timestamp")
        part.insert(0, "symbol", str(symbol).strip().upper())
        part["source"] = str(sources.get(symbol) or "INDSTOCKS_HISTORICAL_5M")
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=ARCHIVE_COLUMNS)
    result = pd.concat(parts, ignore_index=True)
    result["Timestamp"] = pd.to_datetime(result["Timestamp"], errors="coerce")
    result = result.dropna(subset=["Timestamp", "symbol"])
    result = result.sort_values(["symbol", "Timestamp"])
    return result.drop_duplicates(["symbol", "Timestamp"], keep="last").reset_index(drop=True)


def merge_history_archive_frames(
    pending: pd.DataFrame | None,
    current: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge retry rows with the newest slot; duplicate symbol/timestamps keep latest."""
    parts = [x for x in (pending, current) if x is not None and not x.empty]
    if not parts:
        return pd.DataFrame(columns=ARCHIVE_COLUMNS)
    merged = pd.concat(parts, ignore_index=True)
    for column in ARCHIVE_COLUMNS:
        if column not in merged:
            merged[column] = pd.NA
    merged["Timestamp"] = pd.to_datetime(merged["Timestamp"], errors="coerce")
    merged["symbol"] = merged["symbol"].astype(str).str.strip().str.upper()
    merged = merged.dropna(subset=["Timestamp"])
    return (
        merged.sort_values(["symbol", "Timestamp"])
        .drop_duplicates(["symbol", "Timestamp"], keep="last")
        .loc[:, ARCHIVE_COLUMNS]
        .reset_index(drop=True)
    )
