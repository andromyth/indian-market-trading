#!/usr/bin/env python3
"""Import Kaggle NIFTY 50 5-minute bars as close-only model market context.

The model uses only NIFTY close-derived returns/RSI. The source's OHLC
discrepancies and zero-volume column are deliberately not propagated. Source
timestamps are bar-start timestamps in India local time; output timestamps are
shifted forward five minutes to the completed-bar convention used by the
intraday trainer. The first 09:15 bar and incomplete sessions are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
REMOTE_RELATIVE_DIR = Path(
    "production_data/nse_official/intraday_seed/olo_5m_archive/symbol=__NIFTY50__"
)
OUTPUT_NAME = "nifty50_kaggle_5m_close_only.parquet"
SOURCE_URL = "https://www.kaggle.com/datasets/debashis74017/nifty-50-minute-data/data"


def _column_map(frame: pd.DataFrame) -> dict[str, str]:
    return {
        "".join(ch for ch in str(column).strip().lower() if ch.isalnum()): str(column)
        for column in frame.columns
    }


def _parse_timestamps(frame: pd.DataFrame, columns: dict[str, str]) -> pd.Series:
    timestamp_name = next(
        (columns[key] for key in ("timestamp", "datetime", "datetimeist", "datetimeutc", "datetimein") if key in columns),
        None,
    )
    if timestamp_name is not None:
        values = frame[timestamp_name]
    else:
        date_name = columns.get("date")
        time_name = columns.get("time")
        if date_name is None or time_name is None:
            raise ValueError(
                "CSV must contain a timestamp/datetime column, or separate date and time columns."
            )
        values = frame[date_name].astype(str).str.strip() + " " + frame[time_name].astype(str).str.strip()

    parsed = pd.to_datetime(values, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert(IST).dt.tz_localize(None)
    elif not pd.api.types.is_datetime64_ns_dtype(parsed.dtype):
        raise ValueError("Timestamp column contains mixed or unsupported timezone formats.")
    return parsed


def prepare_context_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate full regular sessions and return close-only, close-labelled NIFTY bars."""
    if raw is None or raw.empty:
        raise ValueError("Input NIFTY CSV is empty.")

    columns = _column_map(raw)
    close_name = next(
        (columns[key] for key in ("close", "closeprice", "closingprice") if key in columns),
        None,
    )
    if close_name is None:
        raise ValueError("CSV has no Close price column.")

    timestamps = _parse_timestamps(raw, columns)
    closes = pd.to_numeric(raw[close_name], errors="coerce")
    source = pd.DataFrame({"Timestamp": timestamps, "Close": closes})
    source = source.dropna(subset=["Timestamp"]).copy()
    if source.empty:
        raise ValueError("No parseable timestamps were found in the input CSV.")
    if source["Timestamp"].duplicated().any():
        duplicates = int(source["Timestamp"].duplicated(keep=False).sum())
        raise ValueError(f"Input contains {duplicates} rows with duplicate timestamps.")

    # Kaggle timestamps mark the beginning of each five-minute interval.
    # Retain only the regular NSE session grid, 09:15 through 15:25.
    clock = source["Timestamp"].dt
    regular = (
        (clock.time >= pd.Timestamp("09:15").time())
        & (clock.time <= pd.Timestamp("15:25").time())
        & (clock.minute % 5 == 0)
        & (clock.second == 0)
        & (clock.microsecond == 0)
    )
    regular_source = source.loc[regular].copy()
    total_sessions = int(source["Timestamp"].dt.date.nunique())
    accepted: list[pd.DataFrame] = []
    accepted_dates: list[str] = []

    for session_date, group in regular_source.groupby(
        regular_source["Timestamp"].dt.date, sort=True
    ):
        expected = pd.date_range(
            pd.Timestamp(session_date) + pd.Timedelta(hours=9, minutes=15),
            periods=75,
            freq="5min",
        )
        actual = pd.DatetimeIndex(group.sort_values("Timestamp")["Timestamp"])
        valid_close = (
            pd.to_numeric(group["Close"], errors="coerce").notna().all()
            and (pd.to_numeric(group["Close"], errors="coerce") > 0).all()
        )
        if len(actual) != 75 or not actual.equals(expected) or not valid_close:
            continue
        accepted.append(group.sort_values("Timestamp").copy())
        accepted_dates.append(str(session_date))

    if not accepted:
        raise ValueError("No complete 75-bar regular NSE sessions passed validation.")

    clean = pd.concat(accepted, ignore_index=True)
    # Exclude the 09:15 opening candle. Shift the remaining bar-start labels
    # to bar-close labels (e.g. 09:20 -> 09:25).
    clean = clean.loc[clean["Timestamp"].dt.strftime("%H:%M").ne("09:15")].copy()
    clean["Timestamp"] = clean["Timestamp"] + pd.Timedelta(minutes=5)

    output = pd.DataFrame({
        "Timestamp": clean["Timestamp"].to_numpy(),
        "Open": np.nan,
        "High": np.nan,
        "Low": np.nan,
        "Close": pd.to_numeric(clean["Close"], errors="coerce").to_numpy(dtype=float),
        "Volume": np.nan,
    }).sort_values("Timestamp").reset_index(drop=True)

    stats: dict[str, Any] = {
        "source_rows": int(len(raw)),
        "source_sessions": total_sessions,
        "accepted_complete_sessions": len(accepted_dates),
        "excluded_or_incomplete_sessions": total_sessions - len(accepted_dates),
        "excluded_opening_bars": len(accepted_dates),
        "output_bars": int(len(output)),
        "first_close_timestamp_ist": output["Timestamp"].min().isoformat(),
        "last_close_timestamp_ist": output["Timestamp"].max().isoformat(),
        "features_enabled": ["NIFTY close-derived returns", "NIFTY close-derived RSI"],
        "features_disabled": ["NIFTY open", "NIFTY high", "NIFTY low", "NIFTY volume"],
        "timestamp_conversion": "source bar-start IST timestamp + 5 minutes = completed bar-close IST timestamp",
    }
    return output, stats


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to NIFTY_50_5minute.csv")
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/Indian_Market_Research",
        help="Mounted Indian_Market_Research Google Drive folder",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    drive_root = Path(args.drive_root).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not drive_root.is_dir():
        raise FileNotFoundError(
            f"Drive root is not mounted or does not exist: {drive_root}"
        )

    raw = pd.read_csv(input_path)
    output, stats = prepare_context_frame(raw)

    target_dir = drive_root / REMOTE_RELATIVE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = target_dir / OUTPUT_NAME
    manifest_path = target_dir / "nifty50_kaggle_5m_close_only_manifest.json"

    output.to_parquet(parquet_path, index=False)
    manifest = {
        "dataset": "Kaggle NIFTY 50 5-minute index data",
        "source_url": SOURCE_URL,
        "source_file": input_path.name,
        "source_sha256": _sha256(input_path),
        "output_file": OUTPUT_NAME,
        "output_sha256": _sha256(parquet_path),
        "approved_use": "historical NIFTY market context for the ALL-NSE five-minute intraday model",
        "source_is_official_nse": False,
        "volume_available": False,
        **stats,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "parquet_path": str(parquet_path),
        "manifest_path": str(manifest_path),
        **manifest,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
