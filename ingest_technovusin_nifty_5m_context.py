#!/usr/bin/env python3
"""Extend the NIFTY 50 close-only 5-minute context from technovusin 1-minute CSVs.

Source: https://github.com/technovusin/nifty50-historical-data (1-minute OHLC,
IST bar-start timestamps, 2024/2025 yearly files and 2026 monthly files).

The intraday trainer only consumes NIFTY close-derived returns/RSI and needs
completed-bar, close-labelled timestamps. This script resamples 1-minute bars
into 09:15-aligned 5-minute buckets (Close = last minute close), validates full
75-bar sessions, and reuses ``ingest_kaggle_nifty_5m_context.prepare_context_frame``
so the resulting grid matches the existing Kaggle partition exactly.

Overlap with the existing Kaggle parquet is measured instead of merged blindly:
Kaggle wins on overlapping timestamps and the technovusin bars only extend the
series beyond Kaggle's last timestamp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingest_kaggle_nifty_5m_context import prepare_context_frame

IST = "Asia/Kolkata"
OUTPUT_NAME = "nifty50_5m_merged_close_only.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_1min(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        columns = {str(c).strip().lower(): c for c in frame.columns}
        ts_name = columns.get("timestamp")
        close_name = columns.get("close")
        if ts_name is None or close_name is None:
            raise ValueError(f"{path.name}: expected Timestamp and Close columns, got {list(frame.columns)}")
        part = pd.DataFrame(
            {
                "Timestamp": pd.to_datetime(frame[ts_name], errors="coerce"),
                "Close": pd.to_numeric(frame[close_name], errors="coerce"),
            }
        ).dropna()
        part["source_file"] = path.name
        frames.append(part)
    if not frames:
        raise ValueError("No 1-minute inputs were provided.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates("Timestamp", keep="first")
    return combined.sort_values("Timestamp").reset_index(drop=True)


def resample_to_5min(one_minute: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = one_minute.copy()
    source["Bucket"] = source["Timestamp"].dt.floor("5min")
    bucketed = (
        source.sort_values("Timestamp")
        .groupby("Bucket", as_index=False)
        .agg(Close=("Close", "last"))
        .rename(columns={"Bucket": "Timestamp"})
    )
    minute_counts = source.groupby("Bucket")["Timestamp"].count()
    partial_buckets = int((minute_counts < 5).sum())
    five_minute, stats = prepare_context_frame(bucketed)
    stats = dict(stats)
    stats.update(
        {
            "source_minute_rows": int(len(one_minute)),
            "five_minute_buckets": int(len(bucketed)),
            "partial_buckets_before_validation": partial_buckets,
        }
    )
    return five_minute, stats


def compare_sources(new: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    ref = reference.copy()
    ref["Timestamp"] = pd.to_datetime(ref["Timestamp"])
    ref = ref.dropna(subset=["Timestamp", "Close"]).sort_values("Timestamp")
    merged = new.merge(ref, on="Timestamp", suffixes=("_new", "_ref"))
    if merged.empty:
        return {"overlap_rows": 0}
    diff = (merged["Close_new"] - merged["Close_ref"]).abs()
    return {
        "overlap_rows": int(len(merged)),
        "overlap_first": str(merged["Timestamp"].min()),
        "overlap_last": str(merged["Timestamp"].max()),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "pct_within_1_point": float((diff <= 1.0).mean() * 100.0),
        "pct_within_5_points": float((diff <= 5.0).mean() * 100.0),
        "correlation": float(merged["Close_new"].corr(merged["Close_ref"])),
    }


def build_merged(new: pd.DataFrame, reference: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    new_cols = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    new = new[new_cols].copy()
    report: dict[str, Any] = {"reference_supplied": reference is not None}
    if reference is None or reference.empty:
        merged = new.sort_values("Timestamp").reset_index(drop=True)
        report["mode"] = "technovusin_only"
    else:
        ref = reference.copy()
        ref["Timestamp"] = pd.to_datetime(ref["Timestamp"])
        for column in new_cols:
            if column not in ref.columns:
                ref[column] = np.nan
        ref = ref[new_cols].dropna(subset=["Timestamp"]).sort_values("Timestamp")
        report["comparison"] = compare_sources(new, ref)
        ref_last = ref["Timestamp"].max()
        # Kaggle wins on overlap; technovusin only extends the tail.
        tail = new.loc[new["Timestamp"] > ref_last].copy()
        merged = pd.concat([ref, tail], ignore_index=True).sort_values("Timestamp").reset_index(drop=True)
        report["mode"] = "kaggle_priority_technovusin_tail"
        report["reference_last_timestamp"] = str(ref_last)
        report["technovusin_tail_rows"] = int(len(tail))
        report["technovusin_tail_first"] = str(tail["Timestamp"].min()) if len(tail) else None
        report["technovusin_tail_last"] = str(tail["Timestamp"].max()) if len(tail) else None
    merged = merged.drop_duplicates("Timestamp", keep="first").sort_values("Timestamp").reset_index(drop=True)
    report["merged_rows"] = int(len(merged))
    report["merged_first"] = str(merged["Timestamp"].min())
    report["merged_last"] = str(merged["Timestamp"].max())
    report["merged_sessions"] = int(merged["Timestamp"].dt.date.nunique())
    return merged, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="technovusin 1-minute CSV files")
    parser.add_argument("--kaggle-parquet", default="", help="Existing Kaggle close-only 5m parquet")
    parser.add_argument("--output", required=True, help="Merged close-only 5m parquet to write")
    parser.add_argument("--manifest", default="", help="Optional manifest JSON path")
    args = parser.parse_args()

    inputs = [Path(p) for p in args.input]
    one_minute = load_1min(inputs)
    five_minute, source_stats = resample_to_5min(one_minute)
    print("TECHNOVUSIN_1MIN_ROWS=" + str(len(one_minute)))
    print("TECHNOVUSIN_5M_STATS=" + json.dumps(source_stats, sort_keys=True, default=str))

    reference = None
    if args.kaggle_parquet:
        ref_path = Path(args.kaggle_parquet)
        if ref_path.is_file():
            reference = pd.read_parquet(ref_path)
            print("KAGGLE_REFERENCE_ROWS=" + str(len(reference)))
        else:
            print("KAGGLE_REFERENCE_MISSING=" + str(ref_path))

    merged, report = build_merged(five_minute, reference)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    report = {
        "dataset": "NIFTY 50 5-minute close-only context",
        "sources": {
            "technovusin": "https://github.com/technovusin/nifty50-historical-data",
            "inputs": [p.name for p in inputs],
            "input_sha256": _sha256(inputs[0]) if len(inputs) == 1 else None,
            "input_files": len(inputs),
        },
        "source_stats": source_stats,
        "merge": report,
        "output_file": out_path.name,
        "output_sha256": _sha256(out_path),
    }
    print("MERGE_REPORT=" + json.dumps(report["merge"], sort_keys=True, default=str))
    print("MERGED_ROWS=" + str(len(merged)))
    print("MERGED_LAST_TIMESTAMP=" + str(merged["Timestamp"].max()))
    print("TECHNOVUSIN_CONTEXT_BUILD=PASS")

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
        print("MANIFEST_WRITTEN=" + str(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
