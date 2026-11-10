#!/usr/bin/env python3
"""Audit isolated Hugging Face 5-minute staging output; never writes production data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hf_historical_seed import session_quality


def audit_staging(root: Path, canonical_root: Path | None = None) -> dict:
    files = sorted(root.glob("symbol=*/**/*.parquet"))
    symbols = []
    total_rows = 0
    total_oi_rows = 0
    total_duplicates = 0
    total_invalid_ohlc = 0
    total_invalid_oi = 0
    total_non_monotonic = 0
    total_out_of_session = 0
    missing_oi_symbols = []
    for path in files:
        symbol = path.parent.name.removeprefix("symbol=")
        frame = pd.read_parquet(path)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        duplicates = int(timestamps.duplicated(keep=False).sum())
        numeric = frame[["open", "high", "low", "close", "volume"]].apply(
            pd.to_numeric, errors="coerce"
        )
        valid = numeric.notna().all(axis=1)
        valid &= numeric[["open", "high", "low", "close"]].gt(0).all(axis=1)
        valid &= numeric["high"] >= numeric[["open", "low", "close"]].max(axis=1)
        valid &= numeric["low"] <= numeric[["open", "high", "close"]].min(axis=1)
        valid &= numeric["volume"].ge(0)
        invalid = int((~valid).sum() + timestamps.isna().sum())
        quality_frame = frame.loc[timestamps.notna(), ["timestamp"]].copy()
        quality = session_quality(quality_frame)
        has_oi = "oi" in frame.columns
        oi_numeric = pd.to_numeric(frame["oi"], errors="coerce") if has_oi else pd.Series(dtype="float64")
        oi_rows = int(oi_numeric.notna().sum()) if has_oi else 0
        invalid_oi = int((oi_numeric.notna() & oi_numeric.lt(0)).sum()) if has_oi else 0
        non_monotonic = int((timestamps.diff().dropna() < pd.Timedelta(0)).sum())
        local_clock = timestamps.dt.tz_convert("Asia/Kolkata").dt.strftime("%H:%M")
        out_of_session = int((~local_clock.between("09:20", "15:30")).sum())
        if not has_oi:
            missing_oi_symbols.append(symbol)
        row_count = len(frame)
        total_rows += row_count
        total_oi_rows += oi_rows
        total_duplicates += duplicates
        total_invalid_ohlc += invalid
        total_invalid_oi += invalid_oi
        total_non_monotonic += non_monotonic
        total_out_of_session += out_of_session
        symbols.append({
            "symbol": symbol,
            "rows_5m": row_count,
            **quality,
            "oi_column_present": has_oi,
            "oi_non_null_rows": oi_rows,
            "oi_non_null_pct": round(100 * oi_rows / row_count, 2) if row_count else 0.0,
            "duplicate_timestamp_rows": duplicates,
            "invalid_rows": invalid,
            "negative_oi_rows": invalid_oi,
            "non_monotonic_timestamp_rows": non_monotonic,
            "out_of_session_rows": out_of_session,
        })

    canonical_symbols = set()
    if canonical_root and canonical_root.exists():
        canonical_symbols = {
            p.stem.upper()
            for p in canonical_root.glob("symbol=*/**/*.parquet")
        }
    staged_symbols = {row["symbol"] for row in symbols}
    overlap = sorted(staged_symbols & canonical_symbols) if canonical_root else None

    return {
        "mode": "STAGING_ONLY",
        "canonical_archive_modified": False,
        "staged_symbol_files": len(files),
        "staged_symbols": len(staged_symbols),
        "total_5m_rows": total_rows,
        "complete_symbol_sessions": sum(x["complete_sessions"] for x in symbols),
        "partial_symbol_sessions": sum(x["partial_sessions"] for x in symbols),
        "symbols_with_200_complete_sessions": sum(x["complete_sessions"] >= 200 for x in symbols),
        "symbols_below_200_complete_sessions": sum(x["complete_sessions"] < 200 for x in symbols),
        "oi_column_missing_symbols": missing_oi_symbols,
        "oi_non_null_rows": total_oi_rows,
        "oi_non_null_pct": round(100 * total_oi_rows / total_rows, 2) if total_rows else 0.0,
        "duplicate_timestamp_rows": total_duplicates,
        "invalid_rows": total_invalid_ohlc,
        "negative_oi_rows": total_invalid_oi,
        "non_monotonic_timestamp_rows": total_non_monotonic,
        "out_of_session_rows": total_out_of_session,
        "canonical_symbol_overlap_count": len(overlap) if overlap is not None else None,
        "canonical_symbol_overlap": overlap,
        "canonical_comparison": "performed" if canonical_root else "BLOCKED_NO_READ_ONLY_ARCHIVE_MOUNT",
        "symbols": symbols,
        "merge_readiness": {
            "ready": bool(canonical_root) and bool(files) and not missing_oi_symbols and total_duplicates == 0 and total_invalid_ohlc == 0 and total_invalid_oi == 0 and total_non_monotonic == 0 and total_out_of_session == 0,
            "blocked_reasons": ([] if canonical_root else ["canonical_archive_read_only_comparison_not_run"]),
        },
        "quality_gate": {
            "passed": bool(files) and not missing_oi_symbols and total_duplicates == 0 and total_invalid_ohlc == 0 and total_invalid_oi == 0 and total_non_monotonic == 0 and total_out_of_session == 0,
            "checks": {
                "parquet_files_present": bool(files),
                "oi_column_present_for_every_symbol": not missing_oi_symbols,
                "no_duplicate_timestamps": total_duplicates == 0,
                "no_invalid_ohlcv_rows": total_invalid_ohlc == 0,
                "no_negative_oi": total_invalid_oi == 0,
                "timestamps_monotonic_per_symbol": total_non_monotonic == 0,
                "all_bars_within_regular_session": total_out_of_session == 0,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, default=Path("hf_multishard_staging"))
    parser.add_argument("--canonical-root", type=Path, default=None,
                        help="Optional read-only local canonical archive for symbol overlap only")
    parser.add_argument("--output", type=Path, default=Path("hf_multishard_quality_audit.json"))
    args = parser.parse_args()
    report = audit_staging(args.staging_root, args.canonical_root)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "symbols"}, indent=2))
    print(f"Audit: {args.output}")
    return 0 if report["quality_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
