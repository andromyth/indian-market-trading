#!/usr/bin/env python3
"""Assemble the native per-symbol 5-minute archive into a training dataset.

The archive stores one Parquet per symbol under
``.../olo_5m_archive/symbols/symbol=<SYM>/<SYM>.parquet`` with a DatetimeIndex.
``pretrain_pipeline.py`` expects a long frame with ``symbol, Timestamp, OHLCV``.
This script writes that frame as a directory of Parquet parts so pandas can read
it lazily without materialising every symbol in memory at once.

Read-only with respect to the archive; only ``--out-dir`` is written.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from dhan_backfill_all_nse_5m_history import OHLCV, retain_latest_sessions

IST = ZoneInfo("Asia/Kolkata")
ARCHIVE_REL = "production_data/nse_official/intraday_seed/olo_5m_archive/symbols"


def _default_drive_root() -> Path:
    mounted = Path("/content/drive/MyDrive/Indian_Market_Research")
    if mounted.exists():
        return mounted
    import os

    return Path(os.getenv("INDIAN_MARKET_RESEARCH_ROOT", "/tmp/Indian_Market_Research"))


def _frame(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    columns = {str(column).strip().lower(): column for column in raw.columns}
    if "timestamp" in columns:
        index = pd.DatetimeIndex(pd.to_datetime(raw[columns["timestamp"]], errors="coerce"))
    elif "datetime" in columns:
        index = pd.DatetimeIndex(pd.to_datetime(raw[columns["datetime"]], errors="coerce"))
    else:
        index = pd.DatetimeIndex(pd.to_datetime(raw.index, errors="coerce"))
    index = index.tz_localize(IST) if index.tz is None else index.tz_convert(IST)
    x = pd.DataFrame(index=index)
    for name in OHLCV:
        source = columns.get(name.lower())
        if source is None:
            raise RuntimeError(f"{path.name}: missing {name}")
        x[name] = pd.to_numeric(raw[source].to_numpy(), errors="coerce")
    x = x.loc[~x.index.isna()].sort_index()
    x = x.loc[~x.index.duplicated(keep="last")]
    return x


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, default=_default_drive_root())
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=500)
    parser.add_argument("--min-bars-per-session", type=int, default=68)
    parser.add_argument("--symbols", default="", help="Optional comma-separated allow-list")
    parser.add_argument("--max-symbols", type=int, default=0, help="0 = all archive symbols")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--start", default="", help="Optional inclusive IST date filter")
    parser.add_argument("--end", default="", help="Optional inclusive IST date filter")
    parser.add_argument("--clean", action="store_true", help="Remove existing part-*.parquet in out-dir first")
    args = parser.parse_args(argv)

    archive = args.archive_dir or (args.drive_root / ARCHIVE_REL)
    if not archive.is_dir():
        raise FileNotFoundError(f"Native archive directory not found: {archive}")
    allow = {name.strip().upper() for name in args.symbols.split(",") if name.strip()}
    direct = [
        path for path in archive.glob("symbol=*/*.parquet")
        if "__NIFTY" not in path.parent.name.upper()
    ]
    # Some Drive writers keep checkpoints/source partitions below the symbol
    # directory. Fall back to a recursive scan only when no direct file exists.
    candidates = direct or [
        path for path in archive.glob("symbol=*/**/*.parquet")
        if "__NIFTY" not in path.parts[-2].upper()
    ]
    by_symbol: dict[str, list[Path]] = {}
    for path in candidates:
        symbol_dir = next(
            (part for part in reversed(path.parts) if part.startswith("symbol=")),
            "",
        )
        symbol = symbol_dir.split("=", 1)[1].upper() if symbol_dir else ""
        if not symbol or "__NIFTY" in symbol:
            continue
        by_symbol.setdefault(symbol, []).append(path)
    paths = sorted(by_symbol)
    if allow:
        paths = [symbol for symbol in paths if symbol in allow]
    if args.max_symbols > 0:
        paths = paths[: args.max_symbols]
    if not paths:
        raise RuntimeError("No symbol Parquet files matched")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for old in parts_dir.glob("part-*.parquet"):
            old.unlink()

    start = pd.Timestamp(args.start, tz=IST) if args.start else None
    end = pd.Timestamp(args.end, tz=IST) + pd.Timedelta(days=1) if args.end else None
    sessions: set[str] = set()
    rows = 0
    written = 0
    skipped: list[dict[str, str]] = []
    chunk: list[pd.DataFrame] = []

    def flush() -> None:
        nonlocal chunk, written
        if not chunk:
            return
        part = pd.concat(chunk, ignore_index=True)
        name = f"part-{written:04d}.parquet"
        part.to_parquet(parts_dir / name, index=False, compression="zstd")
        written += 1
        chunk = []

    for index, symbol in enumerate(paths, 1):
        try:
            files = sorted(
                by_symbol[symbol],
                key=lambda path: ("kaggle" in str(path).lower(), str(path)),
            )
            frames = [_frame(path) for path in files]
            x = pd.concat(frames).sort_index()
            x = x.loc[~x.index.duplicated(keep="last")]
            if start is not None:
                x = x.loc[x.index >= start]
            if end is not None:
                x = x.loc[x.index < end]
            x = retain_latest_sessions(x, args.sessions, args.min_bars_per_session)
            if x.empty:
                skipped.append({"symbol": symbol, "reason": "no_usable_sessions"})
                continue
            out = x.reset_index()
            out.columns = ["Timestamp", *OHLCV]
            out.insert(0, "symbol", symbol)
            chunk.append(out)
            sessions |= set(pd.DatetimeIndex(x.index).normalize().strftime("%Y-%m-%d"))
            rows += len(out)
        except Exception as exc:  # pragma: no cover - depends on Drive/archive state
            skipped.append({"symbol": symbol, "reason": f"{type(exc).__name__}: {exc}"})
        if len(chunk) >= max(1, args.chunk_size):
            flush()
        if index % 50 == 0:
            print(f"ASSEMBLE_PROGRESS symbols={index}/{len(paths)} rows={rows:,}", flush=True)
    flush()

    manifest = {
        "source_archive": str(archive),
        "out_dir": str(args.out_dir),
        "generated_at": pd.Timestamp.now(tz=IST).isoformat(),
        "symbols_requested": len(paths),
        "parquet_files": sum(len(files) for files in by_symbol.values()),
        "symbols_written": len(paths) - len(skipped),
        "symbols_skipped": len(skipped),
        "rows": rows,
        "sessions": len(sessions),
        "first_session": min(sessions) if sessions else None,
        "last_session": max(sessions) if sessions else None,
        "sessions_retained_per_symbol": args.sessions,
        "min_bars_per_session": args.min_bars_per_session,
        "dataset_path": str(parts_dir),
        "parts": sorted(path.name for path in parts_dir.glob("part-*.parquet")),
        "schema": ["symbol", "Timestamp", *OHLCV],
        "skipped_sample": skipped[:20],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "skipped_sample"}, indent=2, sort_keys=True))
    print(f"TRAIN_BARS_PARQUET={parts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
