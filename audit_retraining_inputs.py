#!/usr/bin/env python3
"""Audit equity, F&O and NIFTY 5-minute coverage before retraining.

Read-only: this script never writes to the canonical archive or Drive unless
``--json-out`` / ``--md-out`` point at explicit local paths. It answers one
question: is the last 200/500-session window fully covered by the inputs that
``pretrain_pipeline.py`` consumes?
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
CANONICAL_NIFTY_REMOTE = (
    "production_data/nse_official/intraday_seed/olo_5m_archive/"
    "symbols/symbol=__NIFTY50__"
)
KAGGLE_NAME = "nifty50_kaggle_5m_close_only.parquet"
MERGED_NAME = "nifty50_5m_merged_close_only.parquet"
FNO_DEFAULT = "production_data/nse_official/fno/fno_daily_features_2026.parquet"


def _default_drive_root() -> Path:
    mounted = Path("/content/drive/MyDrive/Indian_Market_Research")
    if mounted.exists():
        return mounted
    import os

    return Path(os.getenv("INDIAN_MARKET_RESEARCH_ROOT", "/tmp/Indian_Market_Research"))


def _default_fno(root: Path) -> Path:
    directory = root / "production_data/nse_official/fno"
    for name in (
        "fno_daily_features_2024_2026.parquet",
        "fno_daily_features_2025_2026.parquet",
        "fno_daily_features_2024_2025.parquet",
        "fno_daily_features_2026.parquet",
    ):
        candidate = directory / name
        if candidate.exists():
            return candidate
    return directory / "fno_daily_features_2026.parquet"


def _timestamps(frame: pd.DataFrame) -> pd.DatetimeIndex:
    for column in ("Timestamp", "timestamp", "Datetime", "datetime"):
        if column in frame.columns:
            values = pd.to_datetime(frame[column], errors="coerce")
            break
    else:
        values = pd.to_datetime(frame.index, errors="coerce")
    index = pd.DatetimeIndex(values)
    index = index.tz_localize(IST) if index.tz is None else index.tz_convert(IST)
    return index[~index.isna()]


def _dates(index: pd.DatetimeIndex) -> set[str]:
    return set(index.normalize().strftime("%Y-%m-%d"))


def _equity_scan(root: Path, max_symbols: int) -> dict[str, Any]:
    archive = root / "production_data/nse_official/intraday_seed/olo_5m_archive"
    manifest_path = archive / "backfill_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    files = sorted((archive / "symbols").glob("symbol=*/*.parquet"))
    names = {
        path.parent.name.split("=", 1)[1].upper()
        for path in files
        if "__NIFTY" not in path.parent.name.upper()
    }
    sampled = [path for path in files if "__NIFTY" not in path.parent.name.upper()][: max(1, max_symbols)]
    sessions: set[str] = set()
    read_errors: list[str] = []
    for path in sampled:
        try:
            sessions |= _dates(_timestamps(pd.read_parquet(path)))
        except Exception as exc:  # pragma: no cover - depends on Drive state
            read_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    return {
        "manifest": {
            key: manifest.get(key)
            for key in (
                "overall_status",
                "symbols_resolved",
                "symbols_with_200_plus_sessions",
                "official_symbols_requested",
                "updated_at_utc",
            )
        },
        "symbols_total": len(names),
        "symbols_sampled": len(sampled),
        "sessions": sorted(sessions),
        "read_errors": read_errors,
        "archive_exists": archive.exists(),
    }


def _load_fno(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    frame = pd.read_parquet(path)
    columns = {str(column).strip().lower(): column for column in frame.columns}
    date_column = next((columns[key] for key in ("date", "trade_date", "timestamp") if key in columns), None)
    symbol_column = next((columns[key] for key in ("symbol", "ticker", "underlying") if key in columns), None)
    if date_column is None or symbol_column is None:
        return {"path": str(path), "exists": True, "error": "missing date/symbol columns", "columns": list(frame.columns)}
    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    symbols = frame[symbol_column].astype(str).str.upper().str.replace(".NS", "", regex=False)
    by_date: dict[str, set[str]] = {}
    for day, group in frame.assign(_date=dates, _symbol=symbols).dropna(subset=["_date"]).groupby("_date"):
        by_date[pd.Timestamp(day).strftime("%Y-%m-%d")] = set(group["_symbol"].unique())
    return {
        "path": str(path),
        "exists": True,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dates": sorted(by_date),
        "by_date": by_date,
        "symbols": sorted(set(symbols)),
        "min_date": min(by_date) if by_date else None,
        "max_date": max(by_date) if by_date else None,
    }


def _load_nifty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    frame = pd.read_parquet(path)
    index = _timestamps(frame)
    sessions = _dates(index)
    counts = pd.Series(1, index=index.normalize()).groupby(level=0).sum()
    counts.index = counts.index.strftime("%Y-%m-%d")
    incomplete = {str(day): int(count) for day, count in counts.items() if int(count) < 75}
    return {
        "path": str(path),
        "exists": True,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "sessions": sorted(sessions),
        "by_session_bars": {str(day): int(count) for day, count in counts.items()},
        "incomplete_sessions": incomplete,
        "min_date": min(sessions) if sessions else None,
        "max_date": max(sessions) if sessions else None,
        "first_timestamp": index.min().isoformat() if len(index) else None,
        "last_timestamp": index.max().isoformat() if len(index) else None,
    }


def _coverage(days: list[str], present: set[str]) -> float:
    return (sum(day in present for day in days) / len(days)) if days else 0.0


def _missing_runs(days: list[str], present: set[str]) -> list[dict[str, Any]]:
    runs: list[list[str]] = []
    current: list[str] = []
    for day in days:
        if day not in present:
            current.append(day)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [
        {"start": run[0], "end": run[-1], "sessions": len(run)}
        for run in runs
    ]


def _window_report(
    days: list[str],
    nifty_present: set[str],
    fno_by_date: dict[str, set[str]],
    equity_symbols: set[str],
    fno_symbols: set[str],
) -> dict[str, Any]:
    overlap = equity_symbols & fno_symbols
    per_date = []
    for day in days:
        members = fno_by_date.get(day, set())
        if overlap:
            per_date.append(len(members & equity_symbols) / len(overlap))
    return {
        "sessions": len(days),
        "start": days[0] if days else None,
        "end": days[-1] if days else None,
        "nifty_coverage": round(_coverage(days, nifty_present), 6),
        "fno_date_coverage": round(_coverage(days, set(fno_by_date)), 6),
        "fno_symbol_coverage_mean": round(sum(per_date) / len(per_date), 6) if per_date else 0.0,
        "missing_nifty_sessions": len([day for day in days if day not in nifty_present]),
        "missing_fno_sessions": len([day for day in days if day not in fno_by_date]),
    }


def _largest_ready_window(
    days: list[str],
    nifty_present: set[str],
    fno_present: set[str],
    *,
    nifty_min: float = 0.95,
    fno_min: float = 0.90,
) -> dict[str, Any] | None:
    # Prefer the largest window ending at the newest session (the original
    # behaviour), but fall back to earlier contiguous windows: trailing NIFTY
    # or F&O gaps must not hide an otherwise fully covered historical window.
    for end in range(len(days), 0, -1):
        for size in range(end, 0, -1):
            window = days[end - size : end]
            if _coverage(window, nifty_present) >= nifty_min and _coverage(window, fno_present) >= fno_min:
                return {
                    "sessions": size,
                    "start": window[0],
                    "end": window[-1],
                    "nifty_coverage": round(_coverage(window, nifty_present), 6),
                    "fno_date_coverage": round(_coverage(window, fno_present), 6),
                }
    return None


def build_report(
    root: Path,
    *,
    max_equity_symbols: int = 20,
    fno_path: Path | None = None,
    nifty_path: Path | None = None,
) -> dict[str, Any]:
    equity = _equity_scan(root, max_equity_symbols)
    days = equity["sessions"]
    fno_path = fno_path or _default_fno(root)
    fno = _load_fno(fno_path)
    candidates = []
    if nifty_path is not None:
        candidates.append(nifty_path)
    else:
        candidates.extend(
            [
                root / CANONICAL_NIFTY_REMOTE / MERGED_NAME,
                root / CANONICAL_NIFTY_REMOTE / KAGGLE_NAME,
            ]
        )
    nifty = {"exists": False}
    for candidate in candidates:
        loaded = _load_nifty(candidate)
        if loaded.get("exists"):
            nifty = loaded
            break

    nifty_present = set(nifty.get("sessions", []))
    fno_by_date = fno.get("by_date", {}) if fno.get("exists") else {}
    fno_present = set(fno_by_date)
    equity_symbols = set()
    # Names are cheap; the sampled parquets only define the session calendar.
    archive = root / "production_data/nse_official/intraday_seed/olo_5m_archive"
    for path in (archive / "symbols").glob("symbol=*"):
        name = path.name.split("=", 1)[1].upper()
        if "__NIFTY" not in name:
            equity_symbols.add(name)
    fno_symbols = set(fno.get("symbols", []))

    windows = {
        str(size): _window_report(days[-size:], nifty_present, fno_by_date, equity_symbols, fno_symbols)
        for size in (200, 300, 500)
        if len(days) >= size
    }
    largest = _largest_ready_window(days, nifty_present, fno_present)

    missing_nifty_runs = _missing_runs(days, nifty_present)
    missing_fno_runs = _missing_runs(days, fno_present)
    verdict = "READY"
    if not largest:
        verdict = "NEEDS_NIFTY_AND_FNO"
    elif largest["sessions"] < 200:
        verdict = "INSUFFICIENT_COVERED_SESSIONS"
    elif largest["sessions"] < len(days):
        verdict = "READY_FOR_A_SMALLER_WINDOW"

    return {
        "drive_root": str(root),
        "generated_at": pd.Timestamp.now(tz=IST).isoformat(),
        "equity": {
            "archive_exists": equity["archive_exists"],
            "manifest": equity["manifest"],
            "symbols_total": equity["symbols_total"],
            "symbols_sampled": equity["symbols_sampled"],
            "sessions": len(days),
            "first_session": days[0] if days else None,
            "last_session": days[-1] if days else None,
            "read_errors": equity["read_errors"],
        },
        "fno": {
            "path": fno.get("path"),
            "exists": fno.get("exists"),
            "rows": fno.get("rows"),
            "symbols": len(fno_symbols),
            "dates": len(fno_present),
            "min_date": fno.get("min_date"),
            "max_date": fno.get("max_date"),
            "missing_runs": missing_fno_runs[:20],
            "missing_run_count": len(missing_fno_runs),
        },
        "nifty": {
            "path": nifty.get("path"),
            "exists": nifty.get("exists"),
            "rows": nifty.get("rows"),
            "sessions": len(nifty_present),
            "min_date": nifty.get("min_date"),
            "max_date": nifty.get("max_date"),
            "incomplete_sessions": nifty.get("incomplete_sessions", {}),
            "missing_runs": missing_nifty_runs[:20],
            "missing_run_count": len(missing_nifty_runs),
        },
        "windows": windows,
        "largest_ready_window": largest,
        "verdict": verdict,
    }


def _markdown(report: dict[str, Any]) -> str:
    equity = report["equity"]
    fno = report["fno"]
    nifty = report["nifty"]
    lines = [
        "# Retraining input readiness",
        "",
        f"- Drive root: `{report['drive_root']}`",
        f"- Equity: **{equity['sessions']}** sessions "
        f"({equity['first_session']} .. {equity['last_session']}), "
        f"{equity['symbols_total']} symbols",
        f"- F&O: **{fno['dates']}** dates ({fno['min_date']} .. {fno['max_date']}), "
        f"{fno['symbols']} symbols",
        f"- NIFTY 5m: **{nifty['sessions']}** sessions ({nifty['min_date']} .. {nifty['max_date']}), "
        f"{nifty['rows']} rows",
        f"- Verdict: **{report['verdict']}**",
        "",
        "## Window coverage",
        "",
        "| Window | Start | End | NIFTY | F&O dates | F&O symbols |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for size, window in report["windows"].items():
        lines.append(
            f"| last {size} | {window['start']} | {window['end']} | "
            f"{window['nifty_coverage']:.1%} | {window['fno_date_coverage']:.1%} | "
            f"{window['fno_symbol_coverage_mean']:.1%} |"
        )
    largest = report["largest_ready_window"]
    lines += [
        "",
        "## Largest fully covered window",
        "",
        (
            f"**{largest['sessions']} sessions** ({largest['start']} .. {largest['end']}), "
            f"NIFTY {largest['nifty_coverage']:.1%}, F&O dates {largest['fno_date_coverage']:.1%}"
            if largest
            else "None at the 95% NIFTY / 90% F&O thresholds."
        ),
        "",
        "## Missing runs",
        "",
        f"- NIFTY: {nifty['missing_run_count']} run(s)",
        f"- F&O: {fno['missing_run_count']} run(s)",
    ]
    for run in nifty.get("missing_runs", [])[:5]:
        lines.append(f"  - NIFTY {run['start']} .. {run['end']} ({run['sessions']} sessions)")
    for run in fno.get("missing_runs", [])[:5]:
        lines.append(f"  - F&O {run['start']} .. {run['end']} ({run['sessions']} sessions)")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, default=_default_drive_root())
    parser.add_argument("--max-equity-symbols", type=int, default=20)
    parser.add_argument("--fno-parquet", type=Path, default=None)
    parser.add_argument("--nifty-parquet", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = build_report(
        args.drive_root,
        max_equity_symbols=args.max_equity_symbols,
        fno_path=args.fno_parquet,
        nifty_path=args.nifty_parquet,
    )
    summary = {key: value for key, value in report.items() if key != "windows"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"JSON={args.json_out}")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(_markdown(report), encoding="utf-8")
        print(f"MARKDOWN={args.md_out}")
    return 0 if report["verdict"].startswith("READY") else 2


if __name__ == "__main__":
    raise SystemExit(main())
