#!/usr/bin/env python3
"""Backfill native NSE five-minute history through the INDstocks read-only API.

The provider permits at most five scrip codes and seven calendar days per request.
This runner batches five stocks per request, walks the requested range in seven-day
windows, respects the documented 5 requests/second data limit, and checkpoints each
stock directly to the same Drive archive used by the DhanHQ backfill.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from dhan_backfill_all_nse_5m_history import (
    FRESHNESS_WINDOW_DAYS,
    MANIFEST_REMOTE,
    MIN_SESSION_BARS,
    REMOTE_ROOT,
    _download_existing,
    _thread_drive_store,
    retain_latest_sessions,
    usable_session_summary,
)

IST = ZoneInfo("Asia/Kolkata")
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
MAX_CODES_PER_REQUEST = 5
MAX_DAYS_PER_REQUEST = 7


def normalize_indstocks_5m(frame: pd.DataFrame | None, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Convert INDstocks candle-open timestamps into completed five-minute bar closes."""
    empty = pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=IST, name="Datetime"))
    if frame is None or frame.empty:
        return empty
    x = frame.copy()
    x.columns = [str(column).strip() for column in x.columns]
    aliases = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }
    x = x.rename(columns={column: aliases.get(str(column).strip().lower(), column) for column in x.columns})
    if "Timestamp" in x.columns:
        index = pd.DatetimeIndex(pd.to_datetime(x.pop("Timestamp"), errors="coerce"))
    else:
        index = pd.DatetimeIndex(pd.to_datetime(x.index, errors="coerce"))
    index = index.tz_localize(IST) if index.tz is None else index.tz_convert(IST)
    x.index = index
    for column in OHLCV:
        if column not in x:
            x[column] = np.nan
        x[column] = pd.to_numeric(x[column], errors="coerce")
    x = x.loc[~x.index.isna()].sort_index()
    x = x.loc[~x.index.duplicated(keep="last")]
    starts = pd.DatetimeIndex(x.index)
    regular = (
        (starts.time >= pd.Timestamp("09:15").time())
        & (starts.time < pd.Timestamp("15:30").time())
        & (starts.minute % 5 == 0)
    )
    x = x.loc[regular].copy()
    x.index = x.index + pd.Timedelta(minutes=5)
    valid = x[["Open", "High", "Low", "Close"]].notna().all(axis=1)
    valid &= x[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)
    x = x.loc[valid]
    if cutoff is not None:
        boundary = pd.Timestamp(cutoff)
        boundary = boundary.tz_localize(IST) if boundary.tzinfo is None else boundary.tz_convert(IST)
        x = x.loc[x.index <= boundary]
    x.index.name = "Datetime"
    return x[OHLCV].loc[~x.index.duplicated(keep="last")].sort_index()


def _instrument_codes() -> dict[str, str]:
    """Build the current ordinary NSE-equity symbol-to-provider-code map once."""
    from indstocks_client import _load_instruments

    frame = _load_instruments("NSE")
    required = {"EXCH", "SECURITY_ID", "TRADING_SYMBOL"}
    if not required.issubset(frame.columns):
        raise RuntimeError("INDstocks instrument master is missing NSE equity mapping columns.")
    x = frame.copy()
    x["EXCH"] = x["EXCH"].astype(str).str.strip().str.upper()
    x["SECURITY_ID"] = x["SECURITY_ID"].astype(str).str.strip()
    x["TRADING_SYMBOL"] = x["TRADING_SYMBOL"].astype(str).str.strip().str.upper()
    x = x.loc[
        x["EXCH"].eq("NSE")
        & x["SECURITY_ID"].str.fullmatch(r"\d+")
        & x["TRADING_SYMBOL"].ne("")
    ].copy()
    if "SERIES" in x.columns:
        eq = x["SERIES"].astype(str).str.strip().str.upper().eq("EQ")
        if eq.any():
            x = x.loc[eq]
    x = x.drop_duplicates("TRADING_SYMBOL", keep="first")
    return {
        str(row.TRADING_SYMBOL): f"NSE_{row.SECURITY_ID}"
        for row in x.itertuples(index=False)
    }


def fetch_indstocks_5m_batch(
    symbols_and_codes: list[tuple[str, str]],
    start: datetime,
    end: datetime,
    *,
    pause_seconds: float = 0.22,
) -> tuple[dict[str, pd.DataFrame], int]:
    """Fetch a maximum of five stocks at once over consecutive seven-day windows."""
    if not symbols_and_codes:
        return {}, 0
    if len(symbols_and_codes) > MAX_CODES_PER_REQUEST:
        raise ValueError(f"INDstocks accepts at most {MAX_CODES_PER_REQUEST} scrip codes per request.")
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Historical range must use timezone-aware start/end with start < end.")

    from indstocks_client import _request

    symbols_by_code = {code: symbol for symbol, code in symbols_and_codes}
    accumulated: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol, _ in symbols_and_codes}
    cursor = start
    calls = 0
    while cursor < end:
        window_end = min(end, cursor + timedelta(days=MAX_DAYS_PER_REQUEST))
        response = _request(
            "GET",
            "/market/historical/5minute",
            params={
                "scrip-codes": ",".join(symbols_by_code),
                "start_time": int(cursor.timestamp() * 1000),
                "end_time": int(window_end.timestamp() * 1000),
            },
        )
        body = response.json()
        if not isinstance(body, dict) or body.get("success") is not True:
            raise RuntimeError("INDstocks five-minute historical API did not report success.")
        data = body.get("data") or {}
        for code, symbol in symbols_by_code.items():
            instrument = data.get(code) or {}
            candles = instrument.get("candles") or []
            if not candles:
                continue
            rows = pd.DataFrame(candles)
            required = {"ts", "o", "h", "l", "c", "v"}
            if not required.issubset(rows.columns):
                raise RuntimeError(f"INDstocks candle response is incomplete for {symbol}.")
            index = pd.to_datetime(
                pd.to_numeric(rows["ts"], errors="coerce"),
                unit="s",
                utc=True,
            ).dt.tz_convert(IST)
            part = pd.DataFrame(
                {
                    "Open": pd.to_numeric(rows["o"], errors="coerce").to_numpy(),
                    "High": pd.to_numeric(rows["h"], errors="coerce").to_numpy(),
                    "Low": pd.to_numeric(rows["l"], errors="coerce").to_numpy(),
                    "Close": pd.to_numeric(rows["c"], errors="coerce").to_numpy(),
                    "Volume": pd.to_numeric(rows["v"], errors="coerce").to_numpy(),
                },
                index=pd.DatetimeIndex(index, name="Datetime"),
            )
            if not part.empty:
                accumulated[symbol].append(part)
        calls += 1
        cursor = window_end
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    result: dict[str, pd.DataFrame] = {}
    for symbol, parts in accumulated.items():
        if not parts:
            result[symbol] = pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=IST, name="Datetime"))
        else:
            merged = pd.concat(parts).sort_index()
            result[symbol] = merged[OHLCV].sort_index().loc[~merged.index.duplicated(keep='last')][OHLCV]
    return result, calls


def _write_manifest(store: Any, root: Path, report: dict[str, Any]) -> None:
    path = root / "backfill_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    uploaded = store.upload_or_update(path, MANIFEST_REMOTE, mime_type="application/json")
    if str(uploaded.get("status") or "").upper() != "OK":
        raise RuntimeError(f"INDstocks backfill manifest Drive write was not confirmed: {uploaded}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=500)
    parser.add_argument("--calendar-days", type=int, default=1000)
    parser.add_argument("--initial-calendar-days", type=int, default=1000)
    parser.add_argument("--minimum-usable-sessions", type=int, default=200)
    parser.add_argument("--minimum-bars-per-session", type=int, default=50)
    parser.add_argument("--pause-seconds", type=float, default=0.22)
    parser.add_argument("--minimum-symbol-coverage", type=float, default=0.90)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--work-root", default="/tmp/nse_5m_indstocks_backfill")
    args = parser.parse_args()

    if not 200 <= args.sessions <= 500:
        raise ValueError("--sessions must be between 200 and 500")
    if not 200 <= args.minimum_usable_sessions <= args.sessions:
        raise ValueError("--minimum-usable-sessions must be 200 through --sessions")
    if args.minimum_bars_per_session < MIN_SESSION_BARS:
        raise ValueError(f"--minimum-bars-per-session must be at least {MIN_SESSION_BARS}")
    if not 0.5 <= args.minimum_symbol_coverage <= 1.0:
        raise ValueError("--minimum-symbol-coverage must be between 0.5 and 1.0")
    if args.pause_seconds < 0.20:
        raise ValueError("INDstocks Data API is limited to five requests/second; pause must be at least 0.20s.")

    from indstocks_client import configured
    from delivery_training_universe import official_nse_symbols
    from persistent_store import GoogleDriveStore, drive_status

    if not configured():
        raise RuntimeError("INDstocks manual access token is not configured in GitHub Actions secrets.")
    preflight = drive_status(require_write=True)
    if not preflight.get("configured") or preflight.get("reachable") is not True or preflight.get("write_ready") is not True:
        raise RuntimeError("Google Drive read/write preflight failed: " + str(preflight.get("reason")))

    if args.symbols.strip():
        symbols = sorted({
            value.strip().upper().replace(".NS", "").replace(".BO", "")
            for value in args.symbols.split(",") if value.strip()
        })
    else:
        symbols = sorted({
            str(value).strip().upper().replace(".NS", "").replace(".BO", "")
            for value in official_nse_symbols(force_refresh=True) if str(value).strip()
        })
    if args.max_symbols > 0:
        symbols = symbols[:args.max_symbols]
    if not symbols:
        raise RuntimeError("Official NSE equity universe is empty.")

    codes = _instrument_codes()
    resolvable = [(symbol, codes[symbol]) for symbol in symbols if symbol in codes]
    resolution_errors = [
        {"symbol": symbol, "status": "ERROR", "error": "No NSE equity scrip-code in INDstocks instrument master."}
        for symbol in symbols if symbol not in codes
    ]
    if not resolvable:
        raise RuntimeError("INDstocks instrument master did not resolve any official NSE equity symbols.")

    end = pd.Timestamp(args.end_date, tz=IST).to_pydatetime() if args.end_date else datetime.now(IST)
    initial_start = end - timedelta(days=int(args.initial_calendar_days))
    local_root = Path(args.work_root)
    local_root.mkdir(parents=True, exist_ok=True)
    store = GoogleDriveStore()
    results: list[dict[str, Any]] = list(resolution_errors)
    total_api_calls = 0
    batches = [resolvable[pos:pos + MAX_CODES_PER_REQUEST] for pos in range(0, len(resolvable), MAX_CODES_PER_REQUEST)]

    for batch_no, batch in enumerate(batches, start=1):
        existing_by_symbol: dict[str, pd.DataFrame] = {}
        starts: dict[str, datetime] = {}
        for symbol, _ in batch:
            remote = f"{REMOTE_ROOT}/symbol={symbol}/{symbol}.parquet"
            local = local_root / f"symbol={symbol}" / f"{symbol}.existing.parquet"
            existing = _download_existing(store, remote, local)
            existing_by_symbol[symbol] = existing
            summary = usable_session_summary(existing, args.minimum_bars_per_session)
            starts[symbol] = choose_backfill_start(
                summary,
                end,
                initial_start,
                args.calendar_days,
                args.sessions,
            )

        # Batch ranges must be common per request. Fetch the earliest start needed
        # by any stock in this five-symbol group, then deduplicate against Drive.
        batch_start = min(starts.values())
        try:
            fetched_by_symbol, calls = fetch_indstocks_5m_batch(
                batch, batch_start, end, pause_seconds=args.pause_seconds
            )
            total_api_calls += calls
        except Exception as exc:
            detail = f"{type(exc).__name__}: {str(exc)[:300]}"
            results.extend({"symbol": symbol, "status": "ERROR", "error": detail} for symbol, _ in batch)
            print(f"BACKFILL_BATCH_FAILED batch={batch_no}/{len(batches)} symbols={[s for s, _ in batch]} error={detail}", flush=True)
            continue

        for symbol, _ in batch:
            existing = existing_by_symbol[symbol]
            fetched = fetched_by_symbol.get(symbol, pd.DataFrame())
            fetched = normalize_indstocks_5m(
                fetched,
                cutoff=pd.Timestamp.now(tz=IST).floor("5min"),
            )
            parts = [frame for frame in (existing, fetched) if frame is not None and not frame.empty]
            merged = pd.concat(parts).sort_index() if parts else pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=IST))
            merged = merged.loc[~merged.index.duplicated(keep="last")]
            merged = retain_latest_sessions(merged, args.sessions, args.minimum_bars_per_session)
            summary = usable_session_summary(merged, args.minimum_bars_per_session)
            latest = pd.to_datetime(summary.get("last_timestamp_ist"), errors="coerce")
            fresh = pd.notna(latest) and latest >= pd.Timestamp(end) - pd.Timedelta(days=FRESHNESS_WINDOW_DAYS)
            if merged.empty:
                results.append({"symbol": symbol, "status": "NO_DATA", "usable_sessions": 0, "source": "INDSTOCKS_HISTORICAL_5M"})
                continue

            local = local_root / f"symbol={symbol}" / f"{symbol}.parquet"
            local.parent.mkdir(parents=True, exist_ok=True)
            remote = f"{REMOTE_ROOT}/symbol={symbol}/{symbol}.parquet"
            merged.to_parquet(local, index=True, compression="zstd")
            uploaded = store.upload_or_update(local, remote, mime_type="application/vnd.apache.parquet")
            if str(uploaded.get("status") or "").upper() != "OK":
                results.append({"symbol": symbol, "status": "ERROR", "error": f"Drive upload not confirmed: {uploaded}"})
                continue
            ready = summary["usable_sessions"] >= args.minimum_usable_sessions
            results.append({
                "symbol": symbol,
                "status": "OK" if ready and fresh else "STALE_LATEST_DATA" if ready else "INSUFFICIENT_HISTORY",
                "source": "INDSTOCKS_HISTORICAL_5M",
                "remote_path": remote,
                "rows": int(len(merged)),
                "fetched_rows": int(len(fetched)),
                "usable_sessions": int(summary["usable_sessions"]),
                "raw_sessions": int(summary["raw_sessions"]),
                "fresh_latest_data": bool(fresh),
                "first_timestamp_ist": summary["first_timestamp_ist"],
                "last_timestamp_ist": summary["last_timestamp_ist"],
                "drive_action": uploaded.get("action"),
                "drive_bytes": uploaded.get("size"),
            })

        print(
            f"BACKFILL_PROGRESS provider=INDSTOCKS batch={batch_no}/{len(batches)} "
            f"completed_symbols={min(batch_no * MAX_CODES_PER_REQUEST, len(resolvable))}/{len(resolvable)} "
            f"api_calls={total_api_calls}",
            flush=True,
        )

    results.sort(key=lambda row: row["symbol"])
    ready = [row for row in results if row.get("status") == "OK"]
    with_200 = [row for row in results if int(row.get("usable_sessions") or 0) >= args.minimum_usable_sessions]
    required = max(1, int(math.ceil(len(symbols) * args.minimum_symbol_coverage)))
    report = {
        "audit": "nse_native_five_minute_historical_backfill",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "INDSTOCKS_HISTORICAL_5M_BATCH_API",
        "bar_interval_minutes": 5,
        "timestamp_convention": "provider_candle_start_plus_5_minutes_as_bar_close",
        "maximum_instruments_per_request": MAX_CODES_PER_REQUEST,
        "maximum_calendar_days_per_request": MAX_DAYS_PER_REQUEST,
        "documented_data_api_rate_limit_per_second": 5,
        "request_pause_seconds": args.pause_seconds,
        "api_calls": total_api_calls,
        "target_sessions_per_symbol": args.sessions,
        "minimum_usable_sessions": args.minimum_usable_sessions,
        "minimum_bars_per_usable_session": args.minimum_bars_per_session,
        "calendar_days_requested": args.calendar_days,
        "initial_calendar_days": args.initial_calendar_days,
        "range_start_ist": initial_start.isoformat(),
        "range_end_ist": end.isoformat(),
        "official_symbols_requested": len(symbols),
        "symbols_resolved": len(resolvable),
        "symbols_with_200_plus_sessions": len(with_200),
        "symbols_with_fresh_latest_data": sum(row.get("fresh_latest_data") is True for row in results),
        "symbols_with_200_plus_sessions_and_fresh_data": len(ready),
        "symbols_with_data_after_2026_09_24": sum(
            bool(row.get("last_timestamp_ist")) and row["last_timestamp_ist"][:10] > "2026-09-24"
            for row in results
        ),
        "latest_archive_timestamp_ist": max(
            (row.get("last_timestamp_ist") or "" for row in results), default=None
        ),
        "symbols_with_errors": sum(row.get("status") == "ERROR" for row in results),
        "symbols_below_200_sessions": sum(row.get("status") == "INSUFFICIENT_HISTORY" for row in results),
        "symbols_with_data_but_no_200_sessions": sum(
            row.get("status") in {"INSUFFICIENT_HISTORY", "STALE_LATEST_DATA"} for row in results
        ),
        "overall_status": "PASS" if len(ready) >= required else "NEEDS_ATTENTION",
        "required_symbols_with_200_plus_sessions_and_fresh_data": required,
        "symbols": results,
        "manifest_drive_path": MANIFEST_REMOTE,
        "no_auto_orders": True,
        "production_model_changed": False,
    }
    _write_manifest(store, local_root, report)
    print("BACKFILL_MANIFEST=" + json.dumps({
        key: report.get(key)
        for key in (
            "overall_status", "official_symbols_requested", "symbols_resolved",
            "symbols_with_200_plus_sessions_and_fresh_data",
            "required_symbols_with_200_plus_sessions_and_fresh_data",
            "symbols_with_data_after_2026_09_24", "latest_archive_timestamp_ist",
            "api_calls", "manifest_drive_path",
        )
    }, sort_keys=True))
    return 0 if report["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
