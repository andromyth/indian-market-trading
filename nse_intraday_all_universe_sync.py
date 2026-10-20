#!/usr/bin/env python3
"""Capture NSE public 1-minute intraday data for the official equity universe."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from delivery_training_universe import official_nse_symbols
import nse_data_layer
from market_session_guard import is_nse_trading_day
from persistent_store import GoogleDriveStore

MANIFEST_REMOTE = "production_data/nse_official/intraday_1m/sync_manifests/latest.json"
IST = ZoneInfo("Asia/Kolkata")


def _write_manifest(payload: dict) -> None:
    """Write the local progress manifest and best-effort mirror it to Drive."""
    local = Path("data/nse_official/intraday_1m_sync_manifest.json")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    try:
        result = GoogleDriveStore().upload_or_update(
            local, MANIFEST_REMOTE, mime_type="application/json"
        )
        print(f"INTRADAY_MANIFEST_DRIVE_SYNC: {result.get('status', 'UNKNOWN')}", flush=True)
    except Exception as exc:
        # Continue the capture; make the persistence problem visible in job logs.
        print(
            f"INTRADAY_MANIFEST_DRIVE_SYNC_WARNING: {type(exc).__name__}: {exc}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-symbols", type=int, default=0,
        help="0 means the complete current NSE universe",
    )
    args = parser.parse_args()
    if args.pause_seconds < 0:
        parser.error("--pause-seconds must be >= 0")
    if args.start_index < 0:
        parser.error("--start-index must be >= 0")
    if args.max_symbols < 0:
        parser.error("--max-symbols must be >= 0")

    now_ist = datetime.now(IST)
    market_window = (now_ist.hour, now_ist.minute) >= (9, 10) and (
        now_ist.hour, now_ist.minute
    ) <= (15, 45)
    trading_day = is_nse_trading_day(now_ist.date())
    if not trading_day or not market_window:
        if not trading_day:
            reason = "weekend or NSE exchange holiday"
        else:
            reason = "outside the public intraday quote-chart window"
        print(
            f"INTRADAY SYNC SKIPPED: {reason}; "
            f"now={now_ist.isoformat()}",
            flush=True,
        )
        return

    universe = official_nse_symbols(force_refresh=False)
    symbols = universe[args.start_index:]
    if args.max_symbols:
        symbols = symbols[:args.max_symbols]

    run = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "universe_symbol_count": len(universe),
        "start_index": args.start_index,
        "requested_symbol_count": len(symbols),
        "source": "NSE_OFFICIAL_CHART",
        "target": "production_data/intraday_1m/{YYYY}/{MM}/{DD}/{SYMBOL}.parquet",
        "volume_policy": "never_fabricate_minute_volume",
        "symbols": {},
        "ok_symbols": 0,
        "no_data_symbols": 0,
        "error_symbols": 0,
    }
    print(
        f"INTRADAY NSE ALL-UNIVERSE SYNC: universe={len(universe):,} "
        f"requested={len(symbols):,} start_index={args.start_index}",
        flush=True,
    )

    for i, symbol in enumerate(symbols, start=1):
        try:
            frame = nse_data_layer.fetch_intraday_1m_current(symbol, sync_drive=True)
            if frame.empty:
                # Empty NSE chart payloads previously counted as successful rows.
                # Preserve a minimal, non-sensitive HTTP diagnostic so a fully
                # empty run cannot look like a successful data capture.
                diagnostic = getattr(nse_data_layer, "_last_nse_diagnostic", {}) or {}
                safe_diagnostic = {
                    key: diagnostic.get(key)
                    for key in ("status_code", "content_type", "bytes", "attempt", "path")
                    if diagnostic.get(key) is not None
                }
                run["symbols"][symbol] = {
                    "status": "NO_DATA",
                    "rows": 0,
                    "source_diagnostic": safe_diagnostic,
                }
                print(
                    f"[{i}/{len(symbols)}] {symbol}: NO_DATA 0 1m rows "
                    f"diagnostic={json.dumps(safe_diagnostic, sort_keys=True)}",
                    flush=True,
                )
            else:
                run["symbols"][symbol] = {"status": "OK", "rows": int(len(frame))}
                print(f"[{i}/{len(symbols)}] {symbol}: {len(frame):,} 1m rows", flush=True)
        except Exception as exc:
            run["symbols"][symbol] = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"[{i}/{len(symbols)}] {symbol}: ERROR {type(exc).__name__}: {exc}",
                flush=True,
            )
        if args.pause_seconds:
            time.sleep(args.pause_seconds)
        if i % 25 == 0:
            run["updated_at"] = datetime.now(timezone.utc).isoformat()
            run["completed_symbols"] = i
            run["ok_symbols"] = sum(v.get("status") == "OK" for v in run["symbols"].values())
            run["no_data_symbols"] = sum(v.get("status") == "NO_DATA" for v in run["symbols"].values())
            run["error_symbols"] = sum(v.get("status") == "ERROR" for v in run["symbols"].values())
            _write_manifest(run)

    run["completed_at"] = datetime.now(timezone.utc).isoformat()
    run["completed_symbols"] = len(symbols)
    run["ok_symbols"] = sum(v.get("status") == "OK" for v in run["symbols"].values())
    run["no_data_symbols"] = sum(v.get("status") == "NO_DATA" for v in run["symbols"].values())
    run["error_symbols"] = sum(v.get("status") == "ERROR" for v in run["symbols"].values())
    _write_manifest(run)
    print(
        f"INTRADAY SYNC COMPLETE: ok={run['ok_symbols']:,} "
        f"no_data={run['no_data_symbols']:,} errors={run['error_symbols']:,}",
        flush=True,
    )
    if symbols and run["ok_symbols"] == 0 and run["no_data_symbols"] == len(symbols):
        raise SystemExit(
            "No NSE one-minute rows were returned for any requested symbol on a "
            "declared trading day; treating the capture as failed rather than green."
        )


if __name__ == "__main__":
    main()
