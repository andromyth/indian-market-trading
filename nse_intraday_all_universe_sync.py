def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause-seconds", type=float, default=0.15)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-symbols", type=int, default=0,
                    help="0 means the complete current NSE universe")
    args = ap.parse_args()

    from datetime import datetime
    from zoneinfo import ZoneInfo

    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    market_day = now_ist.weekday() < 5
    market_window = (now_ist.hour, now_ist.minute) >= (9, 10) and (now_ist.hour, now_ist.minute) <= (15, 45)

    if not market_day or not market_window:
        print(
            f"INTRADAY SYNC SKIPPED: NSE public current-chart capture is only useful "
            f"during the trading window; now={now_ist.isoformat()}",
            flush=True,
        )
        return

    universe = official_nse_symbols(force_refresh=False)
    start = max(0, args.start_index)
    symbols = universe[start:]
    if args.max_symbols > 0:
        symbols = symbols[:args.max_symbols]

    run = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "universe_symbol_count": len(universe),
        "start_index": start,
        "requested_symbol_count": len(symbols),
        "source": "NSE_OFFICIAL_CHART",
        "target": "production_data/intraday_1m/{YYYY}/{MM}/{DD}/{SYMBOL}.parquet",
        "volume_policy": "never_fabricate_minute_volume",
        "symbols": {},
    }

    print(
        f"INTRADAY NSE ALL-UNIVERSE SYNC: universe={len(universe):,} "
        f"requested={len(symbols):,} start_index={start}",
        flush=True,
    )

    for i, symbol in enumerate(symbols, start=1):
        try:
            frame = fetch_intraday_1m_current(symbol, sync_drive=True)
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
        if args.pause_seconds > 0:
            time.sleep(args.pause_seconds)
        if i % 25 == 0:
            run["updated_at"] = datetime.now(timezone.utc).isoformat()
            run["completed_symbols"] = i
            _write_manifest(run)

    run["completed_at"] = datetime.now(timezone.utc).isoformat()
    run["completed_symbols"] = len(symbols)
    run["ok_symbols"] = sum(v.get("status") == "OK" for v in run["symbols"].values())
    run["error_symbols"] = sum(v.get("status") == "ERROR" for v in run["symbols"].values())
    _write_manifest(run)

    print(
        f"INTRADAY SYNC COMPLETE: ok={run['ok_symbols']:,} "
        f"errors={run['error_symbols']:,}",
        flush=True,
    )
#!/usr/bin/env python3
"""Capture NSE public 1-minute intraday data for the full official equity universe."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from delivery_training_universe import official_nse_symbols
from nse_data_layer import fetch_intraday_1m_current
from persistent_store import GoogleDriveStore

MANIFEST_REMOTE = "production_data/nse_official/intraday_1m/sync_manifests/latest.json"

def _write_manifest(payload: dict) -> None:
    local = Path("data/nse_official/intraday_1m_sync_manifest.json")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    try:
        GoogleDriveStore().upload_or_update(local, MANIFEST_REMOTE, mime_type="application/json")
    except Exception:
        pass

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause-seconds", type=float, default=0.15)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-symbols", type=int, default=0,
                    help="0 means the complete current NSE universe")
    args = ap.parse_args()

    universe = official_nse_symbols(force_refresh=False)
    start = max(0, args.start_index)
    symbols = universe[start:]
    if args.max_symbols > 0:
        symbols = symbols[:args.max_symbols]

    run = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "universe_symbol_count": len(universe),
        "start_index": start,
        "requested_symbol_count": len(symbols),
        "source": "NSE_OFFICIAL_CHART",
        "target": "production_data/intraday_1m/{YYYY}/{MM}/{DD}/{SYMBOL}.parquet",
        "volume_policy": "never_fabricate_minute_volume",
        "symbols": {},
    }

    print(
        f"INTRADAY NSE ALL-UNIVERSE SYNC: universe={len(universe):,} "
        f"requested={len(symbols):,} start_index={start}",
        flush=True,
    )

    for i, symbol in enumerate(symbols, start=1):
        try:
            frame = fetch_intraday_1m_current(symbol, sync_drive=True)
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
        if args.pause_seconds > 0:
            time.sleep(args.pause_seconds)
        if i % 25 == 0:
            run["updated_at"] = datetime.now(timezone.utc).isoformat()
            run["completed_symbols"] = i
            _write_manifest(run)

    run["completed_at"] = datetime.now(timezone.utc).isoformat()
    run["completed_symbols"] = len(symbols)
    run["ok_symbols"] = sum(v.get("status") == "OK" for v in run["symbols"].values())
    run["error_symbols"] = sum(v.get("status") == "ERROR" for v in run["symbols"].values())
    _write_manifest(run)

    print(
        f"INTRADAY SYNC COMPLETE: ok={run['ok_symbols']:,} "
        f"errors={run['error_symbols']:,}",
        flush=True,
    )

if __name__ == "__main__":
    main()
