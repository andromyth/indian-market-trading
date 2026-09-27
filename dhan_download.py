"""Local historical downloader backed by DhanHQ v2.

No order APIs are used.  The downloader stores Dhan data under:
  data/historical/dhan/NSE_EQ/<interval>/<SYMBOL>.parquet

Dhan's v2 intraday API permits 1/5/15/25/60-minute candles and documents a
maximum 90-day polling window per request. This module automatically chunks
long ranges and deduplicates timestamps before writing Parquet.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from dhan_client import DhanConfig, historical_daily, historical_intraday_range, resolve_equity_symbol, get_access_token

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "data" / "historical" / "dhan" / "NSE_EQ"
MANIFEST = ROOT / "manifest.json"


def load_manifest() -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"updated_at": None, "files": {}}


def save_manifest(m: dict) -> None:
    m["updated_at"] = datetime.now(timezone.utc).isoformat()
    ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.to_parquet(path)


def download_symbol(symbol: str, start: str, end: str, interval: str = "1m", overwrite: bool = False, oi: bool = False) -> dict:
    cfg = DhanConfig.from_env()
    if not cfg.client_id or not get_access_token(cfg):
        raise RuntimeError("Dhan client ID and valid access token are required. Run: python dhan_auth.py login")
    info = resolve_equity_symbol(symbol)
    root = info["symbol"]
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00")) if "T" in end else datetime.fromisoformat(end)
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00")) if "T" in start else datetime.fromisoformat(start)
    if interval.lower() in {"1d", "day", "daily"}:
        df = historical_daily(cfg, info["security_id"], start_dt.date().isoformat(), end_dt.date().isoformat())
    else:
        iv = int(str(interval).lower().replace("min", "m").replace("m", ""))
        df = historical_intraday_range(
            cfg,
            info["security_id"],
            iv,
            start_dt,
            end_dt,
            oi=oi,
            pause_seconds=0.25,
        )
    out_path = ROOT / interval.lower() / f"{root}.parquet"
    if out_path.exists() and not overwrite:
        try:
            old = pd.read_parquet(out_path)
            if not old.empty and not df.empty:
                df = pd.concat([old, df]).sort_index()
                df = df[~df.index.duplicated(keep="last")]
        except Exception:
            pass
    if not df.empty:
        _write(df, out_path)
    m = load_manifest()
    key = f"{interval}:{root}"
    m["files"][key] = {
        "symbol": root,
        "interval": interval,
        "security_id": info["security_id"],
        "start_requested": start,
        "end_requested": end,
        "rows": int(len(df)),
        "path": str(out_path.relative_to(BASE_DIR)),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source": "DhanHQ v2",
        "status": "OK" if not df.empty else "NO_DATA",
    }
    save_manifest(m)
    return m["files"][key]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD or ISO datetime")
    p.add_argument("--end", required=True, help="YYYY-MM-DD or ISO datetime; end is exclusive in Dhan API")
    p.add_argument("--interval", default="1m", choices=["1m", "5m", "15m", "25m", "60m", "1d"])
    p.add_argument("--oi", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    results = []
    for sym in args.symbols:
        try:
            results.append(download_symbol(sym, args.start, args.end, args.interval, args.overwrite, args.oi))
        except Exception as exc:
            results.append({"symbol": sym, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps(results, indent=2, default=str))
    return 0 if all(r.get("status") not in {"ERROR"} for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
