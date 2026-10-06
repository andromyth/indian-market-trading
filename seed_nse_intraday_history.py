#!/usr/bin/env python3
"""Seed pooled NSE 1-minute training data from a public historical minute archive.

This is a historical training seed only. Live/incremental capture remains the
official NSE public chart path. Source provenance is recorded in a manifest.
No orders are placed.
"""
from __future__ import annotations

import argparse, hashlib, json, shutil, zipfile
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

SOURCE_API = "https://api.github.com/repos/olo-labs/olo-testdata-nse-min-candle/contents/Minute/NSE?ref=main&per_page=1000"
SOURCE_RAW = "https://raw.githubusercontent.com/olo-labs/olo-testdata-nse-min-candle/main/Minute/NSE/{name}"
IST = "Asia/Kolkata"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def list_archives() -> list[tuple[date, str]]:
    r = requests.get(
        SOURCE_API,
        headers={"Accept":"application/vnd.github+json","User-Agent":"indian-market-trading-intraday-seeder/1.0"},
        timeout=30,
    )
    r.raise_for_status()
    out=[]
    for item in r.json():
        name=str(item.get("name",""))
        if not name.lower().endswith(".zip"):
            continue
        try:
            d=datetime.strptime(name[:-4], "%d%m%Y").date()
        except ValueError:
            continue
        out.append((d,name))
    return sorted(set(out))


def source_symbol(ticker: str) -> str:
    # The source documents provider-specific identifiers such as
    # SYMBOL.SERIES.NSE. For this training seed, the first component is used
    # only when it exactly matches the current official NSE equity universe.
    return str(ticker).strip().upper().split(".",1)[0]


def read_archive(content: bytes, universe: set[str], src_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(BytesIO(content)) as zf:
        names=[n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            return pd.DataFrame()
        with zf.open(names[0]) as fh:
            df=pd.read_csv(fh)
    df.columns=[str(c).strip() for c in df.columns]
    required={"Ticker","Date","Time","Open","High","Low","Close","Volume"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"{src_name}: missing columns {sorted(required-set(df.columns))}")
    df["symbol"]=df["Ticker"].map(source_symbol)
    df=df[df["symbol"].isin(universe)].copy()
    if df.empty:
        return df
    df["timestamp"]=pd.to_datetime(
        df["Date"].astype(str)+" "+df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    for c in ["Open","High","Low","Close","Volume"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["symbol","timestamp","Open","High","Low","Close"])
    df=df[["symbol","timestamp","Open","High","Low","Close","Volume"]]
    df["source"]="OLO_PUBLIC_NSE_MINUTE_ARCHIVE"
    df["source_file"]=src_name
    return df.sort_values(["symbol","timestamp"])


def build_symbol_partitions(seed_root: Path) -> dict:
    daily_dir = seed_root / "daily"
    symbol_dir = seed_root / "symbols"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    daily_files = sorted(daily_dir.glob("*.parquet"))
    if not daily_files:
        raise RuntimeError("No filtered daily seed files were produced.")

    # Use DuckDB's streaming Parquet scan for the broad history. The previous
    # implementation concatenated every day into one pandas DataFrame, which
    # becomes impractical at 500-750 sessions across the full NSE universe.
    import duckdb

    for p in symbol_dir.glob("symbol=*"):
        if p.is_dir():
            shutil.rmtree(p)

    glob = str(daily_dir / "*.parquet").replace("'", "''")
    conn = duckdb.connect()
    try:
        conn.execute(
            f"""
            COPY (
                SELECT
                    symbol,
                    timestamp,
                    Open,
                    High,
                    Low,
                    Close,
                    Volume,
                    source,
                    source_file
                FROM read_parquet('{glob}')
                WHERE symbol IS NOT NULL
                  AND timestamp IS NOT NULL
            )
            TO '{str(symbol_dir).replace("'", "''")}'
            (
                FORMAT PARQUET,
                PARTITION_BY (symbol),
                COMPRESSION ZSTD,
                OVERWRITE_OR_IGNORE
            );
            """
        )

        row_count = int(
            conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('{glob}')"
            ).fetchone()[0]
        )
        symbol_count = int(
            conn.execute(
                f"SELECT COUNT(DISTINCT symbol) FROM read_parquet('{glob}')"
            ).fetchone()[0]
        )
    finally:
        conn.close()

    return {
        "partition_files": len(list(symbol_dir.rglob("*.parquet"))),
        "partition_symbols": symbol_count,
        "seed_rows": row_count,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--drive-root",default="/content/drive/MyDrive/Indian_Market_Research")
    ap.add_argument(
        "--lookback-calendar-days",
        type=int,
        default=1200,
        help="Calendar-day window used to collect the historical minute seed.",
    )
    ap.add_argument(
        "--max-archives",
        type=int,
        default=750,
        help="Maximum daily archives to retain; 750 is the preferred ~3-year target.",
    )
    ap.add_argument("--end-date",default="")
    ap.add_argument("--force",action="store_true")
    args=ap.parse_args()

    from delivery_training_universe import official_nse_symbols
    universe=set(official_nse_symbols(force_refresh=True))
    if not universe:
        raise RuntimeError("Official NSE universe is empty")

    seed_root=Path(args.drive_root)/"production_data"/"nse_official"/"intraday_seed"/"olo_minute_archive"
    daily_dir=seed_root/"daily"; daily_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=seed_root/"SEED_MANIFEST.json"

    end=date.fromisoformat(args.end_date) if args.end_date else date.today()
    start=end-timedelta(days=max(1,args.lookback_calendar_days))
    available=[x for x in list_archives() if start <= x[0] <= end]
    available=available[-max(1,args.max_archives):]

    manifest={
        "source":"olo-labs/olo-testdata-nse-min-candle",
        "source_url":"https://github.com/olo-labs/olo-testdata-nse-min-candle",
        "historical_training_only":True,
        "official_live_source":"NSE_OFFICIAL_CHART",
        "preferred_training_history_sessions": 750,
        "minimum_preferred_training_history_sessions": 500,
        "universe_size":len(universe),
        "selected_archives":[],
        "updated_at":datetime.now(timezone.utc).isoformat(),
    }
    print(f"INTRADAY HISTORICAL SEED: official_universe={len(universe):,} selected_archives={len(available):,}",flush=True)

    s=requests.Session()
    s.headers.update({"User-Agent":"indian-market-trading-intraday-seeder/1.0"})
    for i,(d,name) in enumerate(available,1):
        out=daily_dir/f"{d:%Y%m%d}.parquet"
        url=SOURCE_RAW.format(name=name)
        if out.exists() and not args.force:
            df=pd.read_parquet(out)
            manifest["selected_archives"].append({"date":str(d),"archive":name,"status":"REUSED","rows":int(len(df)),"symbols":int(df["symbol"].nunique()) if "symbol" in df else 0})
            print(f"[{i}/{len(available)}] {d}: REUSED rows={len(df):,} symbols={df['symbol'].nunique() if 'symbol' in df else 0}",flush=True)
            continue
        r=s.get(url,timeout=120)
        r.raise_for_status()
        raw=r.content
        df=read_archive(raw,universe,name)
        if df.empty:
            print(f"[{i}/{len(available)}] {d}: 0 matching NSE equities",flush=True)
            continue
        df.to_parquet(out,index=False)
        manifest["selected_archives"].append({"date":str(d),"archive":name,"status":"DOWNLOADED","rows":int(len(df)),
            "symbols":int(df["symbol"].nunique()),"sha256":sha256_bytes(raw),"bytes":len(raw),"url":url})
        print(f"[{i}/{len(available)}] {d}: rows={len(df):,} symbols={df['symbol'].nunique():,}",flush=True)

    part=build_symbol_partitions(seed_root)
    manifest["partitioning"]=part
    manifest["updated_at"]=datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
    print(json.dumps({"seed_root":str(seed_root),"universe_size":len(universe),
                      "archives":len(manifest["selected_archives"]),**part},indent=2),flush=True)

if __name__=="__main__":
    main()
