#!/usr/bin/env python3
"""Stage Hugging Face NSE minute bars as validated 5-minute Parquet.

STAGING ONLY: never writes to the canonical archive, Google Drive, production
worker, scheduler, credentials, or model artifacts.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

DEFAULT_REPO = "xxparthparekhxx/indian-stock-market-minute-data"
DEFAULT_OUT = Path("production_data/nse_official/intraday_seed/hf_seed_staging")
IST = ZoneInfo("Asia/Kolkata")
REQUIRED = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
API = "https://huggingface.co/api/datasets/{repo}/tree/main/minute?recursive=true&expand=false"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

def list_shards(repo: str, session: requests.Session) -> list[dict]:
    url = API.format(repo=repo)
    result = []
    while url:
        response = session.get(url, timeout=60)
        response.raise_for_status()
        result.extend(x for x in response.json()
                      if x.get("type") == "file" and x.get("path", "").endswith(".parquet"))
        url = response.links.get("next", {}).get("url")
    if not result:
        raise RuntimeError(f"No minute Parquet shards found in {repo}")
    return sorted(result, key=lambda x: x["path"])

def download_shard(repo: str, item: dict, path: Path, session: requests.Session) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    with session.get(RESOLVE.format(repo=repo, path=item["path"]), stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        with temp.open("wb") as f:
            for chunk in r.iter_content(4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    temp.replace(path)

def five_minute_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complete 5-minute buckets; discard buckets missing any source minute."""
    empty_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    if "oi" in frame.columns:
        empty_cols.append("oi")
    if frame.empty:
        return pd.DataFrame(columns=empty_cols)
    x = frame.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    missing = REQUIRED - set(x.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce").dt.tz_convert(IST)
    for c in ("open", "high", "low", "close", "volume", "oi"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    x = x.loc[x["symbol"].ne("") & x["symbol"].ne("NAN")]
    clock = x["timestamp"].dt.strftime("%H:%M")
    x = x.loc[(clock >= "09:15") & (clock < "15:30")].copy()
    x = x.drop_duplicates(["symbol", "timestamp"], keep="last")
    x["_session"] = x["timestamp"].dt.date
    x["_bucket"] = x["timestamp"].dt.floor("5min")
    aggregations = {
        "open": ("open", "first"), "high": ("high", "max"),
        "low": ("low", "min"), "close": ("close", "last"),
        "volume": ("volume", "sum"), "minutes": ("timestamp", "nunique"),
    }
    if "oi" in x.columns:
        aggregations["oi"] = ("oi", "last")
    bars = x.groupby(["symbol", "_session", "_bucket"], sort=True, observed=True).agg(
        **aggregations
    ).reset_index()
    bars = bars.loc[bars["minutes"].eq(5)].copy()
    bars["timestamp"] = bars.pop("_bucket") + pd.Timedelta(minutes=5)
    bars = bars.drop(columns=["_session", "minutes"])
    valid = bars[["open", "high", "low", "close"]].gt(0).all(axis=1)
    valid &= bars["high"] >= bars[["open", "low", "close"]].max(axis=1)
    valid &= bars["low"] <= bars[["open", "high", "close"]].min(axis=1)
    valid &= bars["volume"].ge(0)
    bars = bars.loc[valid].sort_values(["symbol", "timestamp"])
    bars["timestamp"] = bars["timestamp"].dt.tz_convert("UTC")
    return bars.reset_index(drop=True)

def session_quality(frame: pd.DataFrame) -> dict:
    """Count regular sessions only when all 75 right-labelled bars are present."""
    if frame.empty:
        return {"complete_sessions": 0, "partial_sessions": 0,
                "first_session": None, "last_session": None}
    local = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.tz_convert(IST).dropna()
    if local.empty:
        return {"complete_sessions": 0, "partial_sessions": 0,
                "first_session": None, "last_session": None}
    expected = set(pd.date_range("09:20", "15:30", freq="5min").strftime("%H:%M"))
    complete = 0
    for _, values in local.groupby(local.dt.date):
        labels = set(values.dt.strftime("%H:%M"))
        if labels == expected and len(values) == 75:
            complete += 1
    total = local.dt.date.nunique()
    return {"complete_sessions": complete, "partial_sessions": int(total - complete),
            "first_session": local.dt.date.min().isoformat(),
            "last_session": local.dt.date.max().isoformat()}


def usable_sessions(frame: pd.DataFrame) -> int:
    """Backward-compatible name; counts complete regular sessions only."""
    return session_quality(frame)["complete_sessions"]

def process_shard(path: Path, out: Path, min_sessions: int, max_sessions: int) -> list[dict]:
    parquet = pq.ParquetFile(path)
    columns = [c for c in parquet.schema.names if c in REQUIRED or c == "oi"]
    if REQUIRED - set(columns):
        raise ValueError(f"{path}: missing columns {sorted(REQUIRED - set(columns))}")
    parts = out / ".parts"
    parts.mkdir(parents=True, exist_ok=True)
    touched, piece = set(), 0
    pending = pd.DataFrame(columns=columns)

    def write_bars(bars: pd.DataFrame) -> None:
        nonlocal piece
        for symbol, part in bars.groupby("symbol", sort=False):
            if not re.fullmatch(r"[A-Z0-9&._-]{1,40}", symbol):
                continue
            folder = parts / f"symbol={symbol}"
            folder.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(part.drop(columns=["symbol"]), preserve_index=False),
                           folder / f"part-{path.stem}-{piece:06d}.parquet", compression="snappy")
            piece += 1
            touched.add(symbol)

    # The source is documented as symbol/timestamp sorted. Keep the last bucket
    # across Arrow batch boundaries so a 5-minute bar is never split and lost.
    for batch in parquet.iter_batches(batch_size=250_000, columns=columns):
        current = batch.to_pandas()
        if not pending.empty:
            current = pd.concat([pending, current], ignore_index=True)
            pending = pd.DataFrame(columns=columns)
        ts = pd.to_datetime(current["timestamp"], utc=True, errors="coerce").dt.tz_convert(IST)
        sym = current["symbol"].astype(str).str.strip().str.upper()
        bucket = ts.dt.floor("5min")
        last_symbol, last_bucket = sym.iloc[-1], bucket.iloc[-1]
        hold = sym.eq(last_symbol) & bucket.eq(last_bucket)
        pending = current.loc[hold, columns].copy()
        write_bars(five_minute_bars(current.loc[~hold, columns]))
    if not pending.empty:
        write_bars(five_minute_bars(pending))

    summaries = []
    for symbol in sorted(touched):
        folder = parts / f"symbol={symbol}"
        files = sorted(folder.glob("*.parquet"))
        frames = [pd.read_parquet(p) for p in files]
        combined = pd.concat(frames, ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
        combined = combined.dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        dates = combined["timestamp"].dt.tz_convert(IST).dt.date
        keep = sorted(dates.unique())[-max_sessions:]
        combined = combined.loc[dates.isin(keep)].copy()
        target = out / ".shard_outputs" / path.stem / f"symbol={symbol}" / f"{symbol}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(target, index=False, compression="snappy")
        quality = session_quality(combined)
        sessions = quality["complete_sessions"]
        summaries.append({"symbol": symbol, "rows_5m": len(combined),
                          "usable_sessions": sessions, **quality,
                          "meets_minimum": sessions >= min_sessions, "path": str(target)})
        for p in files:
            p.unlink(missing_ok=True)
        try:
            folder.rmdir()
        except OSError:
            pass
    return summaries

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-sessions", type=int, default=200)
    parser.add_argument("--max-sessions", type=int, default=500)
    parser.add_argument("--max-shards", type=int, default=0, help="Test only first N shards; 0 means all")
    parser.add_argument("--keep-downloads", action="store_true")
    args = parser.parse_args(argv)
    if args.min_sessions < 1 or args.max_sessions < args.min_sessions:
        parser.error("require --max-sessions >= --min-sessions >= 1")
    args.output.mkdir(parents=True, exist_ok=True)
    downloads = args.output / ".downloads"
    session = requests.Session()
    shards = list_shards(args.repo, session)
    if args.max_shards:
        shards = shards[:args.max_shards]
    symbols: dict[str, dict] = {}
    errors = []
    for i, item in enumerate(shards, 1):
        local = downloads / item["path"].replace("/", "__")
        try:
            if not local.exists() or local.stat().st_size == 0:
                download_shard(args.repo, item, local, session)
            for row in process_shard(local, args.output, args.min_sessions, args.max_sessions):
                source_path = Path(row["path"])
                target_path = args.output / f"symbol={row['symbol']}" / f"{row['symbol']}.parquet"
                target_path.parent.mkdir(parents=True, exist_ok=True)
                old = symbols.get(row["symbol"])
                if old:
                    old_path = Path(old["path"])
                    merged = pd.concat([pd.read_parquet(old_path), pd.read_parquet(source_path)], ignore_index=True)
                    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
                    merged = merged.dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
                    dates = merged["timestamp"].dt.tz_convert(IST).dt.date
                    merged = merged.loc[dates.isin(sorted(dates.unique())[-args.max_sessions:])]
                else:
                    merged = pd.read_parquet(source_path)
                merged.to_parquet(target_path, index=False, compression="snappy")
                quality = session_quality(merged)
                sessions = quality["complete_sessions"]
                row.update(path=str(target_path), rows_5m=len(merged),
                           usable_sessions=sessions, **quality,
                           meets_minimum=sessions >= args.min_sessions)
                symbols[row["symbol"]] = row
            print(f"[{i}/{len(shards)}] {item['path']} symbols_written={len(symbols)}", flush=True)
            if not args.keep_downloads:
                local.unlink(missing_ok=True)
        except Exception as exc:
            errors.append({"shard": item.get("path"), "error": f"{type(exc).__name__}: {exc}"})
            print(f"ERROR {item.get('path')}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    rows = sorted(symbols.values(), key=lambda x: x["symbol"])
    report = {"dataset_repo": args.repo, "mode": "STAGING_ONLY",
              "canonical_archive_modified": False, "production_worker_modified": False,
              "shards_selected": len(shards), "symbols_written": len(rows),
              "symbols_meeting_min_sessions": sum(r["meets_minimum"] for r in rows),
              "symbols_below_min_sessions": sum(not r["meets_minimum"] for r in rows),
              "minimum_sessions": args.min_sessions, "maximum_sessions_retained": args.max_sessions,
              "errors": errors, "symbols": rows}
    report_path = args.output / "hf_seed_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "symbols"}, indent=2))
    print(f"Audit: {report_path}")
    return 1 if errors else 0

if __name__ == "__main__":
    raise SystemExit(main())
