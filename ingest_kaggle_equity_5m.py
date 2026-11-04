#!/usr/bin/env python3
"""Stream Kaggle stock CSV/ZIP history into the canonical five-minute seed.

1-minute candles are aggregated to completed 5-minute OHLCV. Existing 5-minute
candles are timestamp-normalized. Kaggle is stored separately from INDstocks;
the trainer overlays Kaggle on duplicate timestamps and uses INDstocks for gaps.
NIFTY index data is deliberately excluded and imported by the dedicated context
importer.
"""
from __future__ import annotations
import argparse, hashlib, io, json, re, shutil, tempfile, zipfile
from pathlib import Path
from typing import Any, Iterator, TextIO
import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
MIN_SESSION_BARS = 50
REMOTE = Path("production_data/nse_official/intraday_seed/olo_5m_archive/symbols")
INDEX_SYMBOLS = {"NIFTY", "NIFTY50", "NIFTY_50", "NIFTY50INDEX", "NIFTY50_INDEX", "NIFTY_INDEX", "NIFTYBANK", "NIFTY_BANK", "BANKNIFTY", "NIFTYIT"}
ALIASES = {
    "Open": ("open", "o", "openprice"),
    "High": ("high", "h", "highprice"),
    "Low": ("low", "l", "lowprice"),
    "Close": ("close", "c", "closeprice", "closingprice", "last", "ltp"),
    "Volume": ("volume", "vol", "qty", "quantity", "tottrdqty", "totaltradedquantity", "tradedqty"),
}
SYMBOL_KEYS = ("symbol", "ticker", "tradingsymbol", "stock", "scrip", "security", "instrument", "stockcode")
TIME_KEYS = ("timestamp", "datetime", "datetimeist", "datetimeutc", "dateandtime", "timestamp")

def key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower().replace("\ufeff", ""))

def norm_symbol(value: Any) -> str | None:
    s = str(value or "").strip().upper()
    s = re.sub(r"^(NSE:|NSE_)", "", s)
    s = re.sub(r"\.(NS|BO|NSE)$", "", s)
    s = re.sub(r"[-_](EQ|BE|BZ)$", "", s)
    if not s or s in {"NAN", "NONE", "NULL"} or s in INDEX_SYMBOLS:
        return None
    return s if re.fullmatch(r"[A-Z0-9][A-Z0-9&._-]{0,30}", s) else None

def filename_symbol(name: str) -> str | None:
    stem = Path(name).stem.upper()
    stem = re.sub(r"([_-](1M|1MIN|1MINUTE|5M|5MIN|5MINUTE|DATA|OHLCV))+$", "", stem)
    stem = re.sub(r"([_-]20\d{2}.*)$", "", stem)
    return norm_symbol(stem)

def column_map(columns) -> dict[str, str]:
    return {key(c): str(c) for c in columns}

def first_col(lookup: dict[str, str], candidates) -> str | None:
    return next((lookup[k] for k in candidates if k in lookup), None)

def parse_times(frame: pd.DataFrame, lookup: dict[str, str]) -> pd.Series:
    col = first_col(lookup, TIME_KEYS)
    if col:
        values = frame[col]
    else:
        date_col, time_col = lookup.get("date"), lookup.get("time")
        if not date_col:
            raise ValueError("No timestamp/datetime/date field.")
        values = frame[date_col].astype(str)
        if time_col:
            values = values.str.strip() + " " + frame[time_col].astype(str).str.strip()
    parsed = pd.to_datetime(values, errors="coerce", dayfirst=True)
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert(IST)
    if not pd.api.types.is_datetime64_any_dtype(parsed.dtype):
        raise ValueError("Mixed/unsupported timestamp timezone format.")
    return parsed.dt.tz_localize(IST, ambiguous="NaT", nonexistent="NaT")

def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")

def normalize_chunk(chunk: pd.DataFrame, member: str) -> pd.DataFrame:
    lookup = column_map(chunk.columns)
    scol = first_col(lookup, SYMBOL_KEYS)
    fallback = filename_symbol(member)
    if not scol and not fallback:
        raise ValueError(f"{member}: no recognized equity symbol field.")
    cols = {name: first_col(lookup, tuple(key(a) for a in aliases)) for name, aliases in ALIASES.items()}
    missing = [name for name, col in cols.items() if not col]
    if missing:
        raise ValueError(f"{member}: missing OHLCV columns {missing}")
    out = pd.DataFrame({"Timestamp": parse_times(chunk, lookup)})
    for name, col in cols.items():
        out[name] = numeric(chunk[col])
    out["symbol"] = chunk[scol].map(norm_symbol) if scol else fallback
    exch = first_col(lookup, ("exchange", "exch", "exchangesegment"))
    if exch:
        out = out.loc[chunk[exch].astype(str).str.upper().str.contains("NSE", na=False)].copy()
    out = out.dropna(subset=["Timestamp", "symbol"])
    out = out.loc[~out["symbol"].isin(INDEX_SYMBOLS)].copy()
    if out.empty:
        return out
    px = out[["Open", "High", "Low", "Close"]]
    valid = px.notna().all(axis=1) & px.gt(0).all(axis=1)
    valid &= out["Volume"].notna() & out["Volume"].ge(0)
    valid &= out["High"].ge(out[["Open", "Close", "Low"]].max(axis=1))
    valid &= out["Low"].le(out[["Open", "Close", "High"]].min(axis=1))
    return out.loc[valid, ["symbol", "Timestamp", *OHLCV]]

def csv_streams(source: Path) -> Iterator[tuple[str, TextIO]]:
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            members = sorted(n for n in archive.namelist() if n.lower().endswith(".csv") and not n.endswith("/"))
            if not members:
                raise ValueError("ZIP contains no CSV members.")
            for member in members:
                # Do not skip a combined equity file merely because its name
                # contains "NIFTY" (e.g. NIFTY500 stocks). Index rows are
                # excluded by normalized symbol identity instead.
                with archive.open(member) as raw:
                    with io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace") as stream:
                        yield member, stream
        return
    if source.is_dir():
        paths = sorted(source.rglob("*.csv"))
    elif source.is_file() and source.suffix.lower() == ".csv":
        paths = [source]
    else:
        raise ValueError("Input must be a ZIP, directory of CSVs, or CSV file.")
    if not paths:
        raise ValueError("Input contains no CSV files.")
    for path in paths:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            yield str(path.relative_to(source)) if source.is_dir() else path.name, stream

def infer_interval(index: pd.DatetimeIndex, requested: str) -> str:
    if requested in {"1m", "5m"}:
        return requested
    differences = []
    for _, group in pd.Series(index, index=index).groupby(index.normalize(), sort=True):
        d = group.sort_values().diff().dt.total_seconds().div(60)
        differences.extend(d.loc[(d > 0) & (d <= 10)].tolist())
    if not differences:
        raise ValueError("Could not infer intraday bar interval.")
    median = float(np.median(differences))
    if median <= 2.5:
        return "1m"
    if median <= 7.5:
        return "5m"
    raise ValueError(f"Unsupported/daily source (median interval {median:.2f} minutes).")

def infer_label(index: pd.DatetimeIndex, interval: str, requested: str) -> str:
    if requested in {"start", "close"}:
        return requested
    first = pd.Series(index, index=index).groupby(index.normalize(), sort=True).min()
    mins = first.dt.hour * 60 + first.dt.minute
    opening = 9 * 60 + 15
    closing = opening + (1 if interval == "1m" else 5)
    a, b = int(mins.eq(opening).sum()), int(mins.eq(closing).sum())
    if a == b:
        raise ValueError(f"Cannot infer {interval} timestamp label; set --timestamp-label start or close.")
    return "start" if a > b else "close"

def prepare_symbol_bars(
    frame: pd.DataFrame, source_interval: str = "auto", timestamp_label: str = "auto"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame is None or frame.empty:
        raise ValueError("No source rows for symbol.")
    x = frame.copy()
    x["Timestamp"] = pd.to_datetime(x["Timestamp"], errors="coerce")
    x = x.dropna(subset=["Timestamp"])
    idx = pd.DatetimeIndex(x["Timestamp"])
    idx = idx.tz_localize(IST, ambiguous="NaT", nonexistent="NaT") if idx.tz is None else idx.tz_convert(IST)
    x.index = idx
    for c in OHLCV:
        x[c] = numeric(x[c])
    x = x[OHLCV].loc[~x.index.isna()].sort_index(kind="stable")
    px = x[["Open", "High", "Low", "Close"]]
    valid = px.notna().all(axis=1) & px.gt(0).all(axis=1)
    valid &= x["Volume"].notna() & x["Volume"].ge(0)
    valid &= x["High"].ge(x[["Open", "Close", "Low"]].max(axis=1))
    valid &= x["Low"].le(x[["Open", "Close", "High"]].min(axis=1))
    invalid = int((~valid).sum())
    x = x.loc[valid]
    if x.empty:
        raise ValueError("No valid OHLCV observations remain.")
    dup_mask = x.index.duplicated(keep=False)
    conflicts = x.loc[dup_mask].groupby(level=0)[OHLCV].nunique(dropna=False).gt(1).any(axis=1)
    conflict_index = conflicts.index[conflicts]
    x = x.loc[~x.index.isin(conflict_index)]
    prior = len(x)
    x = x.loc[~x.index.duplicated(keep="first")]
    interval = infer_interval(pd.DatetimeIndex(x.index), source_interval)
    label = infer_label(pd.DatetimeIndex(x.index), interval, timestamp_label)
    outputs = []
    for _, day in x.groupby(x.index.normalize(), sort=True):
        day = day.sort_index()
        minute = day.index.hour * 60 + day.index.minute
        if interval == "5m":
            if label == "start":
                keep = (minute >= 555) & (minute <= 925) & (minute % 5 == 0)
                bars = day.loc[keep].copy()
                bars.index = bars.index + pd.Timedelta(minutes=5)
            else:
                keep = (minute >= 560) & (minute <= 930) & (minute % 5 == 0)
                bars = day.loc[keep].copy()
        else:
            if label == "start":
                day = day.loc[(minute >= 555) & (minute <= 929)]
                closed = "left"
            else:
                day = day.loc[(minute >= 556) & (minute <= 930)]
                closed = "right"
            bars = day.resample(
                "5min", origin="start_day", offset="15min", label="right", closed=closed
            ).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
            counts = day["Open"].resample(
                "5min", origin="start_day", offset="15min", label="right", closed=closed
            ).count()
            bars = bars.loc[counts.eq(5)]
        if not bars.empty:
            outputs.append(bars)
    if not outputs:
        raise ValueError("No regular-session five-minute bars produced.")
    result = pd.concat(outputs).sort_index()
    result = result.loc[~result.index.duplicated(keep="first")]
    sessions = result.groupby(result.index.normalize()).size()
    good = sessions.index[sessions.ge(MIN_SESSION_BARS)]
    result = result.loc[result.index.normalize().isin(good)]
    if result.empty:
        raise ValueError("No session reached the 50-bar minimum.")
    output = result.reset_index(names="Timestamp")
    stats = {
        "input_rows": int(len(frame)), "invalid_ohlcv_rows": invalid,
        "identical_duplicate_rows_removed": int(prior-len(x)),
        "conflicting_duplicate_timestamps_excluded": int(len(conflict_index)),
        "source_interval": interval, "source_timestamp_label": label,
        "output_bars": int(len(output)), "usable_sessions": int(len(good)),
        "first_timestamp_ist": output["Timestamp"].min().isoformat(),
        "last_timestamp_ist": output["Timestamp"].max().isoformat(),
    }
    return output, stats

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Kaggle ZIP archive, extracted CSV directory or CSV")
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/Indian_Market_Research")
    ap.add_argument("--chunk-rows", type=int, default=500_000)
    ap.add_argument("--source-interval", choices=("auto","1m","5m"), default="auto")
    ap.add_argument("--timestamp-label", choices=("auto","start","close"), default="auto")
    ap.add_argument("--work-root", default="/tmp/kaggle_equity_5m_import")
    args = ap.parse_args()
    if args.chunk_rows < 10_000:
        raise ValueError("--chunk-rows must be at least 10,000.")
    source, root = Path(args.input).expanduser().resolve(), Path(args.drive_root).expanduser().resolve()
    if not source.exists() or not root.is_dir():
        raise FileNotFoundError("Input archive or mounted Google Drive root is missing.")
    work = Path(args.work_root).expanduser().resolve()
    if work.exists():
        shutil.rmtree(work)
    stage = work / "staging"
    stage.mkdir(parents=True)
    part_counts, source_rows = {}, {}
    errors, files, chunks = [], 0, 0
    for member, stream in csv_streams(source):
        files += 1
        try:
            for chunk in pd.read_csv(stream, chunksize=args.chunk_rows):
                chunks += 1
                normalized = normalize_chunk(chunk, member)
                for symbol, group in normalized.groupby("symbol", sort=False):
                    folder = stage / f"symbol={symbol}"
                    folder.mkdir(parents=True, exist_ok=True)
                    part = part_counts.get(symbol, 0) + 1
                    part_counts[symbol] = part
                    source_rows[symbol] = source_rows.get(symbol, 0) + len(group)
                    group.drop(columns=["symbol"]).to_parquet(
                        folder / f"part-{part:08d}.parquet", index=False, compression="zstd"
                    )
        except Exception as exc:
            errors.append({"member":member,"error":f"{type(exc).__name__}: {str(exc)[:250]}"})
            print(f"KAGGLE_IMPORT_FILE_ERROR {member}: {type(exc).__name__}: {exc}", flush=True)
    if not part_counts:
        raise RuntimeError("No equity OHLCV records imported; check archive CSV columns.")
    target_root = root / REMOTE
    reports = []
    for number, symbol in enumerate(sorted(part_counts), 1):
        parts = [pd.read_parquet(p) for p in sorted((stage/f"symbol={symbol}").glob("*.parquet"))]
        raw = pd.concat(parts, ignore_index=True)
        bars, stats = prepare_symbol_bars(raw, args.source_interval, args.timestamp_label)
        folder = target_root / f"symbol={symbol}" / "source=KAGGLE"
        folder.mkdir(parents=True, exist_ok=True)
        output_path = folder / f"{symbol}_kaggle_5m.parquet"
        temp = output_path.with_suffix(".tmp.parquet")
        bars.to_parquet(temp, index=False, compression="zstd")
        temp.replace(output_path)
        reports.append({
            "symbol":symbol,"source_rows":source_rows[symbol],"source_parts":part_counts[symbol],
            "output_path":str(output_path.relative_to(root)),"output_sha256":sha256(output_path),**stats
        })
        print(f"KAGGLE_IMPORT_PROGRESS {number}/{len(part_counts)} {symbol} bars={len(bars)}", flush=True)
    manifest = {
        "status":"IMPORTED_TO_ISOLATED_KAGGLE_SOURCE_PARTITIONS",
        "source":str(source),"source_size_bytes":source.stat().st_size if source.is_file() else None,
        "csv_files":files,"chunks":chunks,"source_errors":errors,
        "symbols_imported":len(reports),"total_output_bars":sum(x["output_bars"] for x in reports),
        "first_timestamp_ist":min(x["first_timestamp_ist"] for x in reports),
        "last_timestamp_ist":max(x["last_timestamp_ist"] for x in reports),
        "priority":"KAGGLE_PRIMARY_INDSTOCKS_GAP_FALLBACK",
        "nifty_index":"EXCLUDED_USE_DEDICATED_NIFTY_CONTEXT_IMPORTER",
        "training_started":False,"production_model_modified":False,"symbols":reports
    }
    manifest_path = root / "production_data/nse_official/intraday_seed/olo_5m_archive/kaggle_equity_history_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k:manifest[k] for k in (
        "status","symbols_imported","total_output_bars","first_timestamp_ist","last_timestamp_ist","source_errors"
    )}, indent=2))
    return 0 if not errors else 2

if __name__ == "__main__":
    raise SystemExit(main())
