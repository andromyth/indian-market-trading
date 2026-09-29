"""Free historical-data downloader and local store for the Indian stock dashboard.

Sources are public research datasets; the software downloads them itself and keeps
raw/source metadata locally.  It does not embed or redistribute the datasets.

Primary free sources used by this module:
- ganeshbiyer/Nse_Historical_Data: 1-minute Nifty 500 history through 2025.
- ganeshbiyer/Nse_Historical_Data_2026: 1-minute Nifty 500 history for 2026.
- voletiramu/nse-fno-1min-data: free 1-minute spot history for 214 F&O underlyings,
  Apr-2024 through Apr-2026, distributed as a GitHub release.
- NSE public report catalogue is documented in README as the authoritative
  reference for EOD/delivery/F&O reports; the dynamic NSE report download itself
  is not hard-coded here because its endpoint/schema can change.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data" / "historical"
FREE_ROOT = DATA_ROOT / "free_sources"
EQUITY_2018_2025 = FREE_ROOT / "nifty500_1m_2018_2025"
EQUITY_2026 = FREE_ROOT / "nifty500_1m_2026"
FNO_2024_2026 = FREE_ROOT / "fno_underlyings_1m_2024_2026"
MANIFEST_PATH = FREE_ROOT / "manifest.json"

EQUITY_REPOS = {
    "2018_2025": "https://raw.githubusercontent.com/ganeshbiyer/Nse_Historical_Data/main/{symbol}.parquet",
    "2026": "https://raw.githubusercontent.com/ganeshbiyer/Nse_Historical_Data_2026/main/{symbol}.parquet",
}
GITHUB_CONTENTS_API = {
    "2018_2025": "https://api.github.com/repos/ganeshbiyer/Nse_Historical_Data/contents",
    "2026": "https://api.github.com/repos/ganeshbiyer/Nse_Historical_Data_2026/contents",
}
FNO_RELEASE_URL = "https://github.com/voletiramu/nse-fno-1min-data/releases/download/v1.0.0/stocks_1m_csvs.zip"
INDEX_RELEASE_URL = "https://github.com/voletiramu/nse-fno-1min-data/releases/download/indices-v1.0.0/NIFTY_5min_5yr_2021_2026.csv"

# The free Nifty-500 archive follows the security/ticker file naming convention.
# Keep the user's core audit universe here so the downloader works without API access.
CORE_SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "TCS", "INFY", "BHARTIARTL", "SBIN",
    "LT", "MARUTI", "SUNPHARMA", "BEL", "HAL", "TRENT", "JIOFIN", "IDFCFIRSTB",
    "SUZLON", "RVNL", "IRFC", "RPOWER",
]


def ensure_dirs() -> None:
    for p in (EQUITY_2018_2025, EQUITY_2026, FNO_2024_2026):
        p.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def normalize_root_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    s = s.replace("^", "")
    if s.endswith(".NS") or s.endswith(".BO"):
        s = s.rsplit(".", 1)[0]
    return s


def load_manifest() -> dict:
    ensure_dirs()
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"updated_at": None, "files": {}, "archives": {}}


def save_manifest(manifest: dict) -> None:
    ensure_dirs()
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _request_headers() -> dict:
    return {
        "User-Agent": "IndianStockSignalDashboard/1.0 historical-research-downloader",
        "Accept": "application/octet-stream,application/json,text/plain,*/*",
    }


def _download_stream(url: str, destination: Path, timeout: tuple[int, int] = (15, 120), retries: int = 3) -> tuple[bool, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return True, "already_present"
    last_error = "unknown"
    for attempt in range(1, retries + 1):
        temp = destination.with_suffix(destination.suffix + ".part")
        try:
            with requests.get(url, headers=_request_headers(), stream=True, timeout=timeout, allow_redirects=True) as r:
                if r.status_code != 200:
                    last_error = f"HTTP {r.status_code}"
                    if r.status_code in (403, 429) and attempt < retries:
                        time.sleep(2 * attempt)
                        continue
                    return False, last_error
                with temp.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                temp.replace(destination)
                return True, "downloaded"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2 * attempt)
        finally:
            if temp.exists() and not destination.exists():
                try:
                    temp.unlink()
                except OSError:
                    pass
    return False, last_error


def _safe_zip_extract(zip_path: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        root = destination.resolve()
        for info in zf.infolist():
            target = (destination / info.filename).resolve()
            if not str(target).startswith(str(root)):
                raise RuntimeError(f"Unsafe ZIP path: {info.filename}")
            if info.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            count += 1
    return count


def discover_repo_symbols(source: str = "2026", timeout: int = 30) -> list[str]:
    """Discover parquet file stems from the public GitHub repo."""
    if source not in GITHUB_CONTENTS_API:
        raise ValueError(f"Unknown source: {source}")
    r = requests.get(GITHUB_CONTENTS_API[source], headers={**_request_headers(), "Accept": "application/vnd.github+json"}, timeout=timeout)
    r.raise_for_status()
    items = r.json()
    symbols = []
    for item in items:
        name = str(item.get("name", ""))
        if name.upper().endswith(".PARQUET"):
            symbols.append(name[:-8])
    return sorted(set(symbols))


def download_equity_symbol(symbol: str, source: str = "2026", force: bool = False) -> dict:
    """Download one 1-minute parquet from a free public GitHub archive."""
    ensure_dirs()
    root = normalize_root_symbol(symbol)
    if source not in EQUITY_REPOS:
        raise ValueError("source must be '2018_2025' or '2026'")
    destination = (EQUITY_2018_2025 if source == "2018_2025" else EQUITY_2026) / f"{root}.parquet"
    if force and destination.exists():
        destination.unlink()
    url = EQUITY_REPOS[source].format(symbol=root)
    ok, msg = _download_stream(url, destination)
    entry = {
        "kind": "equity_1m",
        "source": source,
        "symbol": root,
        "url": url,
        "path": str(destination.relative_to(BASE_DIR)),
        "status": "ok" if ok else "error",
        "message": msg,
        "bytes": destination.stat().st_size if destination.exists() else 0,
        "sha256": _sha256(destination) if destination.exists() else None,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    # 2026 is the current-year archive and is mirrored into the append-oriented
    # persistent store so new daily observations have a durable home. Older
    # backfill archives remain available locally and can be explicitly synced
    # from the dashboard without forcing a large partitioning job on every run.
    if ok and source == "2026":
        try:
            from persistent_store import persist_local_1m_file
            persistent = persist_local_1m_file(destination, root, source=f"free_archive_{source}", sync_drive=True)
            entry["persistent_store"] = {
                "status": persistent.get("status"),
                "rows_added": persistent.get("rows_added", 0),
                "partitions": len(persistent.get("partitions", [])),
            }
        except Exception as exc:
            entry["persistent_store"] = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    manifest = load_manifest()
    manifest.setdefault("files", {})[f"equity_1m:{source}:{root}"] = entry
    save_manifest(manifest)
    return entry


def download_equity_universe(symbols: Iterable[str], sources: Iterable[str] = ("2018_2025", "2026"), workers: int = 4) -> list[dict]:
    symbols = sorted({normalize_root_symbol(s) for s in symbols if str(s).strip()})
    sources = [s for s in sources]
    jobs = [(s, src) for src in sources for s in symbols]
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 8))) as pool:
        future_map = {pool.submit(download_equity_symbol, sym, src): (sym, src) for sym, src in jobs}
        for future in as_completed(future_map):
            sym, src = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"symbol": sym, "source": src, "status": "error", "message": f"{type(exc).__name__}: {exc}"})
    return sorted(results, key=lambda x: (x.get("source", ""), x.get("symbol", "")))


def download_fno_archive(force: bool = False) -> dict:
    """Download and extract the public 214-underlying 1-minute F&O archive."""
    ensure_dirs()
    archive_dir = FNO_2024_2026
    zip_path = archive_dir / "stocks_1m_csvs.zip"
    if force and zip_path.exists():
        zip_path.unlink()
    ok, msg = _download_stream(FNO_RELEASE_URL, zip_path, timeout=(20, 180), retries=4)
    extracted = 0
    if ok:
        try:
            extracted = _safe_zip_extract(zip_path, archive_dir)
            status = "ok"
            err = None
        except Exception as exc:
            status = "error"
            err = f"{type(exc).__name__}: {exc}"
    else:
        status = "error"
        err = msg
    entry = {
        "kind": "fno_underlyings_1m",
        "url": FNO_RELEASE_URL,
        "path": str(archive_dir.relative_to(BASE_DIR)),
        "archive": str(zip_path.relative_to(BASE_DIR)),
        "status": status,
        "message": err or msg,
        "archive_bytes": zip_path.stat().st_size if zip_path.exists() else 0,
        "extracted_files": extracted,
        "sha256": _sha256(zip_path) if zip_path.exists() else None,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest = load_manifest()
    manifest.setdefault("archives", {})["fno_underlyings_1m_2024_2026"] = entry
    save_manifest(manifest)
    return entry


def _read_any_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError("Reading parquet requires pyarrow or fastparquet. Install requirements.txt first.") from exc


def normalize_1m_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column variants into the dashboard's OHLCV schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    x = df.copy()
    rename = {}
    for c in x.columns:
        lc = str(c).strip().lower().replace(" ", "_")
        if lc in {"timestamp", "datetime", "date", "time", "dt"}:
            rename[c] = "_time"
        elif lc in {"open", "o"}:
            rename[c] = "Open"
        elif lc in {"high", "h"}:
            rename[c] = "High"
        elif lc in {"low", "l"}:
            rename[c] = "Low"
        elif lc in {"close", "c", "ltp"}:
            rename[c] = "Close"
        elif lc in {"volume", "vol", "v"}:
            rename[c] = "Volume"
    x = x.rename(columns=rename)
    if "_time" in x.columns:
        idx = pd.to_datetime(x["_time"], errors="coerce", utc=False)
        x = x.drop(columns=["_time"])
    else:
        idx = pd.to_datetime(x.index, errors="coerce", utc=False)
    # Most public Indian intraday archives are already IST-naive. Preserve timezone if present.
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    x.index = pd.DatetimeIndex(idx)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in x.columns:
            x[c] = pd.NA
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x[["Open", "High", "Low", "Close", "Volume"]]
    x = x[~x.index.isna()].sort_index()
    x = x[~x.index.duplicated(keep="last")]
    return x.dropna(subset=["Open", "High", "Low", "Close"])


def load_local_equity(symbol: str, start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None, interval: str = "1m") -> pd.DataFrame:
    """Read the best locally downloaded free equity history.

    The 2018_2025 archive plus the 2026 archive are merged when both exist.
    For 5m/daily, the software resamples the local 1m history.
    """
    root = normalize_root_symbol(symbol)
    frames = []
    for folder in (EQUITY_2018_2025, EQUITY_2026):
        path = folder / f"{root}.parquet"
        if path.exists():
            frames.append(normalize_1m_frame(_read_any_parquet(path)))
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames).sort_index()
    x = x[~x.index.duplicated(keep="last")]
    if start is not None:
        x = x[x.index >= pd.Timestamp(start)]
    if end is not None:
        x = x[x.index <= pd.Timestamp(end)]
    if interval == "1m":
        return x
    if interval in {"5m", "5min"}:
        return resample_intraday(x, "5min")
    if interval in {"1d", "D", "day", "daily"}:
        return resample_daily(x)
    raise ValueError("local equity interval must be 1m, 5m or 1d")


def resample_intraday(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    x = x.between_time("09:15", "15:29")
    out = x.resample(rule, origin="start_day", offset="15min", label="left", closed="left").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    })
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.resample("1D").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    })
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def local_history_provider(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Local-first provider for historical/audit use, falling back to yfinance."""
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    # Only query local files for bare/NSE equity symbols; do not hijack indexes.
    if not str(symbol).startswith("^"):
        days_map = {"5d": 7, "1mo": 40, "3mo": 100, "6mo": 220, "1y": 380, "2y": 760, "5y": 1900, "10y": 3800}
        days = days_map.get(period, 760)
        start = now - pd.Timedelta(days=days)
        # Permanent store first; hosted Streamlit storage is only ephemeral.
        try:
            from persistent_store import load_persistent_1m
            persistent = load_persistent_1m(symbol, start=start, end=now)
            if not persistent.empty:
                if interval in {"1m", "1min"}:
                    return persistent
                if interval in {"5m", "5min"}:
                    return resample_intraday(persistent, "5min")
                if interval in {"1d", "D", "day", "daily"}:
                    return resample_daily(persistent)
        except Exception:
            persistent = pd.DataFrame()
        try:
            local = load_local_equity(symbol, start=start, end=now, interval=interval)
        except Exception:
            local = pd.DataFrame()
        if not local.empty and len(local) >= 20:
            return local
    # Do not call data_fetcher.get_historical() here: that function already
    # calls _local_persistent_history(), which would recurse back into this
    # provider when local data is unavailable. Provider fallback belongs to
    # data_fetcher.get_historical().
    return pd.DataFrame()


def scan_coverage(symbols: Iterable[str] | None = None) -> pd.DataFrame:
    """Return local archive presence/size/row-range information."""
    ensure_dirs()
    wanted = [normalize_root_symbol(s) for s in symbols] if symbols else None
    rows = []
    for source, folder in [("2018_2025", EQUITY_2018_2025), ("2026", EQUITY_2026)]:
        files = sorted(folder.glob("*.parquet"))
        for path in files:
            sym = path.stem.upper()
            if wanted and sym not in wanted:
                continue
            rows.append({
                "dataset": "equity_1m",
                "source": source,
                "symbol": sym,
                "file_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "path": str(path),
            })
    if FNO_2024_2026.exists():
        for path in sorted(FNO_2024_2026.glob("*_1m.csv.gz")):
            sym = path.name[:-11]
            if wanted and sym not in wanted:
                continue
            rows.append({
                "dataset": "fno_underlying_1m",
                "source": "voletiramu_release_v1.0.0",
                "symbol": sym,
                "file_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "path": str(path),
            })
    return pd.DataFrame(rows)



def local_file_quality(path: str | Path, interval: str = "1m") -> dict:
    """Run integrity checks on one local archive file without contacting external services."""
    from data_quality import read_local_file, integrity_check
    x = read_local_file(path)
    return asdict(integrity_check(x, interval=interval))


def scan_quality(symbols: Iterable[str] | None = None, reconcile: bool = False, sample_days: int = 30) -> pd.DataFrame:
    """Scan local free-data files and optionally reconcile recent overlap to official NSE EOD data."""
    from data_quality import audit_file
    ensure_dirs()
    wanted = {normalize_root_symbol(s) for s in symbols} if symbols else None
    rows = []
    locations = [
        ("2018_2025", EQUITY_2018_2025),
        ("2026", EQUITY_2026),
    ]
    for source_id, folder in locations:
        for path in sorted(folder.glob("*.parquet")):
            sym = path.stem.upper()
            if wanted and sym not in wanted:
                continue
            result = audit_file(path, f"ganesh_{source_id}", symbol=sym, reconcile=reconcile, sample_days=sample_days)
            row = {"dataset": "equity_1m", "source": source_id, "symbol": sym, "path": str(path)}
            row.update({f"integrity_{k}": v for k, v in result["integrity"].items()})
            if result.get("reconciliation"):
                row.update({f"recon_{k}": v for k, v in result["reconciliation"].items()})
            rows.append(row)
    for path in sorted(FNO_2024_2026.glob("*_1m.csv.gz")):
        sym = path.name[:-11].upper()
        if wanted and sym not in wanted:
            continue
        try:
            result = audit_file(path, "voletiramu_fno", symbol=sym, reconcile=False)
            row = {"dataset": "fno_underlying_1m", "source": "voletiramu_release_v1.0.0", "symbol": sym, "path": str(path)}
            row.update({f"integrity_{k}": v for k, v in result["integrity"].items()})
            rows.append(row)
        except Exception as exc:
            rows.append({"dataset": "fno_underlying_1m", "source": "voletiramu_release_v1.0.0", "symbol": sym, "path": str(path), "integrity_status": "ERROR", "integrity_error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)

def storage_summary() -> dict:
    ensure_dirs()
    totals = {}
    for name, folder in [("equity_2018_2025", EQUITY_2018_2025), ("equity_2026", EQUITY_2026), ("fno", FNO_2024_2026)]:
        total = 0
        files = 0
        for p in folder.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
                files += 1
        totals[name] = {"files": files, "gb": round(total / (1024 ** 3), 3)}
    return totals


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Free historical data downloader for Indian Stock Signal Dashboard")
    parser.add_argument("--symbols", nargs="*", help="Symbols, e.g. SBIN BHARTIARTL RELIANCE")
    parser.add_argument("--universe", choices=["core", "nifty500"], default="core")
    parser.add_argument("--sources", nargs="+", choices=["2018_2025", "2026"], default=["2026"])
    parser.add_argument("--download-fno", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Validate local OHLCV integrity")
    parser.add_argument("--reconcile-nse", action="store_true", help="Compare recent local daily aggregates with official NSE history")
    parser.add_argument("--quality-symbols", nargs="*", help="Limit quality/reconciliation scan to symbols")
    parser.add_argument("--source-check", action="store_true", help="Check public source/repository availability online")
    args = parser.parse_args()

    ensure_dirs()
    symbols = args.symbols or (CORE_SYMBOLS if args.universe == "core" else discover_repo_symbols("2026"))
    if symbols:
        print(f"Downloading {len(symbols)} symbols from free equity archives: {args.sources}")
        results = download_equity_universe(symbols, sources=args.sources, workers=args.workers)
        ok = sum(1 for r in results if r.get("status") == "ok")
        print(f"Equity files OK: {ok}/{len(results)}")
    if args.download_fno:
        print("Downloading free F&O underlying 1-minute archive (~464 MB compressed, ~3 GB uncompressed)...")
        print(download_fno_archive())
    if args.scan:
        print(scan_coverage(symbols if symbols else None).to_string(index=False))
    if args.validate or args.reconcile_nse:
        print(scan_quality(symbols=args.quality_symbols or symbols, reconcile=args.reconcile_nse).to_string(index=False))
    if args.source_check:
        from data_quality import online_source_check
        print(online_source_check().to_string(index=False))
    print(json.dumps(storage_summary(), indent=2))


if __name__ == "__main__":
    _cli()
