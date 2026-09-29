"""NSE-first data layer for the Indian market dashboard.

Primary source: NSE India's public web data interfaces.

This module deliberately separates four concerns:
1. NSE acquisition (official exchange web interfaces).
2. Normalization into stable internal schemas.
3. Local cache / persistence.
4. Provenance metadata.

It does not bypass authentication, scrape CAPTCHAs, or attempt to defeat rate
limits. Requests are serialized by default, cached, and retried conservatively.
For production/licensed redistribution of NSE market data, follow NSE's current
Data Sharing & Usage Policy.
"""
from __future__ import annotations

import io
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

NSE_BASE = "https://www.nseindia.com"
IST = ZoneInfo("Asia/Kolkata")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.nseindia.com/",
    "X-Requested-With": "XMLHttpRequest",
}

EQUITY_HISTORY_ENDPOINT = "/api/historicalOR/generateSecurityWiseHistoricalData"
EQUITY_QUOTE_ENDPOINT = "/api/quote-equity"
EQUITY_CHART_ENDPOINT = "/api/chart-databyindex"
CORPORATE_ACTIONS_ENDPOINT = "/api/corporates-corporateActions"
CORPORATE_ANNOUNCEMENTS_ENDPOINT = "/api/corporate-announcements"
FNO_HISTORY_ENDPOINT = "/api/historicalOR/foCPV"

LOCAL_ROOT = Path(__file__).resolve().parent / "data" / "nse_official"
RAW_ROOT = LOCAL_ROOT / "raw"
NORMALIZED_ROOT = LOCAL_ROOT / "normalized"
MANIFEST = LOCAL_ROOT / "manifest.json"

_last_request = 0.0


def root_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    for suffix in (".NS", ".BO"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.replace("/", "_")


def _sleep_rate_limit(min_interval: float = 1.5) -> None:
    global _last_request
    now = time.monotonic()
    wait = min_interval - (now - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    # Prime the NSE session. Public website endpoints commonly rely on the
    # session cookies established by the landing/report page.
    s.get(NSE_BASE + "/report-detail/eq_security", timeout=15)
    return s


def _request(path: str, *, params: dict[str, Any] | None = None,
             accept: str | None = None, retries: int = 3,
             min_interval: float = 1.5) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            _sleep_rate_limit(min_interval)
            s = _session()
            headers = {"Accept": accept} if accept else {}
            r = s.get(NSE_BASE + path, params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 403, 503):
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    if last_exc:
        raise RuntimeError(f"NSE request failed: {type(last_exc).__name__}: {last_exc}") from last_exc
    raise RuntimeError(f"NSE request failed after {retries} attempts: {path}")


def _ensure_local() -> None:
    for p in (RAW_ROOT, NORMALIZED_ROOT):
        p.mkdir(parents=True, exist_ok=True)


def _manifest() -> dict[str, Any]:
    _ensure_local()
    try:
        obj = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            obj.setdefault("records", {})
            return obj
    except Exception:
        pass
    return {"version": "nse-official-v1", "records": {}}


def _save_manifest(obj: dict[str, Any]) -> None:
    obj["updated_at"] = datetime.now(tz=IST).isoformat()
    _ensure_local()
    MANIFEST.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _drive_upload(local_path: Path, remote_path: str, mime: str = "application/octet-stream") -> dict[str, Any]:
    try:
        from persistent_store import GoogleDriveStore
        return GoogleDriveStore().upload_or_update(local_path, remote_path, mime_type=mime)
    except Exception as exc:
        return {"status": "NOT_SYNCED", "error": f"{type(exc).__name__}: {exc}"}


def normalize_equity_history(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize current NSE historical equity CSV/JSON into stable columns."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    x = frame.copy()
    x.columns = [str(c).strip() for c in x.columns]
    aliases = {
        "Date": "date", "DATE": "date", "CH_TIMESTAMP": "date",
        "OPEN": "open", "CH_OPENING_PRICE": "open",
        "HIGH": "high", "CH_TRADE_HIGH_PRICE": "high",
        "LOW": "low", "CH_TRADE_LOW_PRICE": "low",
        "CLOSE": "close", "CH_CLOSING_PRICE": "close",
        "LTP": "ltp", "CH_LAST_TRADED_PRICE": "ltp",
        "PREV. CLOSE": "prev_close", "PREV CLOSE": "prev_close", "CH_PREVIOUS_CLS_PRICE": "prev_close",
        "VWAP": "vwap", "VOLUME": "volume", "CH_TOT_TRADED_QTY": "volume",
        "VALUE": "traded_value", "CH_TOT_TRADED_VAL": "traded_value",
        "NO OF TRADES": "num_trades", "CH_TOTAL_TRADES": "num_trades",
        "DELIVERY QTY": "delivery_qty", "COP_DELIV_QTY": "delivery_qty",
        "DELIVERY %": "delivery_pct", "COP_DELIV_PERC": "delivery_pct",
        "CH_SERIES": "series", "CH_SYMBOL": "symbol",
    }
    ren = {c: aliases[c] for c in x.columns if c in aliases}
    x = x.rename(columns=ren)
    if "symbol" not in x.columns:
        x["symbol"] = root_symbol(symbol)
    x["symbol"] = x["symbol"].astype(str).str.upper()
    if "date" not in x.columns:
        raise ValueError(f"NSE equity response has no date column: {list(x.columns)}")
    x["date"] = pd.to_datetime(x["date"], errors="coerce", dayfirst=True)
    for c in ("open", "high", "low", "close", "ltp", "prev_close", "vwap", "volume", "traded_value", "num_trades", "delivery_qty", "delivery_pct"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if "close" not in x.columns and "ltp" in x.columns:
        x["close"] = x["ltp"]
    x["source"] = "NSE_OFFICIAL"
    x["provenance"] = "nseindia.com public historical equity interface"
    return x.sort_values("date").drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)


def fetch_equity_history(symbol: str, start: date | datetime, end: date | datetime,
                         series: str = "EQ", sync_drive: bool = True) -> pd.DataFrame:
    """Fetch official NSE security-wise historical price/volume data.

    Uses the same public report data exposed by NSE's Price/Volume security
    interface, returned as CSV. The request is chunked to reduce timeout/rate-
    limiting risk and persisted locally; each chunk is also mirrored to Drive
    when Google Drive is configured.
    """
    s = root_symbol(symbol)
    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()
    frames: list[pd.DataFrame] = []
    cursor = start_d
    while cursor <= end_d:
        chunk_end = min(cursor + timedelta(days=89), end_d)
        params = {
            "from": cursor.strftime("%d-%m-%Y"),
            "to": chunk_end.strftime("%d-%m-%Y"),
            "symbol": s,
            "type": "priceVolumeDeliverable",
            "series": series,
            "csv": "true",
        }
        r = _request(EQUITY_HISTORY_ENDPOINT, params=params,
                      accept="text/csv,application/csv,*/*")
        text = r.content
        raw_dir = RAW_ROOT / "equity" / s
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{cursor:%Y%m%d}_{chunk_end:%Y%m%d}.csv"
        raw_path.write_bytes(text)
        try:
            csv = pd.read_csv(io.BytesIO(text), thousands=",")
        except Exception:
            csv = pd.DataFrame()
        normalized = normalize_equity_history(csv, s)
        if not normalized.empty:
            frames.append(normalized)
            out_path = NORMALIZED_ROOT / "equity" / s / f"{cursor:%Y%m%d}_{chunk_end:%Y%m%d}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            normalized.to_parquet(out_path, index=False)
            if sync_drive:
                _drive_upload(raw_path, f"production_data/nse_official/raw/equity/{s}/{raw_path.name}", "text/csv")
                _drive_upload(out_path, f"production_data/nse_official/equity/{s}/{out_path.name}", "application/octet-stream")
            m = _manifest()
            key = f"equity:{s}:{cursor.isoformat()}:{chunk_end.isoformat()}"
            m["records"][key] = {
                "source": "NSE_OFFICIAL",
                "symbol": s,
                "from": cursor.isoformat(), "to": chunk_end.isoformat(),
                "raw_local": str(raw_path), "normalized_local": str(out_path),
                "rows": int(len(normalized)), "retrieved_at": datetime.now(tz=IST).isoformat(),
            }
            _save_manifest(m)
        cursor = chunk_end + timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("date").drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)


def fetch_quote(symbol: str, trade_info: bool = False) -> dict[str, Any]:
    s = root_symbol(symbol)
    params = {"symbol": s}
    if trade_info:
        params["section"] = "trade_info"
    r = _request(EQUITY_QUOTE_ENDPOINT, params=params)
    return r.json()


def fetch_corporate_actions(symbol: str) -> list[dict[str, Any]]:
    s = root_symbol(symbol)
    r = _request(CORPORATE_ACTIONS_ENDPOINT, params={"index": "equities", "symbol": s})
    data = r.json()
    return data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []


def fetch_corporate_announcements(symbol: str) -> list[dict[str, Any]]:
    s = root_symbol(symbol)
    r = _request(CORPORATE_ANNOUNCEMENTS_ENDPOINT, params={"index": "equities", "symbol": s})
    data = r.json()
    return data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []


def fetch_intraday_1m_current(symbol: str, sync_drive: bool = True) -> pd.DataFrame:
    """Fetch today's NSE quote-chart path and aggregate available prints to 1-minute OHLC.

    NSE's public quote chart is an LTP/time series, not a full broker websocket.
    Volume is therefore left missing in this path; it must not be invented.
    """
    s = root_symbol(symbol)
    params = {"index": s + "EQN"}
    r = _request(EQUITY_CHART_ENDPOINT, params=params)
    payload = r.json()
    pts = payload.get("grapthData", []) if isinstance(payload, dict) else []
    if not pts:
        return pd.DataFrame()
    frame = pd.DataFrame(pts, columns=["epoch_ms", "ltp"])
    frame["timestamp"] = pd.to_datetime(frame["epoch_ms"], unit="ms", utc=True).dt.tz_convert(IST)
    frame["ltp"] = pd.to_numeric(frame["ltp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "ltp"]).set_index("timestamp").sort_index()
    bars = frame["ltp"].resample("1min", label="left", closed="left").ohlc()
    bars["Volume"] = np.nan
    bars = bars.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    bars["symbol"] = s
    bars["source"] = "NSE_OFFICIAL_CHART"
    bars["provenance"] = "nseindia.com public equity quote chart"
    bars = bars.dropna(subset=["Open", "High", "Low", "Close"])
    if sync_drive and not bars.empty:
        try:
            from persistent_store import persist_1m_frame
            persist_1m_frame(s, bars, source="NSE_OFFICIAL_CHART", sync_drive=True)
        except Exception:
            pass
    return bars


def fetch_fno_history(symbol: str, start: date | datetime, end: date | datetime,
                      instrument_type: str, expiry_date: date | datetime,
                      option_type: str | None = None, strike_price: float | None = None,
                      sync_drive: bool = True) -> pd.DataFrame:
    """Fetch a contract's official NSE historical F&O data.

    NSE currently exposes contract-wise historical data through the historicalOR
    interface. The caller must specify the contract, expiry and (for options)
    option type/strike to keep the data provenance unambiguous.
    """
    s = root_symbol(symbol)
    start_d, end_d = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    expiry = pd.Timestamp(expiry_date).strftime("%d-%b-%Y")
    frames = []
    cursor = start_d
    while cursor <= end_d:
        chunk_end = min(cursor + timedelta(days=59), end_d)
        params: dict[str, Any] = {
            "from": cursor.strftime("%d-%m-%Y"),
            "to": chunk_end.strftime("%d-%m-%Y"),
            "instrumentType": instrument_type,
            "symbol": s,
            "year": str(pd.Timestamp(expiry_date).year),
            "expiryDate": expiry,
        }
        if option_type:
            params["optionType"] = option_type
        if strike_price is not None:
            params["strikePrice"] = str(float(strike_price))
        r = _request(FNO_HISTORY_ENDPOINT, params=params)
        payload = r.json()
        data = payload.get("data", payload if isinstance(payload, list) else [])
        df = pd.DataFrame(data)
        if not df.empty:
            df["source"] = "NSE_OFFICIAL"
            df["provenance"] = "nseindia.com historical F&O contract interface"
            frames.append(df)
            raw_dir = RAW_ROOT / "fno" / s
            raw_dir.mkdir(parents=True, exist_ok=True)
            out_path = NORMALIZED_ROOT / "fno" / s / f"{cursor:%Y%m%d}_{chunk_end:%Y%m%d}_{instrument_type}_{pd.Timestamp(expiry_date):%Y%m%d}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, index=False)
            if sync_drive:
                _drive_upload(out_path, f"production_data/nse_official/fno/{s}/{out_path.name}")
        cursor = chunk_end + timedelta(days=1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def source_status() -> dict[str, Any]:
    try:
        quote = fetch_quote("RELIANCE")
        return {
            "status": "PASS",
            "source": "NSE_OFFICIAL",
            "endpoint": EQUITY_QUOTE_ENDPOINT,
            "symbol": root_symbol("RELIANCE"),
            "last_price": quote.get("priceInfo", {}).get("lastPrice"),
            "quote_available": bool(quote),
        }
    except Exception as exc:
        return {"status": "ERROR", "source": "NSE_OFFICIAL", "error": f"{type(exc).__name__}: {exc}"}
