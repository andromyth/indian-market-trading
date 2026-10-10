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
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import duckdb

NSE_BASE = "https://www.nseindia.com"
IST = ZoneInfo("Asia/Kolkata")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Referer": "https://www.nseindia.com/report-detail/eq_security",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
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
_nse_session: requests.Session | None = None
_nse_session_at = 0.0
_last_nse_diagnostic: dict[str, Any] = {}


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
    global _nse_session, _nse_session_at
    now = time.monotonic()
    if _nse_session is not None and (now - _nse_session_at) < 600 and _nse_session.cookies:
        return _nse_session
    s = requests.Session()
    s.headers.update(HEADERS)
    # Prime both the site root and the historical-report page so the cookies
    # and anti-bot context expected by current NSE endpoints are established.
    for url in (NSE_BASE, NSE_BASE + "/report-detail/eq_security"):
        try:
            s.get(url, timeout=4)
        except requests.RequestException:
            pass
    _nse_session = s
    _nse_session_at = time.monotonic()
    return s


def _request(path: str, *, params: dict[str, Any] | None = None,
             accept: str | None = None, retries: int = 3,
             min_interval: float = 1.5) -> requests.Response:
    global _last_nse_diagnostic, _nse_session, _nse_session_at
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            _sleep_rate_limit(min_interval)
            s = _session()
            headers = {
                "Accept": accept or "application/json,text/plain,*/*",
                "Referer": "https://www.nseindia.com/report-detail/eq_security",
                "X-Requested-With": "XMLHttpRequest",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            }
            r = s.get(NSE_BASE + path, params=params, headers=headers, timeout=(3, 8))
            _last_nse_diagnostic = {
                "path": path, "status_code": r.status_code,
                "content_type": r.headers.get("content-type", ""),
                "bytes": len(r.content), "attempt": attempt + 1,
            }
            if r.status_code == 200:
                return r
            if r.status_code in (401, 403, 408, 429, 500, 502, 503, 504):
                # Refresh the cookie session before the next attempt.
                _nse_session = None
                _nse_session_at = 0.0
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
                continue
            r.raise_for_status()
        except Exception as exc:
            last_exc = exc
            _last_nse_diagnostic = {
                "path": path, "attempt": attempt + 1,
                "error": f"{type(exc).__name__}: {exc}",
            }
            _nse_session = None
            _nse_session_at = 0.0
            if attempt + 1 < retries:
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
    if last_exc:
        raise RuntimeError(f"NSE request failed: {type(last_exc).__name__}: {last_exc}") from last_exc
    raise RuntimeError(f"NSE request failed after {retries} attempts: {path}")


def nse_last_diagnostic() -> dict[str, Any]:
    """Return the last NSE transport diagnostic without exposing response bodies."""
    return dict(_last_nse_diagnostic)


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
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    tmp.replace(MANIFEST)


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
    x["date"] = pd.to_datetime(x["date"], errors="coerce", format="mixed", dayfirst=True)
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
        base_params = {
            "from": cursor.strftime("%d-%m-%Y"),
            "to": chunk_end.strftime("%d-%m-%Y"),
            "symbol": s,
            "type": "priceVolumeDeliverable",
            # The security-wise historical NSE interface uses csv=true for
            # the actual tabular download response. Without it, the endpoint
            # can return HTTP 200 with a JSON response that is not a usable
            # historical row set on hosted runtimes.
            "csv": "true",
        }

        # Prefer the explicit CSV contract, then retry without the csv hint
        # as a compatibility fallback, and finally try ALL-series.
        request_variants = [
            {**base_params, "series": series},
            {**base_params, "series": "ALL"},
            {k: v for k, v in base_params.items() if k != "csv"} | {"series": series},
            {k: v for k, v in base_params.items() if k != "csv"} | {"series": "ALL"},
        ]
        r = None
        csv = pd.DataFrame()
        for params in request_variants:
            try:
                rr = _request(EQUITY_HISTORY_ENDPOINT, params=params,
                               accept="application/json,text/csv,application/csv,*/*", retries=1, min_interval=0.75)
                ctype = (rr.headers.get("content-type") or "").lower()
                body = rr.text.lstrip()

                # csv=true is the authoritative tabular contract for the
                # security-wise historical report. Sniff the body as well as
                # the content type because some NSE responses are served with
                # a generic application/json content type despite being CSV.
                looks_like_csv = (
                    params.get("csv") == "true"
                    or ("csv" in ctype)
                    or (
                        body
                        and not body.startswith(("{", "["))
                        and "," in body[:500]
                    )
                )

                if looks_like_csv:
                    try:
                        csv = pd.read_csv(
                            io.BytesIO(rr.content),
                            thousands=",",
                        )
                    except Exception:
                        csv = pd.DataFrame()

                if csv.empty:
                    try:
                        payload = rr.json()
                        rows = payload.get(
                            "data",
                            payload if isinstance(payload, list) else [],
                        )
                        csv = pd.DataFrame(rows)
                    except Exception:
                        csv = pd.DataFrame()

                if not csv.empty:
                    r = rr
                    break
                r = rr
            except Exception:
                continue
        if r is None:
            raise RuntimeError(f"NSE returned no response for {s} {cursor}..{chunk_end}")
        raw_dir = RAW_ROOT / "equity" / s
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{cursor:%Y%m%d}_{chunk_end:%Y%m%d}.csv"
        raw_path.write_bytes(r.content)
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
    if frames:
        out = pd.concat(frames, ignore_index=True)
        return (
            out.sort_values("date")
            .drop_duplicates(["symbol", "date"], keep="last")
            .reset_index(drop=True)
        )

    # Direct NSE security-wise history can return an HTTP 200 JSON response
    # without usable historical rows on hosted/cloud IPs. The project already
    # maintains the same official NSE CM-UDiFF daily archive in Google Drive;
    # use it as the persistent NSE-first fallback instead of jumping straight
    # to a third-party provider.
    try:
        drive_history = fetch_nse_drive_bulk_symbol_history(
            s,
            start_d,
            end_d,
            max_calendar_days=240,
        )
        if not drive_history.empty:
            return drive_history
    except Exception:
        pass

    return pd.DataFrame()



INDEX_HISTORY_ENDPOINT = "/api/historicalOR/indicesHistory"
INDEX_HISTORY_NAMES = {
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "NIFTY BANK",
}
INDEX_NORMALIZED_ROOT = NORMALIZED_ROOT / "indices"


def _normalize_index_history(frame: pd.DataFrame, requested_index: str) -> pd.DataFrame:
    """Normalize official NSE historical-index JSON into stable daily OHLC."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    x = frame.copy()
    x.columns = [str(c).strip() for c in x.columns]
    aliases = {
        "EOD_TIMESTAMP": "date",
        "TIMESTAMP": "date",
        "Date": "date",
        "EOD_OPEN_INDEX_VAL": "open",
        "EOD_HIGH_INDEX_VAL": "high",
        "EOD_LOW_INDEX_VAL": "low",
        "EOD_CLOSE_INDEX_VAL": "close",
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
    }
    x = x.rename(columns={c: aliases[c] for c in x.columns if c in aliases})
    if "date" not in x.columns or any(c not in x.columns for c in ("open", "high", "low", "close")):
        return pd.DataFrame()

    raw_dates = x["date"].astype(str).str.strip()
    parsed = pd.to_datetime(raw_dates, format="%d-%b-%Y", errors="coerce")
    remaining = parsed.isna()
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y%m%d"):
        if not remaining.any():
            break
        candidate = pd.to_datetime(raw_dates.loc[remaining], format=fmt, errors="coerce")
        parsed.loc[remaining] = candidate
        remaining = parsed.isna()
    if remaining.any():
        candidate = pd.to_datetime(x.loc[remaining, "date"], errors="coerce")
        parsed.loc[remaining] = candidate

    x["date"] = parsed
    for c in ("open", "high", "low", "close"):
        x[c] = pd.to_numeric(x[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if "volume" not in x.columns:
        x["volume"] = 0.0
    else:
        x["volume"] = pd.to_numeric(x["volume"].astype(str).str.replace(",", "", regex=False), errors="coerce")

    x["symbol"] = str(requested_index).upper()
    x["source"] = "NSE_OFFICIAL_INDEX_HISTORY"
    x["provenance"] = "nseindia.com /api/historicalOR/indicesHistory"
    x = x.dropna(subset=["date", "open", "high", "low", "close"])
    return (
        x[["symbol","date","open","high","low","close","volume","source","provenance"]]
        .sort_values("date")
        .drop_duplicates(["symbol","date"], keep="last")
        .reset_index(drop=True)
    )


def _drive_index_history(
    index_symbol: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Read persisted official NSE benchmark parquet files from Google Drive."""
    root = str(index_symbol).strip().upper()

    try:
        from persistent_store import GoogleDriveStore
        from googleapiclient.http import MediaIoBaseDownload

        store = GoogleDriveStore()

        production = store._find_child(
            store.root_id,
            "production_data",
            "application/vnd.google-apps.folder",
        )
        if not production:
            return pd.DataFrame()

        nse_official = store._find_child(
            production["id"],
            "nse_official",
            "application/vnd.google-apps.folder",
        )
        if not nse_official:
            return pd.DataFrame()

        indices = store._find_child(
            nse_official["id"],
            "indices",
            "application/vnd.google-apps.folder",
        )
        if not indices:
            return pd.DataFrame()

        folder_name = root.replace("^", "").replace("/", "_")

        index_folder = store._find_child(
            indices["id"],
            folder_name,
            "application/vnd.google-apps.folder",
        )

        if not index_folder:
            return pd.DataFrame()

        page_token = None
        files = []

        while True:

            kwargs = {
                "q": (
                    f"'{index_folder['id']}' in parents "
                    "and name contains '.parquet' "
                    "and trashed = false"
                ),
                "spaces": "drive",
                "fields": "nextPageToken,files(id,name)",
                "pageSize": 100,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }

            if page_token:
                kwargs["pageToken"] = page_token

            response = store.service.files().list(
                **kwargs
            ).execute()

            files.extend(
                response.get(
                    "files",
                    [],
                )
            )

            page_token = response.get(
                "nextPageToken"
            )

            if not page_token:
                break

        if not files:
            return pd.DataFrame()

        local_root = (
            NORMALIZED_ROOT
            / "indices"
            / folder_name
        )

        parts = []

        for item in files:

            file_id = str(
                item.get("id", "")
            ).strip()

            name = str(
                item.get("name", "")
            ).strip()

            if not file_id or not name:
                continue

            local_path = (
                local_root
                / name
            )

            try:

                if (
                    not local_path.exists()
                    or local_path.stat().st_size == 0
                ):

                    local_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    request = (
                        store.service.files()
                        .get_media(
                            fileId=file_id
                        )
                    )

                    temp = local_path.with_suffix(
                        ".tmp"
                    )

                    with temp.open("wb") as fh:

                        downloader = MediaIoBaseDownload(
                            fh,
                            request,
                        )

                        done = False

                        while not done:
                            _, done = downloader.next_chunk()

                    temp.replace(
                        local_path
                    )

                frame = pd.read_parquet(
                    local_path
                )

                normalized = _normalize_index_history(
                    frame,
                    root,
                )

                if normalized.empty:
                    # Some persisted files may already be normalized but carry
                    # a different column spelling. Try the frame directly.
                    frame = frame.copy()
                    frame.columns = [
                        str(col).strip().lower()
                        for col in frame.columns
                    ]

                    if all(
                        col in frame.columns
                        for col in (
                            "date",
                            "open",
                            "high",
                            "low",
                            "close",
                        )
                    ):
                        for col in (
                            "open",
                            "high",
                            "low",
                            "close",
                        ):
                            frame[col] = pd.to_numeric(
                                frame[col],
                                errors="coerce",
                            )

                        frame["date"] = pd.to_datetime(
                            frame["date"],
                            errors="coerce",
                        ).dt.normalize()

                        frame["symbol"] = root
                        normalized = frame

                if normalized.empty:
                    continue

                normalized["date"] = pd.to_datetime(
                    normalized["date"],
                    errors="coerce",
                ).dt.normalize()

                normalized = normalized.dropna(
                    subset=["date"]
                )

                normalized = normalized[
                    (normalized["date"] >= pd.Timestamp(start))
                    & (normalized["date"] <= pd.Timestamp(end))
                ]

                if not normalized.empty:
                    parts.append(
                        normalized
                    )

            except Exception:
                continue

        if not parts:
            return pd.DataFrame()

        out = pd.concat(
            parts,
            ignore_index=True,
        )

        out["symbol"] = root

        return (
            out
            .sort_values("date")
            .drop_duplicates(
                ["symbol", "date"],
                keep="last",
            )
            .reset_index(drop=True)
        )

    except Exception:
        return pd.DataFrame()


def _fetch_index_history_nse_api(index_symbol: str, start: date | datetime, end: date | datetime,
                        sync_drive: bool = False) -> pd.DataFrame:
    """Fetch/cache official NSE daily index history with bounded date chunks.

    NSE's public historical-index endpoint can return a silently truncated
    window for a large date range. To avoid treating partial history as valid,
    requests are split into short chunks and the combined result is coverage-
    checked against the requested range. Existing full-range caches are reused
    only when they cover both ends of the requested period.
    """
    root = str(index_symbol).strip().upper()
    if root not in INDEX_HISTORY_NAMES:
        raise ValueError(f"Unsupported NSE benchmark: {index_symbol}")

    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()
    if end_d < start_d:
        raise ValueError("end date cannot be before start date")

    cache_dir = INDEX_NORMALIZED_ROOT / root.replace("^", "").replace("/", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{start_d:%Y%m%d}_{end_d:%Y%m%d}.parquet"

    def _coverage_ok(frame: pd.DataFrame, requested_start: date, requested_end: date) -> bool:
        if frame is None or frame.empty or "date" not in frame.columns:
            return False
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
        if dates.empty:
            return False
        first = dates.min().date()
        last = dates.max().date()
        # Allow weekends/holidays only at the start boundary. The requested
        # end date is the PIT session boundary for production, so stale
        # benchmark history must never qualify as current coverage.
        start_slack = 10
        if first > requested_start + timedelta(days=start_slack):
            return False
        if last < requested_end:
            return False
        span_days = max(1, (requested_end - requested_start).days)
        if span_days >= 90:
            expected_weekdays = int(span_days * 5 / 7)
            if len(dates) < max(30, int(expected_weekdays * 0.50)):
                return False
        return True

    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            cached = pd.read_parquet(cache_path)
            if _coverage_ok(cached, start_d, end_d):
                return cached
        except Exception:
            pass

    frames: list[pd.DataFrame] = []
    cursor = start_d
    chunk_days = 31

    while cursor <= end_d:
        chunk_end = min(end_d, cursor + timedelta(days=chunk_days - 1))
        params = {
            "indexType": INDEX_HISTORY_NAMES[root],
            "from": cursor.strftime("%d-%m-%Y"),
            "to": chunk_end.strftime("%d-%m-%Y"),
        }

        response = _request(
            INDEX_HISTORY_ENDPOINT,
            params=params,
            accept="application/json,text/plain,*/*",
            retries=2,
            min_interval=0.75,
        )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"NSE historical index response was not valid JSON for "
                f"{INDEX_HISTORY_NAMES[root]} {cursor}..{chunk_end}"
            ) from exc

        rows = payload.get("data", payload if isinstance(payload, list) else [])
        normalized = _normalize_index_history(pd.DataFrame(rows), root)

        if normalized.empty:
            raise RuntimeError(
                f"NSE historical index request returned no usable rows for "
                f"{INDEX_HISTORY_NAMES[root]} {cursor}..{chunk_end}"
            )

        # A chunk may legitimately begin/end on a weekend or holiday. Reject
        # only material truncation inside the requested chunk.
        chunk_dates = pd.to_datetime(normalized["date"], errors="coerce").dropna()
        if chunk_dates.empty:
            raise RuntimeError(
                f"NSE historical index chunk contained no usable dates for "
                f"{INDEX_HISTORY_NAMES[root]} {cursor}..{chunk_end}"
            )
        if chunk_dates.min().date() > cursor + timedelta(days=10):
            raise RuntimeError(
                f"NSE historical index chunk starts too late for "
                f"{INDEX_HISTORY_NAMES[root]} {cursor}..{chunk_end}: "
                f"{chunk_dates.min().date()}"
            )
        if chunk_dates.max().date() < chunk_end - timedelta(days=10):
            raise RuntimeError(
                f"NSE historical index chunk ends too early for "
                f"{INDEX_HISTORY_NAMES[root]} {cursor}..{chunk_end}: "
                f"{chunk_dates.max().date()}"
            )

        frames.append(normalized)
        cursor = chunk_end + timedelta(days=1)

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )

    if not _coverage_ok(combined, start_d, end_d):
        raise RuntimeError(
            f"NSE historical index coverage incomplete for "
            f"{INDEX_HISTORY_NAMES[root]}: "
            f"requested={start_d}..{end_d}, "
            f"actual={combined['date'].min().date()}..{combined['date'].max().date()}, "
            f"rows={len(combined)}"
        )

    combined.to_parquet(cache_path, index=False)

    if sync_drive:
        _drive_upload(
            cache_path,
            f"production_data/nse_official/indices/{root.replace('^','')}/{cache_path.name}",
            "application/octet-stream",
        )

    return combined



def fetch_index_history(
    index_symbol: str,
    start: date | datetime,
    end: date | datetime,
    sync_drive: bool = False,
) -> pd.DataFrame:
    """Fetch official NSE benchmark history with persistent Drive-first fallback.

    Historical replay should not fail merely because the NSE index-history API
    returns an HTML/non-JSON response from a hosted IP. The project already
    persists official NIFTY 50 / NIFTY BANK history in Google Drive when those
    files were previously retrieved successfully.
    """
    root = str(index_symbol).strip().upper()

    if root not in INDEX_HISTORY_NAMES:
        raise ValueError(
            f"Unsupported NSE benchmark: {index_symbol}"
        )

    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()

    def _drive_coverage_ok(frame: pd.DataFrame) -> bool:
        """Accept Drive benchmark data only when it covers the requested window."""
        if frame is None or frame.empty or "date" not in frame.columns:
            return False
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
        if dates.empty:
            return False
        first = dates.min().date()
        last = dates.max().date()
        start_slack = 10
        # The final benchmark date is a hard requirement for a PIT production
        # snapshot. Accepting a stale index through a calendar-day slack window
        # silently creates missing market-context features for the target
        # session and can leave thousands of rows incorrectly gated.
        if first > start_d + timedelta(days=start_slack):
            return False
        if last < end_d:
            return False
        span_days = max(1, (end_d - start_d).days)
        if span_days >= 90:
            expected_weekdays = int(span_days * 5 / 7)
            if len(dates) < max(30, int(expected_weekdays * 0.50)):
                return False
        return True

    # 1) Prefer already-persisted official NSE benchmark history, but only when
    # the persisted files cover the requested window. Previously a recent but
    # partial Drive slice could be accepted solely because its latest date was
    # fresh, leaving most stock rows without benchmark features.
    try:
        drive_frame = _drive_index_history(
            root,
            start_d,
            end_d,
        )

        if _drive_coverage_ok(drive_frame):
            return drive_frame

    except Exception:
        pass

    # 2) Use the existing official NSE API implementation.
    try:
        return _fetch_index_history_nse_api(
            root,
            start_d,
            end_d,
            sync_drive=sync_drive,
        )

    except Exception as api_exc:

        # 3) One final Drive retry after an API failure. This is deliberately
        # non-invasive and still uses only official persisted data.
        try:
            drive_frame = _drive_index_history(
                root,
                start_d,
                end_d,
            )

            if _drive_coverage_ok(drive_frame):
                return drive_frame

        except Exception:
            pass

        raise api_exc




NSE_CM_UDIFF_URL_TEMPLATE = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"
)
BULK_EOD_ROOT = NORMALIZED_ROOT / "bulk_eod"
BULK_EOD_MANIFEST = BULK_EOD_ROOT / "manifest.json"

CANONICAL_CASH_MASTER = Path(
    "/content/drive/MyDrive/Indian_Market_Research/"
    "production_data/nse_official/"
    "ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.parquet"
)

# Research projection policy for the continuous cash-equity model. The raw
# identity-preserved master retains every historical series, including T0 and
# non-equity instruments. This projection deliberately excludes T0 and chooses
# one deterministic research row per symbol/date.
CANONICAL_RESEARCH_SERIES = ("EQ", "BE", "BZ", "SM", "ST", "SZ")
CANONICAL_SERIES_PRIORITY = {
    "EQ": 0,
    "BE": 1,
    "BZ": 2,
    "SM": 3,
    "ST": 4,
    "SZ": 5,
}



def load_canonical_cash_research_history(
    master_path: str | Path = CANONICAL_CASH_MASTER,
    *,
    symbols: Iterable[str] | None = None,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
) -> pd.DataFrame:
    """Read the identity-preserved NSE cash master as the model research view.

    The canonical master is never modified. This creates a deterministic
    symbol/date projection for downstream feature engineering:
      * only continuous cash-equity series are eligible;
      * T0 and other historical instruments remain preserved in the master but
        are excluded from ordinary Delivery OHLC features;
      * when multiple eligible listings exist for a symbol/date, series
        precedence EQ > BE > BZ > SM > ST > SZ is applied, then listing_id is
        used as a deterministic tie-breaker;
      * no third-party price source is substituted.
    """
    path = Path(master_path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Canonical NSE cash master not found: {path}")

    conditions = [
        "UPPER(TRIM(CAST(series AS VARCHAR))) IN ('EQ','BE','BZ','SM','ST','SZ')",
        "TRY_CAST(date AS DATE) IS NOT NULL",
    ]
    params: list[object] = []

    if start is not None:
        conditions.append("TRY_CAST(date AS DATE) >= ?")
        params.append(pd.Timestamp(start).date())
    if end is not None:
        conditions.append("TRY_CAST(date AS DATE) <= ?")
        params.append(pd.Timestamp(end).date())

    if symbols is not None:
        normalized = sorted({
            root_symbol(str(s)) for s in symbols
            if str(s).strip()
        })
        if not normalized:
            return pd.DataFrame()
        placeholders = ",".join("?" for _ in normalized)
        conditions.append(
            f"UPPER(TRIM(CAST(symbol AS VARCHAR))) IN ({placeholders})"
        )
        params.extend(normalized)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    master_sql = str(path).replace("'", "''")
    where_sql = " AND ".join(conditions)

    query = f"""
        WITH eligible AS (
            SELECT
                symbol,
                CAST(TRY_CAST(date AS DATE) AS DATE) AS date,
                series,
                isin,
                economic_security_id,
                listing_id,
                row_identity_key,
                security_class,
                instrument_type,
                source_format,
                instrument_id,
                instrument_name,
                open,
                high,
                low,
                close,
                last,
                prev_close,
                last_price_for_research,
                research_price,
                volume,
                traded_value,
                num_trades
            FROM read_parquet('{master_sql}')
            WHERE {where_sql}
              AND TRY_CAST(open AS DOUBLE) IS NOT NULL
              AND TRY_CAST(high AS DOUBLE) IS NOT NULL
              AND TRY_CAST(low AS DOUBLE) IS NOT NULL
              AND TRY_CAST(close AS DOUBLE) IS NOT NULL
              AND TRY_CAST(close AS DOUBLE) > 0
        ),
        ranked AS (
            SELECT
                *,
                CASE UPPER(TRIM(CAST(series AS VARCHAR)))
                    WHEN 'EQ' THEN 0
                    WHEN 'BE' THEN 1
                    WHEN 'BZ' THEN 2
                    WHEN 'SM' THEN 3
                    WHEN 'ST' THEN 4
                    WHEN 'SZ' THEN 5
                    ELSE 99
                END AS series_priority
            FROM eligible
        )
        SELECT
            symbol,
            date,
            UPPER(TRIM(CAST(series AS VARCHAR))) AS series,
            isin,
            economic_security_id,
            listing_id,
            row_identity_key,
            security_class,
            instrument_type,
            source_format,
            instrument_id,
            instrument_name,
            open,
            high,
            low,
            close,
            last,
            prev_close,
            last_price_for_research,
            research_price,
            volume,
            traded_value,
            num_trades
        FROM ranked
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY UPPER(TRIM(CAST(symbol AS VARCHAR))), date
            ORDER BY series_priority ASC, CAST(listing_id AS VARCHAR) ASC
        ) = 1
        ORDER BY UPPER(TRIM(CAST(symbol AS VARCHAR))), date
    """

    frame = con.execute(query, params).df()
    if frame.empty:
        return frame

    frame["symbol"] = (
        frame["symbol"].astype(str).str.strip().str.upper().map(root_symbol)
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for col in (
        "open","high","low","close","last","prev_close",
        "last_price_for_research","research_price",
        "volume","traded_value","num_trades",
    ):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return frame.reset_index(drop=True)


def _parse_nse_csv_bytes(payload: bytes) -> pd.DataFrame:
    """Read a CSV payload while tolerating UTF-8/BOM and alternate delimiters."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(io.BytesIO(payload), encoding=encoding)
            if len(frame.columns) > 1:
                return frame
        except Exception:
            continue
    return pd.DataFrame()


def _normalize_udiff_common_bhavcopy(frame: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    x = frame.copy()
    x.columns = [str(c).strip() for c in x.columns]
    aliases = {
        "TradDt": "date", "TRADING_DATE": "date", "Date": "date",
        "TckrSymb": "symbol", "CH_SYMBOL": "symbol", "Symbol": "symbol",
        "SctySrs": "series", "CH_SERIES": "series", "Series": "series",
        "OpnPric": "open", "CH_OPENING_PRICE": "open", "Open": "open",
        "HghPric": "high", "CH_TRADE_HIGH_PRICE": "high", "High": "high",
        "LwPric": "low", "CH_TRADE_LOW_PRICE": "low", "Low": "low",
        "ClsPric": "close", "CH_CLOSING_PRICE": "close", "Close": "close",
        "LastPric": "ltp", "CH_LAST_TRADED_PRICE": "ltp", "Last": "ltp",
        "PrvsClsgPric": "prev_close", "CH_PREVIOUS_CLS_PRICE": "prev_close",
        "TtlTradgVol": "volume", "CH_TOT_TRADED_QTY": "volume", "Volume": "volume",
        "TtlTrfVal": "traded_value", "CH_TOT_TRADED_VAL": "traded_value",
        "TtlNbOfTxsExctd": "num_trades", "CH_TOTAL_TRADES": "num_trades",
    }
    ren = {c: aliases[c] for c in x.columns if c in aliases}
    x = x.rename(columns=ren)
    if "date" not in x.columns:
        x["date"] = pd.Timestamp(trade_date)

    # NSE CM-UDiFF commonly supplies ISO dates such as YYYY-MM-DD.
    # Do not parse these with dayfirst=True: that can silently transform
    # 2026-07-01 into 2026-01-07 and similar month/day inversions.
    raw_dates = x["date"].astype(str).str.strip()
    parsed_dates = pd.to_datetime(
        raw_dates,
        format="%Y-%m-%d",
        errors="coerce",
    )

    # Support the other date representations seen in NSE extracts without
    # applying dayfirst parsing to already-ISO values.
    remaining = parsed_dates.isna()
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%y", "%d/%b/%Y", "%Y%m%d"):
        if not remaining.any():
            break
        candidate = pd.to_datetime(
            raw_dates.loc[remaining],
            format=fmt,
            errors="coerce",
        )
        parsed_dates.loc[remaining] = candidate
        remaining = parsed_dates.isna()

    # Final fallback for datetime-like values that may already have been
    # parsed by pandas before normalization.
    if remaining.any():
        candidate = pd.to_datetime(
            x.loc[remaining, "date"],
            errors="coerce",
        )
        parsed_dates.loc[remaining] = candidate

    x["date"] = parsed_dates
    if x["date"].isna().all():
        x["date"] = pd.Timestamp(trade_date)
    if "symbol" not in x.columns:
        return pd.DataFrame()
    x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
    if "series" in x.columns:
        x["series"] = x["series"].astype(str).str.strip().str.upper()
    else:
        x["series"] = "EQ"
    # Retain equity-like series; exclude ETFs/ETNs/debt from the stock universe.
    x = x[x["series"].isin({"EQ", "BE", "BZ", "SM", "ST", "SZ", "GB", "RR"})].copy()
    for c in ("open", "high", "low", "close", "ltp", "prev_close", "volume", "traded_value", "num_trades"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if "close" not in x.columns and "ltp" in x.columns:
        x["close"] = x["ltp"]
    required = {"symbol", "date", "open", "high", "low", "close"}
    if not required.issubset(x.columns):
        return pd.DataFrame()
    x["source"] = "NSE_OFFICIAL_UDIFF_BHAVCOPY"
    x["provenance"] = "nsearchives.nseindia.com CM-UDiFF Common Bhavcopy"
    return x.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)


def _sync_bulk_eod_to_drive(trade_date: date, parquet_path: Path, zip_path: Path) -> None:
    """Persist the canonical EOD parquet and fail if that required upload fails."""
    remote_dir = f"production_data/nse_official/bulk_eod/{trade_date:%Y/%m/%d}"

    # Retain the source ZIP when possible, but the parquet is the required
    # canonical artifact consumed by production and its freshness gate.
    if zip_path.exists():
        zip_result = _drive_upload(
            zip_path,
            f"{remote_dir}/{zip_path.name}",
            "application/zip",
        )
        if not isinstance(zip_result, dict) or zip_result.get("status") != "OK":
            detail = zip_result.get("error") if isinstance(zip_result, dict) else repr(zip_result)
            print(
                f"EOD ZIP Drive sync warning for {trade_date}: {detail}",
                flush=True,
            )

    parquet_result = _drive_upload(
        parquet_path,
        f"{remote_dir}/{parquet_path.name}",
        "application/octet-stream",
    )
    if not isinstance(parquet_result, dict) or parquet_result.get("status") != "OK":
        detail = parquet_result.get("error") if isinstance(parquet_result, dict) else repr(parquet_result)
        raise RuntimeError(
            f"Required NSE EOD parquet was not persisted to Google Drive for "
            f"{trade_date}: {detail}"
        )


def fetch_nse_eod_bhavcopy(trade_date: date | datetime, sync_drive: bool = False,
                           force_refresh: bool = False) -> pd.DataFrame:
    """Fetch one official NSE CM-UDiFF common bhavcopy containing all equity symbols.

    This is the bulk-EOD path used to expand coverage without issuing one NSE
    request per security. The downloaded day is cached locally and can be
    mirrored to Google Drive when explicitly requested.
    """
    global _last_nse_diagnostic

    d = pd.Timestamp(trade_date).date()
    key = d.strftime("%Y%m%d")
    out_dir = BULK_EOD_ROOT / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"BhavCopy_NSE_CM_0_0_0_{key}_F_0000.parquet"
    zip_path = out_dir / f"BhavCopy_NSE_CM_0_0_0_{key}_F_0000.csv.zip"
    if parquet_path.exists() and parquet_path.stat().st_size > 0 and not force_refresh:
        cached = pd.read_parquet(parquet_path)
        # A prior run may have written the local cache but failed to upload it.
        # When Drive sync is requested, retry the canonical parquet upload even
        # on the cache-hit path; otherwise gap repair can never heal that gap.
        if sync_drive:
            _sync_bulk_eod_to_drive(d, parquet_path, zip_path)
        return cached
    url = NSE_CM_UDIFF_URL_TEMPLATE.format(date=key)
    # Archive endpoint is a direct static download; use one bounded attempt.
    _sleep_rate_limit(0.25)
    try:
        r = requests.get(
            url,
            headers={**HEADERS, "Accept": "application/zip,text/plain,*/*"},
            timeout=(3, 8),
        )
        _last_nse_diagnostic = {
            "path": url,
            "status_code": r.status_code,
            "content_type": r.headers.get("content-type", ""),
            "bytes": len(r.content),
        }
        r.raise_for_status()
    except requests.HTTPError as exc:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        _last_nse_diagnostic.update(
            {
                "path": url,
                "status_code": status_code,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise RuntimeError(
            f"NSE EOD bhavcopy unavailable for {key}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    except requests.RequestException as exc:
        _last_nse_diagnostic.update(
            {
                "path": url,
                "status_code": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise RuntimeError(
            f"NSE EOD bhavcopy unavailable for {key}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    zip_path.write_bytes(r.content)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            return pd.DataFrame()
        # Prefer the main CM CSV, otherwise first CSV-like member.
        name = next((n for n in names if n.lower().endswith(".csv")), names[0])
        raw = zf.read(name)
    frame = _parse_nse_csv_bytes(raw)
    normalized = _normalize_udiff_common_bhavcopy(frame, d)
    if normalized.empty:
        raise RuntimeError(f"NSE EOD bhavcopy parsed but yielded no equity rows for {key}")
    normalized.to_parquet(parquet_path, index=False)
    if sync_drive:
        _sync_bulk_eod_to_drive(d, parquet_path, zip_path)
    if BULK_EOD_MANIFEST.exists():
        try:
            m = json.loads(BULK_EOD_MANIFEST.read_text())
            if not isinstance(m, dict):
                raise ValueError("bulk EOD manifest is not a JSON object")
        except (OSError, ValueError, json.JSONDecodeError):
            # A previously interrupted runner can leave a partial manifest.
            # Reinitialize it from the files already present on disk.
            m = {"version": "nse-cm-udiff-v1", "days": {}}
    else:
        m = {"version": "nse-cm-udiff-v1", "days": {}}
    m.setdefault("days", {})[key] = {
        "source": "NSE_OFFICIAL_UDIFF_BHAVCOPY", "url": url,
        "parquet": str(parquet_path), "rows": int(len(normalized)),
        "retrieved_at": datetime.now(tz=IST).isoformat(),
    }
    BULK_EOD_MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True, default=str))
    return normalized


def _drive_bulk_file_index() -> dict[str, str]:
    """Return a cached Google Drive index of official NSE bulk-EOD parquet files."""
    cache = getattr(_drive_bulk_file_index, "_cache", None)
    if cache is not None:
        return cache

    try:
        from persistent_store import GoogleDriveStore

        store = GoogleDriveStore()

        files = {}
        token = None

        while True:
            kwargs = {
                "q": (
                    "name contains 'BhavCopy_NSE_CM_0_0_0_' "
                    "and name contains '.parquet' and trashed = false"
                ),
                "spaces": "drive",
                "fields": "nextPageToken,files(id,name)",
                "pageSize": 1000,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }

            if token:
                kwargs["pageToken"] = token

            resp = store.service.files().list(**kwargs).execute()

            for item in resp.get("files", []):
                name = str(item.get("name", "")).strip()
                file_id = str(item.get("id", "")).strip()
                if name and file_id:
                    files[name] = file_id

            token = resp.get("nextPageToken")

            if not token:
                break

        _drive_bulk_file_index._cache = files
        return files

    except Exception:
        _drive_bulk_file_index._cache = {}
        return {}


def _drive_download_bulk_day(filename: str, destination: Path) -> bool:
    """Download one official NSE bulk-EOD parquet from Google Drive."""
    try:
        from persistent_store import GoogleDriveStore
        from googleapiclient.http import MediaIoBaseDownload

        index = _drive_bulk_file_index()
        file_id = index.get(filename)

        if not file_id:
            return False

        store = GoogleDriveStore()

        request = store.service.files().get_media(
            fileId=file_id
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp = destination.with_suffix(
            destination.suffix + ".tmp"
        )

        with temp.open("wb") as fh:
            downloader = MediaIoBaseDownload(
                fh,
                request,
            )

            done = False

            while not done:
                _, done = downloader.next_chunk()

        temp.replace(destination)
        return True

    except Exception:
        return False


def fetch_nse_drive_bulk_symbol_history(
    symbol: str,
    start: date | datetime,
    end: date | datetime,
    max_calendar_days: int = 240,
) -> pd.DataFrame:
    """Read official NSE bulk-EOD history from Google Drive when direct NSE
    security-history HTTP is unavailable.

    The Drive archive is the persistent official NSE source already populated
    by the project's EOD ingestion workflow. Only the requested symbol is
    retained from each day, and downloaded day files are cached locally.
    """
    s = root_symbol(symbol)

    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()

    earliest = end_d - timedelta(
        days=int(max_calendar_days)
    )

    if start_d < earliest:
        start_d = earliest

    days = pd.date_range(
        start_d,
        end_d,
        freq="B",
    )

    if len(days) == 0:
        return pd.DataFrame()

    pieces = []

    for day in days:

        d = day.date()

        filename = (
            f"BhavCopy_NSE_CM_0_0_0_"
            f"{d:%Y%m%d}_F_0000.parquet"
        )

        local_path = (
            BULK_EOD_ROOT
            / f"{d:%Y}"
            / f"{d:%m}"
            / f"{d:%d}"
            / filename
        )

        try:
            if local_path.exists() and local_path.stat().st_size > 0:
                day_df = pd.read_parquet(
                    local_path,
                    columns=None,
                )

            else:
                ok = _drive_download_bulk_day(
                    filename,
                    local_path,
                )

                if not ok:
                    continue

                day_df = pd.read_parquet(
                    local_path,
                    columns=None,
                )

            if day_df.empty:
                continue

            if "symbol" not in day_df.columns:
                continue

            part = day_df[
                day_df["symbol"]
                .astype(str)
                .str.strip()
                .str.upper()
                == s
            ].copy()

            if not part.empty:
                pieces.append(part)

        except Exception:
            continue

    if not pieces:
        return pd.DataFrame()

    out = pd.concat(
        pieces,
        ignore_index=True,
    )

    out["symbol"] = (
        out["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out["date"] = pd.to_datetime(
        out["date"],
        errors="coerce",
    ).dt.normalize()

    return (
        out
        .dropna(
            subset=[
                "symbol",
                "date",
            ]
        )
        .sort_values("date")
        .drop_duplicates(
            [
                "symbol",
                "date",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )


def fetch_nse_bulk_symbol_history(symbol: str, start: date | datetime, end: date | datetime,
                                  sync_drive: bool = False, max_days: int | None = None,
                                  download_missing: bool = False) -> pd.DataFrame:
    """Build per-symbol daily history from official NSE bulk CM-UDiFF day files.

    Cached days are read locally; missing days are downloaded one at a time. The
    function is intended for scheduled/backfill jobs, not tight UI loops.
    """
    s = root_symbol(symbol)
    start_d = pd.Timestamp(start).date(); end_d = pd.Timestamp(end).date()
    days = pd.date_range(start_d, end_d, freq="B")
    if max_days is not None:
        days = days[-int(max_days):]
    pieces = []
    for day in days:
        try:
            d = day.date()
            cache = BULK_EOD_ROOT / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}" / f"BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.parquet"
            if cache.exists():
                df = pd.read_parquet(cache)
            elif download_missing:
                df = fetch_nse_eod_bhavcopy(d, sync_drive=sync_drive)
            else:
                continue
            part = df[df["symbol"].astype(str).str.upper() == s]
            if not part.empty:
                pieces.append(part)
        except Exception:
            continue
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values("date").drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)

EQUITY_UNIVERSE_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

def fetch_nse_equity_universe(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch the current NSE-listed/available-for-trading equity universe.

    This is the official NSE securities CSV linked from the current
    "Securities available for Trading" page.
    """
    cache = NORMALIZED_ROOT / "universe" / "EQUITY_L.csv"
    if cache.exists() and not force_refresh and (time.time() - cache.stat().st_mtime) < 86400:
        return pd.read_csv(cache)
    _ensure_local()
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(EQUITY_UNIVERSE_URL, headers=HEADERS, timeout=(3, 8))
        r.raise_for_status()
        cache.write_bytes(r.content)
        df = pd.read_csv(io.BytesIO(r.content))
    except Exception:
        if cache.exists():
            return pd.read_csv(cache)
        raise
    return df

def nse_equity_symbols(force_refresh: bool = False) -> list[str]:
    df = fetch_nse_equity_universe(force_refresh=force_refresh)
    col = next((c for c in df.columns if str(c).strip().upper() in {"SYMBOL", "SYMBOL_NAME", "CH_SYMBOL"}), None)
    if col is None:
        raise ValueError(f"NSE equity universe has no symbol column: {list(df.columns)}")
    vals = (df[col].astype(str).str.strip().str.upper()
            .replace({"NAN": "", "NONE": ""}))
    return sorted({v for v in vals if v and v not in {"SYMBOL", "SCRIPCODE"}})

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


def fetch_intraday_1m_current(
    symbol: str,
    sync_drive: bool = True,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch today's NSE quote-chart path and aggregate available prints to 1-minute OHLC.

    ``retries`` is configurable so interactive Streamlit calls can fail fast
    when a hosted NSE edge is temporarily unavailable, while scheduled
    ingestion retains the conservative default retry policy.
    """
    s = root_symbol(symbol)
    params = {"index": s + "EQN"}
    global _nse_session, _nse_session_at, _last_nse_diagnostic

    # Use the public equity quote page as the referrer context for the chart
    # request. This matches the browser path users reach from NSE.
    session = _session()
    chart_headers = {
        "Accept": "application/json,text/plain,*/*",
        "Referer": f"{NSE_BASE}/get-quote/equity/{s}",
        "X-Requested-With": "XMLHttpRequest",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
    response = None
    last_exc = None
    attempts = max(1, int(retries))
    for attempt in range(attempts):
        try:
            _sleep_rate_limit(0.2 if attempts == 1 else 1.5)
            response = session.get(
                NSE_BASE + EQUITY_CHART_ENDPOINT,
                params=params,
                headers=chart_headers,
                timeout=(3, 6),
            )
            _last_nse_diagnostic = {
                "path": EQUITY_CHART_ENDPOINT,
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "bytes": len(response.content),
                "attempt": attempt + 1,
            }
            if response.status_code == 200:
                break
            if response.status_code in (401, 403, 408, 429, 500, 502, 503, 504):
                _nse_session = None
                _nse_session_at = 0.0
                if attempt + 1 < attempts:
                    session = _session()
                continue
            response.raise_for_status()
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                time.sleep(1.0)

    if response is None or response.status_code != 200:
        if last_exc:
            raise RuntimeError(f"NSE request failed: {type(last_exc).__name__}: {last_exc}") from last_exc
        raise RuntimeError(f"NSE request failed after {attempts} attempts: {EQUITY_CHART_ENDPOINT}")

    payload = response.json()
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
    """Non-blocking NSE status based on official universe archive reachability/cache.

    Quote-equity is intentionally not used as a startup health gate because a
    transient quote-endpoint block must not prevent historical/data-backed
    predictions.
    """
    try:
        df = fetch_nse_equity_universe(force_refresh=False)
        col = next((c for c in df.columns if str(c).strip().upper() in {"SYMBOL", "CH_SYMBOL"}), None)
        count = int(df[col].astype(str).str.strip().nunique()) if col else 0
        return {"status": "PASS" if count else "ERROR", "source": "NSE_OFFICIAL", "endpoint": EQUITY_UNIVERSE_URL, "universe_symbols": count, "quote_health_check": False}
    except Exception as exc:
        return {"status": "ERROR", "source": "NSE_OFFICIAL", "endpoint": EQUITY_UNIVERSE_URL, "error": f"{type(exc).__name__}: {exc}", "quote_health_check": False}
