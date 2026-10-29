"""Read-only INDstocks market-data adapter.

This module deliberately exposes no order-placement method. It supports:
- a dashboard-generated access token (manual refresh; 24-hour lifetime), or
- documented TOTP token generation for unattended read-only data collection.

Credentials are read from environment variables or Streamlit Secrets. Manual-token
mode is the safe default; TOTP token generation requires explicit
INDSTOCKS_AUTH_MODE=totp. Tokens, MPINs, and TOTP secrets are never logged.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import os
import re
import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any

import pandas as pd
import requests

API_BASE = "https://api.indstocks.com"
IST = ZoneInfo("Asia/Kolkata")
DEFAULT_TIMEOUT = 12
TOKEN_GENERATION_FAILURE_COOLDOWN_SECONDS = 60

_TOKEN_LOCK = threading.Lock()
_CACHED_TOKEN: str | None = None
_CACHED_TOKEN_EXPIRES_AT = 0.0
_CACHED_TOKEN_GENERATED_AT = 0.0
_CACHED_TOKEN_GENERATED_VALUE: str | None = None
_TOKEN_GENERATION_RETRY_AFTER = 0.0
_TOKEN_GENERATION_LAST_ERROR: str | None = None
_INSTRUMENTS: dict[str, tuple[float, pd.DataFrame]] = {}
_INSTRUMENTS_LOCK = threading.Lock()


def _setting(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    try:
        import streamlit as st
        value = st.secrets.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    return default


def _timeout() -> int:
    try:
        return max(3, min(60, int(_setting("INDSTOCKS_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT)) or DEFAULT_TIMEOUT)))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT


def _auth_mode() -> str:
    """Return the explicitly selected token owner mode; never infer TOTP from secrets."""
    mode = str(_setting("INDSTOCKS_AUTH_MODE", "manual") or "manual").strip().lower()
    if mode not in {"manual", "totp"}:
        raise RuntimeError("INDSTOCKS_AUTH_MODE must be either 'manual' or 'totp'")
    return mode


def configured() -> bool:
    """Check credentials for the selected mode without triggering authentication."""
    if _auth_mode() == "manual":
        return bool(_setting("INDSTOCKS_ACCESS_TOKEN"))
    return all(
        _setting(k)
        for k in ("INDSTOCKS_CLIENT_ID", "INDSTOCKS_MPIN", "INDSTOCKS_TOTP_SECRET")
    )


def _totp(secret: str, at_time: float | None = None) -> str:
    """RFC 6238 six-digit TOTP using only the Python standard library."""
    normalized = "".join(secret.strip().split()).upper()
    normalized += "=" * ((8 - len(normalized) % 8) % 8)
    key = base64.b32decode(normalized, casefold=True)
    counter = int((time.time() if at_time is None else at_time) // 30)
    digest = hmac.new(key, counter.to_bytes(8, "big"), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    number = (int.from_bytes(digest[offset:offset + 4], "big") & 0x7FFFFFFF) % 1_000_000
    return f"{number:06d}"


def _safe_provider_error_detail(response: requests.Response) -> str:
    """Return allowlisted provider error fields without exposing credentials or PII.

    Provider 403 responses can mean either token/authentication failure or an
    account/endpoint permission denial. Keep only structured diagnostic fields;
    never include the raw response body or arbitrary nested payloads.
    """
    try:
        body = response.json()
    except (ValueError, requests.RequestException):
        return ""
    if not isinstance(body, dict):
        return ""

    aliases = {
        "error_type": ("error_type", "errorType", "type"),
        "code": ("code", "error_code", "errorCode"),
        "message": ("message", "error_message", "errorMessage", "error_description", "errorDescription"),
        "detail": ("detail", "description"),
        "request_id": ("request_id", "requestId", "correlation_id", "correlationId"),
    }

    # Some API gateways wrap diagnostic fields inside an "error" object.
    containers = [body]
    nested = body.get("error")
    if isinstance(nested, dict):
        containers.append(nested)

    parts: list[str] = []
    seen: set[str] = set()
    for label, names in aliases.items():
        value = None
        for container in containers:
            for name in names:
                candidate = container.get(name)
                if isinstance(candidate, (str, int, float, bool)):
                    value = candidate
                    break
            if value is not None:
                break
        if value is None:
            continue

        rendered = re.sub(r"[\r\n\t]+", " ", str(value)).strip()
        rendered = re.sub(
            r"(?i)(authorization\s*[:=]\s*)(?:(?:bearer|token)\s+)?[^\s,;&]+",
            r"\1[REDACTED]",
            rendered,
        )
        rendered = re.sub(
            r"(?i)\b((?:x-api-key|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|token|mpin|totp|password|secret)\s*[:=]\s*)[^\s,;&]+",
            r"\1[REDACTED]",
            rendered,
        )
        rendered = re.sub(
            r"\b[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
            "[REDACTED_JWT]",
            rendered,
        )
        rendered = rendered[:240]
        if rendered and label not in seen:
            parts.append(f"{label}={rendered}")
            seen.add(label)

    if not parts:
        return ""
    return "; provider_response=" + ", ".join(parts)

def _generate_token() -> str:
    client_id = _setting("INDSTOCKS_CLIENT_ID")
    mpin = _setting("INDSTOCKS_MPIN")
    secret = _setting("INDSTOCKS_TOTP_SECRET")
    if not all((client_id, mpin, secret)):
        raise RuntimeError(
            "INDstocks automatic authentication is not configured. Set "
            "INDSTOCKS_CLIENT_ID, INDSTOCKS_MPIN and INDSTOCKS_TOTP_SECRET "
            "in Streamlit Secrets, or provide INDSTOCKS_ACCESS_TOKEN."
        )
    payload = {"mpin": mpin, "totp": _totp(secret)}
    try:
        response = requests.post(
            f"{API_BASE}/generate/token",
            headers={
                "x-api-key": client_id,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=_timeout(),
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"INDstocks token generation network error: {type(exc).__name__}") from None
    if response.status_code >= 400:
        if response.status_code in (401, 403):
            detail = "authentication rejected; verify Client ID, MPIN, TOTP setup and server clock"
        elif response.status_code == 429:
            detail = "token generation rate-limited; wait before retrying"
        else:
            detail = f"HTTP {response.status_code}"
        detail += _safe_provider_error_detail(response)
        raise RuntimeError(f"INDstocks token generation failed: {detail}")
    try:
        body = response.json()
    except ValueError:
        raise RuntimeError("INDstocks token generation returned a non-JSON response") from None
    token = body.get("token")
    if not token and isinstance(body.get("data"), dict):
        token = body["data"].get("token")
    if not token:
        raise RuntimeError("INDstocks token generation did not return a token")
    return str(token)


def _generate_and_cache_token_locked() -> str:
    """Generate/cache one token while holding _TOKEN_LOCK, with failure cooldown."""
    global _CACHED_TOKEN, _CACHED_TOKEN_EXPIRES_AT, _CACHED_TOKEN_GENERATED_AT
    global _CACHED_TOKEN_GENERATED_VALUE
    global _TOKEN_GENERATION_RETRY_AFTER, _TOKEN_GENERATION_LAST_ERROR

    now = time.time()
    if now < _TOKEN_GENERATION_RETRY_AFTER:
        remaining = max(1, int(_TOKEN_GENERATION_RETRY_AFTER - now))
        detail = f"; previous failure: {_TOKEN_GENERATION_LAST_ERROR}" if _TOKEN_GENERATION_LAST_ERROR else ""
        raise RuntimeError(
            f"INDstocks token generation is cooling down for another {remaining}s{detail}"
        )

    try:
        token = _generate_token()
    except Exception as exc:
        # The previous cached token is expired or has already been rejected.
        # Never leave it available as a fallback when refresh fails.
        _CACHED_TOKEN = None
        _CACHED_TOKEN_EXPIRES_AT = 0.0
        _CACHED_TOKEN_GENERATED_AT = 0.0
        _CACHED_TOKEN_GENERATED_VALUE = None
        _TOKEN_GENERATION_RETRY_AFTER = (
            time.time() + TOKEN_GENERATION_FAILURE_COOLDOWN_SECONDS
        )
        _TOKEN_GENERATION_LAST_ERROR = f"{type(exc).__name__}: {str(exc)[:180]}"
        # Do not log credential values or repeat provider calls from every
        # concurrent history batch after a token endpoint failure/rate limit.
        raise RuntimeError(
            "INDstocks token generation failed; further attempts are suppressed "
            f"for {TOKEN_GENERATION_FAILURE_COOLDOWN_SECONDS}s: "
            f"{_TOKEN_GENERATION_LAST_ERROR}"
        ) from None

    _CACHED_TOKEN = token
    _CACHED_TOKEN_GENERATED_AT = time.time()
    _CACHED_TOKEN_GENERATED_VALUE = token
    _CACHED_TOKEN_EXPIRES_AT = _CACHED_TOKEN_GENERATED_AT + 23 * 3600
    _TOKEN_GENERATION_RETRY_AFTER = 0.0
    _TOKEN_GENERATION_LAST_ERROR = None
    return token


def access_token() -> str:
    """Use the explicitly selected credential mode; only TOTP mode can mint tokens."""
    global _CACHED_TOKEN, _CACHED_TOKEN_EXPIRES_AT
    mode = _auth_mode()
    if mode == "manual":
        manual = _setting("INDSTOCKS_ACCESS_TOKEN")
        if not manual:
            raise RuntimeError(
                "INDstocks manual-token mode is active but INDSTOCKS_ACCESS_TOKEN "
                "is not configured; TOTP generation is disabled in this process"
            )
        return manual

    if not _has_totp_credentials():
        raise RuntimeError(
            "INDstocks TOTP mode requires INDSTOCKS_CLIENT_ID, INDSTOCKS_MPIN "
            "and INDSTOCKS_TOTP_SECRET"
        )
    with _TOKEN_LOCK:
        now = time.time()
        if _CACHED_TOKEN and _CACHED_TOKEN_EXPIRES_AT > now + 3600:
            return _CACHED_TOKEN
        return _generate_and_cache_token_locked()


def _has_totp_credentials() -> bool:
    """Return True only when this process is explicitly authorized to mint tokens."""
    return _auth_mode() == "totp" and all(
        _setting(name)
        for name in ("INDSTOCKS_CLIENT_ID", "INDSTOCKS_MPIN", "INDSTOCKS_TOTP_SECRET")
    )


def _is_rejected_access_token(response: requests.Response) -> bool:
    """Detect explicit invalid/expired access-token responses, not generic 403s."""
    if response.status_code not in (401, 403):
        return False
    detail = _safe_provider_error_detail(response).lower()
    names_access_token = "access_token" in detail or "access token" in detail
    signals_invalidity = any(
        marker in detail
        for marker in (
            "incorrect", "expired", "revoked", "re-authenticate",
            "reauthenticate", "invalid token", "token is invalid",
        )
    )
    return names_access_token and signals_invalidity


def _refresh_rejected_access_token(rejected_token: str) -> str:
    """Refresh a TOTP token rejected before its nominal 24-hour lifetime.

    INDstocks may revoke a token early when a new session is created. Refreshes
    are serialized, and failed token generation is cooled down to prevent each
    concurrent history batch from hammering a rate-limited token endpoint.
    """
    global _CACHED_TOKEN, _CACHED_TOKEN_EXPIRES_AT, _CACHED_TOKEN_GENERATED_AT
    global _CACHED_TOKEN_GENERATED_VALUE
    global _TOKEN_GENERATION_RETRY_AFTER, _TOKEN_GENERATION_LAST_ERROR
    if not _has_totp_credentials():
        raise RuntimeError("INDstocks automatic authentication is not configured")

    with _TOKEN_LOCK:
        now = time.time()
        # Another request may already have repaired the shared cache while this
        # response was in flight. Reuse that newer token rather than log in again.
        if (
            _CACHED_TOKEN
            and _CACHED_TOKEN != rejected_token
            and _CACHED_TOKEN_EXPIRES_AT > now
        ):
            return _CACHED_TOKEN

        # The just-issued TOTP token may be rejected immediately by the API.
        # Do not mint a second account-wide token in the same minute: INDstocks
        # rate-limits token issuance, and another login can invalidate the shared
        # token while still being rejected. Quarantine this token and defer the
        # next login attempt until the provider cooldown expires.
        generated_at = _CACHED_TOKEN_GENERATED_AT
        generated_value = _CACHED_TOKEN_GENERATED_VALUE
        _CACHED_TOKEN = None
        _CACHED_TOKEN_EXPIRES_AT = 0.0
        _CACHED_TOKEN_GENERATED_AT = 0.0
        _CACHED_TOKEN_GENERATED_VALUE = None
        if (
            generated_value == rejected_token
            and generated_at
            and 0 <= now - generated_at < TOKEN_GENERATION_FAILURE_COOLDOWN_SECONDS
        ):
            _TOKEN_GENERATION_RETRY_AFTER = max(
                _TOKEN_GENERATION_RETRY_AFTER,
                generated_at + TOKEN_GENERATION_FAILURE_COOLDOWN_SECONDS,
            )
            _TOKEN_GENERATION_LAST_ERROR = (
                "freshly generated token was rejected; waiting for provider cooldown"
            )
            remaining = max(1, int(_TOKEN_GENERATION_RETRY_AFTER - now))
            raise RuntimeError(
                "freshly generated INDstocks token was rejected; "
                f"further token generation is suppressed for {remaining}s"
            )
        return _generate_and_cache_token_locked()


def _send_authenticated_request(
    method: str,
    path: str,
    params: dict[str, Any] | None,
    token: str,
) -> requests.Response:
    try:
        return requests.request(
            method,
            f"{API_BASE}{path}",
            headers={
                "Authorization": token,
                "Accept": "application/json, text/csv, */*",
                "Content-Type": "application/json",
            },
            params=params,
            timeout=_timeout(),
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"INDstocks request failed: {type(exc).__name__}") from None


def _request(method: str, path: str, *, params: dict[str, Any] | None = None) -> requests.Response:
    token = access_token()
    response = _send_authenticated_request(method, path, params, token)

    # Retry exactly once only when the provider explicitly says this access token
    # is invalid/expired/revoked. Do not turn ordinary endpoint-permission 403s
    # into repeated authentication attempts or use this path for manual tokens.
    if _has_totp_credentials() and _is_rejected_access_token(response):
        try:
            refreshed_token = _refresh_rejected_access_token(token)
        except Exception as exc:
            raise RuntimeError(
                "INDstocks rejected the current access token and automatic "
                f"re-authentication failed: {type(exc).__name__}: {str(exc)[:240]}"
            ) from None
        if refreshed_token:
            response = _send_authenticated_request(method, path, params, refreshed_token)

    if response.status_code >= 400:
        if response.status_code == 401:
            detail = "HTTP 401; the access token may have expired, or authentication failed"
        elif response.status_code == 403:
            detail = "HTTP 403; authentication or endpoint permissions were rejected"
        elif response.status_code == 429:
            detail = "HTTP 429; provider rate limit reached"
        else:
            detail = f"HTTP {response.status_code}"
        detail += _safe_provider_error_detail(response)
        raise RuntimeError(f"INDstocks API {detail}")
    return response


def profile() -> dict[str, Any]:
    """Safe authentication smoke test; callers should avoid displaying account PII."""
    body = _request("GET", "/user/profile").json()
    if not isinstance(body, dict):
        raise RuntimeError("INDstocks profile response was not an object")
    if body.get("status") not in (None, "success") and body.get("success") is not True:
        raise RuntimeError("INDstocks profile check did not succeed")
    return body.get("data", body)


def _root_symbol(symbol: str) -> tuple[str, str]:
    value = str(symbol).strip().upper()
    exchange = "BSE" if value.endswith(".BO") else "NSE"
    for suffix in (".NS", ".BO"):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
            break
    if not value or value.startswith("^"):
        raise ValueError(f"Unsupported INDstocks equity symbol: {symbol}")
    return value, exchange


def _load_instruments(exchange: str) -> pd.DataFrame:
    now = time.time()
    with _INSTRUMENTS_LOCK:
        cached = _INSTRUMENTS.get(exchange)
        if cached and now - cached[0] < 6 * 3600:
            return cached[1].copy()
    response = _request("GET", "/market/instruments", params={"source": "equity"})
    try:
        frame = pd.read_csv(io.BytesIO(response.content), dtype=str)
    except Exception:
        raise RuntimeError("Could not parse INDstocks equity instrument master CSV") from None
    frame.columns = [str(c).strip().upper() for c in frame.columns]
    needed = {"EXCH", "SECURITY_ID", "TRADING_SYMBOL"}
    if not needed.issubset(frame.columns):
        raise RuntimeError("INDstocks instrument master is missing required columns")
    frame["EXCH"] = frame["EXCH"].astype(str).str.strip().str.upper()
    frame["TRADING_SYMBOL"] = frame["TRADING_SYMBOL"].astype(str).str.strip().str.upper()
    frame["SECURITY_ID"] = frame["SECURITY_ID"].astype(str).str.strip()
    with _INSTRUMENTS_LOCK:
        _INSTRUMENTS[exchange] = (now, frame)
    return frame.copy()


def resolve_scrip_code(symbol: str) -> str:
    """Resolve an NSE/BSE listed equity to the provider's SEGMENT_SECURITY_ID form."""
    root, exchange = _root_symbol(symbol)
    frame = _load_instruments(exchange)
    matches = frame[
        frame["EXCH"].eq(exchange)
        & frame["TRADING_SYMBOL"].eq(root)
        & frame["SECURITY_ID"].str.fullmatch(r"\d+")
    ].copy()
    if "SERIES" in matches.columns:
        ordinary = matches[matches["SERIES"].astype(str).str.strip().str.upper().eq("EQ")]
        if not ordinary.empty:
            matches = ordinary
    if "SEGMENT" in matches.columns:
        equity_segment = matches[matches["SEGMENT"].astype(str).str.strip().str.upper().isin({"E", "EQ", "EQUITY"})]
        if not equity_segment.empty:
            matches = equity_segment
    if matches.empty:
        raise RuntimeError(f"INDstocks instrument master has no {exchange} equity match for {root}")
    security_id = str(matches.iloc[0]["SECURITY_ID"]).strip()
    return f"{'NSE' if exchange == 'NSE' else 'BSE'}_{security_id}"


def latest_quote(symbol: str) -> dict[str, Any]:
    code = resolve_scrip_code(symbol)
    body = _request("GET", "/market/quotes/full", params={"scrip-codes": code}).json()
    data = body.get("data", {}) if isinstance(body, dict) else {}
    quote = data.get(code)
    if not isinstance(quote, dict) or quote.get("live_price") is None:
        raise RuntimeError(f"INDstocks quote was unavailable for {symbol}")
    return dict(quote)


def latest_price(symbol: str) -> float | None:
    try:
        return float(latest_quote(symbol)["live_price"])
    except (RuntimeError, ValueError, TypeError):
        return None


_INTERVALS = {
    "1m": ("1minute", 7),
    "1min": ("1minute", 7),
    "1minute": ("1minute", 7),
    "2m": ("2minute", 7),
    "3m": ("3minute", 7),
    "4m": ("4minute", 7),
    "5m": ("5minute", 7),
    "10m": ("10minute", 7),
    "15m": ("15minute", 7),
    "30m": ("30minute", 7),
    "60m": ("60minute", 15),
    "1h": ("60minute", 15),
    "120m": ("120minute", 15),
    "180m": ("180minute", 15),
    "240m": ("240minute", 15),
    "1d": ("1day", 365),
    "1day": ("1day", 365),
    "d": ("1day", 365),
    "day": ("1day", 365),
    "1w": ("1week", 365),
    "1week": ("1week", 365),
    "1mo": ("1month", 365),
    "1month": ("1month", 365),
}


def _ist_timestamp(value: Any, default: datetime) -> datetime:
    ts = pd.Timestamp(value) if value is not None else pd.Timestamp(default)
    if ts.tzinfo is None:
        ts = ts.tz_localize(IST)
    else:
        ts = ts.tz_convert(IST)
    return ts.to_pydatetime()


def historical(
    symbol: str,
    start: Any,
    end: Any,
    interval: str = "1minute",
    *,
    max_calls: int = 60,
) -> pd.DataFrame:
    """Download and stitch date-bounded historical OHLCV candles.

    INDstocks limits each request to 7 days for minute candles and 365 days for
    daily/weekly/monthly candles. The end timestamp is exclusive.
    """
    key = str(interval).strip().lower()
    if key not in _INTERVALS:
        raise ValueError(f"Unsupported INDstocks interval: {interval}")
    api_interval, window_days = _INTERVALS[key]
    start_dt = _ist_timestamp(start, datetime.now(IST) - timedelta(days=1))
    end_dt = _ist_timestamp(end, datetime.now(IST))
    if start_dt >= end_dt:
        raise ValueError("INDstocks historical start must be earlier than end")
    if end_dt - start_dt > timedelta(days=window_days * max_calls):
        raise ValueError(f"Requested range exceeds the {max_calls}-call safety cap for {interval}")

    code = resolve_scrip_code(symbol)
    step = timedelta(days=window_days)
    cursor = start_dt
    frames: list[pd.DataFrame] = []
    calls = 0
    while cursor < end_dt:
        window_end = min(end_dt, cursor + step)
        start_ms = int(cursor.timestamp() * 1000)
        end_ms = int(window_end.timestamp() * 1000)
        body = _request(
            "GET",
            f"/market/historical/{api_interval}",
            params={"scrip-codes": code, "start_time": start_ms, "end_time": end_ms},
        ).json()
        if not isinstance(body, dict) or body.get("success") is not True:
            raise RuntimeError("INDstocks historical API did not report success")
        data = body.get("data") or {}
        instrument = data.get(code) or {}
        candles = instrument.get("candles") or []
        if candles:
            df = pd.DataFrame(candles)
            required = {"ts", "o", "h", "l", "c", "v"}
            if required.issubset(df.columns):
                idx = pd.to_datetime(pd.to_numeric(df["ts"], errors="coerce"), unit="s", utc=True).dt.tz_convert(IST)
                # Use arrays rather than indexed Series: otherwise pandas
                # aligns the original integer row index with the DatetimeIndex
                # and silently turns every OHLCV cell into NaN.
                frame = pd.DataFrame(
                    {
                        "Open": pd.to_numeric(df["o"], errors="coerce").to_numpy(),
                        "High": pd.to_numeric(df["h"], errors="coerce").to_numpy(),
                        "Low": pd.to_numeric(df["l"], errors="coerce").to_numpy(),
                        "Close": pd.to_numeric(df["c"], errors="coerce").to_numpy(),
                        "Volume": pd.to_numeric(df["v"], errors="coerce").to_numpy(),
                    },
                    index=pd.DatetimeIndex(idx, name="Datetime"),
                )
                frames.append(frame.dropna(subset=["Open", "High", "Low", "Close"]))
        cursor = window_end
        calls += 1
        if calls > max_calls:
            raise RuntimeError("INDstocks historical API exceeded the configured call cap")

    if not frames:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex([], name="Datetime"))
    combined = pd.concat(frames).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def historical_1m(symbol: str, days: int = 3) -> pd.DataFrame:
    """Convenience wrapper for live-model input. One-minute data maxes at 7 days per call."""
    days = max(1, min(7, int(days)))
    end = datetime.now(IST)
    return historical(symbol, end - timedelta(days=days), end, "1minute", max_calls=2)


def historical_5m(symbol: str, days: int = 3) -> pd.DataFrame:
    """Convenience wrapper for five-minute OHLCV input used by production inference."""
    days = max(1, min(7, int(days)))
    end = datetime.now(IST)
    return historical(symbol, end - timedelta(days=days), end, "5m", max_calls=2)
