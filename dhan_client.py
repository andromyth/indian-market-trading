"""DhanHQ v2 integration for the Indian Stock Signal Dashboard.

Security model:
- Never hard-code credentials in source.
- Credentials are read from a local .env file or environment variables.
- API key/secret are used only for the Dhan OAuth consent flow.
- An access token is cached locally with a timestamp and expiry.
- This module never places orders; it only uses profile, quote and data APIs.

Current endpoints are based on DhanHQ v2 documentation:
- OAuth consent: https://auth.dhan.co/app/generate-consent
- OAuth consume: https://auth.dhan.co/app/consumeApp-consent
- Profile: https://api.dhan.co/v2/profile
- Historical: https://api.dhan.co/v2/charts/historical and /intraday
- Market quote: https://api.dhan.co/v2/marketfeed/ltp
- Instruments: https://images.dhan.co/api-data/api-scrip-master-detailed.csv
- Expired options: https://api.dhan.co/v2/charts/rollingoption
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import socket
import struct
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
Dhan_DIR = BASE_DIR / "data" / "dhan"
TOKEN_PATH = Dhan_DIR / "auth.json"
INSTRUMENTS_PATH = Dhan_DIR / "instruments_detailed.csv"

API_BASE = "https://api.dhan.co/v2"
AUTH_BASE = "https://auth.dhan.co/app"
INSTRUMENT_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"


def load_dotenv_local(path: Path | None = None) -> None:
    """Minimal .env loader; does not override already-exported variables."""
    path = path or (BASE_DIR / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class DhanConfig:
    client_id: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    access_token: str | None = None
    pin: str | None = None
    totp_secret: str | None = None
    redirect_url: str = "http://127.0.0.1:8765/callback"
    timeout_seconds: int = 30

    @classmethod
    def from_env(cls) -> "DhanConfig":
        """Load Dhan credentials from environment/.env, with Streamlit Secrets support.

        Streamlit Community Cloud does not provide a persistent local .env file.
        When running under Streamlit, values configured in ``st.secrets`` are used
        as a fallback and are never written to disk. Environment variables still
        take precedence for local deployments.
        """
        load_dotenv_local()

        def setting(name: str, default: str | None = None) -> str | None:
            value = os.getenv(name)
            if value:
                return value
            try:
                import streamlit as _st
                value = _st.secrets.get(name)
                if value is not None and str(value).strip():
                    return str(value).strip()
            except Exception:
                pass
            return default

        try:
            timeout = int(setting("DHAN_TIMEOUT_SECONDS", "30") or "30")
        except (TypeError, ValueError):
            timeout = 30

        return cls(
            client_id=setting("DHAN_CLIENT_ID"),
            api_key=setting("DHAN_API_KEY"),
            api_secret=setting("DHAN_API_SECRET"),
            access_token=setting("DHAN_ACCESS_TOKEN"),
            pin=setting("DHAN_PIN"),
            totp_secret=setting("DHAN_TOTP_SECRET"),
            redirect_url=setting("DHAN_REDIRECT_URL", "http://127.0.0.1:8765/callback") or "http://127.0.0.1:8765/callback",
            timeout_seconds=timeout,
        )

    def has_api_key_flow(self) -> bool:
        return bool(self.client_id and self.api_key and self.api_secret)

    def has_any_credential(self) -> bool:
        return bool(self.access_token or self.has_api_key_flow())

    def has_totp_flow(self) -> bool:
        return bool(self.client_id and self.pin and self.totp_secret)


def ensure_dirs() -> None:
    Dhan_DIR.mkdir(parents=True, exist_ok=True)


def _headers(access_token: str, client_id: str | None = None) -> dict[str, str]:
    h = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": access_token,
    }
    if client_id:
        h["client-id"] = client_id
    return h


def _response_json(response: requests.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(f"Dhan returned non-JSON HTTP {response.status_code}: {response.text[:500]}")
    if response.status_code >= 400:
        raise RuntimeError(f"Dhan HTTP {response.status_code}: {data}")
    return data if isinstance(data, dict) else {"data": data}


def save_token(payload: dict[str, Any]) -> None:
    ensure_dirs()
    TOKEN_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        TOKEN_PATH.chmod(0o600)
    except OSError:
        pass


def load_cached_token() -> dict[str, Any] | None:
    if not TOKEN_PATH.exists():
        return None
    try:
        return json.loads(TOKEN_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def cached_token_valid(buffer_seconds: int = 120) -> bool:
    data = load_cached_token()
    if not data or not data.get("accessToken"):
        return False
    expiry = data.get("expiryTime")
    if not expiry:
        return False
    try:
        dt = datetime.fromisoformat(str(expiry).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
        return dt.timestamp() > time.time() + buffer_seconds
    except Exception:
        return False


def totp_code(secret: str, at_time: float | None = None) -> str:
    """Generate a six-digit RFC 6238 TOTP code using SHA-1 and a 30-second step."""
    if not str(secret or "").strip():
        raise ValueError("TOTP secret is empty")
    normalized = "".join(str(secret).strip().split()).upper()
    padding = "=" * ((8 - len(normalized) % 8) % 8)
    try:
        key = base64.b32decode(normalized + padding, casefold=True)
    except Exception as exc:
        raise ValueError("DHAN_TOTP_SECRET must be a valid Base32 secret") from exc
    counter = int((time.time() if at_time is None else at_time) // 30)
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    binary = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return f"{binary % 1_000_000:06d}"


def generate_totp_access_token(config: DhanConfig) -> str:
    """Obtain a fresh 24-hour access token through Dhan's documented TOTP endpoint."""
    if not config.has_totp_flow():
        raise RuntimeError("DHAN_CLIENT_ID, DHAN_PIN and DHAN_TOTP_SECRET are required for TOTP authentication.")
    response = requests.post(
        f"{AUTH_BASE}/generateAccessToken",
        params={
            "dhanClientId": config.client_id,
            "pin": config.pin,
            "totp": totp_code(config.totp_secret),
        },
        timeout=config.timeout_seconds,
    )
    payload = _response_json(response)
    token = payload.get("accessToken")
    expiry = payload.get("expiryTime")
    if not token or not expiry:
        raise RuntimeError("Dhan TOTP authentication did not return accessToken and expiryTime.")
    payload["savedAt"] = datetime.now(timezone.utc).isoformat()
    save_token(payload)
    config.access_token = str(token)
    return str(token)


def get_access_token(config: DhanConfig) -> str | None:
    if config.has_totp_flow():
        cached = load_cached_token()
        if cached and cached_token_valid():
            config.access_token = str(cached["accessToken"])
            return config.access_token
        return generate_totp_access_token(config)
    if config.access_token:
        return config.access_token
    cached = load_cached_token()
    if cached and cached_token_valid():
        return str(cached["accessToken"])
    return None


def profile(config: DhanConfig) -> dict[str, Any]:
    token = get_access_token(config)
    if not token or not config.client_id:
        raise RuntimeError("Dhan access token/client ID not configured")
    response = requests.get(f"{API_BASE}/profile", headers=_headers(token), timeout=config.timeout_seconds)
    return _response_json(response)


def generate_consent(config: DhanConfig) -> str:
    if not config.has_api_key_flow():
        raise RuntimeError("Set DHAN_CLIENT_ID, DHAN_API_KEY and DHAN_API_SECRET in .env first.")
    response = requests.post(
        f"{AUTH_BASE}/generate-consent?{urlencode({'client_id': config.client_id})}",
        headers={"app_id": config.api_key, "app_secret": config.api_secret},
        timeout=config.timeout_seconds,
    )
    payload = _response_json(response)
    consent = payload.get("consentAppId")
    if not consent:
        raise RuntimeError(f"Dhan did not return consentAppId: {payload}")
    login_url = f"https://auth.dhan.co/login/consentApp-login?{urlencode({'consentAppId': consent})}"
    return login_url


def consume_consent(config: DhanConfig, token_id: str) -> dict[str, Any]:
    if not config.has_api_key_flow():
        raise RuntimeError("Set DHAN_CLIENT_ID, DHAN_API_KEY and DHAN_API_SECRET in .env first.")
    response = requests.get(
        f"{AUTH_BASE}/consumeApp-consent?{urlencode({'tokenId': token_id})}",
        headers={"app_id": config.api_key, "app_secret": config.api_secret},
        timeout=config.timeout_seconds,
    )
    payload = _response_json(response)
    if not payload.get("accessToken"):
        raise RuntimeError(f"Dhan did not return accessToken: {payload}")
    payload["savedAt"] = datetime.now(timezone.utc).isoformat()
    save_token(payload)
    return payload


class _OAuthHandler(BaseHTTPRequestHandler):
    token_id: str | None = None
    done = False

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        token = query.get("tokenId", [None])[0]
        if token:
            type(self).token_id = token
            type(self).done = True
            body = b"<html><body><h2>Dhan authentication received.</h2><p>You can close this tab and return to the dashboard.</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = b"<html><body><h2>No tokenId received.</h2></body></html>"
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, fmt, *args):  # silence local callback logs
        return


def oauth_login(config: DhanConfig, timeout_seconds: int = 180, open_browser: bool = True) -> dict[str, Any]:
    """Perform Dhan's documented 3-step key/secret consent flow locally.

    The redirect URL must point to this local callback, e.g.:
    http://127.0.0.1:8765/callback
    """
    parsed = urlparse(config.redirect_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if host not in {"127.0.0.1", "localhost"}:
        raise RuntimeError("For the built-in local callback, DHAN_REDIRECT_URL must use localhost/127.0.0.1.")
    server = HTTPServer((host, port), _OAuthHandler)
    _OAuthHandler.token_id = None
    _OAuthHandler.done = False
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        login_url = generate_consent(config)
        if open_browser:
            webbrowser.open(login_url, new=1)
        start = time.time()
        while time.time() - start < timeout_seconds and not _OAuthHandler.done:
            time.sleep(0.25)
        token_id = _OAuthHandler.token_id
        if not token_id:
            raise RuntimeError("Timed out waiting for the Dhan browser callback.")
        return consume_consent(config, token_id)
    finally:
        server.shutdown()
        server.server_close()


def download_instrument_master(force: bool = False, timeout: int = 60) -> Path:
    ensure_dirs()
    if INSTRUMENTS_PATH.exists() and INSTRUMENTS_PATH.stat().st_size > 0 and not force:
        return INSTRUMENTS_PATH
    response = requests.get(INSTRUMENT_MASTER_URL, timeout=timeout, headers={"User-Agent": "IndianStockSignalDashboard/1.0"})
    response.raise_for_status()
    INSTRUMENTS_PATH.write_bytes(response.content)
    return INSTRUMENTS_PATH


def instrument_table(force: bool = False) -> pd.DataFrame:
    path = download_instrument_master(force=force)
    return pd.read_csv(path, low_memory=False)


def _col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def resolve_equity_symbol(symbol: str) -> dict[str, Any]:
    root = str(symbol).upper().replace(".NS", "").replace(".BO", "")
    df = instrument_table()
    seg_col = _col(df, ["EXCH_ID", "SEM_EXM_EXCH_ID"])
    segment_col = _col(df, ["SEGMENT", "SEM_SEGMENT"])
    symbol_col = _col(df, ["SEM_TRADING_SYMBOL", "SM_SYMBOL_NAME", "SYMBOL_NAME", "SEM_CUSTOM_SYMBOL"])
    sec_col = _col(df, ["SECURITY_ID", "SEM_SMST_SECURITY_ID", "SEM_SECURITY_ID"])
    inst_col = _col(df, ["INSTRUMENT", "SEM_INSTRUMENT_NAME"])
    if not all([seg_col, segment_col, symbol_col, sec_col, inst_col]):
        raise RuntimeError(f"Unable to resolve Dhan instrument-master columns: {list(df.columns)[:30]}")
    m = df[(df[seg_col].astype(str).str.upper() == "NSE") & (df[segment_col].astype(str).str.upper() == "E")]
    m = m[m[symbol_col].astype(str).str.upper() == root]
    m = m[m[inst_col].astype(str).str.upper().str.contains("EQUITY", na=False)]
    if m.empty:
        # Some master versions identify regular equities simply as EQ.
        m = df[(df[seg_col].astype(str).str.upper() == "NSE") & (df[symbol_col].astype(str).str.upper() == root)]
        m = m[m[inst_col].astype(str).str.upper().isin({"EQUITY", "EQ"})]
    if m.empty:
        raise KeyError(f"Dhan NSE equity instrument not found: {root}")
    row = m.iloc[0]
    return {
        "symbol": root,
        "security_id": str(row[sec_col]),
        "exchange_segment": "NSE_EQ",
        "instrument": "EQUITY",
        "raw": row.to_dict(),
    }


def _post_json(config: DhanConfig, path: str, payload: dict[str, Any], use_client_id: bool = False) -> dict[str, Any]:
    token = get_access_token(config)
    if not token:
        raise RuntimeError("No valid Dhan access token. Run Dhan login first.")
    response = requests.post(
        f"{API_BASE}{path}",
        headers=_headers(token, config.client_id if use_client_id else None),
        json=payload,
        timeout=config.timeout_seconds,
    )
    return _response_json(response)


def _epoch_frame(data: dict[str, Any], tz: str = "Asia/Kolkata") -> pd.DataFrame:
    if "data" in data and isinstance(data["data"], dict):
        data = data["data"]
    timestamps = data.get("timestamp") or data.get("timestamps") or []
    fields = {}
    for name in ("open", "high", "low", "close", "volume", "oi"):
        if name in data:
            fields[name.title() if name != "oi" else "OI"] = data[name]
    n = len(timestamps)
    if n == 0:
        return pd.DataFrame()
    for k, vals in list(fields.items()):
        if len(vals) != n:
            fields.pop(k)
    idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(tz)
    df = pd.DataFrame(fields, index=idx)
    df.index.name = "Datetime"
    return df.sort_index()


def historical_daily(config: DhanConfig, security_id: str, from_date: str, to_date: str, oi: bool = False) -> pd.DataFrame:
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "expiryCode": 0,
        "oi": bool(oi),
        "fromDate": from_date,
        "toDate": to_date,
    }
    return _epoch_frame(_post_json(config, "/charts/historical", payload))


def historical_intraday(config: DhanConfig, security_id: str, interval: int, from_dt: str, to_dt: str, instrument: str = "EQUITY", exchange_segment: str = "NSE_EQ", oi: bool = False) -> pd.DataFrame:
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": exchange_segment,
        "instrument": instrument,
        "interval": str(interval),
        "oi": bool(oi),
        "fromDate": from_dt,
        "toDate": to_dt,
    }
    return _epoch_frame(_post_json(config, "/charts/intraday", payload))


def historical_intraday_range(config: DhanConfig, security_id: str, interval: int, start: datetime, end: datetime, max_days: int = 90, instrument: str = "EQUITY", exchange_segment: str = "NSE_EQ", oi: bool = False, pause_seconds: float = 0.25) -> pd.DataFrame:
    """Fetch a long range by respecting Dhan's documented 90-day request limit."""
    pieces: list[pd.DataFrame] = []
    cur = start
    while cur < end:
        chunk_end = min(end, cur + timedelta(days=max_days))
        df = historical_intraday(
            config,
            security_id=security_id,
            interval=interval,
            from_dt=cur.strftime("%Y-%m-%d %H:%M:%S"),
            to_dt=chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
            instrument=instrument,
            exchange_segment=exchange_segment,
            oi=oi,
        )
        if not df.empty:
            pieces.append(df)
        cur = chunk_end
        if cur < end:
            time.sleep(pause_seconds)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces).sort_index()
    return out[~out.index.duplicated(keep="last")]


def ltp(config: DhanConfig, security_ids: list[str]) -> dict[str, float]:
    payload = {"NSE_EQ": [int(x) if str(x).isdigit() else x for x in security_ids]}
    data = _post_json(config, "/marketfeed/ltp", payload, use_client_id=True)
    result = {}
    for sid, obj in (data.get("data", {}).get("NSE_EQ", {}) or {}).items():
        if isinstance(obj, dict) and obj.get("last_price") is not None:
            result[str(sid)] = float(obj["last_price"])
    return result


def quote(config: DhanConfig, security_ids: list[str]) -> dict[str, Any]:
    payload = {"NSE_EQ": [int(x) if str(x).isdigit() else x for x in security_ids]}
    return _post_json(config, "/marketfeed/quote", payload, use_client_id=True)


def rolling_expired_option(config: DhanConfig, security_id: str, expiry_code: int, strike: str = "ATM", option_type: str = "CALL", from_date: str = "2024-01-01", to_date: str = "2024-01-31", instrument: str = "OPTSTK", exchange_segment: str = "NSE_FNO", expiry_flag: str = "MONTH", interval: int = 5) -> pd.DataFrame:
    payload = {
        "exchangeSegment": exchange_segment,
        "interval": str(interval),
        "securityId": str(security_id),
        "instrument": instrument,
        "expiryFlag": expiry_flag,
        "expiryCode": int(expiry_code),
        "strike": strike,
        "drvOptionType": option_type,
        "requiredData": ["open", "high", "low", "close", "iv", "volume", "strike", "oi", "spot"],
        "fromDate": from_date,
        "toDate": to_date,
    }
    data = _post_json(config, "/charts/rollingoption", payload)
    # Response has ce/pe structures. Select requested option type when available.
    d = data.get("data", {}) if isinstance(data, dict) else {}
    key = "ce" if option_type.upper() == "CALL" else "pe"
    return _epoch_frame(d.get(key, {}) or {})


def status(config: DhanConfig) -> dict[str, Any]:
    out = {
        "configured": config.has_any_credential(),
        "key_secret_configured": config.has_api_key_flow(),
        "access_token_loaded": bool(get_access_token(config)),
        "cached_token_valid": cached_token_valid(),
        "profile": None,
        "error": None,
    }
    if out["access_token_loaded"] and config.client_id:
        try:
            out["profile"] = profile(config)
        except Exception as exc:
            out["error"] = str(exc)
    return out


def save_env_template() -> Path:
    path = BASE_DIR / ".env.example"
    path.write_text(
        "# DhanHQ local credentials. DO NOT COMMIT .env.\n"
        "DHAN_CLIENT_ID=\n"
        "DHAN_API_KEY=\n"
        "DHAN_API_SECRET=\n"
        "# Leave blank when using the local OAuth callback.\n"
        "DHAN_REDIRECT_URL=http://127.0.0.1:8765/callback\n"
        "# Optional: temporary 24-hour token instead of OAuth cache.\n"
        "DHAN_ACCESS_TOKEN=\n"
        "DHAN_TIMEOUT_SECONDS=30\n",
        encoding="utf-8",
    )
    return path
