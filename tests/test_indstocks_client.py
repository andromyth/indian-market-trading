from __future__ import annotations

import pandas as pd

import indstocks_client as client


def test_totp_matches_rfc6238_sha1_six_digit_vector():
    # RFC 6238 test secret, Unix time 59 seconds, SHA-1, six digits.
    assert client._totp("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ", at_time=59) == "287082"


def test_root_symbol_and_exchange_normalization():
    assert client._root_symbol("RELIANCE.NS") == ("RELIANCE", "NSE")
    assert client._root_symbol("TCS.BO") == ("TCS", "BSE")


def test_manual_access_token_is_read_without_network(monkeypatch):
    for name in ("INDSTOCKS_CLIENT_ID", "INDSTOCKS_MPIN", "INDSTOCKS_TOTP_SECRET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INDSTOCKS_ACCESS_TOKEN", "unit-test-token")
    monkeypatch.setattr(client, "_CACHED_TOKEN", None)
    assert client.configured()
    assert client.access_token() == "unit-test-token"


def test_resolve_scrip_code_prefers_ordinary_nse_equity(monkeypatch):
    frame = pd.DataFrame(
        [
            {"EXCH": "NSE", "TRADING_SYMBOL": "RELIANCE", "SECURITY_ID": "2885", "SERIES": "EQ", "SEGMENT": "E"},
            {"EXCH": "BSE", "TRADING_SYMBOL": "RELIANCE", "SECURITY_ID": "500325", "SERIES": "A", "SEGMENT": "E"},
        ]
    )
    monkeypatch.setattr(client, "_load_instruments", lambda exchange: frame)
    assert client.resolve_scrip_code("RELIANCE.NS") == "NSE_2885"


def test_historical_candles_normalize_to_ohlcv(monkeypatch):
    start = pd.Timestamp("2026-10-07 09:15:00", tz=client.IST)
    end = start + pd.Timedelta(days=1)
    epoch = int(start.timestamp())

    class FakeResponse:
        def json(self):
            return {
                "success": True,
                "data": {
                    "NSE_2885": {
                        "candles": [
                            {"ts": epoch, "o": 100, "h": 102, "l": 99, "c": 101, "v": 5000}
                        ]
                    }
                },
            }

    monkeypatch.setattr(client, "resolve_scrip_code", lambda symbol: "NSE_2885")
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: FakeResponse())
    frame = client.historical("RELIANCE.NS", start, end, "1m")

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(frame) == 1
    assert frame.iloc[0]["Close"] == 101
    assert str(frame.index.tz) == "Asia/Kolkata"

def test_request_403_preserves_sanitized_provider_error(monkeypatch):
    class FakeResponse:
        status_code = 403

        def json(self):
            return {
                "success": False,
                "error_type": "UserException",
                "code": "MARKET_DATA_PERMISSION_DENIED",
                "message": "This account cannot access this quote endpoint",
                "request_id": "req-123",
                "access_token": "must-not-be-included",
            }

    monkeypatch.setattr(client, "access_token", lambda: "unit-test-token")
    monkeypatch.setattr(client.requests, "request", lambda *args, **kwargs: FakeResponse())

    try:
        client._request("GET", "/market/quotes/full", params={"scrip-codes": "NSE_2885"})
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("HTTP 403 should raise RuntimeError")

    assert "HTTP 403" in message
    assert "UserException" in message
    assert "MARKET_DATA_PERMISSION_DENIED" in message
    assert "This account cannot access this quote endpoint" in message
    assert "req-123" in message
    assert "must-not-be-included" not in message
    assert "unit-test-token" not in message


def test_provider_error_detail_redacts_inline_credentials():
    class FakeResponse:
        def json(self):
            return {
                "error_type": "TokenException",
                "message": "Authorization: Bearer bearer-secret token=inline-secret MPIN=123456",
            }

    detail = client._safe_provider_error_detail(FakeResponse())
    assert "TokenException" in detail
    assert "Authorization: [REDACTED]" in detail
    assert "inline-secret" not in detail
    assert "123456" not in detail
    assert "bearer-secret" not in detail

