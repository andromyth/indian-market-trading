from __future__ import annotations

import pandas as pd
import pytest

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
    monkeypatch.delenv("INDSTOCKS_AUTH_MODE", raising=False)
    monkeypatch.setenv("INDSTOCKS_ACCESS_TOKEN", "unit-test-token")
    monkeypatch.setattr(client, "_CACHED_TOKEN", None)
    assert client._auth_mode() == "manual"
    assert client.configured()
    assert client.access_token() == "unit-test-token"


def test_default_mode_never_mints_a_totp_token_when_credentials_are_present(monkeypatch):
    for name, value in (
        ("INDSTOCKS_CLIENT_ID", "test-client-id"),
        ("INDSTOCKS_MPIN", "test-mpin"),
        ("INDSTOCKS_TOTP_SECRET", "JBSWY3DPEHPK3PXP"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("INDSTOCKS_ACCESS_TOKEN", "streamlit-manual-token")
    monkeypatch.delenv("INDSTOCKS_AUTH_MODE", raising=False)
    monkeypatch.setattr(client, "_CACHED_TOKEN", None)
    monkeypatch.setattr(
        client, "_generate_token",
        lambda: pytest.fail("Default/manual mode must never mint a TOTP token"),
    )

    assert client._auth_mode() == "manual"
    assert client.configured()
    assert not client._has_totp_credentials()
    assert client.access_token() == "streamlit-manual-token"


def test_totp_token_generation_requires_explicit_mode_even_with_manual_token(monkeypatch):
    for name, value in (
        ("INDSTOCKS_CLIENT_ID", "test-client-id"),
        ("INDSTOCKS_MPIN", "test-mpin"),
        ("INDSTOCKS_TOTP_SECRET", "JBSWY3DPEHPK3PXP"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("INDSTOCKS_ACCESS_TOKEN", "streamlit-manual-token")
    monkeypatch.setenv("INDSTOCKS_AUTH_MODE", "totp")
    monkeypatch.setattr(client, "_CACHED_TOKEN", None)
    monkeypatch.setattr(client, "_CACHED_TOKEN_EXPIRES_AT", 0.0)
    monkeypatch.setattr(client, "_TOKEN_GENERATION_RETRY_AFTER", 0.0)
    generated = []
    monkeypatch.setattr(
        client, "_generate_token",
        lambda: generated.append(True) or "cloud-run-generated-token",
    )

    assert client.configured()
    assert client.access_token() == "cloud-run-generated-token"
    assert generated == [True]


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



def test_rejected_cached_token_refreshes_once_with_totp_credentials(monkeypatch):
    for name, value in (
        ("INDSTOCKS_CLIENT_ID", "test-client-id"),
        ("INDSTOCKS_MPIN", "test-mpin"),
        ("INDSTOCKS_TOTP_SECRET", "JBSWY3DPEHPK3PXP"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("INDSTOCKS_AUTH_MODE", "totp")
    monkeypatch.delenv("INDSTOCKS_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(client, "_CACHED_TOKEN", "expired-token")
    monkeypatch.setattr(client, "_CACHED_TOKEN_EXPIRES_AT", client.time.time() + 20 * 3600)

    generated = []
    monkeypatch.setattr(
        client,
        "_generate_token",
        lambda: generated.append("generated") or "refreshed-token",
    )
    sent_tokens = []

    class FakeResponse:
        def __init__(self, status_code, body):
            self.status_code = status_code
            self._body = body

        def json(self):
            return self._body

    def fake_request(method, url, *, headers, params, timeout):
        sent_tokens.append(headers["Authorization"])
        if headers["Authorization"] == "expired-token":
            return FakeResponse(403, {
                "success": False,
                "message": (
                    "The provided access_token is either incorrect, expired, "
                    "or has been revoked. The user needs to re-authenticate."
                ),
            })
        return FakeResponse(200, {"success": True, "data": {"ok": True}})

    monkeypatch.setattr(client.requests, "request", fake_request)
    response = client._request("GET", "/market/quotes/full", params={"scrip-codes": "NSE_2885"})

    assert response.status_code == 200
    assert sent_tokens == ["expired-token", "refreshed-token"]
    assert generated == ["generated"]
    assert client._CACHED_TOKEN == "refreshed-token"



def test_recently_generated_rejected_token_waits_for_cooldown(monkeypatch):
    for name, value in (
        ("INDSTOCKS_CLIENT_ID", "test-client-id"),
        ("INDSTOCKS_MPIN", "test-mpin"),
        ("INDSTOCKS_TOTP_SECRET", "JBSWY3DPEHPK3PXP"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("INDSTOCKS_AUTH_MODE", "totp")
    monkeypatch.delenv("INDSTOCKS_ACCESS_TOKEN", raising=False)
    generated_at = client.time.time()
    monkeypatch.setattr(client, "_CACHED_TOKEN", "fresh-rejected-token")
    monkeypatch.setattr(client, "_CACHED_TOKEN_EXPIRES_AT", generated_at + 20 * 3600)
    monkeypatch.setattr(client, "_CACHED_TOKEN_GENERATED_AT", generated_at)
    monkeypatch.setattr(client, "_CACHED_TOKEN_GENERATED_VALUE", "fresh-rejected-token")
    monkeypatch.setattr(client, "_TOKEN_GENERATION_RETRY_AFTER", 0.0)
    monkeypatch.setattr(client, "_TOKEN_GENERATION_LAST_ERROR", None)
    calls = []

    def forbidden_second_login():
        calls.append("generate")
        raise AssertionError("must not mint a second TOTP token inside the one-minute window")

    monkeypatch.setattr(client, "_generate_token", forbidden_second_login)
    with pytest.raises(RuntimeError, match="freshly generated INDstocks token was rejected"):
        client._refresh_rejected_access_token("fresh-rejected-token")
    with pytest.raises(RuntimeError, match="cooling down"):
        client.access_token()

    assert calls == []
    assert client._CACHED_TOKEN is None
    assert client._TOKEN_GENERATION_RETRY_AFTER >= generated_at + 60

def test_endpoint_permission_403_does_not_trigger_token_refresh(monkeypatch):
    for name, value in (
        ("INDSTOCKS_CLIENT_ID", "test-client-id"),
        ("INDSTOCKS_MPIN", "test-mpin"),
        ("INDSTOCKS_TOTP_SECRET", "JBSWY3DPEHPK3PXP"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("INDSTOCKS_AUTH_MODE", "totp")
    monkeypatch.delenv("INDSTOCKS_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(client, "_CACHED_TOKEN", "current-token")
    monkeypatch.setattr(client, "_CACHED_TOKEN_EXPIRES_AT", client.time.time() + 20 * 3600)
    monkeypatch.setattr(
        client,
        "_generate_token",
        lambda: (_ for _ in ()).throw(AssertionError("Must not refresh for endpoint permissions")),
    )

    class FakeResponse:
        status_code = 403

        def json(self):
            return {
                "success": False,
                "error_type": "UserException",
                "code": "MARKET_DATA_PERMISSION_DENIED",
                "message": "This account cannot access this quote endpoint",
            }

    sent_tokens = []
    monkeypatch.setattr(
        client.requests,
        "request",
        lambda *args, **kwargs: (sent_tokens.append(kwargs["headers"]["Authorization"]) or FakeResponse()),
    )

    with pytest.raises(RuntimeError, match="MARKET_DATA_PERMISSION_DENIED"):
        client._request("GET", "/market/quotes/full", params={"scrip-codes": "NSE_2885"})
    assert sent_tokens == ["current-token"]


def test_token_generation_failure_starts_cooldown_and_prevents_repeated_calls(monkeypatch):
    for name, value in (
        ("INDSTOCKS_CLIENT_ID", "test-client-id"),
        ("INDSTOCKS_MPIN", "test-mpin"),
        ("INDSTOCKS_TOTP_SECRET", "JBSWY3DPEHPK3PXP"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("INDSTOCKS_AUTH_MODE", "totp")
    monkeypatch.delenv("INDSTOCKS_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(client, "_CACHED_TOKEN", "stale-expired-token")
    monkeypatch.setattr(client, "_CACHED_TOKEN_EXPIRES_AT", 0.0)
    monkeypatch.setattr(client, "_CACHED_TOKEN_GENERATED_AT", client.time.time() - 7200)
    monkeypatch.setattr(client, "_CACHED_TOKEN_GENERATED_VALUE", "stale-expired-token")
    monkeypatch.setattr(client, "_TOKEN_GENERATION_RETRY_AFTER", 0.0)
    monkeypatch.setattr(client, "_TOKEN_GENERATION_LAST_ERROR", None)
    calls = []

    def fail_rate_limited():
        calls.append("attempt")
        raise RuntimeError("token generation rate-limited; wait before retrying")

    monkeypatch.setattr(client, "_generate_token", fail_rate_limited)
    with pytest.raises(RuntimeError, match="suppressed for 60s"):
        client.access_token()
    with pytest.raises(RuntimeError, match="cooling down"):
        client.access_token()

    assert calls == ["attempt"]
    assert client._TOKEN_GENERATION_RETRY_AFTER > client.time.time()
    assert client._CACHED_TOKEN is None
    assert client._CACHED_TOKEN_EXPIRES_AT == 0.0
    assert client._CACHED_TOKEN_GENERATED_AT == 0.0
    assert client._CACHED_TOKEN_GENERATED_VALUE is None


def test_rejected_token_refresh_failure_does_not_spam_rate_limited_endpoint(monkeypatch):
    for name, value in (
        ("INDSTOCKS_CLIENT_ID", "test-client-id"),
        ("INDSTOCKS_MPIN", "test-mpin"),
        ("INDSTOCKS_TOTP_SECRET", "JBSWY3DPEHPK3PXP"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("INDSTOCKS_AUTH_MODE", "totp")
    monkeypatch.delenv("INDSTOCKS_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(client, "_CACHED_TOKEN", "expired-token")
    monkeypatch.setattr(client, "_CACHED_TOKEN_EXPIRES_AT", client.time.time() + 20 * 3600)
    monkeypatch.setattr(client, "_TOKEN_GENERATION_RETRY_AFTER", 0.0)
    monkeypatch.setattr(client, "_TOKEN_GENERATION_LAST_ERROR", None)
    calls = []

    def fail_rate_limited():
        calls.append("attempt")
        raise RuntimeError("token generation rate-limited; wait before retrying")

    monkeypatch.setattr(client, "_generate_token", fail_rate_limited)
    with pytest.raises(RuntimeError, match=r"suppressed for \d+s"):
        client._refresh_rejected_access_token("expired-token")
    with pytest.raises(RuntimeError, match="cooling down"):
        client._refresh_rejected_access_token("expired-token")

    assert calls == ["attempt"]
    assert client._TOKEN_GENERATION_RETRY_AFTER > client.time.time()
