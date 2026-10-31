from __future__ import annotations

import base64

from dhan_client import DhanConfig, generate_totp_access_token, totp_code


def test_totp_matches_rfc_6238_sha1_six_digit_vector():
    secret = base64.b32encode(b"12345678901234567890").decode("ascii").rstrip("=")
    assert totp_code(secret, at_time=59) == "287082"


def test_totp_access_token_uses_dhan_endpoint_and_persists_expiry(monkeypatch):
    import dhan_client

    secret = base64.b32encode(b"12345678901234567890").decode("ascii").rstrip("=")
    config = DhanConfig(
        client_id="1000000001",
        pin="123456",
        totp_secret=secret,
    )
    saved = {}
    calls = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "accessToken": "fresh-token",
                "expiryTime": "2026-11-01T00:00:00",
            }

    def fake_post(url, params, timeout):
        calls.update({"url": url, "params": params, "timeout": timeout})
        return Response()

    monkeypatch.setattr(dhan_client.requests, "post", fake_post)
    monkeypatch.setattr(dhan_client, "save_token", lambda payload: saved.update(payload))
    monkeypatch.setattr(dhan_client, "totp_code", lambda secret: "123456")

    token = generate_totp_access_token(config)

    assert token == "fresh-token"
    assert config.access_token == "fresh-token"
    assert calls["url"].endswith("/app/generateAccessToken")
    assert calls["params"] == {
        "dhanClientId": "1000000001",
        "pin": "123456",
        "totp": "123456",
    }
    assert saved["accessToken"] == "fresh-token"
    assert saved["expiryTime"] == "2026-11-01T00:00:00"


def test_totp_flow_requires_all_three_credentials():
    assert DhanConfig(client_id="id", pin="pin").has_totp_flow() is False
    assert DhanConfig(client_id="id", pin="pin", totp_secret="secret").has_totp_flow() is True
