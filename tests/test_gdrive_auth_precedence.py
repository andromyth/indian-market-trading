from __future__ import annotations

import json

import pytest

from persistent_store import gdrive_config


def test_service_account_preferred_when_legacy_oauth_token_also_present(monkeypatch):
    service_info = {
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "test-key-id",
        "private_key": "test-private-key",
        "client_email": "market-data-bot@example.iam.gserviceaccount.com",
        "client_id": "123456789",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    oauth_info = {
        "type": "authorized_user",
        "client_id": "legacy-client-id",
        "client_secret": "legacy-client-secret",
        "refresh_token": "legacy-or-expired-refresh-token",
    }
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", json.dumps(service_info))
    monkeypatch.setenv("GOOGLE_OAUTH_TOKEN_JSON", json.dumps(oauth_info))
    monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "test-root-folder-id")
    monkeypatch.delenv("GDRIVE_ROOT_FOLDER_ID", raising=False)
    monkeypatch.delenv("GOOGLE_DRIVE_ROOT_FOLDER_ID", raising=False)
    monkeypatch.delenv("DRIVE_ROOT_FOLDER_ID", raising=False)

    cfg = gdrive_config()

    assert cfg["configured"] is True
    assert cfg["auth_method"] == "service_account"
    assert cfg["client_email"] == service_info["client_email"]
    assert cfg["root_folder_id"] == "test-root-folder-id"
    assert cfg["oauth_write_token_info"]["refresh_token"] == oauth_info["refresh_token"]
    assert cfg["oauth_write_configured"] is True


def test_oauth_remains_supported_when_no_service_account_is_configured(monkeypatch):
    oauth_info = {
        "type": "authorized_user",
        "client_id": "oauth-client-id",
        "client_secret": "oauth-client-secret",
        "refresh_token": "oauth-refresh-token",
    }
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.setenv("GOOGLE_OAUTH_TOKEN_JSON", json.dumps(oauth_info))
    monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "test-root-folder-id")

    cfg = gdrive_config()

    assert cfg["configured"] is True
    assert cfg["auth_method"] == "oauth"
    assert cfg["oauth_token_info"]["refresh_token"] == "oauth-refresh-token"


def test_service_account_writer_rejects_invalid_user_oauth_token(monkeypatch):
    import persistent_store

    store = persistent_store.GoogleDriveStore.__new__(persistent_store.GoogleDriveStore)
    store.cfg = {
        "auth_method": "service_account",
        "oauth_write_token_info": {"refresh_token": "test-refresh-token"},
    }
    store._write_service = None

    def rejected_oauth(_cfg):
        raise ValueError("invalid_grant")

    monkeypatch.setattr(persistent_store, "_drive_credentials", rejected_oauth)
    with pytest.raises(RuntimeError, match="OAuth user token required to upload"):
        store._get_write_service()


def test_invalid_oauth_write_json_is_reported_without_exposing_secret(monkeypatch):
    import persistent_store

    monkeypatch.setenv(
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        json.dumps({
            "type": "service_account",
            "client_email": "market-data-bot@example.iam.gserviceaccount.com",
        }),
    )
    monkeypatch.setenv("GOOGLE_OAUTH_TOKEN_JSON", json.dumps({
        "type": "authorized_user",
        "client_id": "test-client-id",
        "client_secret": "do-not-print-this",
        # Intentionally missing refresh_token.
    }))
    monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "test-root-folder-id")
    monkeypatch.delenv("GDRIVE_ROOT_FOLDER_ID", raising=False)
    monkeypatch.delenv("GOOGLE_DRIVE_ROOT_FOLDER_ID", raising=False)
    monkeypatch.delenv("DRIVE_ROOT_FOLDER_ID", raising=False)

    cfg = persistent_store.gdrive_config()

    assert cfg["auth_method"] == "service_account"
    assert cfg["oauth_write_configured"] is True
    assert cfg["oauth_write_token_info"] is None
    assert cfg["oauth_write_error"] == "OAuth token JSON has no refresh_token"
    assert "do-not-print-this" not in str(cfg)
