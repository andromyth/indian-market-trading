import ssl
import sys
import types

import pytest

import persistent_store


def test_drive_execute_retries_transient_ssl(monkeypatch):
    calls = {"count": 0}
    monkeypatch.setattr(persistent_store.time, "sleep", lambda _: None)
    monkeypatch.setattr(persistent_store.random, "uniform", lambda _a, _b: 0)

    def request():
        calls["count"] += 1
        if calls["count"] == 1:
            raise ssl.SSLError("UNEXPECTED_EOF_WHILE_READING")
        return {"ok": True}

    assert persistent_store._drive_execute_with_retry("test_metadata", request) == {"ok": True}
    assert calls["count"] == 2


def test_drive_execute_does_not_retry_permanent_error(monkeypatch):
    calls = {"count": 0}
    monkeypatch.setattr(persistent_store.time, "sleep", lambda _: None)

    def request():
        calls["count"] += 1
        raise ValueError("invalid request")

    with pytest.raises(ValueError, match="invalid request"):
        persistent_store._drive_execute_with_retry("test_metadata", request)
    assert calls["count"] == 1


def test_drive_download_restarts_after_ssl_eof_and_replaces_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(persistent_store.time, "sleep", lambda _: None)
    monkeypatch.setattr(persistent_store.random, "uniform", lambda _a, _b: 0)
    package = types.ModuleType("googleapiclient")
    package.__path__ = []
    http = types.ModuleType("googleapiclient.http")

    class FakeDownloader:
        instances = 0

        def __init__(self, output, request, chunksize):
            self.output = output
            FakeDownloader.instances += 1
            self.instance = FakeDownloader.instances

        def next_chunk(self):
            if self.instance == 1:
                raise ssl.SSLError("UNEXPECTED_EOF_WHILE_READING")
            self.output.write(b"complete-download")
            return None, True

    http.MediaIoBaseDownload = FakeDownloader
    monkeypatch.setitem(sys.modules, "googleapiclient", package)
    monkeypatch.setitem(sys.modules, "googleapiclient.http", http)

    class Files:
        def get_media(self, fileId):
            return {"file_id": fileId}

    class Service:
        def files(self):
            return Files()

    destination = tmp_path / "checkpoint.parquet"
    destination.write_bytes(b"previous-valid-file")

    result = persistent_store.download_drive_file_with_retry(
        Service(), "drive-file-id", destination, operation="test_checkpoint"
    )

    assert result == destination
    assert destination.read_bytes() == b"complete-download"
    assert FakeDownloader.instances == 2
    assert list(tmp_path.glob("*.download")) == []
