from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from delivery_production_refresh import _download_file


class _FakeDriveFiles:
    def get_media(self, fileId):
        return fileId


class _FakeDriveService:
    def files(self):
        return _FakeDriveFiles()


class _FakeDriveStore:
    root_id = "root"
    service = _FakeDriveService()

    def _find_child(self, parent, name):
        if (parent, name) == ("root", "folder"):
            return {"id": "folder-id"}
        if (parent, name) == ("folder-id", "data.parquet"):
            return {"id": "file-id"}
        return None


def test_drive_download_retries_transient_timeout_and_replaces_atomically(tmp_path):
    attempts = {"count": 0}

    class _FakeDownloader:
        def __init__(self, handle, request):
            self.handle = handle

        def next_chunk(self):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise TimeoutError("simulated Drive read timeout")
            self.handle.write(b"downloaded-payload")
            return None, True

    target = Path(tmp_path) / "features" / "data.parquet"
    with patch("googleapiclient.http.MediaIoBaseDownload", _FakeDownloader), patch(
        "delivery_production_refresh.time.sleep"
    ) as sleep:
        assert _download_file(_FakeDriveStore(), "folder/data.parquet", target)

    assert attempts["count"] == 2
    assert target.read_bytes() == b"downloaded-payload"
    assert not target.with_suffix(target.suffix + ".tmp").exists()
    sleep.assert_called_once_with(1)
