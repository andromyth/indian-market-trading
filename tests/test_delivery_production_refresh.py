from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from delivery_production_refresh import _download_file, _walk


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


def test_archive_walk_retries_transient_drive_list_http_500():
    class _TransientDriveError(Exception):
        def __init__(self):
            super().__init__("Google Drive Internal Error")
            self.resp = SimpleNamespace(status=500)

    class _ListRequest:
        def __init__(self, calls):
            self.calls = calls

        def execute(self):
            self.calls["count"] += 1
            if self.calls["count"] == 1:
                raise _TransientDriveError()
            return {"files": []}

    class _DriveFiles:
        def __init__(self, calls):
            self.calls = calls

        def list(self, **kwargs):
            return _ListRequest(self.calls)

    class _DriveService:
        def __init__(self, calls):
            self.calls = calls

        def files(self):
            return _DriveFiles(self.calls)

    class _Store:
        def __init__(self, calls):
            self.service = _DriveService(calls)

    calls = {"count": 0}
    out = []
    with patch("delivery_production_refresh.time.sleep") as sleep:
        _walk(
            _Store(calls),
            "bulk-eod-folder",
            "production_data/nse_official/bulk_eod",
            1,
            out,
        )

    assert calls["count"] == 2
    assert out == []
    sleep.assert_called_once_with(1)


def test_fundamental_refresh_uses_explicit_target_session_and_persists_complete_index(tmp_path):
    from datetime import date
    from delivery_production_refresh import _refresh_fundamental_availability

    class Store:
        def __init__(self):
            self.uploads = []

        def upload_or_update(self, path, remote, mime_type=None):
            self.uploads.append((str(path), remote, mime_type))

    store = Store()
    with (
        patch(
            "colab_build_all_nse_delivery_multifactor_candidate._catalog_all",
            return_value=[{"symbol": "AAA"}],
        ) as catalog,
        patch(
            "colab_build_all_nse_delivery_multifactor_candidate._build_fundamental_availability",
            return_value={
                "coverage_start": "2025-01-01",
                "coverage_end": "2026-10-30",
                "coverage_complete": True,
                "rows": 1,
            },
        ),
        patch(
            "colab_build_all_nse_delivery_multifactor_candidate.FUND_CATALOG_CACHE",
            tmp_path / "catalog.json",
        ),
        patch(
            "colab_build_all_nse_delivery_multifactor_candidate.FUND_AVAILABILITY",
            tmp_path / "availability.parquet",
        ),
        patch(
            "colab_build_all_nse_delivery_multifactor_candidate.FUND_AVAILABILITY_META",
            tmp_path / "availability.meta.json",
        ),
    ):
        report = _refresh_fundamental_availability(
            store, ["AAA"], as_of_date=date(2026, 10, 30)
        )

    catalog.assert_called_once_with(
        force_refresh=False, as_of_date=date(2026, 10, 30)
    )
    assert report["current_target_covered"] is True
    assert report["fallback_used"] is False
    assert len(store.uploads) == 3


def test_fundamental_refresh_timeout_fallback_keeps_stale_coverage_unextended(tmp_path):
    import json
    from datetime import date
    from delivery_production_refresh import _refresh_fundamental_availability

    availability = tmp_path / "availability.parquet"
    availability.write_bytes(b"cached-index")
    availability_meta = tmp_path / "availability.meta.json"
    availability_meta.write_text(
        json.dumps({
            "coverage_start": "2025-01-01",
            "coverage_end": "2026-10-10",
            "coverage_complete": True,
            "rows": 123,
        }),
        encoding="utf-8",
    )

    class Store:
        def __init__(self):
            self.uploads = []

        def upload_or_update(self, *args, **kwargs):
            self.uploads.append((args, kwargs))

    store = Store()
    with (
        patch(
            "colab_build_all_nse_delivery_multifactor_candidate._catalog_all",
            side_effect=RuntimeError("NSE ReadTimeout"),
        ),
        patch(
            "colab_build_all_nse_delivery_multifactor_candidate.FUND_CATALOG_CACHE",
            tmp_path / "catalog.json",
        ),
        patch(
            "colab_build_all_nse_delivery_multifactor_candidate.FUND_AVAILABILITY",
            availability,
        ),
        patch(
            "colab_build_all_nse_delivery_multifactor_candidate.FUND_AVAILABILITY_META",
            availability_meta,
        ),
    ):
        report = _refresh_fundamental_availability(
            store, ["AAA"], as_of_date=date(2026, 10, 30),
            allow_stale_fallback=True,
        )

    assert report["fallback_used"] is True
    assert report["current_target_covered"] is False
    assert report["coverage_end"] == "2026-10-10"
    assert store.uploads == []
    assert json.loads(availability_meta.read_text(encoding="utf-8"))["coverage_end"] == "2026-10-10"


def test_fundamental_refresh_still_fails_closed_without_explicit_fallback():
    from datetime import date
    import pytest
    from delivery_production_refresh import _refresh_fundamental_availability

    with patch(
        "colab_build_all_nse_delivery_multifactor_candidate._catalog_all",
        side_effect=RuntimeError("NSE ReadTimeout"),
    ):
        with pytest.raises(RuntimeError, match="NSE ReadTimeout"):
            _refresh_fundamental_availability(
                object(), ["AAA"], as_of_date=date(2026, 10, 30)
            )
