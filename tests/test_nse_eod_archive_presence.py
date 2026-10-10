from datetime import date

import pytest

import nse_bulk_eod_ingest
import nse_data_layer


FOLDER_MIME = "application/vnd.google-apps.folder"


class _Request:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class _Files:
    def __init__(self, children):
        self.children = children

    def list(self, **kwargs):
        # The ingestion discovery query is always "'<parent-id>' in parents".
        parent_id = kwargs["q"].split("'", 2)[1]
        return _Request({"files": self.children.get(parent_id, [])})


class _Service:
    def __init__(self, children):
        self._files = _Files(children)

    def files(self):
        return self._files


class _Store:
    def __init__(self, children):
        self.service = _Service(children)


def _folder(name, folder_id):
    return {"name": name, "id": folder_id, "mimeType": FOLDER_MIME}


def _file(name, file_id):
    return {"name": name, "id": file_id, "mimeType": "application/octet-stream"}


def test_archive_discovery_requires_real_canonical_parquet_inside_date_folder():
    children = {
        "root": [_folder("2026", "year")],
        "year": [_folder("10", "month")],
        "month": [
            _folder("06", "day06"),
            _folder("07", "day07"),
            _folder("08", "day08"),
            _folder("09", "day09"),
        ],
        "day06": [_file("BhavCopy_NSE_CM_0_0_0_20261006_F_0000.parquet", "p06")],
        # A ZIP-only date, an empty date folder, and unrelated files must not count.
        "day07": [_file("BhavCopy_NSE_CM_0_0_0_20261007_F_0000.csv.zip", "z07")],
        "day08": [],
        "day09": [
            _file("README.txt", "readme"),
            _file("BhavCopy_NSE_CM_0_0_0_20261009_F_0000.parquet", "p09"),
        ],
    }

    found = nse_bulk_eod_ingest._collect_dates_under_folder(
        _Store(children),
        "root",
        label="test",
    )

    assert found == {date(2026, 10, 6), date(2026, 10, 9)}


def test_eod_sync_raises_if_required_drive_parquet_upload_fails(monkeypatch, tmp_path):
    day = date(2026, 10, 7)
    parquet_path = tmp_path / "BhavCopy_NSE_CM_0_0_0_20261007_F_0000.parquet"
    zip_path = tmp_path / "BhavCopy_NSE_CM_0_0_0_20261007_F_0000.csv.zip"
    parquet_path.write_bytes(b"parquet-placeholder")
    calls = []

    def fake_upload(path, remote_path, mime):
        calls.append((path, remote_path, mime))
        if str(path).endswith(".parquet"):
            return {"status": "NOT_SYNCED", "error": "simulated upload failure"}
        return {"status": "OK", "remote_path": remote_path}

    monkeypatch.setattr(nse_data_layer, "_drive_upload", fake_upload)
    with pytest.raises(RuntimeError, match="Required NSE EOD parquet was not persisted"):
        nse_data_layer._sync_bulk_eod_to_drive(day, parquet_path, zip_path)

    assert any(
        call[1].endswith("/BhavCopy_NSE_CM_0_0_0_20261007_F_0000.parquet")
        for call in calls
    )
