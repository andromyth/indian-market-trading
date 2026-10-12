from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import requests

from refresh_canonical_cash_master_from_bulk_eod import _download_drive_archives


class _FakeRequest:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


class _FakeFiles:
    def __init__(self, items):
        self.items = items

    def list(self, **kwargs):
        return _FakeRequest({"files": self.items})

    def get_media(self, **kwargs):
        raise AssertionError("Drive media download should not be used for a missing ZIP fallback test")


class _FakeService:
    def __init__(self, items):
        self._files = _FakeFiles(items)

    def files(self):
        return self._files


class _FakeStore:
    def __init__(self, items):
        self.service = _FakeService(items)


class _FakeResponse:
    def __init__(self, content: bytes = b"", error: Exception | None = None):
        self.content = content
        self.error = error
        self.status_code = 200 if error is None else getattr(error, "status_code", 404)

    def raise_for_status(self):
        if self.error:
            raise self.error


def _valid_source_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "BhavCopy_NSE_CM_0_0_0_20261009.csv",
            "TradDt,TckrSymb,SctySrs,ISIN,FinInstrmTp,ClsPric\n"
            "2026-10-09,AAA,EQ,INE000A01000,STK,100.0\n",
        )
    return buffer.getvalue()


class MissingOfficialZipRecoveryTests(unittest.TestCase):
    def test_recovers_original_zip_when_drive_only_has_parquet(self):
        store = _FakeStore([
            {"id": "parquet-id", "name": "BhavCopy_NSE_CM_0_0_0_20261009_F_0000.parquet", "size": "100"}
        ])
        with tempfile.TemporaryDirectory(prefix="bulk-eod-zip-test-") as root:
            with patch("requests.get", return_value=_FakeResponse(_valid_source_zip())) as get:
                result = _download_drive_archives(
                    store,
                    start=__import__("datetime").date(2026, 10, 1),
                    end=__import__("datetime").date(2026, 10, 12),
                    destination=Path(root),
                )
            expected = Path(root) / "2026" / "10" / "09" / "BhavCopy_NSE_CM_0_0_0_20261009_F_0000.csv.zip"
            self.assertTrue(expected.exists())
            self.assertEqual(result["downloaded_directly_from_official_nse"], ["2026-10-09"])
            self.assertEqual(result["missing_source_zip_dates"], [])
            self.assertEqual(get.call_count, 1)

    def test_reports_unrecoverable_source_zip_instead_of_silent_skip(self):
        store = _FakeStore([
            {"id": "parquet-id", "name": "BhavCopy_NSE_CM_0_0_0_20261009_F_0000.parquet", "size": "100"}
        ])
        error = requests.HTTPError("404 archive unavailable")
        with tempfile.TemporaryDirectory(prefix="bulk-eod-zip-test-") as root:
            with patch("requests.get", return_value=_FakeResponse(error=error)):
                result = _download_drive_archives(
                    store,
                    start=__import__("datetime").date(2026, 10, 1),
                    end=__import__("datetime").date(2026, 10, 12),
                    destination=Path(root),
                )
        self.assertEqual(len(result["missing_source_zip_dates"]), 1)
        self.assertEqual(result["missing_source_zip_dates"][0]["date"], "2026-10-09")
        self.assertFalse(
            (Path(root) / "2026" / "10" / "09" / "BhavCopy_NSE_CM_0_0_0_20261009_F_0000.csv.zip").exists()
        )


if __name__ == "__main__":
    unittest.main()
