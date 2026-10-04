from __future__ import annotations

import io
import zipfile
from pathlib import Path

from rebuild_nse_cash_history_identity_aware import read_zip_csv


def _zip_csv(text: str) -> Path:
    path = Path("/tmp/nse_identity_test.csv.zip")
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sample.csv", text)
    path.write_bytes(payload.getvalue())
    return path


def test_literal_na_series_is_preserved():
    path = _zip_csv(
        "SYMBOL,SERIES,ISIN,TIMESTAMP,CLOSE\n"
        "1035UCL26,NA,INE123456789,09-May-2024,99.5\n"
    )
    df = read_zip_csv(path)
    assert df.loc[0, "SERIES"] == "NA"


def test_blank_series_remains_blank_not_literal_nan():
    path = _zip_csv(
        "SYMBOL,SERIES,ISIN,TIMESTAMP,CLOSE\n"
        "X,,INE123456789,09-May-2024,99.5\n"
    )
    df = read_zip_csv(path)
    assert df.loc[0, "SERIES"] == ""
