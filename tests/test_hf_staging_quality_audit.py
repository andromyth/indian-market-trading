import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hf_staging_quality_audit import audit_staging


def test_audit_reports_oi_duplicates_and_session_quality(tmp_path):
    labels = pd.date_range("2026-11-09 09:20", "2026-11-09 15:30",
                           freq="5min", tz="Asia/Kolkata").tz_convert("UTC")
    frame = pd.DataFrame({
        "timestamp": labels,
        "open": [100.0] * 75, "high": [101.0] * 75,
        "low": [99.0] * 75, "close": [100.5] * 75,
        "volume": [10] * 75, "oi": [500] * 75,
    })
    path = tmp_path / "symbol=RELIANCE" / "RELIANCE.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)

    report = audit_staging(tmp_path)
    assert report["staged_symbols"] == 1
    assert report["complete_symbol_sessions"] == 1
    assert report["oi_non_null_pct"] == 100.0
    assert report["duplicate_timestamp_rows"] == 0
    assert report["quality_gate"]["passed"]


def test_audit_fails_when_oi_is_missing(tmp_path):
    frame = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-11-09 03:50:00+00:00")],
        "open": [100.0], "high": [101.0], "low": [99.0],
        "close": [100.5], "volume": [10],
    })
    path = tmp_path / "symbol=SBIN" / "SBIN.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)

    report = audit_staging(tmp_path)
    assert report["oi_column_missing_symbols"] == ["SBIN"]
    assert not report["quality_gate"]["passed"]
