from datetime import date

import pandas as pd

import nse_data_layer as nse


def _sample_index_frame():
    return pd.DataFrame(
        {
            "symbol": ["^NSEI", "^NSEI"],
            "date": pd.to_datetime(["2016-01-04", "2026-10-01"]),
            "open": [7900.0, 25000.0],
            "high": [8000.0, 25100.0],
            "low": [7800.0, 24900.0],
            "close": [7950.0, 25050.0],
            "volume": [0.0, 0.0],
            "source": ["NSE_OFFICIAL_INDEX_HISTORY"] * 2,
            "provenance": ["test fixture"] * 2,
        }
    )


def test_drive_upload_uses_mounted_research_root(tmp_path, monkeypatch):
    mounted_root = tmp_path / "Indian_Market_Research"
    mounted_root.mkdir()
    monkeypatch.setattr(nse, "_mounted_drive_root", lambda: mounted_root)

    source = tmp_path / "index-cache.parquet"
    source.write_bytes(b"official-index-cache")

    result = nse._drive_upload(
        source,
        "production_data/nse_official/indices/NSEI/20160101_20261001.parquet",
    )

    destination = (
        mounted_root / "production_data" / "nse_official" / "indices"
        / "NSEI" / "20160101_20261001.parquet"
    )
    assert result["status"] == "OK"
    assert result["mode"] == "mounted_drive_copy"
    assert destination.read_bytes() == source.read_bytes()
    assert not destination.with_name(destination.name + ".uploading").exists()


def test_drive_index_history_reads_complete_range_from_mounted_drive(tmp_path, monkeypatch):
    mounted_root = tmp_path / "Indian_Market_Research"
    cache_dir = (
        mounted_root / "production_data" / "nse_official" / "indices" / "NSEI"
    )
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / "20160101_20261001.parquet"
    _sample_index_frame().to_parquet(cache_file, index=False)
    monkeypatch.setattr(nse, "_mounted_drive_root", lambda: mounted_root)

    actual = nse._drive_index_history(
        "^NSEI",
        date(2016, 1, 1),
        date(2026, 10, 1),
    )

    assert not actual.empty
    assert actual["date"].min().date() == date(2016, 1, 4)
    assert actual["date"].max().date() == date(2026, 10, 1)
    assert actual["close"].tolist() == [7950.0, 25050.0]
