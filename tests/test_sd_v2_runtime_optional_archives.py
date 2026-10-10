from __future__ import annotations

import pandas as pd

from selfdriving import sd_v2_runtime as runtime


def test_optional_parquet_falls_back_from_beta_v2_to_legacy_archive(monkeypatch, tmp_path):
    local = tmp_path / "global_cues.parquet"
    calls: list[str] = []
    expected = pd.DataFrame(
        {
            "key": ["NIFTY"],
            "date": ["2026-10-09"],
            "value": [25_000.0],
        }
    )

    def fake_download(remote: str, destination):
        calls.append(remote)
        if remote == "selfdriving/beta_version2/global_cues.parquet":
            raise FileNotFoundError(remote)
        if remote == "selfdriving/global_cues.parquet":
            expected.to_parquet(destination, index=False)
            return
        raise AssertionError(f"Unexpected remote path: {remote}")

    monkeypatch.setattr(runtime, "_download", fake_download)
    frame, source = runtime._load_optional_parquet(
        (
            "selfdriving/beta_version2/global_cues.parquet",
            "selfdriving/global_cues.parquet",
        ),
        local,
    )

    assert calls == [
        "selfdriving/beta_version2/global_cues.parquet",
        "selfdriving/global_cues.parquet",
    ]
    assert source == "selfdriving/global_cues.parquet"
    assert len(frame) == 1
    assert frame.loc[0, "key"] == "NIFTY"


def test_optional_parquet_marks_missing_archive_as_no_data(monkeypatch, tmp_path):
    local = tmp_path / "raw_news.parquet"

    def missing(remote: str, destination):
        raise FileNotFoundError(remote)

    monkeypatch.setattr(runtime, "_download", missing)
    frame, source = runtime._load_optional_parquet(
        (
            "selfdriving/beta_version2/raw_news.parquet",
            "selfdriving/raw_news.parquet",
        ),
        local,
    )

    assert frame.empty
    assert source == "MISSING"
    assert not local.exists()


def test_optional_parquet_does_not_hide_drive_transport_errors(monkeypatch, tmp_path):
    local = tmp_path / "global_cues.parquet"

    def unavailable(remote: str, destination):
        raise RuntimeError("Drive permission or transport failure")

    monkeypatch.setattr(runtime, "_download", unavailable)
    try:
        runtime._load_optional_parquet(
            ("selfdriving/beta_version2/global_cues.parquet",),
            local,
        )
    except RuntimeError as exc:
        assert "Drive permission or transport failure" in str(exc)
    else:
        raise AssertionError("Transport failures must not be silently treated as missing optional data")
