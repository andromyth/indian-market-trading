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

def test_runtime_cache_is_reused_only_for_matching_drive_version(tmp_path):
    local = tmp_path / "global_cues.parquet"
    local.write_bytes(b"validated-parquet-bytes")
    metadata = {
        "id": "file-1",
        "name": "global_cues.parquet",
        "mimeType": "application/octet-stream",
        "size": str(local.stat().st_size),
        "modifiedTime": "2026-10-10T10:00:00.000Z",
        "md5Checksum": "checksum-1",
    }

    # Without a sidecar, existing local bytes cannot be assumed fresh.
    assert runtime._cache_matches_remote(local, metadata) is False

    runtime._cache_metadata_path(local).write_text(
        __import__("json").dumps(runtime._remote_signature(metadata)),
        encoding="utf-8",
    )
    assert runtime._cache_matches_remote(local, metadata) is True

    changed_version = {**metadata, "modifiedTime": "2026-10-10T10:05:00.000Z"}
    assert runtime._cache_matches_remote(local, changed_version) is False

    changed_size = {**metadata, "size": str(local.stat().st_size + 1)}
    assert runtime._cache_matches_remote(local, changed_size) is False

