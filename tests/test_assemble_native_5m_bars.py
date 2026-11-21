from __future__ import annotations

import json

import pandas as pd

import assemble_native_5m_bars as assembler


def _write_symbol(root, symbol, sessions):
    path = root / f"symbol={symbol}" / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    index = pd.DatetimeIndex(
        [
            stamp
            for day in sessions
            for stamp in pd.date_range(f"{day} 09:20", f"{day} 15:30", freq="5min", tz=assembler.IST)
        ],
        name="Datetime",
    )
    pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000.0},
        index=index,
    ).to_parquet(path)


def test_assemble_writes_long_frame_and_manifest(tmp_path):
    archive = tmp_path / "archive"
    _write_symbol(archive, "AAA", ["2026-01-02", "2026-01-05"])
    _write_symbol(archive, "BBB", ["2026-01-02"])
    out = tmp_path / "out"

    rc = assembler.main(
        [
            "--archive-dir", str(archive),
            "--out-dir", str(out),
            "--sessions", "2",
            "--chunk-size", "10",
            "--clean",
        ]
    )

    assert rc == 0
    frame = pd.read_parquet(out / "parts")
    assert list(frame.columns) == ["symbol", "Timestamp", "Open", "High", "Low", "Close", "Volume"]
    assert set(frame["symbol"]) == {"AAA", "BBB"}
    assert len(frame) == 225
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["symbols_written"] == 2
    assert manifest["sessions"] == 2
    assert manifest["rows"] == 225
