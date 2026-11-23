from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import retrain_to_close_job as job


def test_training_window_tiers_and_floor():
    assert job.training_window(500) == (150, 60)
    assert job.training_window(499) == (80, 40)
    assert job.training_window(200) == (80, 40)
    assert job.training_window(199) == (50, 30)
    assert job.training_window(120) == (50, 30)
    assert job.training_window(119) == (35, 20)
    assert job.training_window(80) == (35, 20)
    assert job.training_window(79) == (20, 12)
    assert job.training_window(45) == (20, 12)
    with pytest.raises(ValueError):
        job.training_window(44)


def test_archive_freshness_compares_sessions():
    assert job.archive_freshness("2026-11-23", "2026-11-23")[0] is True
    assert job.archive_freshness("2026-11-24", "2026-11-23")[0] is True
    assert job.archive_freshness("2026-11-20", "2026-11-23")[0] is False
    assert job.archive_freshness(None, "2026-11-23")[0] is False


def test_choose_universe_prefers_fno_liquidity_and_drops_index():
    fno = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "NIFTY", "CCC", "CCC", "CCC"],
        }
    )
    available = {name: name for name in ("AAA", "BBB", "CCC", "DDD")}
    chosen = job.choose_universe(available, fno, universe="fno", max_symbols=2)
    assert chosen == ["CCC", "AAA"]
    assert job.choose_universe(available, fno, universe="all", max_symbols=0) == [
        "AAA",
        "BBB",
        "CCC",
        "DDD",
    ]


def _write_local_inputs(root, symbols: tuple[str, ...], n_sessions: int = 200):
    archive = root / job.ARCHIVE_REL
    sessions = [day.strftime("%Y-%m-%d") for day in pd.bdate_range("2025-01-01", periods=n_sessions)]
    for symbol in symbols:
        frames = []
        for day in sessions:
            index = pd.date_range(f"{day} 09:20", f"{day} 15:30", freq="5min", tz="Asia/Kolkata")
            base = 100.0 + (hash(symbol) % 13)
            rng = np.random.default_rng(abs(hash((symbol, day))) % (2**32))
            values = base + np.cumsum(rng.normal(0.0, 0.25, len(index)))
            frames.append(
                pd.DataFrame(
                    {
                        "Open": values,
                        "High": values + 0.2,
                        "Low": values - 0.2,
                        "Close": values,
                        "Volume": 1000.0,
                    },
                    index=index,
                )
            )
        directory = archive / f"symbol={symbol}"
        directory.mkdir(parents=True, exist_ok=True)
        pd.concat(frames).to_parquet(directory / f"{symbol}.parquet")

    nifty_dir = archive / f"symbol={job.NIFTY_SYMBOL}"
    nifty_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for day in sessions:
        index = pd.date_range(f"{day} 09:20", f"{day} 15:30", freq="5min", tz="Asia/Kolkata")
        rng = np.random.default_rng(abs(hash(("NIFTY", day))) % (2**32))
        rows.append(
            pd.DataFrame(
                {
                    "Timestamp": index,
                    "Close": 25000.0 + np.cumsum(rng.normal(0.0, 0.5, len(index))),
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_parquet(nifty_dir / job.NIFTY_FILES[0], index=False)

    fno_dir = root / job.FNO_REL
    fno_dir.mkdir(parents=True, exist_ok=True)
    fno = pd.DataFrame(
        {
            "date": np.repeat(pd.to_datetime(sessions), len(symbols)),
            "symbol": np.tile(symbols, n_sessions),
            "fno_pcr": 1.0,
            "fno_oi_change": 0.01,
            "fno_fut_return": 0.001,
            "fno_fut_oi": 1000.0,
            "fno_fut_basis": 0.2,
        }
    )
    fno.to_parquet(fno_dir / job.FNO_FILES[0], index=False)
    return sessions


def test_local_mode_runs_full_pipeline(tmp_path):
    symbols = tuple(f"S{i:02d}" for i in range(20))
    _write_local_inputs(tmp_path / "drive", symbols)
    out = tmp_path / "out"
    work = tmp_path / "work"
    code = job.main(
        [
            "--drive-root", str(tmp_path / "drive"),
            "--work-root", str(work),
            "--out", str(out),
            "--no-upload",
            "--universe", "fno",
            "--max-symbols", "20",
            "--sessions", "200",
            "--n-splits", "1",
            "--max-train-rows", "50000",
            "--cost", "0.0011",
            "--warmup-bars", "3",
        ]
    )
    assert code == 0
    status = json.loads((work / "retrain_status.json").read_text())
    assert status["status"] in {"COMPLETED_PASS", "COMPLETED_FAIL"}, status
    assert status["covered_sessions"] == 200
    assert status["symbols_selected"] == 20
    assert (out / "candidate_bundle.joblib").is_file()
    assert (out / "training_report.json").is_file()


def test_local_mode_continues_below_200_and_flags(tmp_path):
    symbols = tuple(f"L{i:02d}" for i in range(10))
    _write_local_inputs(tmp_path / "drive", symbols, n_sessions=60)
    out = tmp_path / "out"
    work = tmp_path / "work"
    code = job.main(
        [
            "--drive-root", str(tmp_path / "drive"),
            "--work-root", str(work),
            "--out", str(out),
            "--no-upload",
            "--universe", "fno",
            "--max-symbols", "10",
            "--sessions", "60",
            "--n-splits", "1",
            "--max-train-rows", "20000",
            "--cost", "0.0011",
            "--warmup-bars", "3",
        ]
    )
    assert code == 0
    status = json.loads((work / "retrain_status.json").read_text())
    assert status["status"] in {"COMPLETED_PASS", "COMPLETED_FAIL"}, status
    assert status["covered_sessions"] == 60
    assert status["insufficient_history"] is True
    assert "history_warning" in status
    assert (out / "training_report.json").is_file()
