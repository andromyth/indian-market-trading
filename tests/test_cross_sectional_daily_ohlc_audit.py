from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from cross_sectional_daily_ohlc_audit import (
    BLOCK_LENGTH_SESSIONS,
    completed_session_cutoff,
    moving_block_bootstrap_ci,
    prepare_symbol_data,
    select_shared_holdout_positions,
)

IST = ZoneInfo("Asia/Kolkata")


def _synthetic_ohlcv(n: int = 280) -> pd.DataFrame:
    rng = np.random.default_rng(94171)
    dates = pd.bdate_range("2025-01-02", periods=n)
    close = 200.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, size=n)))
    gap = rng.normal(0.0, 0.004, size=n)
    open_ = np.r_[close[0], close[:-1]] * np.exp(gap)
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.007, size=n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.007, size=n)))
    volume = rng.integers(100_000, 2_000_000, size=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def test_completed_session_cutoff_excludes_intraday_daily_bar():
    before_close = datetime(2026, 10, 14, 13, 35, tzinfo=IST)
    after_close = datetime(2026, 10, 14, 16, 5, tzinfo=IST)

    assert completed_session_cutoff(before_close).isoformat() == "2026-10-13"
    assert completed_session_cutoff(after_close).isoformat() == "2026-10-14"


def test_missing_expected_stock_session_is_not_shifted_to_next_available_bar():
    frame = _synthetic_ohlcv()
    calendar = pd.DatetimeIndex(frame.index)
    missing_date = calendar[210]
    incomplete = frame.drop(index=missing_date)
    prepared = prepare_symbol_data(incomplete, calendar)

    # The +7 target for an origin exactly seven sessions before the missing
    # bar is unavailable; it must not be assigned the next available stock bar.
    missing_origin = 210 - 7
    assert not bool(prepared["complete_origin"][missing_origin])
    stack = prepared["stacked"]
    assert not (
        stack["origin_position"].eq(missing_origin)
        & stack["forecast_horizon"].eq(7.0)
    ).any()


def test_shared_holdout_uses_identical_complete_origins_for_every_symbol():
    n = 280
    first = np.ones(n, dtype=bool)
    second = np.ones(n, dtype=bool)
    first[255] = False
    second[260] = False

    split, positions = select_shared_holdout_positions(
        {"AAA": first, "BBB": second},
        session_count=n,
        holdout_fraction=0.20,
        horizon_count=7,
    )

    assert split == 224
    assert 255 not in positions
    assert 260 not in positions
    assert len(positions) == (n - 7 - split) - 2


def test_moving_block_bootstrap_reports_skill_uncertainty_deterministically():
    rng = np.random.default_rng(1)
    dates, symbols, horizons, targets = 35, 3, 7, 4
    baseline = rng.uniform(0.5, 1.5, size=(dates, symbols, horizons, targets))
    model = baseline * 0.8

    result_a = moving_block_bootstrap_ci(
        model, baseline, [0, 1, 2], replicates=100, seed=77
    )
    result_b = moving_block_bootstrap_ci(
        model, baseline, [0, 1, 2], replicates=100, seed=77
    )

    assert result_a == result_b
    assert result_a["block_length_sessions"] == BLOCK_LENGTH_SESSIONS
    assert result_a["mean_stock_skill_ci95"][0] > 0
    assert result_a["mean_stock_skill_ci95"][1] > 0
    assert result_a["positive_stock_rate_ci95"] == [1.0, 1.0]


def test_purge_gap_is_more_than_longest_target_horizon():
    from cross_sectional_daily_ohlc_audit import PURGE_SESSIONS

    assert PURGE_SESSIONS > max(1, 7)



def test_nifty_history_loader_does_not_request_drive_sync(monkeypatch):
    from cross_sectional_daily_ohlc_audit import load_daily_history
    import nse_data_layer as layer

    dates = pd.bdate_range("2025-01-01", periods=220)
    source = pd.DataFrame({
        "date": dates,
        "open": np.full(len(dates), 100.0),
        "high": np.full(len(dates), 102.0),
        "low": np.full(len(dates), 99.0),
        "close": np.full(len(dates), 101.0),
        "volume": np.zeros(len(dates)),
    })
    calls = []

    def fake_fetch_index_history(symbol, start, end, *, sync_drive):
        calls.append((symbol, start, end, sync_drive))
        return source.copy()

    monkeypatch.setattr(layer, "fetch_index_history", fake_fetch_index_history)

    daily, meta = load_daily_history("^NSEI", "900d")

    assert len(calls) == 1
    assert calls[0][0] == "^NSEI"
    assert calls[0][2] == audit.completed_session_cutoff()
    assert calls[0][3] is False
    assert len(daily) >= 200
    assert meta["provider_path"] == "NSE_official_index_adapter_sync_drive_false"
    assert not daily.empty
