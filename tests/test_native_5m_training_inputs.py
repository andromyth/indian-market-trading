from __future__ import annotations

import numpy as np
import pandas as pd

from colab_build_all_nse_intraday_multifactor import (
    LIVE_5M_REL,
    NATIVE_5M_SEED_REL,
    SEED_REL,
    load_live_history,
    load_symbol,
    load_symbol_5m,
)

IST = "Asia/Kolkata"


def _frame(index, base):
    close = np.arange(len(index), dtype=float) + base
    return pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.3,
            "Low": close - 0.3,
            "Close": close,
            "Volume": 1000.0,
        },
        index=index,
    )


def test_existing_shared_loader_remains_one_minute_while_new_loader_merges_native_5m(tmp_path):
    symbol = "TEST"
    minute_index = pd.date_range("2026-10-30 09:15", periods=15, freq="min", tz=IST)
    minute = _frame(minute_index, 100.0)
    seed_dir = tmp_path / SEED_REL / f"symbol={symbol}"
    seed_dir.mkdir(parents=True)
    minute.to_parquet(seed_dir / f"{symbol}.parquet")

    native_index = pd.date_range("2026-10-30 09:20", periods=3, freq="5min", tz=IST)
    native = _frame(native_index, 200.0)
    native_dir = tmp_path / NATIVE_5M_SEED_REL / f"symbol={symbol}"
    native_dir.mkdir(parents=True)
    native.to_parquet(native_dir / f"{symbol}.parquet")

    legacy = load_symbol(tmp_path, symbol)
    merged = load_symbol_5m(tmp_path, symbol)

    assert len(legacy) == 15
    assert legacy.index.freq is None or legacy.index.inferred_freq == "min"
    assert len(merged) == 3
    assert merged.loc[pd.Timestamp("2026-10-30 09:20", tz=IST), "Close"] == 200.0


def test_legacy_one_minute_archive_is_not_used_by_five_minute_training(tmp_path):
    symbol = "TEST"
    minute_index = pd.date_range("2026-10-30 09:15", periods=15, freq="min", tz=IST)
    minute = _frame(minute_index, 100.0)
    seed_dir = tmp_path / SEED_REL / f"symbol={symbol}"
    seed_dir.mkdir(parents=True)
    minute.to_parquet(seed_dir / f"{symbol}.parquet")

    assert len(load_symbol(tmp_path, symbol)) == 15
    assert load_symbol_5m(tmp_path, symbol).empty

def test_live_slot_archive_is_loaded_by_symbol_and_keeps_close_time(tmp_path):
    index = pd.date_range("2026-10-30 10:00", periods=2, freq="5min", tz=IST)
    bars = _frame(index, 300.0).reset_index(names="Timestamp")
    bars.insert(0, "symbol", "RELIANCE")
    bars["source"] = "INDSTOCKS_HISTORICAL_5M"
    slot = tmp_path / LIVE_5M_REL / "2026" / "10" / "30"
    slot.mkdir(parents=True)
    bars.to_parquet(slot / "capture_20261030_1005.parquet", index=False)

    loaded = load_live_history(tmp_path, now=pd.Timestamp("2026-10-31 10:00", tz=IST))

    assert "RELIANCE" in loaded
    assert loaded["RELIANCE"].index.tolist() == index.tolist()
