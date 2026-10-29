from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import pytest

import colab_build_all_nse_intraday_multifactor as snapshot_trainer
from snapshot_5m_features import (
    FEATURES,
    FEATURE_SCHEMA_VERSION,
    SNAPSHOT_CADENCE_MINUTES,
    READINESS_FEATURES,
    latest_snapshot_features,
    make_snapshot_feature_frame,
    resample_minute_candles_to_snapshot_5m,
)

IST = "Asia/Kolkata"


def test_minute_training_candles_align_to_five_minute_snapshot_endpoints():
    index = pd.date_range("2026-10-28 09:15", periods=16, freq="min", tz=IST, name="Datetime")
    close = pd.Series(np.arange(100.0, 116.0), index=index)
    minute = pd.DataFrame({
        "Open": close - 0.25,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": np.arange(1.0, 17.0),
    }, index=index)

    out = resample_minute_candles_to_snapshot_5m(minute)

    assert list(out.index.strftime("%H:%M")) == ["09:20", "09:25", "09:30"]
    assert out.loc[pd.Timestamp("2026-10-28 09:20", tz=IST), "Close"] == pytest.approx(104.0)
    assert out.loc[pd.Timestamp("2026-10-28 09:20", tz=IST), "Volume"] == pytest.approx(15.0)
    assert out.loc[pd.Timestamp("2026-10-28 09:20", tz=IST), "day_open"] == pytest.approx(99.75)
    assert "Open" not in out.columns
    assert "High" not in out.columns
    assert "Low" not in out.columns


def test_snapshot_features_have_five_minute_semantics_and_inference_parity():
    idx = pd.date_range("2026-10-27 09:20", periods=60, freq="5min", tz=IST, name="Datetime")
    close = 100.0 * np.cumprod(1.0 + np.linspace(0.0002, 0.001, len(idx)))
    volume = np.linspace(1000.0, 5000.0, len(idx))
    frame = pd.DataFrame({
        "Close": close,
        "Volume": volume,
        "day_open": 99.5,
        "day_high": close * 1.01,
        "day_low": close * 0.99,
        "day_volume": np.cumsum(volume),
    }, index=idx)

    feature_frame = make_snapshot_feature_frame(frame)
    latest = latest_snapshot_features(frame)

    assert SNAPSHOT_CADENCE_MINUTES == 5
    assert FEATURE_SCHEMA_VERSION == "nse_5m_snapshot_close_v1"
    assert len(FEATURES) == 24
    assert list(feature_frame.columns) == FEATURES
    assert latest["ret_5m"] == pytest.approx(close[-1] / close[-2] - 1.0)
    assert latest["ret_15m"] == pytest.approx(close[-1] / close[-4] - 1.0)
    assert latest["session_return"] == pytest.approx(close[-1] / 99.5 - 1.0)
    assert latest["fno_available"] == 0.0
    for name in FEATURES:
        left, right = feature_frame.iloc[-1][name], latest[name]
        if pd.isna(left):
            assert pd.isna(right)
        else:
            assert right == pytest.approx(float(left))


def test_snapshot_feature_adapter_uses_supplied_day_context_not_fake_candle_ohlc():
    idx = pd.date_range("2026-10-28 09:15", periods=20, freq="5min", tz=IST, name="Datetime")
    close = np.linspace(200.0, 204.0, len(idx))
    frame = pd.DataFrame({
        "Close": close,
        "Volume": np.arange(1.0, 21.0),
        "day_open": 199.0,
        "day_high": 205.0,
        "day_low": 198.5,
    }, index=idx)
    normalized = frame.copy()
    normalized["day_volume"] = np.cumsum(frame["Volume"])

    latest = latest_snapshot_features(normalized, fno={"pcr": 1.2, "oi_change": 200.0})
    assert latest["session_return"] == pytest.approx(close[-1] / 199.0 - 1.0)
    assert latest["day_high_ratio"] == pytest.approx(close[-1] / 205.0 - 1.0)
    assert latest["day_low_ratio"] == pytest.approx(close[-1] / 198.5 - 1.0)
    assert latest["fno_pcr"] == pytest.approx(1.2)
    assert latest["fno_oi_change"] == pytest.approx(200.0)
    assert latest["fno_available"] == 1.0


def test_missing_five_minute_snapshot_stays_a_gap_in_return_features():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-10-28 09:15", tz=IST),
        pd.Timestamp("2026-10-28 09:20", tz=IST),
        pd.Timestamp("2026-10-28 09:30", tz=IST),
    ], name="Datetime")
    frame = pd.DataFrame({
        "Close": [100.0, 101.0, 103.0],
        "Volume": [100.0, 200.0, 500.0],
        "day_open": [100.0, 100.0, 100.0],
        "day_high": [100.0, 101.0, 103.0],
        "day_low": [100.0, 100.0, 100.0],
        "day_volume": [100.0, 300.0, 800.0],
    }, index=idx)

    features = make_snapshot_feature_frame(frame)

    assert pd.isna(features.loc[pd.Timestamp("2026-10-28 09:30", tz=IST), "ret_5m"])



def test_training_cache_reads_only_persisted_five_minute_snapshots_and_preserves_gaps(tmp_path, monkeypatch):
    root = tmp_path / "drive"
    snapshot_root = root / "production_data" / "all_universe_minute_snapshots" / "2026" / "10" / "28"
    snapshot_root.mkdir(parents=True)
    slots = [
        pd.Timestamp("2026-10-28 09:15", tz=IST),
        pd.Timestamp("2026-10-28 09:20", tz=IST),
        # Deliberately omit 09:25 to verify interval gaps are not filled.
        pd.Timestamp("2026-10-28 09:30", tz=IST),
    ]
    for i, slot in enumerate(slots):
        rows = [
            {
                "symbol": "RELIANCE",
                "cycle_minute_ist": slot.isoformat(),
                "quote_status": "PASS",
                "actual_observed_price": 100.0 + i,
                "day_volume": 1000.0 + i * 100,
                "day_open": 99.0,
                "day_high": 101.0 + i,
                "day_low": 98.0,
            },
            {
                "symbol": "SBIN",
                "cycle_minute_ist": slot.isoformat(),
                "quote_status": "QUOTE_REQUEST_ERROR" if i == 1 else "PASS",
                "actual_observed_price": None if i == 1 else 500.0 + i,
                "day_volume": None if i == 1 else 2000.0 + i * 50,
                "day_open": 499.0 if i != 1 else None,
                "day_high": 501.0 + i if i != 1 else None,
                "day_low": 498.0 if i != 1 else None,
            },
        ]
        pd.DataFrame(rows).to_parquet(
            snapshot_root / f"snapshot_{slot:%Y%m%d_%H%M}.parquet",
            index=False,
        )

    cache_root = tmp_path / "snapshot-training-cache"
    manifest = snapshot_trainer.build_snapshot_training_cache(root, cache_root)
    monkeypatch.setattr(snapshot_trainer, "_SNAPSHOT_CACHE_ROOT", cache_root, raising=False)
    reliance = snapshot_trainer.load_symbol(root, "RELIANCE")
    sbin = snapshot_trainer.load_symbol(root, "SBIN")

    assert manifest["training_input_source"] == "PERSISTED_FIVE_MINUTE_NSE_QUOTE_SNAPSHOTS_ONLY"
    assert manifest["snapshot_cadence_minutes"] == 5
    assert manifest["snapshot_file_count"] == 3
    assert list(reliance.index.strftime("%H:%M")) == ["09:15", "09:20", "09:30"]
    assert reliance["Close"].tolist() == [100.0, 101.0, 102.0]
    assert np.isnan(reliance.loc[pd.Timestamp("2026-10-28 09:30", tz=IST), "Volume"])
    assert pd.isna(sbin.loc[pd.Timestamp("2026-10-28 09:20", tz=IST), "Close"])
    assert pd.isna(sbin.loc[pd.Timestamp("2026-10-28 09:30", tz=IST), "Volume"])
    assert reliance["Open"].isna().all()
    assert reliance["High"].isna().all()
    assert reliance["Low"].isna().all()


def test_training_features_use_five_minute_rows_without_resampling_minute_candles():
    idx = pd.date_range("2026-10-27 09:15", periods=15, freq="5min", tz=IST, name="Datetime")
    close = np.linspace(100.0, 105.0, len(idx))
    frame = pd.DataFrame({
        "Close": close,
        "Volume": np.full(len(idx), 100.0),
        "day_open": 99.0,
        "day_high": close + 1.0,
        "day_low": close - 1.0,
        "day_volume": np.arange(len(idx), dtype=float) * 100.0,
    }, index=idx)

    out = snapshot_trainer.make_features(frame)

    assert out.index.equals(idx)
    assert "ret_5m" in out.columns
    assert "target_30m" in out.columns
    assert out.loc[idx[0], "ret_5m"] != out.loc[idx[-1], "ret_5m"]
    assert np.isnan(out.loc[idx[-1], "target_30m"])



def test_rolling_indicators_warm_from_prior_snapshots_without_overnight_return():
    prior_idx = pd.date_range("2026-10-27 09:15", periods=60, freq="5min", tz=IST)
    current_idx = pd.date_range("2026-10-28 09:15", periods=2, freq="5min", tz=IST)
    idx = prior_idx.append(current_idx).rename("Datetime")
    close = np.concatenate([np.linspace(100.0, 159.0, len(prior_idx)), [220.0, 221.0]])
    day_open = np.concatenate([np.full(len(prior_idx), 100.0), [219.0, 219.0]])
    day_high = np.concatenate([np.linspace(101.0, 160.0, len(prior_idx)), [221.0, 222.0]])
    day_low = np.concatenate([np.linspace(99.0, 158.0, len(prior_idx)), [218.0, 218.0]])
    frame = pd.DataFrame({
        "Close": close,
        "Volume": np.concatenate([np.full(len(prior_idx), 50.0), [np.nan, 10.0]]),
        "day_open": day_open,
        "day_high": day_high,
        "day_low": day_low,
        "day_volume": np.concatenate([np.arange(1000.0, 1060.0), [1000.0, 1010.0]]),
    }, index=idx)

    features = make_snapshot_feature_frame(frame)
    open_slot = pd.Timestamp("2026-10-28 09:15", tz=IST)
    next_slot = pd.Timestamp("2026-10-28 09:20", tz=IST)

    assert pd.notna(features.loc[open_slot, "ma_250m_ratio"])
    assert pd.notna(features.loc[open_slot, "rsi_70m"])
    assert pd.isna(features.loc[open_slot, "ret_5m"])
    assert features.loc[next_slot, "ret_5m"] == pytest.approx(close[-1] / close[-2] - 1.0)



def test_core_readiness_does_not_penalize_expected_missing_fno_fields():
    prior_idx = pd.date_range("2026-10-27 09:15", periods=60, freq="5min", tz=IST)
    current_idx = pd.date_range("2026-10-28 09:15", periods=6, freq="5min", tz=IST)
    idx = prior_idx.append(current_idx).rename("Datetime")
    close = np.concatenate([np.linspace(100.0, 159.0, len(prior_idx)), [220.0, 220.5, 221.0, 221.5, 222.0, 222.5]])
    day_open = np.concatenate([np.full(len(prior_idx), 100.0), np.full(len(current_idx), 219.0)])
    day_high = np.concatenate([np.linspace(101.0, 160.0, len(prior_idx)), [220.0, 220.5, 221.0, 221.5, 222.0, 223.0]])
    day_low = np.concatenate([np.linspace(99.0, 158.0, len(prior_idx)), np.full(len(current_idx), 218.0)])
    volume = np.concatenate([np.full(len(prior_idx), 50.0), [np.nan, 10.0, 12.0, 11.0, 9.0, 10.0]])
    day_volume = np.concatenate([np.arange(1000.0, 1060.0), [1000.0, 1010.0, 1022.0, 1033.0, 1042.0, 1052.0]])
    frame = pd.DataFrame({
        "Close": close,
        "Volume": volume,
        "day_open": day_open,
        "day_high": day_high,
        "day_low": day_low,
        "day_volume": day_volume,
    }, index=idx)

    feature_frame = make_snapshot_feature_frame(frame)
    row = latest_snapshot_features(frame)
    readiness = pd.Series(row).reindex(READINESS_FEATURES).notna().mean()
    five_minute_slot_readiness = feature_frame.loc[
        pd.Timestamp("2026-10-28 09:20", tz=IST), list(READINESS_FEATURES)
    ].notna().mean()

    assert len(READINESS_FEATURES) == 19
    assert all(name not in READINESS_FEATURES for name in (
        "fno_pcr", "fno_oi_change", "fno_fut_return", "fno_fut_oi", "fno_basis"
    ))
    assert row["fno_available"] == 0.0
    assert readiness >= 0.70
    assert five_minute_slot_readiness >= 0.70
