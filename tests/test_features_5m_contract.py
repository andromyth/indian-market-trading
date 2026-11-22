from __future__ import annotations

import numpy as np
import pandas as pd

from features_5m import FEATURES_5M, _fno_asof, make_features_5m, market_features_5m, prepare_fno


def _session(day: str = "2026-05-20", close: float = 100.0) -> pd.DataFrame:
    index = pd.date_range(f"{day} 09:20", f"{day} 15:30", freq="5min", tz="Asia/Kolkata")
    values = close + np.arange(len(index), dtype=float) * 0.1
    return pd.DataFrame(
        {
            "Open": values,
            "High": values + 0.2,
            "Low": values - 0.2,
            "Close": values,
            "Volume": 1000.0,
        },
        index=pd.DatetimeIndex(index, name="Datetime"),
    )


def test_feature_contract_is_42_columns() -> None:
    assert len(FEATURES_5M) == 42
    assert len(set(FEATURES_5M)) == 42


def test_fno_availability_requires_a_finite_value() -> None:
    index = pd.DatetimeIndex(
        pd.date_range("2026-05-20 09:20", "2026-05-20 10:20", freq="5min", tz="Asia/Kolkata")
    )
    empty_values = pd.DataFrame(
        {
            "_date": pd.to_datetime(["2026-05-19"]),
            "fno_pcr": [np.nan],
            "fno_oi_change": [np.nan],
            "fno_fut_return": [np.nan],
            "fno_fut_oi": [np.nan],
            "fno_basis": [np.nan],
        }
    )
    populated_values = empty_values.copy()
    populated_values.loc[0, "fno_pcr"] = 1.1

    empty = _fno_asof(index, empty_values)
    populated = _fno_asof(index, populated_values)

    assert empty["fno_available"].eq(0.0).all()
    assert populated["fno_available"].eq(1.0).all()


def test_prepare_fno_accepts_derived_fut_basis_alias() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-19"]),
            "symbol": ["AAA"],
            "fno_pcr": [1.0],
            "fno_oi_change": [0.1],
            "fno_fut_return": [0.002],
            "fno_fut_oi": [123.0],
            "fno_fut_basis": [0.5],
        }
    )
    prepared = prepare_fno(frame, "AAA")
    assert prepared is not None
    assert float(prepared.loc[0, "fno_basis"]) == 0.5


def test_make_features_5m_returns_the_contract_for_a_full_session() -> None:
    stock = _session()
    market = _session(close=25000.0)
    fno = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-19"]),
            "symbol": ["AAA"],
            "fno_pcr": [1.1],
            "fno_oi_change": [0.01],
            "fno_fut_return": [0.001],
            "fno_fut_oi": [1000.0],
            "fno_fut_basis": [0.2],
        }
    )
    features = make_features_5m(
        stock,
        market_features_5m(market),
        prepare_fno(fno, "AAA"),
        warmup=3,
    )
    assert list(features.columns) == FEATURES_5M
    assert len(features) == 71
    assert features["fno_available"].eq(1.0).all()
    assert features["mkt_ret_1b"].notna().sum() > 0


def test_fno_asof_accepts_millisecond_parquet_dates() -> None:
    # Rebuilt F&O parquets round-trip dates as datetime64[ms] while 5-minute bar
    # indexes are typically datetime64[us]; merge_asof must tolerate the mix.
    index = pd.DatetimeIndex(
        pd.date_range("2026-03-25 09:20", "2026-03-25 10:20", freq="5min", tz="Asia/Kolkata")
    )
    fno = pd.DataFrame(
        {
            "_date": pd.Series(pd.to_datetime(["2026-03-24", "2026-03-25"])).astype("datetime64[ms]"),
            "fno_pcr": [1.0, 1.1],
            "fno_oi_change": [0.0, 0.1],
            "fno_fut_return": [0.0, 0.01],
            "fno_fut_oi": [10.0, 11.0],
            "fno_basis": [0.0, 0.02],
        }
    )
    out = _fno_asof(index, fno)
    assert len(out) == len(index)
    assert (out["fno_available"] == 1.0).all()
