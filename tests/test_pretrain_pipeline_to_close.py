from __future__ import annotations

import numpy as np
import pandas as pd

import pretrain_pipeline as P
from features_5m import FEATURES_5M


def _bars(symbols: tuple[str, ...], sessions: list[str]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        for day in sessions:
            index = pd.date_range(f"{day} 09:20", f"{day} 15:30", freq="5min", tz="Asia/Kolkata")
            base = 100.0 + (hash(symbol) % 17)
            values = base + np.arange(len(index), dtype=float) * 0.05
            rows.append(
                pd.DataFrame(
                    {
                        "symbol": symbol,
                        "Timestamp": index,
                        "Open": values,
                        "High": values + 0.2,
                        "Low": values - 0.2,
                        "Close": values,
                        "Volume": 1000.0,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_build_training_table_uses_close_only_market_and_fno(tmp_path):
    sessions = [day.strftime("%Y-%m-%d") for day in pd.bdate_range("2026-01-01", periods=25)]
    bars = _bars(("AAA", "BBB"), sessions)

    market_index = pd.DatetimeIndex(
        [
            stamp
            for day in sessions
            for stamp in pd.date_range(f"{day} 09:20", f"{day} 15:30", freq="5min", tz="Asia/Kolkata")
        ]
    )
    market = pd.DataFrame(
        {
            "Timestamp": market_index,
            "Close": 25000.0 + np.arange(len(market_index), dtype=float) * 0.1,
        }
    )
    market_path = tmp_path / "nifty.parquet"
    market.to_parquet(market_path, index=False)

    fno = pd.DataFrame(
        {
            "date": pd.to_datetime(sessions),
            "symbol": ["AAA"] * len(sessions),
            "fno_pcr": [1.0] * len(sessions),
            "fno_oi_change": [0.01] * len(sessions),
            "fno_fut_return": [0.001] * len(sessions),
            "fno_fut_oi": [1000.0] * len(sessions),
            "fno_fut_basis": [0.2] * len(sessions),
        }
    )
    fno_path = tmp_path / "fno.parquet"
    fno.to_parquet(fno_path, index=False)

    unified, rep = P.unify_bars(bars, None, stamp="close")
    market_loaded = P.load_market(str(market_path))
    table = P.build_training_table_5m(
        unified,
        market_loaded,
        pd.read_parquet(fno_path),
        stride=3,
        warmup=3,
    )

    assert rep["sessions"] == 25
    assert rep["symbols"] == 2
    assert len(table) > 0
    assert set(FEATURES_5M).issubset(table.columns)
    assert table.loc[table["symbol"].eq("AAA"), "fno_available"].mean() > 0.9
    assert table.loc[table["symbol"].eq("BBB"), "fno_available"].eq(0.0).all()
    assert table["mkt_ret_1b"].notna().sum() > 0

    target = P.prepare_target(table, FEATURES_5M)
    assert len(target) > 0
    assert target["future_return"].notna().all()
    assert set(FEATURES_5M).issubset(target.columns)
