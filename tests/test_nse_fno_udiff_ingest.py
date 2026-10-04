import pandas as pd
import pytest

import nse_fno_udiff_ingest as fno


def test_black76_iv_round_trip():
    forward = 100.0
    strike = 100.0
    t = 30.0 / 365.0
    sigma = 0.25
    price = fno._black76_price(forward, strike, t, sigma, True)
    recovered = fno._implied_vol_black76(price, forward, strike, t, True)
    assert recovered == pytest.approx(sigma, rel=0, abs=1e-6)


def test_atm_iv_uses_near_expiry_option_prices():
    day = pd.Timestamp("2026-01-02")
    options = pd.DataFrame(
        {
            "symbol": ["ABC", "ABC", "ABC", "ABC"],
            "expiry": [day + pd.Timedelta(days=30)] * 4,
            "strike_price": [95.0, 100.0, 105.0, 100.0],
            "close": [8.0, 4.5, 2.0, 5.0],
            "settle": [8.0, 4.5, 2.0, 5.0],
            "option_type": ["CE", "CE", "CE", "PE"],
            "underlying_price": [100.0] * 4,
        }
    )
    futures = pd.Series({"close": 100.0, "settle": 100.0})
    iv = fno._atm_iv(options, futures, day)
    assert pd.notna(iv)
    assert 0.01 < iv < 3.0


def test_normalize_supports_strike_price():
    raw = pd.DataFrame(
        {
            "TCKRSYMB": ["ABC"],
            "FININSTRMTP": ["OPTSTK"],
            "XPRYDT": ["29-Jan-2026"],
            "OPTN_TP": ["CE"],
            "STRKPRIC": [100.0],
            "CLSPRIC": [5.0],
            "UNDRLYGPRIC": [100.0],
        }
    )
    out = fno._normalize(raw, pd.Timestamp("2026-01-02").date())
    assert "strike_price" in out.columns
    assert out.loc[0, "strike_price"] == 100.0
