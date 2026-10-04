import pandas as pd

from nse_data_layer import load_canonical_cash_research_history


def test_canonical_research_view_excludes_t0_and_prefers_eq(tmp_path):
    path = tmp_path / "master.parquet"
    df = pd.DataFrame(
        {
            "symbol": ["ABC","ABC","ABC","XYZ"],
            "date": pd.to_datetime(["2026-01-02"]*4),
            "series": ["T0","BE","EQ","SM"],
            "isin": ["I1","I1","I1","I2"],
            "economic_security_id": ["I1","I1","I1","I2"],
            "listing_id": ["I1|T0","I1|BE","I1|EQ","I2|SM"],
            "row_identity_key": ["k1","k2","k3","k4"],
            "security_class": ["T0","FULLY_PAID_EQUITY","FULLY_PAID_EQUITY","SME_EQUITY"],
            "instrument_type": ["STK"]*4,
            "source_format": ["NSE_CM_UDIFF"]*4,
            "instrument_id": ["1","2","3","4"],
            "instrument_name": ["","","",""],
            "open": [100,101,102,50],
            "high": [100,102,103,51],
            "low": [100,100,101,49],
            "close": [101,102,103,50],
            "last": [101,102,103,50],
            "prev_close": [99,100,101,49],
            "last_price_for_research": [101,102,103,50],
            "research_price": [101,102,103,50],
            "volume": [1,10,20,30],
            "traded_value": [100,1000,2000,1500],
            "num_trades": [1,2,3,4],
        }
    )
    df.to_parquet(path, index=False)

    out = load_canonical_cash_research_history(path)

    assert len(out) == 2
    assert set(out["symbol"]) == {"ABC", "XYZ"}
    abc = out[out["symbol"] == "ABC"].iloc[0]
    assert abc["series"] == "EQ"
    assert abc["close"] == 103


def test_canonical_research_view_uses_explicit_series_precedence(tmp_path):
    path = tmp_path / "master.parquet"
    base = {
        "symbol": ["ABC","ABC"],
        "date": pd.to_datetime(["2026-01-02","2026-01-02"]),
        "series": ["BE","BZ"],
        "isin": ["I1","I1"],
        "economic_security_id": ["I1","I1"],
        "listing_id": ["I1|BE","I1|BZ"],
        "row_identity_key": ["k1","k2"],
        "security_class": ["FULLY_PAID_EQUITY"]*2,
        "instrument_type": ["STK"]*2,
        "source_format": ["NSE_CM_UDIFF"]*2,
        "instrument_id": ["1","2"],
        "instrument_name": ["",""],
        "open": [101,100],
        "high": [102,101],
        "low": [100,99],
        "close": [101.5,100.5],
        "last": [101.5,100.5],
        "prev_close": [100,99],
        "last_price_for_research": [101.5,100.5],
        "research_price": [101.5,100.5],
        "volume": [10,20],
        "traded_value": [1000,2000],
        "num_trades": [2,3],
    }
    pd.DataFrame(base).to_parquet(path, index=False)

    out = load_canonical_cash_research_history(path)
    assert len(out) == 1
    assert out.iloc[0]["series"] == "BE"


def test_canonical_research_view_supports_symbol_and_date_filters(tmp_path):
    path = tmp_path / "master.parquet"
    df = pd.DataFrame(
        {
            "symbol": ["AAA","BBB","AAA"],
            "date": pd.to_datetime(["2025-01-02","2025-01-02","2025-01-03"]),
            "series": ["EQ","EQ","EQ"],
            "isin": ["I1","I2","I1"],
            "economic_security_id": ["I1","I2","I1"],
            "listing_id": ["I1|EQ","I2|EQ","I1|EQ"],
            "row_identity_key": ["a","b","c"],
            "security_class": ["FULLY_PAID_EQUITY"]*3,
            "instrument_type": ["STK"]*3,
            "source_format": ["NSE_CM_UDIFF"]*3,
            "instrument_id": ["1","2","3"],
            "instrument_name": ["","",""],
            "open": [1,2,3],
            "high": [2,3,4],
            "low": [0.5,1.5,2.5],
            "close": [1.5,2.5,3.5],
            "last": [1.5,2.5,3.5],
            "prev_close": [1,2,3],
            "last_price_for_research": [1.5,2.5,3.5],
            "research_price": [1.5,2.5,3.5],
            "volume": [10,20,30],
            "traded_value": [15,50,105],
            "num_trades": [1,2,3],
        }
    )
    df.to_parquet(path, index=False)

    out = load_canonical_cash_research_history(
        path, symbols=["AAA"], start="2025-01-03", end="2025-01-03"
    )

    assert len(out) == 1
    assert out.iloc[0]["symbol"] == "AAA"
    assert out.iloc[0]["date"] == pd.Timestamp("2025-01-03")
