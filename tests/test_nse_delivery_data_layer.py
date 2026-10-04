import pandas as pd

from nse_delivery_data_layer import (
    RESEARCH_SERIES,
    load_delivery_history,
    parse_mto_bytes,
)


def test_parse_mto_record_type_20():
    payload = (
        b"Security Wise Delivery Position\n"
        b"Record Type,Sr No,Name of Security,Series,"
        b"Quantity Traded,Deliverable Quantity,Percent\n"
        b"20,1,RELIANCE,EQ,100000,50000,50.00\n"
        b"20,2,ABC,SM,1000,700,70.00\n"
        b"10,3,HEADER,EQ,0,0,0.00\n"
    )

    out = parse_mto_bytes(
        payload,
        "2019-01-14",
        archive_url="https://nsearchives.nseindia.com/test",
    )

    assert len(out) == 2
    assert set(out["symbol"]) == {"RELIANCE", "ABC"}
    assert set(out["series"]) == {"EQ", "SM"}
    assert float(
        out.loc[out["symbol"].eq("RELIANCE"), "delivery_qty"].iloc[0]
    ) == 50000
    assert float(
        out.loc[out["symbol"].eq("RELIANCE"), "delivery_pct"].iloc[0]
    ) == 50.0


def test_research_series_policy():
    assert RESEARCH_SERIES == (
        "EQ",
        "BE",
        "BZ",
        "SM",
        "ST",
        "SZ",
    )


def test_load_delivery_history_missing_root(tmp_path):
    out = load_delivery_history(tmp_path)
    assert out.empty
