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


def test_persist_derived_rows_accepts_custom_paths_and_skips_upload(tmp_path):
    rows = pd.DataFrame(
        {
            "symbol": ["ABC", "ABC"],
            "date": ["2026-01-02", "2026-01-05"],
            "fno_fut_close": [100.0, 101.0],
            "fno_atm_iv": [0.20, 0.21],
        }
    )
    derived = tmp_path / "fno_2026_custom.parquet"
    meta = tmp_path / "fno_2026_custom.meta.json"

    fno._persist_derived_rows(rows, derived=derived, meta_path=meta, upload=False)

    out = pd.read_parquet(derived)
    assert len(out) == 2
    assert out["fno_fut_return"].iloc[1] == pytest.approx(0.01)
    assert out["fno_iv_change"].iloc[1] == pytest.approx(0.01)
    report = pd.read_json(meta, typ="series")
    assert int(report["rows"]) == 2
    assert report["min_date"] == "2026-01-02"
    assert report["max_date"] == "2026-01-05"


def test_rebuild_derived_range_walks_weekdays_and_persists(tmp_path, monkeypatch):
    seen = []

    def fake_normalize(raw, current):
        return pd.DataFrame({"date": [pd.Timestamp(current)]})

    def fake_build(day):
        current = pd.Timestamp(day["date"].iloc[0])
        seen.append(current.date().isoformat())
        return pd.DataFrame(
            {
                "symbol": ["ABC"],
                "date": [current],
                "fno_fut_close": [100.0],
                "fno_atm_iv": [0.2],
            }
        )

    fake_zip = tmp_path / "x.zip"
    fake_zip.write_bytes(b"PK\x03\x04")
    monkeypatch.setattr(fno, "_existing_local_archive", lambda current: fake_zip)
    monkeypatch.setattr(fno, "_read_zip", lambda content: pd.DataFrame())
    monkeypatch.setattr(fno, "_normalize", fake_normalize)
    monkeypatch.setattr(fno, "_build_features", fake_build)

    derived = tmp_path / "fno_range.parquet"
    meta = tmp_path / "fno_range.meta.json"
    fno._rebuild_derived_range(
        pd.Timestamp("2026-01-02").date(),
        pd.Timestamp("2026-01-05").date(),
        derived=derived,
        meta_path=meta,
        upload=False,
    )

    assert seen == ["2026-01-02", "2026-01-05"]
    out = pd.read_parquet(derived)
    assert len(out) == 2
    assert out["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-05"]
