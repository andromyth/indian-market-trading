from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

import prediction_universe as pu

PLAN = pu.load_capital_plan()


def _ledger(rows):
    return pd.DataFrame(rows, columns=[
        "prediction_date", "symbol", "signal", "reference_close", "actual_close",
        "direction_match", "validation_status",
    ])


def test_budget_ladder_and_hard_price_cap():
    budget = pu.position_budget(PLAN)
    assert budget["positions"] == 2
    assert budget["capital"] == 7000
    assert budget["per_trade"] == pytest.approx(3425.0)
    assert budget["price_cap"] == pytest.approx(2500.0)

    grown = pu.position_budget(PLAN, realised_pnl=12000)
    assert grown["positions"] == 4
    assert grown["price_cap"] <= 2500


def test_delivery_stats_marks_profitable_only_with_evidence():
    ledger = _ledger([
        ["2026-11-02", "AAA", "BUY", 100, 103, True, "MATURED"],
        ["2026-11-03", "AAA", "BUY", 103, 106, True, "MATURED"],
        ["2026-11-04", "AAA", "SELL", 106, 101, True, "MATURED"],
        ["2026-11-02", "BBB", "BUY", 200, 196, False, "MATURED"],
        ["2026-11-03", "BBB", "BUY", 196, 194, False, "MATURED"],
        ["2026-11-04", "BBB", "BUY", 194, 197, True, "MATURED"],
        ["2026-11-05", "CCC", "BUY", 50, None, None, "PENDING"],
    ])
    stats = pu.delivery_stats(ledger, lookback_sessions=20, min_records=3)
    assert stats["AAA"]["profitable"] is True
    assert stats["BBB"]["profitable"] is False
    assert "CCC" not in stats
    assert pu.latest_closes(ledger)["AAA"] == pytest.approx(106)


def test_intraday_stats_use_signed_return():
    frame = pd.DataFrame({
        "symbol": ["AAA", "AAA", "AAA", "BBB"],
        "actual_return": [0.01, 0.02, 0.005, -0.01],
        "ledger_date": ["2026-11-13"] * 4,
    })
    stats = pu.intraday_stats([frame], min_records=3)
    assert stats["AAA"]["profitable"] is True
    assert stats["BBB"]["records"] == 1
    assert stats["BBB"]["profitable"] is False


def test_select_universe_drops_unaffordable_and_caps_size():
    closes = {f"S{i:03d}": 100.0 + i for i in range(40)}
    closes["EXPENSIVE"] = 5000.0
    fno = {"S000", "S001", "S002"}
    selection = pu.select_universe(closes, {}, {}, fno, PLAN)
    assert "EXPENSIVE" not in selection["symbols"]
    assert len(selection["symbols"]) == 40
    assert selection["signals"] if "signals" in selection else True
    assert selection["stats"]["fno"] == 3
    assert selection["universe_version"]


def test_select_universe_includes_market_context_and_respects_limit():
    plan = dict(PLAN, universe_size=10)
    closes = {f"S{i:02d}": 50.0 for i in range(30)}
    closes[pu.MARKET_CONTEXT_SYMBOL] = 24000.0
    selection = pu.select_universe(closes, {}, {}, set(), plan)
    assert pu.MARKET_CONTEXT_SYMBOL in selection["symbols"]
    assert len(selection["symbols"]) == 11


def test_tradable_picks_filter_price_and_rank():
    predictions = pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC", "DDD"],
        "signal": ["BUY", "BUY", "BUY", "SELL"],
        "probability_up_30m": [0.7, 0.6, 0.9, 0.9],
        "expected_return_30m": [0.02, 0.018, 0.03, 0.05],
        "prediction_status": ["PREDICTED"] * 4,
    })
    prices = {"AAA": 500.0, "BBB": 2400.0, "CCC": 5000.0, "DDD": 100.0}
    result = pu.tradable_picks(predictions, PLAN, prices=prices)
    symbols = [p["symbol"] for p in result["picks"]]
    assert symbols == ["AAA", "BBB"]
    assert result["picks"][0]["quantity"] == 6
    assert result["skipped"] >= 1


def test_tradable_picks_ignores_sub_threshold_moves():
    predictions = pd.DataFrame({
        "symbol": ["AAA"],
        "signal": ["BUY"],
        "probability_up_30m": [0.9],
        "expected_return_30m": [0.004],
        "prediction_status": ["PREDICTED"],
    })
    assert pu.tradable_picks(predictions, PLAN, prices={"AAA": 100.0})["picks"] == []


def test_active_symbols_falls_back_without_store():
    symbols, meta = pu.active_symbols(store=None)
    assert symbols is None
    assert meta["status"] == "FALLBACK_FULL_ROSTER"


def test_build_manifest_aborts_without_closes(monkeypatch):
    monkeypatch.setattr(pu, "_download_bytes", lambda store, remote: None)
    manifest = pu.build_manifest(store=object(), plan=PLAN, sync=False)
    assert manifest["status"] == "ABORTED"
    assert "closes" in manifest["sources"]


def test_build_manifest_selects_from_fake_sources(monkeypatch):
    ledger_csv = _ledger([
        ["2026-11-02", s, "BUY", 100, 102, True, "MATURED"]
        for s in ["AAA", "BBB", "CCC"]
    ]).to_csv(index=False).encode()

    def fake_download(store, remote):
        if remote == pu.DELIVERY_LEDGER:
            return ledger_csv
        return None

    monkeypatch.setattr(pu, "_download_bytes", fake_download)
    monkeypatch.setattr(pu, "_outcomes_frames", lambda store, now, days: ([], []))
    monkeypatch.setattr(pu, "fno_universe", lambda frame: {"AAA"})
    plan = dict(PLAN, universe_size=50)
    manifest = pu.build_manifest(store=object(), plan=plan, sync=False,
                                 now=datetime(2026, 11, 16, 16, 0))
    assert manifest["status"] == "ABORTED"
    assert manifest["stats"]["selected"] == 3
