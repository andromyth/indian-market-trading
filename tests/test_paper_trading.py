from __future__ import annotations

import pytest

import paper_trading as paper


def test_calculate_portfolio_buy_then_partial_sell_includes_costs():
    state = paper.calculate_portfolio(
        100_000,
        [
            {"id": 1, "symbol": "SBIN", "side": "BUY", "quantity": 10,
             "fill_price": 100, "gross": 1000, "charges": 1},
            {"id": 2, "symbol": "SBIN", "side": "SELL", "quantity": 4,
             "fill_price": 120, "gross": 480, "charges": 0.48},
        ],
    )

    assert state["cash"] == pytest.approx(99_478.52)
    assert state["realised_pnl"] == pytest.approx(79.12)
    assert state["positions"]["SBIN"]["quantity"] == 6
    assert state["positions"]["SBIN"]["average_cost"] == pytest.approx(100.1)


def test_calculate_portfolio_rejects_short_position():
    with pytest.raises(ValueError, match="short position"):
        paper.calculate_portfolio(
            10_000,
            [{"id": 1, "symbol": "SBIN", "side": "SELL", "quantity": 1,
              "fill_price": 100, "gross": 100, "charges": 0}],
        )


def test_place_order_rejects_insufficient_cash_and_overselling(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_DB_PATH", str(tmp_path / "paper.sqlite3"))
    paper.init_db()
    paper.create_account(1_000, fee_bps=0, slippage_bps=0)
    account = paper.get_account()

    with pytest.raises(ValueError, match="Insufficient paper cash"):
        paper.place_order("SBIN", "BUY", 11, 100, account)

    paper.place_order("SBIN", "BUY", 5, 100, account)
    with pytest.raises(ValueError, match="only 5 shares are held"):
        paper.place_order("SBIN", "SELL", 6, 110, account)


def test_place_order_applies_buy_slippage_and_charges(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_DB_PATH", str(tmp_path / "paper.sqlite3"))
    paper.init_db()
    paper.create_account(10_000, fee_bps=10, slippage_bps=5)
    fill = paper.place_order("RELIANCE", "BUY", 2, 1_000, paper.get_account())

    assert fill["fill_price"] == pytest.approx(1_000.5)
    assert fill["gross"] == pytest.approx(2_001.0)
    assert fill["charges"] == pytest.approx(2.0)
