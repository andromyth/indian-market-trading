from __future__ import annotations

import pandas as pd
"""Unit tests for long/short paper ledger, margin and backward compatibility."""

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
    assert state["positions"]["SBIN"]["side"] == "LONG"
    assert state["positions"]["SBIN"]["average_cost"] == pytest.approx(100.1)


def test_calculate_portfolio_short_then_partial_buy_to_cover_includes_both_fees():
    state = paper.calculate_portfolio(
        10_000,
        [
            {"id": 1, "symbol": "IDEA", "side": "SELL", "quantity": 10,
             "fill_price": 100, "gross": 1000, "charges": 1},
            {"id": 2, "symbol": "IDEA", "side": "BUY", "quantity": 4,
             "fill_price": 80, "gross": 320, "charges": 0.32},
        ],
    )

    assert state["cash"] == pytest.approx(10_678.68)
    assert state["realised_pnl"] == pytest.approx(79.28)
    assert state["positions"]["IDEA"]["quantity"] == -6
    assert state["positions"]["IDEA"]["side"] == "SHORT"
    assert state["positions"]["IDEA"]["average_cost"] == pytest.approx(99.9)


def test_sell_reverses_long_to_short_and_buy_reverses_short_to_long():
    state = paper.calculate_portfolio(
        1_000,
        [
            {"id": 1, "symbol": "SBIN", "side": "BUY", "quantity": 5,
             "fill_price": 100, "gross": 500, "charges": 0},
            {"id": 2, "symbol": "SBIN", "side": "SELL", "quantity": 8,
             "fill_price": 120, "gross": 960, "charges": 0},
            {"id": 3, "symbol": "SBIN", "side": "BUY", "quantity": 5,
             "fill_price": 90, "gross": 450, "charges": 0},
        ],
    )

    # SELL closes five longs and opens three shorts; BUY covers all three
    # shorts and opens two longs.
    assert state["positions"]["SBIN"]["quantity"] == 2
    assert state["positions"]["SBIN"]["side"] == "LONG"
    assert state["realised_pnl"] == pytest.approx(190.0)


def test_short_position_marks_to_market_and_reserves_margin():
    state = paper.calculate_portfolio(
        10_000,
        [{"id": 1, "symbol": "IDEA", "side": "SELL", "quantity": 100,
          "fill_price": 50, "gross": 5_000, "charges": 0}],
        market_prices={"IDEA": 60},
    )

    assert state["cash"] == pytest.approx(15_000)
    assert state["short_liability"] == pytest.approx(6_000)
    assert state["equity"] == pytest.approx(9_000)
    assert state["unrealised_pnl"] == pytest.approx(-1_000)
    assert state["initial_margin_required"] == pytest.approx(9_000)
    assert state["maintenance_margin_required"] == pytest.approx(1_800)
    assert state["available_cash"] == pytest.approx(6_000)
    assert state["margin_excess"] == pytest.approx(7_200)
    assert state["margin_status"] == "OK"


def test_short_mark_to_market_flags_maintenance_margin_call():
    state = paper.calculate_portfolio(
        10_000,
        [{"id": 1, "symbol": "IDEA", "side": "SELL", "quantity": 100,
          "fill_price": 50, "gross": 5_000, "charges": 0}],
        market_prices={"IDEA": 150},
    )

    assert state["equity"] == pytest.approx(0)
    assert state["maintenance_margin_required"] == pytest.approx(4_500)
    assert state["margin_status"] == "CALL"

    stressed = paper.calculate_portfolio(
        10_000,
        [{"id": 1, "symbol": "IDEA", "side": "SELL", "quantity": 100,
          "fill_price": 50, "gross": 5_000, "charges": 0}],
        market_prices={"IDEA": 250},
    )
    assert stressed["equity"] == pytest.approx(-10_000)
    assert stressed["margin_status"] == "CALL"


def test_place_order_rejects_insufficient_long_cash_but_allows_short_and_cover(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_DB_PATH", str(tmp_path / "paper.sqlite3"))
    paper.init_db()
    paper.create_account(1_000, fee_bps=0, slippage_bps=0)
    account = paper.get_account()

    with pytest.raises(ValueError, match="Insufficient free cash"):
        paper.place_order("SBIN", "BUY", 11, 100, account)

    short_fill = paper.place_order("IDEA", "SELL", 20, 100, account)
    assert short_fill["side"] == "SELL"
    state = paper.calculate_portfolio(account["initial_cash"], paper.list_fills())
    assert state["positions"]["IDEA"]["quantity"] == -20
    assert state["available_cash"] == pytest.approx(0)

    cover_fill = paper.place_order("IDEA", "BUY", 5, 90, account)
    assert cover_fill["side"] == "BUY"
    state = paper.calculate_portfolio(account["initial_cash"], paper.list_fills())
    assert state["positions"]["IDEA"]["quantity"] == -15


def test_place_order_blocks_new_short_when_initial_margin_is_insufficient(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_DB_PATH", str(tmp_path / "paper.sqlite3"))
    paper.init_db()
    paper.create_account(1_000, fee_bps=0, slippage_bps=0)
    account = paper.get_account()

    with pytest.raises(ValueError, match="Insufficient free cash"):
        paper.place_order("IDEA", "SELL", 21, 100, account)

    fill = paper.place_order("IDEA", "SELL", 20, 100, account)
    assert fill["quantity"] == 20


def test_place_order_applies_buy_slippage_and_charges(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_DB_PATH", str(tmp_path / "paper.sqlite3"))
    paper.init_db()
    paper.create_account(10_000, fee_bps=10, slippage_bps=5)
    fill = paper.place_order("RELIANCE", "BUY", 2, 1_000, paper.get_account())

    assert fill["fill_price"] == pytest.approx(1_000.5)
    assert fill["gross"] == pytest.approx(2_001.0)
    assert fill["charges"] == pytest.approx(2.0)


def test_existing_long_only_ledger_history_remains_readable_and_can_open_short(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_DB_PATH", str(tmp_path / "paper.sqlite3"))
    paper.init_db()
    paper.create_account(5_000, fee_bps=0, slippage_bps=0)
    account = paper.get_account()

    # A legacy ledger created by the previous long-only implementation.
    paper.place_order("SBIN", "BUY", 10, 100, account)
    paper.place_order("SBIN", "SELL", 4, 110, account)
    old_fills = paper.list_fills()
    assert len(old_fills) == 2
    assert paper.calculate_portfolio(5_000, old_fills)["positions"]["SBIN"]["quantity"] == 6

    # Append a short on another symbol without resetting or migrating the ledger.
    paper.place_order("IDEA", "SELL", 20, 100, account)
    new_fills = paper.list_fills()
    state = paper.calculate_portfolio(5_000, new_fills)
    assert len(new_fills) == 3
    assert state["positions"]["SBIN"]["quantity"] == 6
    assert state["positions"]["IDEA"]["quantity"] == -20


def test_live_market_overview_uses_selected_and_watchlist_quotes(monkeypatch):
    calls = []
    class FakeSt:
        def expander(self, *args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
        def button(self, *args, **kwargs):
            return False
        def dataframe(self, frame, **kwargs):
            self.frame = frame
        def caption(self, value):
            self.note = value
        session_state = {}

    fake = FakeSt()
    monkeypatch.setattr(paper, "st", fake)
    def loader(symbol, revision, include_predictions):
        calls.append((symbol, include_predictions))
        return {
            "symbol": symbol.replace(".NS", ""),
            "price": 125.5,
            "price_fresh": True,
            "source": "YFINANCE",
            "quote_timestamp": "2026-11-03T10:00:00+05:30",
            "change_percent": 1.25,
        }

    paper._render_live_market_overview(
        [{"symbol": "IDEA"}],
        {
            "symbol": "SBIN",
            "price": 800.0,
            "price_fresh": True,
            "source": "NSE",
            "quote_timestamp": "2026-11-03T10:00:00+05:30",
            "change_percent": -0.5,
        },
        loader,
        1,
    )
    assert calls[0] == ("IDEA.NS", False)
    assert all(include_predictions is False for _, include_predictions in calls)
    frame = fake.frame
    selected = frame.loc[frame["Symbol"] == "SBIN"].iloc[0]
    assert selected["Last price"] == 800.0
    assert selected["Source"] == "NSE"
    assert selected["Status"] == "FRESH"
    idea = frame.loc[frame["Symbol"] == "IDEA"].iloc[0]
    assert idea["Last price"] == 125.5
    assert idea["Change %"] == 1.25


def test_live_market_overview_preserves_unavailable_quote_status(monkeypatch):
    class FakeSt:
        def expander(self, *args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
        def button(self, *args, **kwargs):
            return False
        def dataframe(self, frame, **kwargs):
            self.frame = frame
        def caption(self, value):
            self.note = value
        session_state = {}

    fake = FakeSt()
    monkeypatch.setattr(paper, "st", fake)
    paper._render_live_market_overview(
        [],
        {"symbol": "SBIN", "price": None, "price_fresh": False,
         "price_freshness_status": "NO_USABLE_PRICE"},
        lambda *args: {},
        0,
    )
    row = fake.frame.loc[fake.frame["Symbol"] == "SBIN"].iloc[0]
    assert pd.isna(row["Last price"])
    assert row["Status"] == "NO_USABLE_PRICE"
    assert row["Source"] == "UNAVAILABLE"
