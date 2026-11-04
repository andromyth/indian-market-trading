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
        paper.place_order("SBIN", "BUY", 11, 100, account, segment="INTRADAY")

    short_fill = paper.place_order("IDEA", "SELL", 20, 100, account, segment="INTRADAY")
    assert short_fill["side"] == "SELL"
    state = paper.calculate_portfolio(account["initial_cash"], paper.list_fills())
    assert state["positions"]["IDEA"]["quantity"] == -20
    assert state["available_cash"] == pytest.approx(0)

    cover_fill = paper.place_order("IDEA", "BUY", 5, 90, account, segment="INTRADAY")
    assert cover_fill["side"] == "BUY"
    state = paper.calculate_portfolio(account["initial_cash"], paper.list_fills())
    assert state["positions"]["IDEA"]["quantity"] == -15


def test_place_order_blocks_new_short_when_initial_margin_is_insufficient(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_DB_PATH", str(tmp_path / "paper.sqlite3"))
    paper.init_db()
    paper.create_account(1_000, fee_bps=0, slippage_bps=0)
    account = paper.get_account()

    with pytest.raises(ValueError, match="Insufficient free cash"):
        paper.place_order("IDEA", "SELL", 21, 100, account, segment="INTRADAY")

    fill = paper.place_order("IDEA", "SELL", 20, 100, account, segment="INTRADAY")
    assert fill["quantity"] == 20


def test_place_order_applies_buy_slippage_and_charges(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_DB_PATH", str(tmp_path / "paper.sqlite3"))
    paper.init_db()
    paper.create_account(10_000, fee_bps=10, slippage_bps=5)
    fill = paper.place_order("RELIANCE", "BUY", 2, 1_000, paper.get_account())

    assert fill["fill_price"] == pytest.approx(1_000.5)
    assert fill["gross"] == pytest.approx(2_001.0)
    assert fill["charges"] == pytest.approx(2.73)
    assert fill["brokerage"] == pytest.approx(2.00)
    assert fill["stt"] == pytest.approx(2.00)
    assert fill["stamp_duty"] == pytest.approx(0.30)
    assert fill["gst"] == pytest.approx(0.37)


def test_existing_long_only_ledger_history_remains_readable_and_can_open_short(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_DB_PATH", str(tmp_path / "paper.sqlite3"))
    paper.init_db()
    paper.create_account(5_000, fee_bps=0, slippage_bps=0)
    account = paper.get_account()

    # A legacy ledger created by the previous long-only implementation.
    paper.place_order("SBIN", "BUY", 10, 100, account, segment="INTRADAY")
    paper.place_order("SBIN", "SELL", 4, 110, account, segment="INTRADAY")
    old_fills = paper.list_fills()
    assert len(old_fills) == 2
    assert paper.calculate_portfolio(5_000, old_fills)["positions"]["SBIN"]["quantity"] == 6

    # Append a short on another symbol without resetting or migrating the ledger.
    paper.place_order("IDEA", "SELL", 20, 100, account, segment="INTRADAY")
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



def test_open_positions_marks_report_live_quote_source():
    state = {
        "positions": {
            "SBIN": {"quantity": 10, "cost_basis": 1_000.0, "average_cost": 100.0},
            "IDEA": {"quantity": -5, "cost_basis": 500.0, "average_cost": 100.0},
        }
    }

    def loader(symbol, revision, include_predictions):
        return {
            "price": 90.0,
            "quote_timestamp": "2026-11-03T12:00:00+05:30",
        }

    frame, prices = paper._open_positions_with_marks(
        state,
        {
            "symbol": "SBIN",
            "price": 110.0,
            "quote_timestamp": "2026-11-03T12:01:00+05:30",
        },
        loader,
        0,
    )

    assert prices == {"SBIN": 110.0, "IDEA": 90.0}
    assert frame.set_index("Symbol").loc["SBIN", "Price source"] == "LIVE_QUOTE"
    assert frame.set_index("Symbol").loc["IDEA", "Price source"] == "LIVE_QUOTE"


def test_open_positions_marks_report_entry_basis_fallback():
    state = {
        "positions": {
            "IDEA": {"quantity": -5, "cost_basis": 500.0, "average_cost": 100.0},
        }
    }

    frame, prices = paper._open_positions_with_marks(
        state,
        {"symbol": "SBIN", "price": 800.0},
        lambda *args: {},
        0,
    )

    assert prices == {"IDEA": 100.0}
    assert frame.iloc[0]["Price source"] == "ENTRY_BASIS_FALLBACK"
    assert frame.iloc[0]["Mark price"] == 100.0



def test_segmented_portfolio_keeps_intraday_and_delivery_positions_separate():
    fills = [
        {"id": 1, "segment": "INTRADAY", "symbol": "SBIN", "side": "BUY",
         "quantity": 10, "fill_price": 100, "gross": 1000, "charges": 0},
        {"id": 2, "segment": "DELIVERY", "symbol": "SBIN", "side": "SELL",
         "quantity": 5, "fill_price": 100, "gross": 500, "charges": 0},
    ]

    state = paper.calculate_segmented_portfolio(
        10_000, fills, market_prices={"SBIN": 110}
    )

    intraday = state["segments"]["INTRADAY"]
    delivery = state["segments"]["DELIVERY"]
    assert intraday["positions"]["SBIN"]["quantity"] == 10
    assert delivery["positions"]["SBIN"]["quantity"] == -5
    assert intraday["unrealised_pnl"] == pytest.approx(100)
    assert delivery["unrealised_pnl"] == pytest.approx(-50)
    assert state["cash"] == pytest.approx(9_500)
    assert state["long_market_value"] == pytest.approx(1_100)
    assert state["short_liability"] == pytest.approx(550)
    assert state["unrealised_pnl"] == pytest.approx(50)
    assert state["equity"] == pytest.approx(10_050)


def test_place_order_persists_separate_intraday_and_delivery_books(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_DB_PATH", str(tmp_path / "paper.sqlite3"))
    paper.init_db()
    paper.create_account(10_000, fee_bps=0, slippage_bps=0)
    account = paper.get_account()

    intraday_fill = paper.place_order(
        "SBIN", "BUY", 10, 100, account, segment="INTRADAY"
    )
    delivery_fill = paper.place_order(
        "SBIN", "SELL", 5, 100, account, segment="DELIVERY"
    )

    assert intraday_fill["segment"] == "INTRADAY"
    assert delivery_fill["segment"] == "DELIVERY"
    assert len(paper.list_fills(segment="INTRADAY")) == 1
    assert len(paper.list_fills(segment="DELIVERY")) == 1
    assert paper.list_fills(segment="LEGACY") == []

    state = paper.calculate_segmented_portfolio(
        account["initial_cash"], paper.list_fills(), market_prices={"SBIN": 110}
    )
    assert state["segments"]["INTRADAY"]["positions"]["SBIN"]["quantity"] == 10
    assert state["segments"]["DELIVERY"]["positions"]["SBIN"]["quantity"] == -5


def test_existing_unclassified_database_fills_migrate_to_legacy_without_loss(tmp_path, monkeypatch):
    import sqlite3

    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setenv("PAPER_TRADING_DB_PATH", str(db_path))
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE paper_account (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                initial_cash REAL NOT NULL,
                fee_bps REAL NOT NULL,
                slippage_bps REAL NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            """INSERT INTO paper_account VALUES (1, 10000, 10, 5, '2026-11-03T09:00:00+05:30')"""
        )
        connection.execute(
            """CREATE TABLE paper_fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ist TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                reference_price REAL NOT NULL,
                fill_price REAL NOT NULL,
                gross REAL NOT NULL,
                charges REAL NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                quote_timestamp TEXT NOT NULL DEFAULT '',
                prediction_signal TEXT NOT NULL DEFAULT '',
                prediction_probability_up REAL,
                prediction_expected_return REAL,
                prediction_model_version TEXT NOT NULL DEFAULT '',
                prediction_decision_timestamp TEXT NOT NULL DEFAULT '',
                prediction_horizon_min INTEGER
            )"""
        )
        connection.execute(
            """INSERT INTO paper_fills(
                timestamp_ist, symbol, side, quantity, reference_price,
                fill_price, gross, charges
            ) VALUES ('2026-11-03T09:10:00+05:30', 'SBIN', 'BUY', 2, 100, 100, 200, 0)"""
        )

    paper.init_db()
    legacy = paper.list_fills(segment="LEGACY")
    assert len(legacy) == 1
    assert legacy[0]["symbol"] == "SBIN"
    assert legacy[0]["quantity"] == 2
    assert legacy[0]["segment"] == "LEGACY"
    assert paper.list_fills(segment="INTRADAY") == []
    assert paper.list_fills(segment="DELIVERY") == []



def test_nse_cash_equity_charge_rates_are_side_and_segment_specific():
    intraday_buy = paper.calculate_trade_charges("INTRADAY", "BUY", 100_000, 0)
    intraday_sell = paper.calculate_trade_charges("INTRADAY", "SELL", 100_000, 0)
    delivery_buy = paper.calculate_trade_charges("DELIVERY", "BUY", 100_000, 0)
    delivery_sell = paper.calculate_trade_charges("DELIVERY", "SELL", 100_000, 0)

    assert intraday_buy["stt"] == 0
    assert intraday_buy["stamp_duty"] == pytest.approx(3.00)
    assert intraday_sell["stt"] == pytest.approx(25.00)
    assert intraday_sell["stamp_duty"] == 0
    assert delivery_buy["stt"] == pytest.approx(100.00)
    assert delivery_buy["stamp_duty"] == pytest.approx(15.00)
    assert delivery_sell["stt"] == pytest.approx(100.00)
    assert delivery_sell["stamp_duty"] == 0
    assert intraday_buy["exchange_charges"] == pytest.approx(3.07)
    assert intraday_buy["sebi_charges"] == pytest.approx(0.10)
    assert intraday_buy["ipft_charges"] == 0


def test_realised_pnl_deducts_entry_and_exit_statutory_charges(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_DB_PATH", str(tmp_path / "paper.sqlite3"))
    paper.init_db()
    paper.create_account(5_000, fee_bps=10, slippage_bps=0)
    account = paper.get_account()

    buy = paper.place_order(
        "SBIN", "BUY", 10, 100, account, segment="INTRADAY"
    )
    sell = paper.place_order(
        "SBIN", "SELL", 10, 110, account, segment="INTRADAY"
    )
    state = paper.calculate_portfolio(5_000, paper.list_fills())

    expected = 100.00 - buy["charges"] - sell["charges"]
    assert state["positions"] == {}
    assert state["realised_pnl"] == pytest.approx(expected)
    assert state["realised_pnl"] == pytest.approx(97.14)
    assert state["cash"] == pytest.approx(5_000 + expected)


def test_cash_allocations_are_separate_and_sum_to_total(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_DB_PATH", str(tmp_path / "paper.sqlite3"))
    paper.init_db()
    paper.create_account(100_000, fee_bps=0, slippage_bps=0)
    account = paper.get_account()

    allocations = paper._segment_capitals_from_account(account)
    assert allocations == {
        "INTRADAY": 5_000.0,
        "DELIVERY": 95_000.0,
        "LEGACY": 0.0,
    }

    paper.update_segment_capital_allocations(7_500, 92_500, 0)
    updated = paper.get_account()
    assert paper._segment_capitals_from_account(updated) == {
        "INTRADAY": 7_500.0,
        "DELIVERY": 92_500.0,
        "LEGACY": 0.0,
    }


def test_segment_cash_and_risk_are_not_shared(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_DB_PATH", str(tmp_path / "paper.sqlite3"))
    paper.init_db()
    paper.create_account(100_000, fee_bps=0, slippage_bps=0)
    account = paper.get_account()

    # Intraday cannot use the unused ₹95,000 Delivery allocation.
    with pytest.raises(ValueError, match="Insufficient free cash in the Intraday book"):
        paper.place_order(
            "SBIN", "BUY", 51, 100, account, segment="INTRADAY"
        )

    fill = paper.place_order(
        "SBIN", "BUY", 50, 100, account, segment="INTRADAY"
    )
    assert fill["segment"] == "INTRADAY"
    state = paper.calculate_segmented_portfolio(
        100_000,
        paper.list_fills(),
        segment_capitals=paper._segment_capitals_from_account(account),
    )
    assert state["segments"]["INTRADAY"]["cash"] == pytest.approx(0)
    assert state["segments"]["DELIVERY"]["cash"] == pytest.approx(95_000)
    assert state["cash"] == pytest.approx(95_000)
