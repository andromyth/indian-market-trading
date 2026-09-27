"""Causal backtesting for the stock signal engine.

Key properties of the rebuilt backtest:
* Signal is evaluated at bar i using information known by the close of bar i.
* Entry happens no earlier than bar i+1 (next bar open).
* Stop/target can trigger intrabar; if both are hit in one bar, the stop is
  assumed first (conservative).
* Estimated transaction cost is configurable.
* Intraday positions are force-closed before a new market session.
* Metrics include compounded strategy return, drawdown, profit factor and
  holdout-period results.
"""
from __future__ import annotations

import random

import pandas as pd

from signals import add_indicators, combined_signal, market_regime_score, technical_score


def _session_key(idx) -> object:
    try:
        ts = pd.Timestamp(idx)
        return ts.tz_localize(None).date() if ts.tzinfo is not None else ts.date()
    except Exception:
        return idx


def _trade_return(entry: float, exit_: float, cost_bps: float) -> float:
    gross = (exit_ - entry) / entry
    cost = cost_bps / 10000.0
    return gross - cost


def _summarize(trades_df: pd.DataFrame, closes: pd.Series, start_idx: int) -> dict:
    if trades_df.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "avg_return_per_trade_pct": 0.0,
            "strategy_total_return_pct": 0.0,
            "buy_and_hold_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "avg_holding_bars": 0.0,
        }

    returns = trades_df["pnl_net_pct"] / 100.0
    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()

    start_price = float(closes.iloc[start_idx])
    end_price = float(closes.iloc[-1])
    buy_hold = (end_price - start_price) / start_price * 100.0

    return {
        "trades": int(len(trades_df)),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 1),
        "avg_return_per_trade_pct": round(float(returns.mean() * 100), 3),
        "strategy_total_return_pct": round(float((equity.iloc[-1] - 1) * 100), 2),
        "buy_and_hold_return_pct": round(float(buy_hold), 2),
        "max_drawdown_pct": round(float(drawdown.min() * 100), 2),
        "profit_factor": round(float(gains / losses), 2) if losses > 0 else float("inf"),
        "avg_holding_bars": round(float(trades_df["holding_bars"].mean()), 1),
    }


def backtest(
    df: pd.DataFrame,
    style: str = "Delivery",
    market_df: pd.DataFrame | None = None,
    fast: int = 20,
    slow: int = 50,
    rsi_period: int = 14,
    stop_atr: float | None = None,
    target_atr: float | None = None,
    max_hold: int | None = None,
    cost_bps: float = 15.0,
    slippage_bps: float = 5.0,
):
    """Backtest directional long-only cash-equity logic with causal execution."""
    if df is None or df.empty or len(df) < slow + 10:
        return pd.DataFrame(), {"trades": 0}

    if style == "Delivery":
        stop_atr = 1.8 if stop_atr is None else stop_atr
        target_atr = 3.0 if target_atr is None else target_atr
        max_hold = 20 if max_hold is None else max_hold
    else:
        stop_atr = 1.2 if stop_atr is None else stop_atr
        target_atr = 2.0 if target_atr is None else target_atr
        max_hold = 30 if max_hold is None else max_hold

    df = add_indicators(df, fast=fast, slow=slow, rsi_period=rsi_period)
    market_df = add_indicators(market_df, fast=fast, slow=slow, rsi_period=rsi_period) if market_df is not None and not market_df.empty else None

    warmup = max(slow, 26, 14) + 2
    if len(df) <= warmup + 2:
        return pd.DataFrame(), {"trades": 0}

    trades = []
    position = None
    equity = 1.0
    total_cost_bps = float(cost_bps + 2.0 * slippage_bps)

    for i in range(warmup, len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        date = df.index[i]

        mkt_score = 0.0
        if market_df is not None and not market_df.empty:
            try:
                date_key = pd.Timestamp(df.index[i])
                eligible_market = market_df.loc[market_df.index <= date_key]
            except Exception:
                eligible_market = market_df.iloc[: min(i + 1, len(market_df))]
            if len(eligible_market) >= slow:
                mkt_score = market_regime_score(eligible_market, fast=fast, slow=slow)
        tech = technical_score(df.iloc[: i + 1], fast=fast, slow=slow, rsi_period=rsi_period)
        # Backtest intentionally has no look-ahead news input. Current implementation
        # validates the price/risk engine; live news is shown separately in the dashboard.
        label, score = combined_signal(tech, 0.0, mkt_score, event_risk=0.0)

        if position is None and label == "BUY":
            if style == "Intraday" and _session_key(next_row.name) != _session_key(date):
                # Never carry an intraday signal across the close into a new session.
                continue
            entry_price = float(next_row["Open"])
            atr = float(row["ATR14"]) if pd.notna(row["ATR14"]) else 0.0
            if atr <= 0 or entry_price <= 0:
                continue
            position = {
                "entry_i": i + 1,
                "entry_date": df.index[i + 1],
                "entry_price": entry_price,
                "stop": entry_price - stop_atr * atr,
                "target": entry_price + target_atr * atr,
                "score": score,
                "session": _session_key(df.index[i + 1]),
            }
            continue

        if position is None:
            continue

        bar_date = df.index[i]
        current_session = _session_key(bar_date)
        entry_session = position["session"]
        holding = i - position["entry_i"] + 1

        exit_price = None
        exit_reason = None

        # Intraday is flat overnight: exit at session close before the session changes.
        exit_date = date
        if style == "Intraday" and current_session != entry_session:
            prev_bar = df.iloc[i - 1]
            exit_price = float(prev_bar["Close"])
            exit_date = prev_bar.name
            exit_reason = "session_close"
        else:
            low = float(row["Low"])
            high = float(row["High"])
            open_price = float(row["Open"])
            if open_price <= position["stop"]:
                exit_price = open_price
                exit_reason = "gap_stop"
            elif open_price >= position["target"]:
                exit_price = open_price
                exit_reason = "gap_target"
            elif low <= position["stop"] and high >= position["target"]:
                exit_price = position["stop"]
                exit_reason = "stop_first_conservative"
            elif low <= position["stop"]:
                exit_price = position["stop"]
                exit_reason = "stop"
            elif high >= position["target"]:
                exit_price = position["target"]
                exit_reason = "target"
            elif label == "SELL" or holding >= max_hold:
                exit_price = float(row["Close"])
                exit_reason = "signal_exit" if label == "SELL" else "max_hold"

        if exit_price is not None:
            pnl_net = _trade_return(position["entry_price"], exit_price, total_cost_bps) * 100.0
            trades.append(
                {
                    "entry_date": position["entry_date"],
                    "exit_date": exit_date,
                    "entry_price": round(position["entry_price"], 4),
                    "exit_price": round(exit_price, 4),
                    "pnl_net_pct": round(pnl_net, 4),
                    "holding_bars": int(holding),
                    "entry_score": round(position["score"], 4),
                    "exit_reason": exit_reason,
                }
            )
            equity *= 1.0 + pnl_net / 100.0
            position = None

    # Close an open position at last close.
    if position is not None:
        exit_price = float(df["Close"].iloc[-1])
        pnl_net = _trade_return(position["entry_price"], exit_price, total_cost_bps) * 100.0
        trades.append(
            {
                "entry_date": position["entry_date"],
                "exit_date": df.index[-1],
                "entry_price": round(position["entry_price"], 4),
                "exit_price": round(exit_price, 4),
                "pnl_net_pct": round(pnl_net, 4),
                "holding_bars": int(len(df) - 1 - position["entry_i"] + 1),
                "entry_score": round(position["score"], 4),
                "exit_reason": "end_of_data",
            }
        )

    trades_df = pd.DataFrame(trades)
    summary = _summarize(trades_df, df["Close"], warmup)
    summary["cost_model_bps"] = round(total_cost_bps, 2)
    summary["style"] = style
    return trades_df, summary


def random_baseline(
    df: pd.DataFrame,
    trade_durations: list[int] | None = None,
    n_sims: int = 300,
    seed: int = 42,
    cost_bps: float = 15.0,
    slippage_bps: float = 5.0,
):
    """Matched-duration random benchmark. It preserves the strategy's trade-count
    and holding-period distribution, reducing the old baseline's mismatch.
    """
    if df is None or df.empty or not trade_durations:
        return {"sims": 0, "mean_return_pct": 0.0, "p05_return_pct": 0.0, "p95_return_pct": 0.0}
    rng = random.Random(seed)
    closes = df["Close"].tolist()
    n = len(closes)
    costs = (cost_bps + 2 * slippage_bps) / 10000.0
    totals = []
    for _ in range(n_sims):
        equity = 1.0
        for dur in trade_durations:
            max_start = n - dur - 2
            if max_start <= 1:
                continue
            start = rng.randint(1, max_start)
            end = min(n - 1, start + max(1, int(dur)))
            r = (closes[end] - closes[start]) / closes[start] - costs
            equity *= 1 + r
        totals.append((equity - 1) * 100)
    totals.sort()
    p05 = totals[max(0, int(len(totals) * 0.05) - 1)]
    p95 = totals[min(len(totals) - 1, int(len(totals) * 0.95))]
    return {
        "sims": n_sims,
        "mean_return_pct": round(sum(totals) / len(totals), 2),
        "p05_return_pct": round(p05, 2),
        "p95_return_pct": round(p95, 2),
    }


def holdout_backtest(df: pd.DataFrame, style: str, market_df: pd.DataFrame | None = None, **kwargs) -> dict:
    """Evaluate the same fixed rules on the last 30% of the supplied history."""
    if df is None or len(df) < 100:
        return {"holdout_bars": 0, "summary": {"trades": 0}}
    split = int(len(df) * 0.70)
    holdout = df.iloc[split:].copy()
    m_hold = market_df.iloc[split:].copy() if market_df is not None and len(market_df) >= len(df) else market_df
    _, summary = backtest(holdout, style=style, market_df=m_hold, **kwargs)
    return {"holdout_bars": len(holdout), "summary": summary}
