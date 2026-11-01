# Paper Trading Workspace

## Launch

Run the existing Streamlit application:

```bash
streamlit run app.py
```

Choose **Paper Trading** from the Interface selector. The paper-trading page is part of the existing application; it is not a separate production service.

## Scope and execution model

- NSE cash-equity symbols only; long-only positions.
- Manual simulated BUY and SELL fills. A user must submit and confirm each order.
- No broker order endpoints, real-money execution, leverage, short selling, or F&O.
- By default, the order reference is the latest completed intraday bar supplied by the repository's existing broker-first market-data adapter.
- The simulator applies directional slippage to the reference price and configurable estimated charges to turnover.
- A stale or last-session quote requires an explicit override. A manual reference price can also be entered.
- The model panel calls the existing Beta V2 intraday runtime. It only calculates model signals when both the selected stock and its benchmark pass the current-session freshness guard. Model output is context for the user; it does not submit orders.

The initial assumptions are 10 bps estimated charges and 5 bps slippage per fill. These are simulation parameters, not a statutory tax schedule or broker contract note. The MVP does not model order-book depth, queue priority, partial fills, market impact, or exact NSE/broker fees.

## Ledger and persistence

The account, settings, and immutable simulated fills are stored in SQLite, independently of the production prediction and actuals ledgers.

Default local path:

```text
data/paper_trading.sqlite3
```

Set `PAPER_TRADING_DB_PATH` to a mounted persistent path when the hosting filesystem is ephemeral, for example:

```bash
PAPER_TRADING_DB_PATH=/mnt/persistent/paper_trading.sqlite3
```

Resetting the paper account deletes its fills and starts a new ledger; the UI requires an explicit confirmation checkbox.

## Tests

Run the focused ledger and risk-control tests:

```bash
pytest -q tests/test_paper_trading.py
python -m compileall -q app.py paper_trading.py
```
