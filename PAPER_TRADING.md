# Paper Trading Workspace

## Launch

Run the existing Streamlit application:

```bash
streamlit run app.py
```

Choose **Paper Trading** from the Interface selector. The paper-trading page is part of the existing application; it is not a separate production service.

## Scope and execution model

- NSE cash-equity symbols only; long and short positions are supported.
- Manual simulated BUY and SELL fills. A user must submit and confirm each order.
- A SELL closes existing long shares first, then opens or increases a short position for any excess quantity. A BUY covers existing short shares first, then opens or increases a long position for any excess quantity.
- No broker order endpoints, real-money execution, leveraged long buying, or F&O.
- Short-sale proceeds are credited to simulated cash but constrained by a 150% initial-margin reserve against the marked short liability. Maintenance margin is 30% of marked short liability.
- New risk-increasing orders are blocked when free cash after initial-margin reserve is negative or a new short would breach maintenance margin. Risk-reducing orders, including buy-to-cover, remain available during a margin call.
- Open shorts are marked to market using the latest available quote for each held symbol. If a usable quote is unavailable, the position uses its entry basis and the UI identifies that fallback. Net liquidation value includes short liabilities.
- By default, the order reference is the latest completed intraday bar supplied by the repository's existing broker-first market-data adapter.
- The simulator applies directional slippage to the reference price and configurable estimated charges to turnover.
- A stale or last-session quote requires an explicit override. A manual reference price can also be entered.
- The model panel calls the existing Beta V2 intraday runtime. It only calculates model signals when both the selected stock and its benchmark pass the current-session freshness guard. Model output is context for the user; it does not submit orders.

The initial assumptions are 10 bps estimated charges, 5 bps slippage per fill, 150% initial short margin, and 30% maintenance margin against short liability. Margin rates are illustrative simulator assumptions, not exchange, SEBI, or broker requirements. Charges are estimates, not a statutory tax schedule or broker contract note. The MVP does not model order-book depth, queue priority, partial fills, borrow availability/fees, corporate-action recalls, market impact, or exact NSE/broker fees.

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

The existing SQLite schema and fill history are retained. Existing long-only BUY/SELL fills replay as long positions; new excess SELL quantities may open shorts without resetting or migrating the existing ledger. Resetting the paper account still deletes its fills and starts a new ledger; the UI requires an explicit confirmation checkbox.

## Tests

Run the focused ledger and risk-control tests:

```bash
pytest -q tests/test_paper_trading.py
python -m compileall -q app.py paper_trading.py
```
