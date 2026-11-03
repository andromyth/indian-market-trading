# Paper Trading Workspace

## Launch

Run the existing Streamlit application:

```bash
streamlit run app.py
```

Choose **Paper Trading** from the Interface selector. The paper-trading page is part of the existing application; it is not a separate production service.

## Scope and execution model

- NSE cash-equity symbols only; long and short positions are supported.
- Separate **Intraday** and **Delivery** strategy books. Each book has its own positions, fills, realised P&L and unrealised P&L; positions in the same symbol do not net across books.
- Both books share one virtual account cash balance and one account-wide initial/maintenance-margin pool. This prevents the starting capital from being duplicated across the two tracks.
- Existing fills without a strategy tag are preserved in **Legacy / Unassigned** and are not silently classified as Intraday or Delivery.
- Manual simulated BUY and SELL fills. A user must select a strategy book and submit and confirm each order.
- A SELL closes existing long shares first, then opens or increases a short position for any excess quantity. A BUY covers existing short shares first, then opens or increases a long position for any excess quantity.
- No broker order endpoints, real-money execution, leveraged long buying, or F&O.
- Short-sale proceeds are credited to simulated cash but constrained by a 150% initial-margin reserve against the marked short liability. Maintenance margin is 30% of marked short liability.
- New risk-increasing orders are blocked when free cash after initial-margin reserve is negative or a new short would breach maintenance margin. Risk-reducing orders, including buy-to-cover, remain available during a margin call.
- Open shorts are marked to market using the latest available quote for each held symbol. If a usable quote is unavailable, the position uses its entry basis and the UI identifies that fallback. Net liquidation value includes short liabilities.
- By default, the order reference is the latest completed intraday bar supplied by the repository's existing broker-first market-data adapter.
- The simulator applies directional slippage to the reference price and configurable estimated charges to turnover.
- A stale or last-session quote requires an explicit override. A manual reference price can also be entered.
- The Intraday model panel calls the existing Beta V2 intraday runtime at a 5-minute decision interval. It only calculates model signals when both the selected stock and its benchmark pass the current-session freshness guard.
- The Delivery model panel calls the existing Beta V2 Delivery champion and its point-in-time daily feature snapshot. During an active target session, a prior-close forecast is identified as such; it is not presented as a newly recalculated intraday signal.
- Model outputs are stored with each fill in the selected track as context; neither model submits orders.

The initial assumptions are 10 bps estimated charges, 5 bps slippage per fill, 150% initial short margin, and 30% maintenance margin against short liability. Margin rates are illustrative simulator assumptions, not exchange, SEBI, or broker requirements. Charges are estimates, not a statutory tax schedule or broker contract note. The MVP does not model order-book depth, queue priority, partial fills, borrow availability/fees, corporate-action recalls, market impact, or exact NSE/broker fees.

## Ledger and persistence

The account, settings, and immutable simulated fills are stored in SQLite, independently of the production prediction and actuals ledgers. Each fill has a `segment` tag (`INTRADAY`, `DELIVERY`, or `LEGACY`). A schema-only migration adds that tag to an existing database; previous untagged rows receive `LEGACY` and remain intact.

Default local path:

```text
data/paper_trading.sqlite3
```

Set `PAPER_TRADING_DB_PATH` to a mounted persistent path when the hosting filesystem is ephemeral, for example:

```bash
PAPER_TRADING_DB_PATH=/mnt/persistent/paper_trading.sqlite3
```

The existing fill history is retained. Existing untagged BUY/SELL fills replay in Legacy / Unassigned; new fills are written to the selected Intraday or Delivery book. The combined account summary aggregates cash and margin across all books, while book tabs show separate positions, P&L and ledgers. Resetting the paper account clears every book and still requires an explicit confirmation checkbox.

## Tests

Run the focused ledger and risk-control tests:

```bash
pytest -q tests/test_paper_trading.py
python -m compileall -q app.py paper_trading.py
```
