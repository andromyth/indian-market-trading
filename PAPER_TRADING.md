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
- Intraday, Delivery and Legacy have separate opening cash allocations, cash balances, positions, and margin calculations. The combined account sums the books without duplicating starting capital.
- A new ₹100,000 account allocates ₹5,000 to Intraday and ₹95,000 to Delivery by default. Existing accounts are migrated without deleting fills; old unclassified fills remain funded in Legacy, with ₹5,000 allocated to Intraday. Allocations can be edited without resetting the ledger; they must sum to the account's total opening capital.
- Risk-increasing orders are checked against the selected book's free cash and margin, not the combined account's unused funds.
- Existing fills without a strategy tag are preserved in **Legacy / Unassigned** and are not silently classified as Intraday or Delivery.
- Manual simulated BUY and SELL fills. A user must select a strategy book and submit and confirm each order.
- A SELL closes existing long shares first, then opens or increases a short position for any excess quantity. A BUY covers existing short shares first, then opens or increases a long position for any excess quantity.
- No broker order endpoints, real-money execution, leveraged long buying, or F&O.
- Short-sale proceeds are credited to simulated cash but constrained by a 150% initial-margin reserve against the marked short liability. Maintenance margin is 30% of marked short liability.
- New risk-increasing orders are blocked when free cash after initial-margin reserve is negative or a new short would breach maintenance margin. Risk-reducing orders, including buy-to-cover, remain available during a margin call.
- Open shorts are marked to market using the latest available quote for each held symbol. If a usable quote is unavailable, the position uses its entry basis and the UI identifies that fallback. Net liquidation value includes short liabilities.
- By default, the order reference is the latest completed intraday bar supplied by the repository's existing broker-first market-data adapter.
- The simulator applies directional slippage to the reference price and itemises brokerage, STT, stamp duty, NSE transaction charges, NSE IPFT, SEBI turnover fees and GST for each fill. Brokerage remains configurable because tariffs differ by broker.
- A stale or last-session quote requires an explicit override. A manual reference price can also be entered.
- The Intraday model panel calls the existing Beta V2 intraday runtime at a 5-minute decision interval. It only calculates model signals when both the selected stock and its benchmark pass the current-session freshness guard.
- The Delivery model panel calls the existing Beta V2 Delivery champion and its point-in-time daily feature snapshot. During an active target session, a prior-close forecast is identified as such; it is not presented as a newly recalculated intraday signal.
- Model outputs are stored with each fill in the selected track as context; neither model submits orders.

The initial assumptions are 10 bps configurable brokerage, 5 bps slippage per fill, 150% initial short margin, and 30% maintenance margin against short liability. The 2026 NSE cash-equity charge schedule is estimated per executed side: delivery STT 0.1% on buy and sell; intraday STT 0.025% on sell only; stamp duty 0.015% delivery / 0.003% intraday on buy only; NSE transaction charges ₹306.99/crore; NSE IPFT ₹0.01/crore; SEBI turnover fee ₹10/crore; and 18% GST on brokerage, exchange transaction, SEBI and IPFT charges. The rates are sourced from NSE's 2026 schedules and circulars. Per-fill paise rounding can differ slightly from an exchange/broker contract note.

Brokerage is an editable estimate and is not asserted to match a particular broker's tariff, per-order cap or free-delivery rules. DP/depository charges, auto square-off fees, penalties, borrow fees, and other broker-specific items are not modelled. Entry charges are included in open-position cost basis; exit charges are deducted from realised P&L when shares are closed. Short-margin percentages are illustrative simulator assumptions, not exchange, SEBI or broker requirements. The MVP does not model order-book depth, queue priority, partial fills, borrow availability/fees, corporate-action recalls or market impact.

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

The existing fill history is retained. Existing untagged BUY/SELL fills replay in Legacy / Unassigned and are not reclassified. For an existing account, the allocation migration assigns ₹5,000 to Intraday and the remaining opening capital to Legacy when unclassified fills exist, otherwise to Delivery. Classified historical fills retain their strategy tag. The combined account summary aggregates cash, market value, liabilities and P&L across books; each book tab also displays its own starting allocation, cash, free cash, margin excess, positions, P&L and itemised ledger.

The schema migration preserves old fill rows. For already classified Intraday/Delivery fills, the previous generic charges estimate is retained as the brokerage component and missing statutory levies are added once; unclassified Legacy fills retain their previous unitemised charge value. Resetting the paper account clears every book and still requires an explicit confirmation checkbox.

## Tests

Run the focused ledger and risk-control tests:

```bash
pytest -q tests/test_paper_trading.py
python -m compileall -q app.py paper_trading.py
```
