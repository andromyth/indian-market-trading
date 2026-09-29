# Hotfix 4 — 2026-09-29

Fixes the watchlist interaction in Streamlit.

- Watchlist text entry is now inside a Streamlit form.
- Added explicit **Analyze** submit button.
- Typed text is not treated as the active watchlist until Analyze is submitted.
- Existing production ML, data normalization, persistent ledger, and validation logic are preserved.
- No order execution added.

## Hotfix 5
- Analyze submission increments a cache-busting watchlist revision.
- History loading now falls back local/persistent -> NSE -> Dhan -> yfinance.download -> Ticker.history.
- Empty/insufficient data produces an explicit diagnostic instead of a blank dashboard.
- Build marker: 2026-09-29-HOTFIX5.

HOTFIX7
- Removed yfinance from runtime market-data path.
- Added direct Yahoo public Chart API adapter with bounded timeout.
- Removed repeated yfinance `Connection already opened` failure mode.
- Updated get_historical/get_intraday/get_latest_price fallbacks.
