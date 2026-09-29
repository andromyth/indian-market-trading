# Release notes — 2026-09-29 HOTFIX9

This release addresses interactive-dashboard latency/regression without changing the trained production model artifacts.

## Root cause
The Streamlit page spinner labeled `load_history(...)` occurs before model inference. For NSE daily Delivery mode the application was attempting persistent/local lookup, then NSE historical retrieval, and in some deployments Drive synchronization. The previous local fallback also contained a recursive provider path. These could make a simple search take a long time or repeat provider attempts.

## Changes
- NSE remains the first remote source for `.NS` daily history.
- Interactive NSE history fetches no longer block the user request on Google Drive upload.
- Local-provider recursion removed; fallback ownership is centralized in `data_fetcher.get_historical()`.
- Delivery live history window reduced to 120 calendar days, sufficient for the 50-day rolling feature and 48-bar return feature with trading-day margin.
- 7-day validation history lookup reduced to 45 calendar days.
- Streamlit cache loaders for history/benchmark use `show_spinner=False` so the UI does not imply that the ML model itself is loading while market data is being fetched.
- Production model files/contracts are unchanged.
