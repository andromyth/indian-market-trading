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


## HOTFIX10 — NSE-FIRST DATA PATH
- Fixed NSE historical request handling to accept both current JSON and CSV response forms.
- Reuses a warmed NSE cookie session with current browser-like request headers and refreshes it on 401/403/408/429/5xx responses.
- Tries both EQ and ALL series forms for the official security-wise historical endpoint.
- Added current NSE official equity-universe ingestion from `https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv`.
- Added safe NSE transport diagnostics to the UI when no history is available.
- Interactive search does not sync freshly fetched NSE data to Google Drive before returning a result.
- Build marker updated to HOTFIX10.


NSE-FIRST ARCHITECTURE UPDATE
- Removed automatic /api/quote-equity startup health checks from Streamlit.
- Added official NSE CM-UDiFF bulk EOD ingestion path for scalable daily-universe coverage.
- Kept NSE official Securities available for Equity segment as the universe authority.
- Interactive history no longer downloads missing bulk days; scheduled ingestion/backfill owns that work.
- Direct NSE security-wise history remains an explicit primary path with bounded retries/timeouts.
- Dhan/Yahoo remain explicit fallbacks only.

NSE-FIRST ARCHITECTURE BUILD — 2026-09-29-NSE-FIRST11
- Removed /api/quote-equity from the normal Streamlit startup/health path.
- NSE official Securities available for Equity segment remains the universe authority.
- Added official NSE CM-UDiFF Common Bhavcopy bulk-EOD ingestion for scalable all-equity coverage.
- Interactive daily history reads cached official NSE bulk data without downloading missing days in the UI request path.
- Direct NSE security-wise history remains a bounded primary remote path.
- Dhan and Yahoo remain explicit fallbacks only.
- Added a scheduled GitHub Actions ingestion workflow with optional Google Drive mirroring.
- Added a Colab candidate builder for a broader-NSE next-session Delivery model; current three-stock production artifacts remain unchanged until the broader candidate passes validation.
