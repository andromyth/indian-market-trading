# Unattended intraday prediction service (Google Cloud Run)

This private Cloud Run worker is scheduled once per minute during regular NSE cash-market hours. It retains the existing prediction engine, checks freshness, and writes prediction snapshots and daily ledgers to Google Drive.

**Full-universe market observations are separate from forecasts.** Each tick also calls the read-only INDstocks full-quote endpoint in batches of up to 900 instruments and writes one Parquet row for every symbol in the official NSE equity universe, including explicit missing/unmapped quote rows. Minute snapshots are stored under `production_data/all_universe_minute_snapshots/YYYY/MM/DD/snapshot_YYYYMMDD_HHMM.parquet`; `latest_manifest.json` tracks expected/quoted/missing symbols, matched outcomes, and Drive persistence status. An actual forecast is attached only for symbols and horizons emitted by the existing model. The current forecast engine still focuses on its dynamic news-selected watchlist (five symbols by default); this change does not claim the locked classifier predicts all NSE stocks.

INDstocks full quotes provide a **collector-time live price snapshot**, not an exchange-timestamped one-minute OHLCV candle. Outcome matches are therefore explicitly labelled as observed target-minute quote matches. For true exchange-stamped minute-bar evaluation, use a validated one-minute candle feed and retain its source timestamp. No orders are placed.

## Live website-news collection

The dedicated `market_news_pit_collector.py` module runs through the scheduled GitHub Actions workflow every 15 minutes during NSE trading hours. It uses direct publisher RSS where dependable (Economic Times Markets, Mint Markets, Business Standard Markets) and publisher-filtered Google News RSS for Economic Times, Moneycontrol, Financial Express and CNBC-TV18. It records article title/summary, URL, publisher, publication and first-seen timestamps, sentiment, source health and point-in-time symbol matching. It does not scrape full articles or bypass paywalls.

Google Drive outputs:
- `processed/live_news/website_news_articles_2026.parquet` — deduplicated article archive.
- `processed/live_news/website_news_pit_2026.parquet` — rolling seven-day point-in-time article/symbol dataset.
- `processed/live_news/website_news_manifest_2026.json` — source health, last collection time, row counts and coverage status.

The intraday worker checks the small manifest each minute and reloads the PIT dataset before inference when a new collection is published. It attaches the derived per-symbol/market news context and transparent `NEWS_BUY_BIAS`, `NEWS_SELL_BIAS`, `NEWS_NEUTRAL`, or `NEWS_INSUFFICIENT_DATA` context to the minute-level forecast snapshot and prediction ledger. These news labels are context—not a substitute for the trained model's signal—and the locked classifier's numeric feature contract is unchanged. Incorporating news into model probabilities would require a separately validated feature-contract/training change.

A successful refresh means the configured feeds met the collector's source-health gate; it does **not** guarantee exhaustive coverage for every NSE stock. Missing stock-specific news must not be treated as proof that no news exists.

## Why Cloud Run rather than an all-day GitHub Actions job?

The repository is private. An all-day GitHub-hosted worker job every trading day would consume substantial included/billed runner minutes. Cloud Run is configured with one instance, CPU always allocated, and a durable Google Drive output path. Google documents that background threads need at least one minimum instance and that instance-based billing charges for the full instance lifecycle, including idle time. Credits can cover this temporarily; it is not guaranteed free after they expire.

References:
- https://docs.cloud.google.com/run/docs/about-instance-autoscaling
- https://docs.cloud.google.com/run/docs/configuring/billing-settings
- https://cloud.google.com/run/pricing

## Prerequisites

1. A Google Cloud project with billing enabled and sufficient remaining credits.
2. The gcloud CLI installed and authenticated as an account that can deploy Cloud Run, enable APIs, create service accounts, grant IAM roles, and use Secret Manager.
3. The current Google Drive root and write credentials already used by the repository's production workflows.

## 1. Transfer existing credentials into Secret Manager

Open Google Cloud Console → **Security → Secret Manager**. Create the required secrets below and add their latest versions. Copy the values from the existing GitHub repository's **Settings → Secrets and variables → Actions** or the existing Streamlit Cloud Secrets; do not commit credentials into files, notebooks, or logs.

| Secret Manager name | Required? | Value |
|---|---|---|
| google-drive-root-id | Yes | The existing GOOGLE_DRIVE_ROOT_ID folder ID |
| google-oauth-token-json | Yes | Full GOOGLE_OAUTH_TOKEN_JSON JSON, including its refresh token |
| google-service-account-json | Optional | Full GOOGLE_SERVICE_ACCOUNT_JSON service-account key JSON |

The worker can use the OAuth user token for both Google Drive reads and writes. A valid service-account JSON is optional; when present and valid, it can be used for Drive read/list operations while OAuth handles writes to the owner's My Drive. The deployment script validates this optional secret silently and falls back to OAuth-only access if it is missing or malformed.

The deployment explicitly sets `INDSTOCKS_AUTH_MODE=totp` only on Cloud Run when all three TOTP secrets are present. Streamlit is forced to `manual` mode in `app.py`, so an existing dashboard token can remain configured without triggering token generation or replacing Cloud Run's token.

The deployment preflight checks actual Google Drive read/write access and refuses to run if it fails. JSON formatting alone is not proof that the account can access the configured Drive folder.

For the required full-universe minute-quote archive, provision **either** a valid `indstocks-access-token` secret (manual token, expires after 24 hours) **or all three** secrets `indstocks-client-id`, `indstocks-mpin` and `indstocks-totp-secret` for automatic read-only authentication. The deploy script maps these secrets into the Cloud Run worker when they exist. Without an available INDstocks authentication method, full-universe snapshots will be marked `BLOCKED_PROVIDER_CREDENTIALS` and the service health will be degraded; a collection attempt is not proof that Drive data was written.

Do not expose these values in terminal output. Never paste credentials into GitHub source files or chat.

## 2. Deploy the worker

From a trusted local checkout of this repository on a machine with gcloud authenticated, update to main after this PR is merged, then run from the repository root:

~~~bash
git fetch origin main
git switch main
git pull --ff-only origin main

export PROJECT_ID="YOUR_GOOGLE_CLOUD_PROJECT_ID"
export REGION="asia-south1"
bash deploy_autonomous_intraday_cloud_run.sh
~~~

The script enables the required APIs, creates a runtime service account, grants it access to Secret Manager, builds the root Dockerfile, and deploys a private Cloud Run service with one instance (min=1, max=1), one vCPU, 2 GiB memory, and CPU allocated outside requests. The service runs its own background loop; **no Cloud Scheduler and no open browser are required**.

The service is intentionally private (--no-allow-unauthenticated). Do not make it public just to view its health endpoint.

## 3. Verify it is alive

Open Google Cloud Console → **Cloud Run** → indian-market-intraday-worker → **Logs**. Check for these structured log events:

1. google_drive_preflight with configured=true, reachable=true, and write_ready=true.
2. cloud_run_worker_loop_started.
3. During an NSE session: minute_tick_started, stock_data_collected, intraday_prediction_completed, and cloud_run_tick_result.
4. Every tick: `all_universe_capture_status=OK` plus a snapshot path under `production_data/all_universe_minute_snapshots/YYYY/MM/DD/`. Confirm `quoted_symbol_count` equals `expected_symbol_count`; `PARTIAL`, `PERSISTENCE_ERROR` or `BLOCKED_PROVIDER_CREDENTIALS` indicates incomplete coverage.
5. Check `latest_manifest.json`, each `manifest_YYYYMMDD_HHMM.json`, and the snapshot Parquet row count for each minute. Rows whose quote status is not PASS are explicit gaps, not valid prices.
6. The daily workflow also audits 375 regular-session minute slots and stores JSON/CSV results under `production_data/all_universe_minute_snapshots/audits/YYYY/MM/DD/`. A strict audit fails if minute snapshots/manifests are missing, duplicated, incomplete, or have quote gaps.
7. The separate per-minute prediction snapshot remains under `production_data/intraday_minute_snapshots/YYYY/MM/DD/`; the forecast model itself is still watchlist scoped.
8. After market close, review forecast outcome reports under `production_data/intraday_predictions/outcomes_audit/YYYY/MM/DD/`. Target-minute quote matches in the full-universe archive are not the same as exchange-stamped one-minute candle closes.


A weekend, NSE holiday, pre-open or post-close log should show MARKET_CLOSED/no market data cycle. Stale/missing stock or benchmark candles cause the affected prediction to be withheld rather than guessed.

## Prediction learning and error diagnosis

The separate `prediction_learning_engine.py` module runs at **16:30 IST on weekdays** through `.github/workflows/autonomous_prediction_learning.yml`, after the existing 16:00 IST outcomes audit. It reads persisted forecast ledgers and matching actual one-minute bar partitions, de-duplicates repeated forecast identities, and produces diagnosis slices by symbol, horizon, active model version, signal, input source, feature coverage and time of day.

It tests probability calibration and expected-return bias corrections with an earlier-time training segment and a later, untouched holdout. Corrections are saved as review-only candidate profiles under `production_data/intraday_predictions/learning/YYYY/MM/DD/`; no candidate is automatically applied to the live model or signal thresholds. The report includes outcome coverage gaps, duplicate records, stale input age, poor-performing segments, probability calibration and price-forecast MAE versus the no-change baseline.

New intraday inference rows also store the exact numeric model inputs in `diagnostic_feature_snapshot`, with missing/non-finite values encoded as null, together with the champion's bundle/model version. This enables association analysis between feature values and probability/return residuals after target bars mature. Those associations are diagnostic evidence, not causal proof. Since this instrumentation is code running in Cloud Run, **redeploy the updated service** before expecting new forecast records to contain it. Historical predictions that did not store exact feature values cannot be reconstructed reliably after the fact.

Learning outputs include a detailed matured-outcome CSV, segment-metrics CSV, feature-error-associations CSV, JSON diagnostic report and candidate-corrections JSON. The candidate profiles require subsequent shadow testing and an explicit promotion decision before any production use.

## Important constraints

- One minute is a polling target, not an exchange-feed latency guarantee. The loop can take longer than one minute if an upstream source or model artifact download is slow; it never fabricates bars to catch up.
- If the service restarts, it rebuilds its in-memory watchlist from recent headlines and resumes. The daily prediction ledger is hydrated from Drive before appending so a cold instance does not overwrite earlier same-day records. A per-minute prediction/actual snapshot is written to Drive on each completed tick; completed bars, the daily ledger and worker status are additionally synced every five cycles and on shutdown.\n- The GitHub Actions workflow `Autonomous intraday prediction-vs-actual audit` runs at 16:00 IST on weekdays, skips NSE holidays, and uploads the daily audit CSV/JSON after joining persisted forecasts to actual one-minute closes. Outcomes are finalized only when the forecast horizon has elapsed and a target bar is available.
- Existing trained classifier bundles must have their price-regression heads fitted once for numerical predicted prices to appear. Use COLAB_SD_INTRADAY_RETURN_HEAD_RETRAIN.py once if the active horizon bundles still report RETRAIN_REQUIRED.
- Cloud Run minimum-instance and instance-based CPU settings are billable. Monitor the Google Cloud credits/billing page and scale down/delete the service if you no longer need it.


## Market-news collection and PIT integration

The `Market news PIT collector` GitHub Actions workflow runs every 15 minutes from 08:50 through 14:50 IST, plus a final 15:00 IST refresh, on weekdays; the Python market-session guard skips weekends and NSE holidays. It reads publisher RSS where available and publisher-filtered Google News RSS for additional financial publishers. It stores publisher-attributed headline/short-summary metadata, first-seen time, point-in-time availability, feed health, finance-oriented event category, directional score, news-bias label, explanation and event-impact context beneath `processed/live_news/` in Google Drive.

Website headline rows are compiled into Delivery's point-in-time panel as separate `web_news_available`, `news_direction_score`, `event_impact_score` and sentiment/source context fields. They do not overwrite the locked classifier's existing `news_available` feature, which retains its training-time official-announcement semantics. Feed silence is not treated as proof that no other news exists, and this workflow does not claim corporate-event coverage: the separate NSE event readiness gate remains strict. The intraday worker adds a separate `NEWS_BUY_BIAS`, `NEWS_SELL_BIAS`, `NEWS_NEUTRAL` or `NEWS_INSUFFICIENT_DATA` overlay using finance phrase cues, VADER sentiment, event impact, recency and independent-source agreement. Its strength score is heuristic, not a calibrated probability or order instruction. The deployed Intraday classifier was trained without news features, so its original BUY/SELL/ABSTAIN output remains unchanged; both outputs are logged for outcome-based validation before any fusion into the production model.
