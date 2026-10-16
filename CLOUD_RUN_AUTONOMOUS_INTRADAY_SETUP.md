# Unattended intraday prediction service (Google Cloud Run)

This deploys an unattended Cloud Run worker, so prediction continues when the Streamlit dashboard/browser is closed. It polls once per minute during the regular NSE cash session, refreshes the dynamic news-based watchlist every 15 minutes, uses completed one-minute bars, checks stock/benchmark freshness, calls the existing saved Beta V2 horizons, and writes an immutable minute-level snapshot containing observed prices and the predictions made at that time. The full daily prediction ledger and bar partitions are also synced to Google Drive.

It deliberately scores the top five news-matched watchlist symbols by default, not the entire NSE universe each minute. The existing hourly full-universe ingestion workflow remains separate. No orders are placed.

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

The deployment preflight checks actual Google Drive read/write access and refuses to run if it fails. JSON formatting alone is not proof that the account can access the configured Drive folder.

Optional broker credentials can be added to Secret Manager using these names to enable the authenticated INDstocks source:
- indstocks-client-id
- indstocks-mpin
- indstocks-totp-secret
- indstocks-access-token (manual token fallback)

If optional broker secrets are absent, the worker still tries the configured NSE/historical fallbacks. Do not expose these values in terminal output.

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
4. Every tick: minute_prediction_actual_snapshot_persisted with status OK and a path under production_data/intraday_minute_snapshots/YYYY/MM/DD/. Each snapshot contains the latest observed stock/benchmark prices and that minute's forecast rows.\n5. Every five cycles: drive_flush_completed with status OK and the daily prediction/bar partitions synced to Google Drive.\n6. After market close: the scheduled Autonomous intraday prediction-vs-actual audit validates matured forecasts and writes CSV/JSON reports under production_data/intraday_predictions/outcomes_audit/YYYY/MM/DD/.

A weekend, NSE holiday, pre-open or post-close log should show MARKET_CLOSED/no market data cycle. Stale/missing stock or benchmark candles cause the affected prediction to be withheld rather than guessed.

## Important constraints

- One minute is a polling target, not an exchange-feed latency guarantee. The loop can take longer than one minute if an upstream source or model artifact download is slow; it never fabricates bars to catch up.
- If the service restarts, it rebuilds its in-memory watchlist from recent headlines and resumes. The daily prediction ledger is hydrated from Drive before appending so a cold instance does not overwrite earlier same-day records. A per-minute prediction/actual snapshot is written to Drive on each completed tick; completed bars, the daily ledger and worker status are additionally synced every five cycles and on shutdown.\n- The GitHub Actions workflow `Autonomous intraday prediction-vs-actual audit` runs at 16:00 IST on weekdays, skips NSE holidays, and uploads the daily audit CSV/JSON after joining persisted forecasts to actual one-minute closes. Outcomes are finalized only when the forecast horizon has elapsed and a target bar is available.
- Existing trained classifier bundles must have their price-regression heads fitted once for numerical predicted prices to appear. Use COLAB_SD_INTRADAY_RETURN_HEAD_RETRAIN.py once if the active horizon bundles still report RETRAIN_REQUIRED.
- Cloud Run minimum-instance and instance-based CPU settings are billable. Monitor the Google Cloud credits/billing page and scale down/delete the service if you no longer need it.
