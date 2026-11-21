# INDstocks API access — Cloud Run only

INDstocks is a read-only market-data source. **The singleton Cloud Run service is the only process permitted to authenticate to or call INDstocks.** Streamlit, local scripts, notebooks and GitHub Actions must not access the provider or generate/refresh its token.

## Cloud Run credentials

Create these Google Cloud Secret Manager secrets:

| Secret Manager name | Value |
|---|---|
| `indstocks-client-id` | INDstocks Client ID |
| `indstocks-mpin` | Account MPIN |
| `indstocks-totp-secret` | Base32 TOTP secret |

The deployment maps these secrets only into `indian-market-intraday-worker` and sets `INDSTOCKS_AUTH_MODE=totp`. Manual access-token secrets are not supported by the deployment.

Cloud Run is configured with **one maximum instance and concurrency one**. The INDstocks client also serializes token generation in-process and applies a cooldown after token-generation failures. The `K_SERVICE` runtime guard rejects INDstocks access outside Cloud Run, even if a process happens to contain copied credentials.

Do not put INDstocks credentials or tokens in GitHub repository secrets, Streamlit Secrets, local environment files, notebooks, or source code. Do not run `indstocks_smoke_test.py` locally. A second login can invalidate the active account token.

## Data flow

1. **Cloud Run** owns the INDstocks token and collects five-minute full-universe quote snapshots. It runs the resumable 5-minute historical backfill only through the authenticated private Cloud Run endpoint, sharing the same process-local token cache.
2. Cloud Run persists quote snapshots and symbol-wise five-minute history Parquet files to Google Drive.
3. **GitHub Actions** reads those Drive artifacts, compiles the existing 30-minute ALL-NSE model features, loads the existing model/F&O context from Drive, and writes prediction Parquet files and manifests back to Drive. It does not import or call INDstocks.
4. The news collector is a separate GitHub Actions workflow. It reads publisher RSS and writes the point-in-time news archive to Drive. News is optional context; absence of a matching news headline must not block the independent ALL-NSE prediction pipeline.
5. Streamlit is a read-only dashboard and uses other configured market-data fallbacks. INDstocks access is disabled in the dashboard.

Prediction output:
`production_data/all_nse_minute_predictions/YYYY/MM/DD/predictions_YYYYMMDD_HHMM.parquet`

Source quote snapshots:
`production_data/all_universe_minute_snapshots/YYYY/MM/DD/snapshot_YYYYMMDD_HHMM.parquet`

## Recovery rules

- If Cloud Run reports an INDstocks token-generation timeout or rejection, inspect Cloud Run logs and the Secret Manager TOTP configuration. Do not generate another token from a second process.
- If the provider remains unavailable, Cloud Run must record the capture as blocked/degraded; GitHub must not invent quotes or candles and must not generate predictions from a stale snapshot.
- The GitHub compiler skips a slot when its Drive snapshot is missing, not persisted, or older than its freshness limit.
- A stored prediction file is not proof of a complete feature row. Inspect `PREDICTED` versus `PREDICTED_DATA_LIMITED` counts and feature coverage.

## Provider limits and scope

- Historical INDstocks requests are issued only by Cloud Run.
- Retraining notebooks (Colab/Kaggle) read the merged NIFTY 5m artifact and per-symbol Parquets from Drive. They do not import `indstocks_client` or hold a token; see `INTEGRATION.md` section 0.
- The native five-minute backfill is resumable from the per-symbol Drive checkpoints and retains the 200–500 usable-session target.
- This integration is read-only. It does not place, modify, or cancel orders.
