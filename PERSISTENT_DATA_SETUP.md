# Permanent 1-minute historical store

The application treats Streamlit Cloud storage as an ephemeral cache. The system-of-record for captured 1-minute data is the persistent store under:

`data/persistent_market_data/intraday_1m/YYYY/MM/DD/SYMBOL.parquet`

Each trading day is a separate partition. New bars are merged by timestamp so repeated downloads do not create duplicate observations. Historical trading days are not deleted by normal capture/sync operations.

## Local mode

The application writes the daily partitions locally and records them in:

`data/persistent_market_data/manifest.json`

This is useful for local runs and as a cache on Streamlit Cloud, but Streamlit Cloud's filesystem is not the durable system-of-record.

## Google Drive mode (recommended for Streamlit Cloud)

1. Create a Google Cloud project.
2. Enable the Google Drive API.
3. Create a service account and download its JSON key.
4. Create a normal Google Drive folder that will be the root of the trading-data archive.
5. Share that folder with the service-account `client_email` as Editor.
6. Copy the folder ID from the Drive URL.
7. In Streamlit Cloud, open the existing app's settings -> Secrets.
8. Paste the structure from `STREAMLIT_SECRETS_GDRIVE.example.toml`, replacing the placeholders with the real values.
9. Restart/redeploy the existing app.

The app then stores persistent partitions under:

`production_data/intraday_1m/YYYY/MM/DD/SYMBOL.parquet`

and maintains a local manifest with the Drive file IDs and hashes.

## What is automatically persisted

- Dhan 1-minute downloads are copied into the persistent daily partitions.
- Current-year free 1-minute archive downloads are partitioned into the persistent store.
- The `Capture current 1-minute data` control appends the latest 1-minute bars to the persistent store.
- The `Sync local 1-minute archive to Google Drive` control mirrors local persistent partitions to Drive.

Historical replay reads the persistent store first, then the older local free-data cache, then the public fallback where permitted.

## Important limitation

This revision adds persistent capture and storage infrastructure. It does not yet create a true autonomous one-minute scheduler on Streamlit Cloud. The next live-intraday revision can add a timed refresh/capture loop once the persistent store is configured and verified.
