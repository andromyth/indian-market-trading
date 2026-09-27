# Free browser deployment

This is the clean application package for Streamlit Community Cloud.

## Upload to GitHub

Upload the contents of this directory to a **private GitHub repository**.
Do not upload `.env`, credentials, `data/dhan/auth.json`, `.venv`, `__pycache__`, research datasets, model checkpoints, or generated reports.

## Streamlit Secrets

For Dhan live data, configure these secrets in the Streamlit app settings:

```toml
DHAN_CLIENT_ID = ""
DHAN_ACCESS_TOKEN = ""
DHAN_TIMEOUT_SECONDS = "30"
```

`DHAN_API_KEY` and `DHAN_API_SECRET` should only be configured if you have a separate secure workflow for obtaining the token. The hosted dashboard intentionally does not launch the local OAuth callback.

If Dhan is not configured, the dashboard uses its existing free fallback sources.

## Deploy

In Streamlit Community Cloud: Create app -> select the private GitHub repository -> select `main` -> choose `app.py` -> Deploy.

## Important

The application is decision-support software and does not place orders. Hosted storage is not a replacement for Google Drive. Research/model archives should remain in Google Drive and should not be copied into the Git repository.

## Historical Replay & Audit

The sidebar now includes **Historical Replay**. Select a past date, a symbol, and Delivery, Intraday, or F&O-underlying replay. The app reconstructs the selected timestamp from historical data, runs the relevant production/rule engine, and compares the result with subsequent actual prices.

Intraday uses the certified `target_30m` 137-feature production model. Delivery uses the existing causal rule engine with a selectable forward holding horizon. The replay never injects today's news or F&O chain into a past prediction. Timestamped historical news is best-effort from public Google News RSS search; historical option-chain/OI/IV replay is explicitly not substituted and the F&O option is labelled as an underlying replay.

The replay ledger can be exported to CSV from the dashboard for later prediction-vs-actual analysis.
