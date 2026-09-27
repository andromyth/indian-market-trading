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
