# INDstocks API setup (read-only market data)

The application uses INDstocks for authenticated quotes and one-minute OHLCV candles when configured. Existing official NSE, local/Google Drive archive, Dhan (if configured), and Yahoo fallback paths remain in place. The application has no order-placement implementation.

## Option A — first connection check with the generated access token

In Streamlit Community Cloud, open your app → Settings → Secrets and add:

    INDSTOCKS_ACCESS_TOKEN = "PASTE_YOUR_GENERATED_ACCESS_TOKEN"
    INDSTOCKS_TIMEOUT_SECONDS = "12"

Save secrets and restart/reboot the app. Never paste the token into chat, source code, GitHub files, screenshots, or logs. The token is treated as a password and expires after 24 hours, so this option is only suitable for an initial connection test or manual renewal.

## Option B — unattended access-token renewal (recommended for autonomy)

First, open the official INDstocks API access page:

https://www.indstocks.com/app/api-trading/access-tokens

Complete the site's Setup TOTP flow. Securely save the TOTP secret when it is shown; it is not retrievable later. Record the Client ID shown after setup and use your INDstocks account MPIN. Then replace the manual-token setting in Streamlit Secrets with:

    INDSTOCKS_CLIENT_ID = "YOUR_CLIENT_ID"
    INDSTOCKS_MPIN = "YOUR_ACCOUNT_MPIN"
    INDSTOCKS_TOTP_SECRET = "YOUR_BASE32_TOTP_SECRET"
    INDSTOCKS_TIMEOUT_SECONDS = "12"

Do not keep the token-generation secret in GitHub. The app computes the six-digit TOTP locally and calls the documented token-generation endpoint. The client caches the resulting token in process memory and refreshes it before the 24-hour expiry; it does not write credentials or tokens to disk.

**Token ownership:** the Streamlit dashboard is forced into manual-token mode and must not mint TOTP tokens. Its existing `INDSTOCKS_ACCESS_TOKEN` setting is left untouched. Cloud Run explicitly uses `INDSTOCKS_AUTH_MODE=totp` and is the only deployed process allowed to generate tokens. In every other process, the default mode is `manual`; merely having TOTP secrets present cannot trigger token generation.

Important token constraint: INDstocks documents that only one TOTP-generated token can be live per account. A successful token-generation call invalidates the previous TOTP token. Do not configure multiple independent applications to generate tokens for the same account; use one application/process to own token generation.

## Verify the connection

From a local terminal where credentials are supplied as environment variables, run:

    python indstocks_smoke_test.py

The test checks API authentication without printing profile details, requests one live quote for RELIANCE, and retrieves recent one-minute candles. It reports success/failure and the row count, not your credentials.

If testing inside Streamlit Community Cloud, configure Secrets first; the production dashboard automatically uses the provider on its next run. Confirm the data source label shows INDSTOCKS_API for the live one-minute feed while the market is open. Outside market hours, the quote may be the last available quote; do not interpret it as a live tradable price without checking its timestamp/market status.

## Static IP

INDstocks' current documentation says a whitelisted static IP is required for order placement, modification, and cancellation. Read-only quotes, historical data, profile, and funds endpoints do not require whitelisting. This application uses only read-only market-data endpoints; no orders are sent.

## Scope and limitations

- The one-minute historical endpoint supports a maximum seven-day window per API call; this adapter requests up to seven days for the live intraday model.
- The broker's daily, weekly and monthly history endpoint supports up to one year per call. Long-term bulk research should continue to use the project's existing persistent historical archive and official NSE ingestion.
- A broker quote or candle is not by itself a validated prediction. Prediction quality still depends on feature readiness, the existing locked model/feature contract, correct timestamps, and out-of-sample validation.
- This integration is a data-source change only. It does not create a new model version, alter trained model weights, or enable automated orders.
