# Google Drive upload authentication recovery

The project stores its canonical market archive under the existing Indian_Market_Research folder in My Drive. The service account can read that folder, but Google Drive service accounts have no personal storage quota and cannot upload owned files to My Drive. The integration now keeps the service account for read/list operations and uses a user OAuth token for writes.

## 1. Download an OAuth desktop-client file

1. Open Google Cloud Console and select the same Cloud project used by this application.
2. Open APIs & Services → Credentials.
3. Under OAuth 2.0 Client IDs, select the existing Desktop app client. If the existing client is a Web application client, create a Desktop app OAuth client for this local authorization helper.
4. Download its client JSON file to your computer. Do not put this file in the Git repository.

Google Drive API must remain enabled.

## 2. Reauthorize your Google account locally

From the repository directory, install the helper dependency if needed:

    python -m pip install google-auth-oauthlib google-auth google-api-python-client

Run the helper using the downloaded OAuth client JSON path:

    python google_drive_oauth_login.py --client-secrets ~/Downloads/client_secret.json --output google_drive_oauth_token.json

Replace the sample path with the actual downloaded filename. A browser will open. Sign into the Google account that owns the Indian_Market_Research folder and approve Drive access. The helper writes a new authorized-user token file locally and does not print token values.

## 3. Replace the GitHub Actions secret

Open repository Settings → Secrets and variables → Actions:
https://github.com/andromyth/indian-market-trading/settings/secrets/actions

Replace the value of GOOGLE_OAUTH_TOKEN_JSON with the full contents of google_drive_oauth_token.json. Keep GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_DRIVE_ROOT_ID unchanged. Do not add the OAuth token to Streamlit source code or commit either JSON file. The repository .gitignore already ignores the helper output and client JSON filenames.

## 4. Ensure the token will not expire after seven days

In Google Cloud Console, open Google Auth Platform → Audience and check the app's Publishing status. Google's documentation states that refresh tokens for external apps left in Testing expire after seven days when scopes beyond basic identity are requested. For unattended operation, review the project's publishing/verification requirements and avoid leaving this Drive-scoped app in Testing indefinitely:
https://support.google.com/cloud/answer/15549945

If Google requires app verification for your configuration, follow Google's published requirements for the requested Drive scope. Do not bypass OAuth consent or share token values.

## 5. Rerun the market-data refresh

Open the NSE Delivery production cycle:
https://github.com/andromyth/indian-market-trading/actions/workflows/nse_delivery_production_cycle.yml

Run it on main with target date 2026-10-09. The first step now validates both read authentication and the OAuth write identity before fetching new data. A healthy result should report the service account as the read identity and OAuth as the write identity. After that, NSE Delivery files and the point-in-time snapshot can be uploaded to the same existing Drive folder.

## Why the service account alone is not enough

The current Drive root metadata reports driveId as null, meaning the folder is in My Drive rather than a Shared Drive. Google documents that service accounts have no storage quota; uploads must use OAuth on behalf of a human user or a Shared Drive available under a supported Google Workspace edition:
https://developers.google.com/workspace/drive/api/guides/about-shareddrives
