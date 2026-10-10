#!/usr/bin/env python3
"""Create a fresh Google Drive OAuth user token for GitHub Actions.

Run locally after downloading an OAuth 2.0 Client ID JSON of type "Desktop app"
from Google Cloud Console. The script writes the authorized-user JSON to a
permission-restricted file and never prints access/refresh tokens.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-secrets", required=True, help="Path to downloaded Desktop-app OAuth client JSON")
    parser.add_argument("--output", default="google_drive_oauth_token.json", help="Output path for the new authorized-user token JSON")
    args = parser.parse_args()

    client_path = Path(args.client_secrets).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not client_path.is_file():
        raise SystemExit(f"OAuth client JSON not found: {client_path}")

    flow = InstalledAppFlow.from_client_secrets_file(str(client_path), scopes=SCOPES)
    credentials = flow.run_local_server(
        host="localhost",
        port=0,
        access_type="offline",
        prompt="consent",
        include_granted_scopes="true",
        authorization_prompt_message="Open this URL in your browser to authorize Drive access: {url}",
        success_message="Google Drive authorization completed. You may close this browser tab.",
    )
    if not credentials.refresh_token:
        raise SystemExit(
            "Google did not return a refresh token. Re-run consent with prompt=consent "
            "and verify the OAuth client/app audience settings."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(credentials.to_json(), encoding="utf-8")
    try:
        os.chmod(output_path, 0o600)
    except OSError:
        pass

    print("OAuth authorization completed.")
    print(f"New token JSON saved to: {output_path}")
    print("A refresh token is present. No token value was printed.")
    print("Copy the file contents into the GitHub Actions secret GOOGLE_OAUTH_TOKEN_JSON; never commit or share this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
