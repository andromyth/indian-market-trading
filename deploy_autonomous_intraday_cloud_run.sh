#!/usr/bin/env bash
set -euo pipefail

# Deploy from the repository root after required secrets are provisioned in
# Google Secret Manager. Secret values are never printed or embedded in files.
: "${PROJECT_ID}"

REGION="${REGION:-asia-south1}"
SERVICE_NAME="${SERVICE_NAME:-indian-market-intraday-worker}"
RUN_SA_NAME="${RUN_SA_NAME:-indian-market-intraday-runtime}"
SCHEDULER_SA_NAME="${SCHEDULER_SA_NAME:-indian-market-intraday-scheduler}"
RUN_SA="${RUN_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SCHEDULER_SA="${SCHEDULER_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

required_secrets=(
  "google-drive-root-id"
  "google-oauth-token-json"
)

echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Cloud Run service: ${SERVICE_NAME}"

gcloud config set project "${PROJECT_ID}" >/dev/null
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  cloudscheduler.googleapis.com \
  --project "${PROJECT_ID}"

ensure_service_account() {
  local name="$1"
  local email="$2"
  local display="$3"
  if ! gcloud iam service-accounts describe "$email" --project "$PROJECT_ID" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$name" \
      --project "$PROJECT_ID" \
      --display-name "$display"
  fi
}

ensure_service_account "$RUN_SA_NAME" "$RUN_SA" "Indian Market Intraday Runtime"
ensure_service_account "$SCHEDULER_SA_NAME" "$SCHEDULER_SA" "Indian Market Intraday Scheduler"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:${RUN_SA}" \
  --role "roles/secretmanager.secretAccessor" \
  --quiet >/dev/null

for secret_name in "${required_secrets[@]}"; do
  if ! gcloud secrets describe "$secret_name" --project "$PROJECT_ID" >/dev/null 2>&1; then
    echo "Required Secret Manager secret missing: $secret_name"
    echo "Create it and add its latest version before deploying."
    exit 2
  fi
  if ! gcloud secrets versions access latest \
      --secret "$secret_name" --project "$PROJECT_ID" >/dev/null 2>&1; then
    echo "Secret has no readable enabled version: $secret_name"
    exit 2
  fi
done

# OAuth user credentials are sufficient for My Drive read/write. An optional
# valid service-account key may be used for Drive read/list while OAuth handles writes.
secret_mappings=(
  "GOOGLE_DRIVE_ROOT_ID=google-drive-root-id:latest"
  "GDRIVE_ROOT_FOLDER_ID=google-drive-root-id:latest"
  "GOOGLE_OAUTH_TOKEN_JSON=google-oauth-token-json:latest"
)

# Do not fail deployment because an optional legacy service-account secret is
# malformed. Validate it silently before injecting it into the container.
if gcloud secrets describe "google-service-account-json" --project "$PROJECT_ID" >/dev/null 2>&1; then
  if gcloud secrets versions access latest \
      --secret "google-service-account-json" --project "$PROJECT_ID" 2>/dev/null \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if isinstance(d,dict) and d.get("type")=="service_account" and d.get("client_email") and d.get("private_key") else 1)' >/dev/null 2>&1; then
    secret_mappings+=("GOOGLE_SERVICE_ACCOUNT_JSON=google-service-account-json:latest")
  else
    echo "WARNING: Optional google-service-account-json is invalid or unreadable; using OAuth-only Google Drive access." >&2
  fi
fi
optional_secrets=(
  "INDSTOCKS_CLIENT_ID=indstocks-client-id"
  "INDSTOCKS_MPIN=indstocks-mpin"
  "INDSTOCKS_TOTP_SECRET=indstocks-totp-secret"
  "INDSTOCKS_ACCESS_TOKEN=indstocks-access-token"
)
for mapping in "${optional_secrets[@]}"; do
  env_name="${mapping%%=*}"
  secret_name="${mapping#*=}"
  if gcloud secrets describe "$secret_name" --project "$PROJECT_ID" >/dev/null 2>&1 \
      && gcloud secrets versions access latest --secret "$secret_name" --project "$PROJECT_ID" >/dev/null 2>&1; then
    secret_mappings+=("$env_name=$secret_name:latest")
  fi
done
secrets_arg="$(IFS=,; echo "${secret_mappings[*]}")"

# Cloud Scheduler calls /tick during the NSE session. Request-based billing and
# min-instances=0 let Cloud Run scale down overnight/weekends instead of charging
# for always-allocated CPU when the market is closed.
gcloud run deploy "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source . \
  --service-account "$RUN_SA" \
  --port 8080 \
  --cpu 1 \
  --memory 2Gi \
  --min 0 \
  --max 1 \
  --concurrency 1 \
  --timeout 300 \
  --no-allow-unauthenticated \
  --set-env-vars "AUTONOMOUS_INTRADAY_ROOT=/tmp/autonomous_intraday,AUTONOMOUS_INTRADAY_MAX_STOCKS=5,AUTONOMOUS_INTRADAY_INTERVAL_SECONDS=60,QUANT_TIMEFRAME_DIAGNOSTICS=1" \
  --set-secrets "$secrets_arg"

SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)')"
if [[ -z "$SERVICE_URL" ]]; then
  echo "Could not resolve Cloud Run service URL after deploy."
  exit 3
fi

# Grant the Cloud Scheduler identity permission to invoke this private service.
gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --member "serviceAccount:$SCHEDULER_SA" \
  --role "roles/run.invoker" \
  --quiet >/dev/null

# Cloud Scheduler jobs use OIDC. Allow the deploying identity to attach the
# scheduler service account when creating/updating jobs.
DEPLOYER_ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
if [[ -n "$DEPLOYER_ACCOUNT" && "$DEPLOYER_ACCOUNT" != "(unset)" ]]; then
  if [[ "$DEPLOYER_ACCOUNT" == *".gserviceaccount.com" ]]; then
    DEPLOYER_MEMBER="serviceAccount:$DEPLOYER_ACCOUNT"
  else
    DEPLOYER_MEMBER="user:$DEPLOYER_ACCOUNT"
  fi
  gcloud iam service-accounts add-iam-policy-binding "$SCHEDULER_SA" \
    --project "$PROJECT_ID" \
    --member "$DEPLOYER_MEMBER" \
    --role "roles/iam.serviceAccountUser" \
    --quiet >/dev/null
fi

upsert_scheduler_job() {
  local job_name="$1"
  local schedule="$2"
  local description="$3"
  local common_args=(
    --project "$PROJECT_ID"
    --location "$REGION"
    --schedule "$schedule"
    --time-zone "Asia/Kolkata"
    --uri "$SERVICE_URL/tick"
    --http-method POST
    --oidc-service-account-email "$SCHEDULER_SA"
    --oidc-token-audience "$SERVICE_URL"
    --attempt-deadline 180s
    --max-retry-attempts 0
  )

  if gcloud scheduler jobs describe "$job_name" \
      --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
    gcloud scheduler jobs update http "$job_name" "${common_args[@]}"
  else
    gcloud scheduler jobs create http "$job_name" "${common_args[@]}" \
      --description "$description"
  fi
}

upsert_scheduler_job \
  "intraday-worker-0915-0959" \
  "15-59 9 * * 1-5" \
  "Indian Market intraday worker, 09:15-09:59 Asia/Kolkata"
upsert_scheduler_job \
  "intraday-worker-1000-1459" \
  "* 10-14 * * 1-5" \
  "Indian Market intraday worker, 10:00-14:59 Asia/Kolkata"
upsert_scheduler_job \
  "intraday-worker-1500-1530" \
  "0-30 15 * * 1-5" \
  "Indian Market intraday worker, 15:00-15:30 Asia/Kolkata"

echo
echo "Cloud Run service deployed: $SERVICE_URL"
echo "Cloud Scheduler jobs configured in timezone Asia/Kolkata:"
echo "  09:15-09:59, 10:00-14:59, 15:00-15:30 on weekdays."
echo "NSE holiday/session guards run inside the worker; weekends and holidays do not fetch/predict."
echo "Check Cloud Run logs for google_drive_preflight, cloud_run_worker_api_started,"
echo "minute_tick_started, intraday_prediction_completed, and drive_flush_completed."
echo "Billing uses request-based Cloud Run calls and scheduler jobs; monitor project credits/billing."
