#!/usr/bin/env bash
set -euo pipefail

# Deploy from the checked-out repository root. Required secrets must already
# exist in Google Secret Manager; this script never prints or embeds secret values.
: "${PROJECT_ID}"

REGION="${REGION:-asia-south1}"
SERVICE_NAME="${SERVICE_NAME:-indian-market-intraday-worker}"
RUN_SA_NAME="${RUN_SA_NAME:-indian-market-intraday-runtime}"
RUN_SA="${RUN_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

required_secrets=(
  "google-drive-root-id"
  "google-service-account-json"
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
  --project "${PROJECT_ID}"

if ! gcloud iam service-accounts describe "${RUN_SA}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${RUN_SA_NAME}" \
    --project "${PROJECT_ID}" \
    --display-name "Indian Market Intraday Worker"
fi

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:${RUN_SA}" \
  --role "roles/secretmanager.secretAccessor" \
  --quiet >/dev/null

for secret_name in "${required_secrets[@]}"; do
  if ! gcloud secrets describe "${secret_name}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
    echo "Required secret missing: ${secret_name}"
    echo "Create it in Secret Manager and add a version before deploying."
    exit 2
  fi
  if ! gcloud secrets versions access latest \
      --secret "${secret_name}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
    echo "Secret has no readable enabled version: ${secret_name}"
    exit 2
  fi
done

secret_mappings=(
  "GOOGLE_DRIVE_ROOT_ID=google-drive-root-id:latest"
  "GDRIVE_ROOT_FOLDER_ID=google-drive-root-id:latest"
  "GOOGLE_SERVICE_ACCOUNT_JSON=google-service-account-json:latest"
  "GOOGLE_OAUTH_TOKEN_JSON=google-oauth-token-json:latest"
)

# Broker credentials are optional. If the matching Secret Manager secrets exist,
# they are passed through; otherwise the worker tries configured NSE/fallback
# providers. Secret values are never read into shell variables or printed.
optional_secrets=(
  "INDSTOCKS_CLIENT_ID=indstocks-client-id"
  "INDSTOCKS_MPIN=indstocks-mpin"
  "INDSTOCKS_TOTP_SECRET=indstocks-totp-secret"
  "INDSTOCKS_ACCESS_TOKEN=indstocks-access-token"
)
for mapping in "${optional_secrets[@]}"; do
  env_name="${mapping%%=*}"
  secret_name="${mapping#*=}"
  if gcloud secrets describe "${secret_name}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
    secret_mappings+=("${env_name}=${secret_name}:latest")
  fi
done

secrets_arg="$(IFS=,; echo "${secret_mappings[*]}")"

gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --source . \
  --service-account "${RUN_SA}" \
  --port 8080 \
  --cpu 1 \
  --memory 2Gi \
  --min 1 \
  --max 1 \
  --concurrency 4 \
  --timeout 300 \
  --no-allow-unauthenticated \
  --no-cpu-throttling \
  --set-env-vars "AUTONOMOUS_INTRADAY_ROOT=/tmp/autonomous_intraday,AUTONOMOUS_INTRADAY_MAX_STOCKS=5,AUTONOMOUS_INTRADAY_INTERVAL_SECONDS=60,QUANT_TIMEFRAME_DIAGNOSTICS=1" \
  --set-secrets "${secrets_arg}"

echo
echo "Deployment command completed. Check startup logs for:"
echo "  google_drive_preflight"
echo "  cloud_run_worker_loop_started"
echo "  minute_tick_started"
echo "  cloud_run_tick_result"
echo
echo "This service uses one always-allocated Cloud Run instance. Check Google Cloud Billing"
echo "and remaining credits; costs can apply after the credit balance is exhausted."
