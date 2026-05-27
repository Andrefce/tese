#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# CardioSDF — Deploy to Google Cloud Run
# ═══════════════════════════════════════════════════════════════════
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated
#   2. A GCP project with Cloud Run, Artifact Registry, and Cloud Storage APIs enabled
#   3. The model checkpoint at inference-api/model/inr_sdf_combined_fresh_ed_mix_v1_final.ptrom
#      (copy from webapp/model/ before building)
#
# Usage:
#   export GCP_PROJECT=your-project-id
#   export GCP_REGION=europe-west1        # or your preferred region
#   bash deploy.sh
#
set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud CLI is required but was not found in PATH."
  exit 127
fi

PROJECT="${GCP_PROJECT:-cardiosdf}"
REGION="${GCP_REGION:-europe-west1}"
BUCKET_NAME="${GCS_BUCKET_NAME:-cardiosdf-results}"
REPO="cardiosdf"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEBAPP_DIR="$SCRIPT_DIR"
API_DIR="$SCRIPT_DIR/inference-api"

echo "=== CardioSDF Deployment ==="
echo "Project:  $PROJECT"
echo "Region:   $REGION"
echo "Bucket:   $BUCKET_NAME"
echo ""

REGISTRY="$REGION-docker.pkg.dev/$PROJECT/$REPO"

# ── 1. One-time infra setup (idempotent, fast to skip) ──
echo ">>> Ensuring Artifact Registry repo '$REPO' exists ..."
gcloud artifacts repositories describe "$REPO" \
  --location="$REGION" --project="$PROJECT" >/dev/null 2>&1 \
|| gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --project="$PROJECT"

echo ">>> Ensuring GCS bucket gs://$BUCKET_NAME exists ..."
gcloud storage buckets describe "gs://$BUCKET_NAME" --project "$PROJECT" >/dev/null 2>&1 \
|| gcloud storage buckets create "gs://$BUCKET_NAME" --location "$REGION" --project "$PROJECT"

# CORS / lifecycle / public-read (cheap calls, always sync)
gcloud storage buckets update "gs://$BUCKET_NAME" \
  --cors-file "$SCRIPT_DIR/infrastructure/gcs-cors.json" \
  --project "$PROJECT" --quiet
gcloud storage buckets update "gs://$BUCKET_NAME" \
  --lifecycle-file "$SCRIPT_DIR/infrastructure/gcs-lifecycle.json" \
  --project "$PROJECT" --quiet
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET_NAME" \
  --quiet \
  --member="allUsers" \
  --role="roles/storage.objectViewer" \
  --project "$PROJECT" 2>/dev/null || true

# ── 2. Copy model checkpoint to inference-api build context ──
MODEL_SRC="$WEBAPP_DIR/model"
MODEL_DST="$API_DIR/model"
if [ ! -d "$MODEL_DST" ]; then
  echo ">>> Copying model checkpoint to inference-api/model/ ..."
  cp -r "$MODEL_SRC" "$MODEL_DST"
fi

# ── 3. Build BOTH images in parallel using Cloud Build ──
echo ""
echo ">>> Submitting inference API build ..."
gcloud builds submit "$API_DIR" \
  --tag "$REGISTRY/cardiosdf-inference:latest" \
  --project "$PROJECT" \
  --timeout=1800 \
  --async \
  --format='value(name)' > /tmp/_cbuild_inference.txt

echo ">>> Submitting webapp build ..."
gcloud builds submit "$WEBAPP_DIR" \
  --tag "$REGISTRY/cardiosdf-webapp:latest" \
  --project "$PROJECT" \
  --timeout=600 \
  --async \
  --format='value(name)' > /tmp/_cbuild_webapp.txt

INFERENCE_BUILD=$(cat /tmp/_cbuild_inference.txt)
WEBAPP_BUILD=$(cat /tmp/_cbuild_webapp.txt)

echo "Inference build: $INFERENCE_BUILD"
echo "Webapp build:    $WEBAPP_BUILD"

# Wait for both builds to complete
echo ">>> Waiting for both Cloud Builds to finish (this runs in parallel) ..."
wait_build() {
  local build_id="$1"
  local label="$2"
  while true; do
    STATUS=$(gcloud builds describe "$build_id" --project "$PROJECT" --format='value(status)')
    case "$STATUS" in
      SUCCESS) echo "  ✓ $label build succeeded"; return 0 ;;
      FAILURE|CANCELLED|TIMEOUT|INTERNAL_ERROR)
        echo "  ✗ $label build FAILED (status=$STATUS)"
        gcloud builds log "$build_id" --project "$PROJECT" | tail -40
        return 1 ;;
      *) echo "  … $label build $STATUS — waiting 15s ..."; sleep 15 ;;
    esac
  done
}
wait_build "$INFERENCE_BUILD" "inference-api" &
wait_build "$WEBAPP_BUILD"    "webapp" &
wait  # wait for both background waits

# ── 4. Deploy inference API ──
echo ""
echo ">>> Deploying inference API to Cloud Run ..."
gcloud run deploy cardiosdf-inference \
  --image "$REGISTRY/cardiosdf-inference:latest" \
  --region "$REGION" \
  --project "$PROJECT" \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --memory 16Gi \
  --cpu 4 \
  --timeout 300 \
  --max-instances 2 \
  --min-instances 1 \
  --no-cpu-throttling \
  --no-allow-unauthenticated \
  --set-env-vars "GCS_BUCKET_NAME=$BUCKET_NAME"

INFERENCE_URL=$(gcloud run services describe cardiosdf-inference \
  --region "$REGION" --project "$PROJECT" \
  --format 'value(status.url)')
echo "Inference API URL: $INFERENCE_URL"

# ── 5. Deploy webapp ──
echo ""
echo ">>> Deploying webapp to Cloud Run ..."
gcloud run deploy cardiosdf-webapp \
  --image "$REGISTRY/cardiosdf-webapp:latest" \
  --region "$REGION" \
  --project "$PROJECT" \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --max-instances 5 \
  --allow-unauthenticated \
  --set-env-vars "INFERENCE_API_URL=$INFERENCE_URL,GCS_BUCKET_NAME=$BUCKET_NAME"

# ── 6. Grant webapp SA permission to call inference API ──
WEBAPP_SA=$(gcloud run services describe cardiosdf-webapp \
  --region "$REGION" --project "$PROJECT" \
  --format 'value(spec.template.spec.serviceAccountName)')
if [ -z "$WEBAPP_SA" ]; then
  PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
  WEBAPP_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
fi

echo ">>> Granting webapp SA ($WEBAPP_SA) invoker role on inference API ..."
gcloud run services add-iam-policy-binding cardiosdf-inference \
  --region "$REGION" \
  --project "$PROJECT" \
  --member "serviceAccount:$WEBAPP_SA" \
  --role "roles/run.invoker" 2>/dev/null || true

# ── Done ──
WEBAPP_URL=$(gcloud run services describe cardiosdf-webapp \
  --region "$REGION" --project "$PROJECT" \
  --format 'value(status.url)')

echo ""
echo "=== Deployment Complete ==="
echo "Webapp:        $WEBAPP_URL"
echo "Inference API: $INFERENCE_URL  (internal, not public)"
echo "GCS Bucket:    gs://$BUCKET_NAME"
