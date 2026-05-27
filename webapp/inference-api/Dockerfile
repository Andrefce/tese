# ══════════════════════════════════════════════════════════════════
# CardioSDF Inference API — Multi-stage build
# ══════════════════════════════════════════════════════════════════

# ── Stage 1: Install Python dependencies (cached layer) ──
# pytorch/pytorch already ships CUDA 12.1 + Python 3.11 + torch — no re-download needed
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS deps

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install only the non-torch dependencies (fast — no 2 GB download)
COPY requirements.txt .
RUN uv pip install --system \
        fastapi "uvicorn[standard]" python-multipart \
        numpy nibabel scipy scikit-image \
        google-cloud-storage

# ── Stage 2: Copy model (separate layer — large but rarely changes) ──
FROM deps AS model
COPY model/ model/

# ── Stage 3: Final image with application code ──
FROM model AS final

# Copy application code (changes most frequently — smallest layer)
COPY core/ core/
COPY main.py .

ENV MODEL_PATH=/app/model/inr_sdf_combined_fresh_ed_mix_v1_final.ptrom \
    GCS_BUCKET_NAME=cardiosdf-results

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
