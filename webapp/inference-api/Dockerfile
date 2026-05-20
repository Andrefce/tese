# ── Stage 1: Install Python dependencies ──
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Python dependencies
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# ── Stage 2: Copy application code + model ──
COPY core/ core/
COPY main.py .

# Copy the model checkpoint into the image.
# Expects the model file at build context: model/inr_sdf_combined_fresh_ed_mix_v1_final.ptrom
# If your model is in the parent webapp/model/ folder, build with:
#   docker build -t cardiosdf-inference -f Dockerfile --build-context model=../model .
# Or copy the model file into inference-api/model/ before building.
COPY model/ model/

ENV MODEL_PATH=/app/model/inr_sdf_combined_fresh_ed_mix_v1_final.ptrom \
    GCS_BUCKET_NAME=cardiosdf-results

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
