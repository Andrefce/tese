"""CardioSDF Inference API — FastAPI service for SDF model inference.

Receives a NIfTI segmentation file + metadata, runs the SDF neural model,
saves the result JSON to a GCS bucket, and returns the hash + URL.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from core.nifti import load_nifti_from_bytes, select_frame
from core.sdf_model import extract_contours, load_model, predict_sdf_meshes

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Configuration ──
MODEL_PATH = Path(os.environ.get(
    "MODEL_PATH",
    Path(__file__).parent / "model" / "inr_sdf_combined_fresh_ed_mix_v1_final.ptrom",
))
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "cardiosdf-results")
# For local development: save results to a local directory instead of GCS
LOCAL_RESULTS_DIR = os.environ.get("LOCAL_RESULTS_DIR", "")

# ── Global model (loaded once at startup) ──
SDF_MODEL = None
SDF_CFG = None

app = FastAPI(title="CardioSDF Inference API", version="1.0.0")


# ── Startup: load the model ──
@app.on_event("startup")
async def startup_load_model():
    global SDF_MODEL, SDF_CFG
    if not MODEL_PATH.exists():
        log.warning("Model checkpoint not found at %s — inference will fail", MODEL_PATH)
        return
    log.info("Loading SDF model from %s …", MODEL_PATH)
    SDF_MODEL, SDF_CFG = load_model(MODEL_PATH)
    device = next(SDF_MODEL.parameters()).device
    log.info("SDF model loaded on %s", device)


# ── GCS helpers ──
_gcs_client = None


def _get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client


def _upload_to_gcs(result_bytes: bytes, filename: str) -> str:
    """Upload bytes to GCS and return the public URL."""
    client = _get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(filename)
    blob.upload_from_string(result_bytes, content_type="application/json")
    return f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{filename}"


def _save_local(result_bytes: bytes, filename: str) -> str:
    """Save result to local directory for development."""
    out_dir = Path(LOCAL_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_bytes(result_bytes)
    return str(path)


# ── Health check ──
@app.get("/health")
async def health():
    return {
        "ok": True,
        "model_loaded": SDF_MODEL is not None,
        "device": str(next(SDF_MODEL.parameters()).device) if SDF_MODEL is not None else None,
    }


# ── Main inference endpoint ──
@app.post("/infer")
async def infer(
    segmentation: UploadFile = File(..., description="NIfTI segmentation file (.nii or .nii.gz)"),
    affine: str = Form(..., description="4x4 affine matrix as JSON array"),
    spacing: str = Form(..., description="Voxel spacing [dx, dy, dz] as JSON array"),
    phase: str = Form("ed", description="Cardiac phase: 'ed' or 'es'"),
    frame: int = Form(0, description="Frame index for 4D volumes"),
):
    if SDF_MODEL is None or SDF_CFG is None:
        raise HTTPException(status_code=503, detail="SDF model not loaded")

    # Parse metadata
    try:
        affine_arr = np.array(json.loads(affine), dtype=np.float32)
        if affine_arr.shape != (4, 4):
            raise ValueError(f"Affine must be 4x4, got {affine_arr.shape}")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid affine matrix: {e}")

    try:
        spacing_arr = tuple(float(v) for v in json.loads(spacing))
        if len(spacing_arr) != 3:
            raise ValueError(f"Spacing must have 3 values, got {len(spacing_arr)}")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid spacing: {e}")

    if phase not in ("ed", "es"):
        raise HTTPException(status_code=400, detail="Phase must be 'ed' or 'es'")

    # Read segmentation NIfTI
    try:
        seg_bytes = await segmentation.read()
        seg_nifti = load_nifti_from_bytes(seg_bytes)
        seg_volume = select_frame(seg_nifti["data"], frame)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load segmentation: {e}")

    # Run SDF inference
    try:
        dz = float(spacing_arr[2])
        contours = extract_contours(seg_volume, affine_arr, dz)
        phase_val = 0.0 if phase == "ed" else 1.0

        result = predict_sdf_meshes(
            model=SDF_MODEL,
            contour_xyz=contours["xyz"],
            tissue_labels=contours["tissue"],
            cfg=SDF_CFG,
            phase_val=phase_val,
            scale=contours["scale"],
            centroid=contours["centroid"],
            seg_volume=seg_volume,
            spacing=spacing_arr,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
    except Exception as e:
        log.exception("SDF inference error")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Serialize result to JSON bytes
    result_json = json.dumps(result, default=_json_default).encode("utf-8")

    # Generate hash from result content
    result_hash = hashlib.sha256(result_json).hexdigest()[:16]
    filename = f"{result_hash}.json"

    # Save to GCS or local
    try:
        if LOCAL_RESULTS_DIR:
            url = _save_local(result_json, filename)
        else:
            url = _upload_to_gcs(result_json, filename)
    except Exception as e:
        log.exception("Failed to save result")
        raise HTTPException(status_code=500, detail=f"Failed to save result: {e}")

    log.info("Inference complete: hash=%s, endo_verts=%d, epi_verts=%d",
             result_hash,
             result.get("metrics", {}).get("endoVertices", 0),
             result.get("metrics", {}).get("epiVertices", 0))

    return JSONResponse({
        "hash": result_hash,
        "url": url,
        "metrics": result.get("metrics", {}),
        # Include mesh data inline so the webapp never needs a separate GCS fetch.
        "meshes": result.get("meshes"),
        "meshMethod": result.get("meshMethod"),
        "regionalThickness": result.get("regionalThickness"),
        "aha17": result.get("aha17", []),
    })


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
