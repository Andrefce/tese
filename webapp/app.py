from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from core.inference import analyse_segmentation, make_demo_case, segmentation_from_drawings
from core.nifti import (
    NiftiLoadError,
    case_metadata,
    decode_mask_base64,
    load_nifti,
    select_frame,
    slice_payload,
)

# ── SDF model (lazy-loaded) ──
SDF_MODEL = None
SDF_CFG = None

log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB

UPLOAD_ROOT = Path(os.environ.get("ACDC_WEBAPP_UPLOADS", Path(app.instance_path) / "uploads"))
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

CASES: dict[str, dict[str, Any]] = {}

ALLOWED_EXTENSIONS = {".nii", ".nii.gz"}
MAX_CASES = 50

MODEL_PATH = Path(__file__).parent / "model" / "inr_sdf_combined_fresh_ed_mix_v1_final.ptrom"
DEMO_DATA_ROOT = Path(__file__).parent / "demo-data" / "training"


def _json_error(message: str, status: int = 400):
    response = jsonify({"error": message})
    response.status_code = status
    return response


def _is_nifti(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in ALLOWED_EXTENSIONS)


def _save_upload(file_storage, case_dir: Path) -> Path:
    raw_name = file_storage.filename or ""
    filename = secure_filename(raw_name)
    if not filename or not _is_nifti(filename):
        raise ValueError("Only .nii and .nii.gz files are accepted.")
    target = case_dir / filename
    file_storage.save(target)
    return target


def _register_case(
    name: str,
    mri: dict[str, Any],
    seg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if seg is not None and tuple(seg["data"].shape[:3]) != tuple(mri["data"].shape[:3]):
        raise ValueError(
            f"MRI shape {mri['data'].shape[:3]} and segmentation shape "
            f"{seg['data'].shape[:3]} do not match."
        )

    # Evict oldest case if at capacity
    if len(CASES) >= MAX_CASES:
        oldest = next(iter(CASES))
        del CASES[oldest]
        log.info("Evicted case %s (capacity %d)", oldest, MAX_CASES)

    case_id = uuid.uuid4().hex[:12]
    case = {
        "id": case_id,
        "name": name,
        "mri": mri,
        "seg": seg,
        "drawn_masks": {},
    }
    CASES[case_id] = case
    return case


def _get_case(case_id: str) -> dict[str, Any] | None:
    return CASES.get(case_id)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template(
        "index.html",
        model_checkpoint=os.environ.get("ACDC_SDF_CHECKPOINT", ""),
    )


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "cases": len(CASES)})


def _load_sdf_model():
    """Load SDF model once on first use."""
    global SDF_MODEL, SDF_CFG
    if SDF_MODEL is not None:
        return SDF_MODEL, SDF_CFG
    if not MODEL_PATH.exists():
        log.warning("SDF model checkpoint not found at %s", MODEL_PATH)
        return None, None
    try:
        from core.sdf_model import load_model
        log.info("Loading SDF model from %s …", MODEL_PATH)
        SDF_MODEL, SDF_CFG = load_model(MODEL_PATH)
        log.info("SDF model loaded (device=%s)", next(SDF_MODEL.parameters()).device)
        return SDF_MODEL, SDF_CFG
    except Exception:
        log.exception("Failed to load SDF model")
        return None, None


def _list_acdc_patients() -> list[Path]:
    """Return sorted list of patient directories in demo-data."""
    if not DEMO_DATA_ROOT.is_dir():
        return []
    return sorted(p for p in DEMO_DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("patient"))


def _load_acdc_patient(patient_dir: Path) -> tuple[dict, dict, dict]:
    """Load a real ACDC patient: 4D MRI + ED frame + ED GT segmentation."""
    from core.nifti import resolve_nifti_path

    info_path = patient_dir / "Info.cfg"
    info = {}
    if info_path.exists():
        for line in info_path.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()

    patient_name = patient_dir.name
    ed_frame = int(info.get("ED", 1))
    group = info.get("Group", "")

    # Try loading 4D volume
    mri_4d_path = patient_dir / f"{patient_name}_4d.nii"
    mri = load_nifti(mri_4d_path)

    # Load ED frame segmentation (ground truth)
    gt_pattern = f"{patient_name}_frame{ed_frame:02d}_gt.nii"
    gt_path = patient_dir / gt_pattern
    seg = load_nifti(gt_path)

    patient_info = {
        "ed_frame": ed_frame,
        "group": group,
        "height": info.get("Height"),
        "weight": info.get("Weight"),
    }
    return mri, seg, patient_info


@app.get("/api/patients")
def list_patients():
    """List available ACDC patients from demo-data."""
    patients = _list_acdc_patients()
    result = []
    for p in patients:
        info_path = p / "Info.cfg"
        info = {}
        if info_path.exists():
            for line in info_path.read_text().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k.strip()] = v.strip()
        result.append({
            "id": p.name,
            "group": info.get("Group", ""),
            "ed": int(info.get("ED", 1)),
            "es": int(info.get("ES", 1)),
        })
    return jsonify({"patients": result})


@app.post("/api/demo")
def demo_case():
    """Load a real ACDC patient as demo. Accepts optional {patient: 'patient001'}."""
    payload = request.get_json(silent=True) or {}
    patients = _list_acdc_patients()

    if not patients:
        # Fallback to synthetic demo
        log.info("No ACDC demo data found, using synthetic demo")
        mri_data, seg_data, spacing = make_demo_case()
        case = _register_case(
            "ACDC Synthetic LV Demo",
            {
                "path": "synthetic://acdc-demo-mri",
                "data": mri_data,
                "affine": None,
                "zooms": spacing,
                "shape": tuple(int(v) for v in mri_data.shape),
            },
            {
                "path": "synthetic://acdc-demo-gt",
                "data": seg_data,
                "affine": None,
                "zooms": spacing,
                "shape": tuple(int(v) for v in seg_data.shape),
            },
        )
        return jsonify({"case": case_metadata(case)})

    # Pick requested patient or random
    requested = payload.get("patient")
    if requested:
        patient_dir = DEMO_DATA_ROOT / requested
        if not patient_dir.is_dir():
            return _json_error(f"Patient '{requested}' not found.", 404)
    else:
        import random
        patient_dir = random.choice(patients)

    try:
        mri, seg, patient_info = _load_acdc_patient(patient_dir)
        group_label = f" ({patient_info['group']})" if patient_info.get('group') else ""
        case = _register_case(
            f"{patient_dir.name}{group_label}",
            mri,
            seg,
        )
        case["patient_info"] = patient_info
    except Exception as exc:
        log.exception("Failed to load ACDC patient %s", patient_dir.name)
        return _json_error(f"Failed to load patient: {exc}", 500)

    return jsonify({"case": case_metadata(case)})


@app.post("/api/upload")
def upload_case():
    if "mri" not in request.files:
        return _json_error("Upload an MRI NIfTI file using the field name 'mri'.")

    case_dir = UPLOAD_ROOT / uuid.uuid4().hex
    case_dir.mkdir(parents=True, exist_ok=True)

    try:
        mri_path = _save_upload(request.files["mri"], case_dir)
        seg_path = None
        if "segmentation" in request.files and request.files["segmentation"].filename:
            seg_path = _save_upload(request.files["segmentation"], case_dir)

        mri = load_nifti(mri_path)
        seg = load_nifti(seg_path) if seg_path is not None else None
        case_name = request.form.get("caseName") or Path(mri["path"]).stem
        case = _register_case(case_name, mri, seg)
    except (ValueError, FileNotFoundError, NiftiLoadError) as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        log.exception("Upload failed")
        return _json_error(f"Could not load case: {exc}", 500)

    return jsonify({"case": case_metadata(case)})


@app.get("/api/case/<case_id>/slice")
def get_slice(case_id: str):
    case = _get_case(case_id)
    if case is None:
        return _json_error("Case not found.", 404)
    z = request.args.get("z", case["mri"]["data"].shape[2] // 2, type=int)
    frame = request.args.get("frame", 0, type=int)
    try:
        return jsonify(slice_payload(case, z=z, frame=frame))
    except Exception as exc:
        return _json_error(f"Could not render slice: {exc}", 500)


@app.post("/api/case/<case_id>/mask")
def save_drawn_mask(case_id: str):
    case = _get_case(case_id)
    if case is None:
        return _json_error("Case not found.", 404)
    payload = request.get_json(silent=True) or {}
    try:
        z = int(payload["z"])
        width = int(payload["width"])
        height = int(payload["height"])
        mask = decode_mask_base64(payload["mask"], width=width, height=height)
    except Exception as exc:
        return _json_error(f"Invalid mask payload: {exc}", 400)

    expected_width = int(case["mri"]["data"].shape[0])
    expected_height = int(case["mri"]["data"].shape[1])
    if width != expected_width or height != expected_height:
        return _json_error(
            f"Mask size {width}×{height} does not match "
            f"slice size {expected_width}×{expected_height}.",
            400,
        )
    case["drawn_masks"][z] = (mask > 0).astype(np.uint8)
    return jsonify({"ok": True, "z": z, "paintedPixels": int(mask.sum())})


@app.post("/api/case/<case_id>/clear-masks")
def clear_masks(case_id: str):
    case = _get_case(case_id)
    if case is None:
        return _json_error("Case not found.", 404)
    case["drawn_masks"] = {}
    return jsonify({"ok": True})


@app.post("/api/case/<case_id>/infer")
def infer_case(case_id: str):
    case = _get_case(case_id)
    if case is None:
        return _json_error("Case not found.", 404)

    payload = request.get_json(silent=True) or {}
    frame = int(payload.get("frame", 0))
    use_sdf = payload.get("useSdfModel", True)  # prefer SDF model when available

    try:
        # Determine segmentation source
        seg_volume = None
        source = None
        full_seg = None

        if case.get("seg") is not None:
            seg_volume = select_frame(case["seg"]["data"], frame)
            source = "ACDC ground-truth segmentation"
            full_seg = case["seg"]["data"]
        elif case.get("drawn_masks"):
            mri_volume = select_frame(case["mri"]["data"], frame)
            seg_volume = segmentation_from_drawings(
                tuple(int(v) for v in mri_volume.shape),
                case["drawn_masks"],
                case["mri"]["zooms"],
            )
            source = "Drawn LV mask"
            full_seg = None
        else:
            return _json_error(
                "Provide a GT segmentation or draw an LV mask before running inference.",
                400,
            )

        # Try SDF neural model inference first
        sdf_result = None
        if use_sdf and seg_volume is not None:
            sdf_result = _try_sdf_inference(case, seg_volume, frame)

        # Always compute voxel-based metrics (volumes, EF, etc.)
        voxel_result = analyse_segmentation(
            seg_volume=seg_volume,
            spacing=case["mri"]["zooms"],
            source=source,
            full_segmentation=full_seg,
        )

        # Merge: use SDF meshes if available, voxel metrics otherwise
        if sdf_result is not None:
            result = voxel_result
            result["meshes"] = sdf_result["meshes"]
            result["meshMethod"] = sdf_result["meshMethod"]
            result["regionalThickness"] = sdf_result["regionalThickness"]
            result["aha17"] = sdf_result.get("aha17", [])
            # Add SDF-specific metrics
            for k in ("meanWallThicknessMm", "p95WallThicknessMm",
                      "endoSurfaceAreaCm2", "epiSurfaceAreaCm2"):
                if sdf_result["metrics"].get(k) is not None:
                    result["metrics"][k] = sdf_result["metrics"][k]
            result["source"] = f"{source} + SDF model"
        else:
            result = voxel_result

    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        log.exception("Inference failed for case %s", case_id)
        return _json_error(f"Inference failed: {exc}", 500)

    return jsonify(result)


def _try_sdf_inference(case: dict, seg_volume: np.ndarray, frame: int) -> dict | None:
    """Attempt SDF model inference. Returns result dict or None on failure."""
    model, cfg = _load_sdf_model()
    if model is None:
        return None

    try:
        from core.sdf_model import extract_contours, predict_sdf_meshes

        affine = case.get("seg", {}).get("affine")
        if affine is None:
            affine = case.get("mri", {}).get("affine")
        if affine is None:
            log.warning("No affine matrix available, skipping SDF inference")
            return None

        dz = float(case["mri"]["zooms"][2])

        contours = extract_contours(seg_volume, affine, dz)
        phase_val = 0.0  # ED by default

        # Check patient_info for ED/ES phase
        patient_info = case.get("patient_info", {})
        ed_frame = patient_info.get("ed_frame")
        if ed_frame is not None and frame != ed_frame - 1:
            phase_val = 1.0  # ES

        result = predict_sdf_meshes(
            model=model,
            contour_xyz=contours["xyz"],
            tissue_labels=contours["tissue"],
            cfg=cfg,
            phase_val=phase_val,
            scale=contours["scale"],
            centroid=contours["centroid"],
            seg_volume=seg_volume,
            spacing=case["mri"]["zooms"],
        )
        log.info("SDF inference OK: endo=%d verts, epi=%d verts",
                 result["metrics"].get("endoVertices", 0),
                 result["metrics"].get("epiVertices", 0))
        return result
    except Exception:
        log.exception("SDF inference failed, falling back to voxel-based")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", "8003"))
    app.run(host="127.0.0.1", port=port, debug=True)

