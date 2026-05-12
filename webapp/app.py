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


def _load_acdc_patient(patient_dir: Path) -> tuple[dict, dict, dict, dict | None, dict | None]:
    """Load a real ACDC patient: 4D MRI + ED frame + ED GT segmentation + ES if available."""
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
    es_frame = int(info.get("ES", 1))
    group = info.get("Group", "")

    # Try loading 4D volume
    mri_4d_path = patient_dir / f"{patient_name}_4d.nii"
    mri = load_nifti(mri_4d_path)

    # Load ED frame segmentation (ground truth)
    gt_pattern = f"{patient_name}_frame{ed_frame:02d}_gt.nii"
    gt_path = patient_dir / gt_pattern
    seg = load_nifti(gt_path)

    # Load ES frame MRI and segmentation if available
    es_mri = None
    es_seg = None
    try:
        es_mri_path = patient_dir / f"{patient_name}_frame{es_frame:02d}.nii"
        es_mri = load_nifti(es_mri_path)
    except Exception:
        pass
    try:
        es_gt_path = patient_dir / f"{patient_name}_frame{es_frame:02d}_gt.nii"
        es_seg = load_nifti(es_gt_path)
    except Exception:
        pass

    patient_info = {
        "ed_frame": ed_frame,
        "es_frame": es_frame,
        "group": group,
        "height": info.get("Height"),
        "weight": info.get("Weight"),
    }
    return mri, seg, patient_info, es_mri, es_seg


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
        mri, seg, patient_info, es_mri, es_seg = _load_acdc_patient(patient_dir)
        group_label = f" ({patient_info['group']})" if patient_info.get('group') else ""
        case = _register_case(
            f"{patient_dir.name}{group_label}",
            mri,
            seg,
        )
        case["patient_info"] = patient_info
        if es_mri is not None:
            case["mri_es"] = es_mri
        if es_seg is not None:
            case["seg_es"] = es_seg
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

        # Wall-thickness mode: also accept ES files
        if "mri_es" in request.files and request.files["mri_es"].filename:
            es_mri_path = _save_upload(request.files["mri_es"], case_dir)
            es_seg_path = None
            if "segmentation_es" in request.files and request.files["segmentation_es"].filename:
                es_seg_path = _save_upload(request.files["segmentation_es"], case_dir)
            case["mri_es"] = load_nifti(es_mri_path)
            case["seg_es"] = load_nifti(es_seg_path) if es_seg_path else None

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
    phase = request.args.get("phase", "ed")  # "ed" or "es"
    try:
        if phase == "es" and case.get("mri_es") is not None:
            es_case = {
                "id": case["id"],
                "mri": case["mri_es"],
                "seg": case.get("seg_es"),
                "drawn_masks": case.get("drawn_masks_es", {}),
            }
            return jsonify(slice_payload(es_case, z=z, frame=frame))
        return jsonify(slice_payload(case, z=z, frame=frame))
    except Exception as exc:
        return _json_error(f"Could not render slice: {exc}", 500)


@app.get("/api/case/<case_id>/slice-contours")
def get_slice_contours(case_id: str):
    """Return ordered segmentation contours in the same world-mm space as the 3D mesh."""
    case = _get_case(case_id)
    if case is None:
        return _json_error("Case not found.", 404)
    phase = request.args.get("phase", "ed")
    frame = request.args.get("frame", 0, type=int)

    seg_dict = case.get("seg")
    mri_dict = case["mri"]
    if phase == "es" and case.get("seg_es") is not None:
        seg_dict = case["seg_es"]
        mri_dict = case.get("mri_es", mri_dict)

    if seg_dict is None:
        return jsonify({"slices": []})

    seg_vol = select_frame(seg_dict["data"], frame)
    spacing = mri_dict["zooms"]
    affine = seg_dict.get("affine")
    if affine is None:
        affine = mri_dict.get("affine")
    dz = float(spacing[2])
    n_slices = int(seg_vol.shape[2])

    try:
        from skimage import measure as sk_measure
    except ImportError:
        return jsonify({"slices": [], "spacing": list(spacing)})

    # Use the same vox2world as extract_contours in sdf_model.py
    def vox2world(rows, cols, z_idx):
        """Transform voxel coords to world-mm coords using affine (XY) + dz (Z)."""
        if affine is not None:
            vox = np.column_stack([cols, rows, np.zeros(len(rows)), np.ones(len(rows))])
            world = (np.asarray(affine) @ vox.T).T
            world[:, 2] = z_idx * dz
            return world[:, :3]
        else:
            # Fallback: simple spacing
            return np.column_stack([
                rows * spacing[0],
                cols * spacing[1],
                np.full(len(rows), z_idx * dz),
            ])

    slices_out = []
    for z in range(n_slices):
        seg_slice = np.rint(seg_vol[:, :, z]).astype(np.int16)
        if not np.any(seg_slice > 0):
            continue
        slice_data = {"z": z, "zMm": float(z * dz), "contours": []}
        for lbl, name in [(2, "myo"), (3, "lv")]:
            mask = (seg_slice == lbl).astype(np.float64)
            if mask.sum() == 0:
                continue
            contours = sk_measure.find_contours(mask, 0.5)
            for contour in contours:
                if len(contour) < 5:
                    continue
                step = max(1, len(contour) // 60)
                pts = contour[::step]
                # Transform to world coordinates (same as mesh)
                world_pts = vox2world(pts[:, 0], pts[:, 1], z)
                slice_data["contours"].append({
                    "label": name,
                    "points": world_pts.tolist(),
                })
        if slice_data["contours"]:
            slices_out.append(slice_data)

    return jsonify({"slices": slices_out, "spacing": list(spacing)})


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
    phase = payload.get("phase", "ed")  # "ed" or "es"

    try:
        # Pick the right data based on phase
        if phase == "es" and case.get("mri_es") is not None:
            mri_data = case["mri_es"]
            seg_data = case.get("seg_es")
            drawn_masks = case.get("drawn_masks_es", {})
        else:
            mri_data = case["mri"]
            seg_data = case.get("seg")
            drawn_masks = case.get("drawn_masks", {})

        # Determine segmentation source
        seg_volume = None
        source = None
        full_seg = None

        if seg_data is not None:
            seg_volume = select_frame(seg_data["data"], frame)
            source = f"ACDC ground-truth segmentation ({phase.upper()})"
            full_seg = seg_data["data"]
        elif drawn_masks:
            mri_volume = select_frame(mri_data["data"], frame)
            seg_volume = segmentation_from_drawings(
                tuple(int(v) for v in mri_volume.shape),
                drawn_masks,
                mri_data["zooms"],
            )
            source = f"Drawn LV mask ({phase.upper()})"
            full_seg = None
        else:
            return _json_error(
                "Provide a GT segmentation or draw an LV mask before running inference.",
                400,
            )

        # Try SDF neural model inference first
        sdf_result = None
        if use_sdf and seg_volume is not None:
            sdf_result = _try_sdf_inference(case, seg_volume, frame, phase=phase)

        # Always compute voxel-based metrics (volumes, EF, etc.)
        voxel_result = analyse_segmentation(
            seg_volume=seg_volume,
            spacing=mri_data["zooms"],
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


def _try_sdf_inference(case: dict, seg_volume: np.ndarray, frame: int, phase: str = "ed") -> dict | None:
    """Attempt SDF model inference. Returns result dict or None on failure."""
    model, cfg = _load_sdf_model()
    if model is None:
        return None

    try:
        from core.sdf_model import extract_contours, predict_sdf_meshes

        affine = case.get("seg", {}).get("affine")
        if phase == "es" and case.get("seg_es"):
            affine = case["seg_es"].get("affine", affine)
        if affine is None:
            affine = case.get("mri", {}).get("affine")
        if phase == "es" and case.get("mri_es"):
            affine = case["mri_es"].get("affine", affine)
        if affine is None:
            log.warning("No affine matrix available, skipping SDF inference")
            return None

        dz = float(case["mri"]["zooms"][2])

        contours = extract_contours(seg_volume, affine, dz)
        phase_val = 0.0 if phase == "ed" else 1.0  # ED=0, ES=1

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
    port = int(os.environ.get("PORT", "8009"))
    app.run(host="127.0.0.1", port=port, debug=True)

