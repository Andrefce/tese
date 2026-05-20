from __future__ import annotations

import io
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor as _TpExec
from pathlib import Path
from typing import Any

import numpy as np
import requests as http_requests
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from core.inference import (
    analyse_segmentation,
    compare_wall_thickness_results,
    make_demo_case,
    segmentation_from_drawings,
)
from core.nifti import (
    NiftiLoadError,
    case_metadata,
    decode_mask_base64,
    load_nifti,
    select_frame,
    slice_payload,
)

# ── Inference API (Cloud Run service) ──
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://localhost:8080")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "cardiosdf-results")

log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB

UPLOAD_ROOT = Path(os.environ.get("ACDC_WEBAPP_UPLOADS", Path(app.instance_path) / "uploads"))
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

CASES: dict[str, dict[str, Any]] = {}
JOBS: dict[str, dict[str, Any]] = {}

ALLOWED_EXTENSIONS = {".nii", ".nii.gz"}
MAX_CASES = 50
MAX_JOBS = 200

_job_executor = _TpExec(max_workers=4)


def _evict_jobs() -> None:
    """Evict the oldest jobs when the job table is at capacity."""
    while len(JOBS) >= MAX_JOBS:
        oldest = next(iter(JOBS))
        del JOBS[oldest]


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


def _spatial_shape(volume: dict[str, Any]) -> tuple[int, int, int]:
    return tuple(int(v) for v in volume["data"].shape[:3])


def _validate_segmentation_shape(mri: dict[str, Any], seg: dict[str, Any] | None, label: str) -> None:
    if seg is not None and _spatial_shape(seg) != _spatial_shape(mri):
        raise ValueError(
            f"{label} MRI shape {_spatial_shape(mri)} and segmentation shape "
            f"{_spatial_shape(seg)} do not match."
        )


def _attach_es_phase(
    case: dict[str, Any],
    es_mri: dict[str, Any],
    es_seg: dict[str, Any] | None = None,
) -> None:
    if _spatial_shape(es_mri) != _spatial_shape(case["mri"]):
        raise ValueError(
            f"ED MRI shape {_spatial_shape(case['mri'])} and ES MRI shape "
            f"{_spatial_shape(es_mri)} do not match."
        )
    _validate_segmentation_shape(es_mri, es_seg, "ES")
    case["mri_es"] = es_mri
    case["seg_es"] = es_seg
    case.setdefault("drawn_masks_es", {})


def _phase_data(case: dict[str, Any], phase: str) -> tuple[dict[str, Any], dict[str, Any] | None, dict[int, np.ndarray]]:
    if phase == "es":
        if case.get("mri_es") is None:
            raise ValueError("ES MRI is not loaded for this case.")
        return case["mri_es"], case.get("seg_es"), case.setdefault("drawn_masks_es", {})
    return case["mri"], case.get("seg"), case.setdefault("drawn_masks", {})


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
        asset_version=os.environ.get("K_REVISION", os.environ.get("APP_VERSION", "local")),
    )


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "cases": len(CASES)})


def _get_inference_auth_headers() -> dict[str, str]:
    """Get authentication headers for Cloud Run service-to-service calls.

    When running on Cloud Run, uses the metadata server to obtain an
    identity token.  Locally, returns empty headers (no auth needed).
    """
    if os.environ.get("K_SERVICE"):  # running on Cloud Run
        try:
            import google.auth.transport.requests
            import google.oauth2.id_token
            auth_req = google.auth.transport.requests.Request()
            token = google.oauth2.id_token.fetch_id_token(auth_req, INFERENCE_API_URL)
            return {"Authorization": f"Bearer {token}"}
        except Exception:
            log.exception("Failed to obtain Cloud Run identity token for %s", INFERENCE_API_URL)
    return {}


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
            _attach_es_phase(case, es_mri, es_seg)
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
            _attach_es_phase(
                case,
                load_nifti(es_mri_path),
                load_nifti(es_seg_path) if es_seg_path else None,
            )

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
        phase = str(payload.get("phase", "ed")).lower()
    except Exception as exc:
        return _json_error(f"Invalid mask payload: {exc}", 400)

    try:
        mri_data, _, drawn_masks = _phase_data(case, phase)
    except ValueError as exc:
        return _json_error(str(exc), 400)

    expected_width = int(mri_data["data"].shape[0])
    expected_height = int(mri_data["data"].shape[1])
    if width != expected_width or height != expected_height:
        return _json_error(
            f"Mask size {width}×{height} does not match "
            f"slice size {expected_width}×{expected_height}.",
            400,
        )
    drawn_masks[z] = (mask > 0).astype(np.uint8)
    return jsonify({"ok": True, "z": z, "phase": phase, "paintedPixels": int(mask.sum())})


@app.post("/api/case/<case_id>/clear-masks")
def clear_masks(case_id: str):
    case = _get_case(case_id)
    if case is None:
        return _json_error("Case not found.", 404)
    payload = request.get_json(silent=True) or {}
    phase = str(payload.get("phase", "ed")).lower()
    if phase == "es":
        case["drawn_masks_es"] = {}
    else:
        case["drawn_masks"] = {}
    return jsonify({"ok": True, "phase": phase})


def _run_phase_inference(
    case: dict[str, Any],
    phase: str,
    frame: int,
    use_sdf: bool,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    mri_data, seg_data, drawn_masks = _phase_data(case, phase)

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
    else:
        raise ValueError(
            f"Provide a GT segmentation or draw an LV mask for {phase.upper()} before running inference."
        )

    sdf_result = None
    if use_sdf and seg_volume is not None:
        sdf_result = _try_sdf_inference(case, seg_volume, frame, phase=phase)

    voxel_result = analyse_segmentation(
        seg_volume=seg_volume,
        spacing=mri_data["zooms"],
        source=source,
        full_segmentation=full_seg,
    )

    if sdf_result is not None:
        result = voxel_result
        if "meshes" in sdf_result:
            result["meshes"] = sdf_result["meshes"]
            result["meshMethod"] = sdf_result["meshMethod"]
            result["regionalThickness"] = sdf_result["regionalThickness"]
            result["aha17"] = sdf_result.get("aha17", [])
        if "sdfHash" in sdf_result:
            result["sdfHash"] = sdf_result["sdfHash"]
            result["sdfResultUrl"] = sdf_result["sdfResultUrl"]
        sdf_metrics = sdf_result.get("metrics", {})
        for k in ("meanWallThicknessMm", "p95WallThicknessMm",
                  "endoSurfaceAreaCm2", "epiSurfaceAreaCm2"):
            if sdf_metrics.get(k) is not None:
                result["metrics"][k] = sdf_metrics[k]
        result["source"] = f"{source} + SDF model"
    else:
        result = voxel_result

    result["phase"] = phase
    result["frame"] = int(frame)
    return result, seg_volume, mri_data


def _run_infer_single_job(
    job_id: str,
    case_id: str,
    phase: str,
    frame: int,
    use_sdf: bool,
) -> None:
    try:
        JOBS[job_id]["status"] = "running"
        case = _get_case(case_id)
        if case is None:
            JOBS[job_id] = {"status": "error", "error": "Case not found (may have been evicted)"}
            return
        result, _, _ = _run_phase_inference(case, phase, frame, use_sdf)
        JOBS[job_id] = {"status": "done", "result": result}
    except ValueError as exc:
        JOBS[job_id] = {"status": "error", "error": str(exc)}
    except Exception as exc:
        log.exception("Background single inference failed for case %s", case_id)
        JOBS[job_id] = {"status": "error", "error": f"Inference failed: {exc}"}


def _run_infer_paired_job(
    job_id: str,
    case_id: str,
    ed_frame: int,
    es_frame: int,
    use_sdf: bool,
) -> None:
    try:
        JOBS[job_id]["status"] = "running"
        case = _get_case(case_id)
        if case is None:
            JOBS[job_id] = {"status": "error", "error": "Case not found (may have been evicted)"}
            return

        with _TpExec(max_workers=2) as pool:
            fut_ed = pool.submit(_run_phase_inference, case, "ed", ed_frame, use_sdf)
            fut_es = pool.submit(_run_phase_inference, case, "es", es_frame, use_sdf)
            ed_result, ed_seg_volume, ed_mri = fut_ed.result()
            es_result, _, _ = fut_es.result()

        ed_lv = ed_result.get("metrics", {}).get("lvVolumeMl")
        es_lv = es_result.get("metrics", {}).get("lvVolumeMl")
        if ed_lv is not None and es_lv is not None:
            stroke = float(ed_lv) - float(es_lv)
            ef = stroke / float(ed_lv) * 100.0 if float(ed_lv) > 0 else None
            for r in (ed_result, es_result):
                r["metrics"]["edvMl"] = round(float(ed_lv), 2)
                r["metrics"]["esvMl"] = round(float(es_lv), 2)
                r["metrics"]["strokeVolumeMl"] = round(stroke, 2)
                r["metrics"]["ejectionFractionPct"] = round(ef, 1) if ef is not None else None

        difference = compare_wall_thickness_results(
            ed_result,
            es_result,
            ed_seg_volume=ed_seg_volume,
            spacing=ed_mri["zooms"],
        )
        JOBS[job_id] = {
            "status": "done",
            "result": {
                "source": "Paired ED/ES wall-thickness inference",
                "ed": ed_result,
                "es": es_result,
                "difference": difference,
            },
        }
    except ValueError as exc:
        JOBS[job_id] = {"status": "error", "error": str(exc)}
    except Exception as exc:
        log.exception("Background paired inference failed for case %s", case_id)
        JOBS[job_id] = {"status": "error", "error": f"Paired inference failed: {exc}"}


@app.post("/api/case/<case_id>/infer/start")
def start_infer(case_id: str):
    case = _get_case(case_id)
    if case is None:
        return _json_error("Case not found.", 404)
    payload = request.get_json(silent=True) or {}
    frame = int(payload.get("frame", 0))
    use_sdf = payload.get("useSdfModel", True)
    phase = payload.get("phase", "ed")
    _evict_jobs()
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "pending"}
    _job_executor.submit(_run_infer_single_job, job_id, case_id, phase, frame, use_sdf)
    return jsonify({"jobId": job_id})


@app.post("/api/case/<case_id>/infer-paired/start")
def start_infer_paired(case_id: str):
    case = _get_case(case_id)
    if case is None:
        return _json_error("Case not found.", 404)
    if case.get("mri_es") is None:
        return _json_error("Load both ED and ES MRI data before running paired wall-thickness inference.", 400)
    payload = request.get_json(silent=True) or {}
    use_sdf = payload.get("useSdfModel", True)
    ed_frame = int(payload.get("edFrame", payload.get("frame", 0)))
    es_frame = int(payload.get("esFrame", 0))
    _evict_jobs()
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "pending"}
    _job_executor.submit(_run_infer_paired_job, job_id, case_id, ed_frame, es_frame, use_sdf)
    return jsonify({"jobId": job_id})


@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return _json_error("Job not found.", 404)
    return jsonify(job)


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
        result, _, _ = _run_phase_inference(case, phase, frame, use_sdf)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        log.exception("Inference failed for case %s", case_id)
        return _json_error(f"Inference failed: {exc}", 500)

    return jsonify(result)


@app.post("/api/case/<case_id>/infer-paired")
def infer_paired_case(case_id: str):
    case = _get_case(case_id)
    if case is None:
        return _json_error("Case not found.", 404)
    if case.get("mri_es") is None:
        return _json_error("Load both ED and ES MRI data before running paired wall-thickness inference.", 400)

    payload = request.get_json(silent=True) or {}
    use_sdf = payload.get("useSdfModel", True)
    ed_frame = int(payload.get("edFrame", payload.get("frame", 0)))
    es_frame = int(payload.get("esFrame", 0))

    try:
        ed_result, ed_seg_volume, ed_mri = _run_phase_inference(case, "ed", ed_frame, use_sdf)
        es_result, _, _ = _run_phase_inference(case, "es", es_frame, use_sdf)

        ed_lv = ed_result.get("metrics", {}).get("lvVolumeMl")
        es_lv = es_result.get("metrics", {}).get("lvVolumeMl")
        if ed_lv is not None and es_lv is not None:
            stroke = float(ed_lv) - float(es_lv)
            ef = stroke / float(ed_lv) * 100.0 if float(ed_lv) > 0 else None
            for result in (ed_result, es_result):
                result["metrics"]["edvMl"] = round(float(ed_lv), 2)
                result["metrics"]["esvMl"] = round(float(es_lv), 2)
                result["metrics"]["strokeVolumeMl"] = round(stroke, 2)
                result["metrics"]["ejectionFractionPct"] = round(ef, 1) if ef is not None else None

        difference = compare_wall_thickness_results(
            ed_result,
            es_result,
            ed_seg_volume=ed_seg_volume,
            spacing=ed_mri["zooms"],
        )
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        log.exception("Paired inference failed for case %s", case_id)
        return _json_error(f"Paired inference failed: {exc}", 500)

    return jsonify({
        "source": "Paired ED/ES wall-thickness inference",
        "ed": ed_result,
        "es": es_result,
        "difference": difference,
    })


def _try_sdf_inference(case: dict, seg_volume: np.ndarray, frame: int, phase: str = "ed") -> dict | None:
    """Call the remote inference API for SDF model inference.

    Returns a dict with meshes, meshMethod, regionalThickness, aha17, and metrics
    taken directly from the API response.  Also stores sdfHash/sdfResultUrl if
    present so the frontend can cache-bust.  Returns None on failure.
    """
    try:
        import nibabel as nib
    except ImportError:
        log.warning("nibabel not available, cannot serialize segmentation for API")
        return None

    mri_data, seg_data, _ = _phase_data(case, phase)
    affine = seg_data.get("affine") if seg_data is not None else None
    if affine is None:
        affine = mri_data.get("affine")
    if affine is None:
        log.warning("No affine matrix available, skipping SDF inference")
        return None

    spacing = mri_data["zooms"]

    try:
        from nibabel.fileholders import FileHolder
        seg_img = nib.Nifti1Image(seg_volume.astype(np.float32), np.eye(4))
        buf = io.BytesIO()
        file_map = seg_img.make_file_map()
        file_map["image"] = FileHolder(fileobj=buf)
        file_map["header"] = FileHolder(fileobj=buf)
        seg_img.to_file_map(file_map)
        seg_bytes = buf.getvalue()
    except Exception:
        log.exception("Failed to serialize segmentation to NIfTI")
        return None

    try:
        headers = _get_inference_auth_headers()
        resp = http_requests.post(
            f"{INFERENCE_API_URL}/infer",
            files={"segmentation": ("seg.nii", seg_bytes, "application/octet-stream")},
            data={
                "affine": json.dumps(np.asarray(affine, dtype=float).tolist()),
                "spacing": json.dumps([float(v) for v in spacing]),
                "phase": phase,
                "frame": str(frame),
            },
            headers=headers,
            timeout=300,
        )
        resp.raise_for_status()
        api_result = resp.json()
    except http_requests.exceptions.Timeout:
        log.error("Inference API timed out")
        return None
    except Exception:
        log.exception("Inference API call failed")
        return None

    log.info("SDF inference API OK: hash=%s", api_result.get("hash"))
    out: dict = {"metrics": api_result.get("metrics", {})}
    # Prefer inline mesh data (returned by the API directly)
    for key in ("meshes", "meshMethod", "regionalThickness", "aha17"):
        if key in api_result:
            out[key] = api_result[key]
    # Keep GCS URL for optional cache usage by the frontend
    if "hash" in api_result:
        out["sdfHash"] = api_result["hash"]
    if "url" in api_result:
        out["sdfResultUrl"] = api_result["url"]
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", "8009"))
    app.run(host="127.0.0.1", port=port, debug=True)

