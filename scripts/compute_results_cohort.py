"""Multi-patient wall-thickness cohort pipeline for the Results chapter.

Runs the CardioSDF reconstruction on every available demo patient (ED phase),
computes four wall-thickness estimators per endocardial vertex
(Laplace field, Yezzi-Prince, SDF cone rays, EDT boundary sum), the physical
segmentation reference (EDT boundary sum on the ground-truth labels), the AHA-17
segment assignment, and both the raw and segmentation-calibrated statistics.

It also stores a representative-patient NPZ (meshes, contours, per-vertex
thickness, AHA ids) that the figure scripts consume.

Outputs are written to scripts/webapp/notebooks/outputs/cohort/.

Run:
    cd scripts/webapp && python ../compute_results_cohort.py
"""
from __future__ import annotations

# Cap thread usage BEFORE importing numeric libraries so a single patient cannot
# saturate every core / oversubscribe BLAS and exhaust memory.
import os

_THREADS = os.environ.get("RESULTS_THREADS", "2")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import gc
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import (
    distance_transform_edt,
    binary_fill_holes,
    binary_dilation,
    generate_binary_structure,
)
from scipy.spatial import cKDTree
from scipy.sparse import linalg as spla
from skimage import measure
import trimesh

# ── Paths ─────────────────────────────────────────────────────────
WEBAPP_DIR = (Path(__file__).resolve().parent / "webapp")
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

from core.nifti import load_nifti  # noqa: E402
from core.sdf_model import (  # noqa: E402
    FLIP_Z,
    extract_contours,
    load_model,
    predict_sdf_meshes,
    _reference_wall_thickness_from_segmentation,
    _build_contour_tensor,
    _build_grid_and_query,
    _mc_field,
    _snap_mesh_to_contours,
)

try:
    from pyezzi import compute_thickness_cardiac
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Yezzi-Prince requires `pyezzi` (pip install pyezzi)."
    ) from exc

import torch  # noqa: E402

try:
    torch.set_num_threads(int(_THREADS))
except Exception:
    pass

LBL_BG, LBL_RV, LBL_MYO, LBL_LV = 0, 1, 2, 3
MODEL_PATH = WEBAPP_DIR / "model" / "inr_sdf_combined_fresh_ed_mix_v1_final.ptrom"
DATA_ROOT = WEBAPP_DIR / "demo-data" / "training"
OUT_DIR = WEBAPP_DIR / "notebooks" / "outputs" / "cohort"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GRID_RES = int(os.environ.get("CARDIOSDF_GRID_RES", "96"))
CONE_RAY_COUNT = 7
CONE_RAY_SAMPLE_LIMIT = 1800
THICKNESS_MAX_MM = 15.0
REPRESENTATIVE = "patient001"

AHA_17_NAMES = [
    "Basal Anterior", "Basal Anteroseptal", "Basal Inferoseptal",
    "Basal Inferior", "Basal Inferolateral", "Basal Anterolateral",
    "Mid Anterior", "Mid Anteroseptal", "Mid Inferoseptal",
    "Mid Inferior", "Mid Inferolateral", "Mid Anterolateral",
    "Apical Anterior", "Apical Septal", "Apical Inferior", "Apical Lateral",
    "Apex",
]

METHODS = ["Laplace field", "Yezzi-Prince", "SDF cone rays", "EDT boundary sum"]


# ── Geometry helpers (faithful to the single-patient notebook) ────
def payload_array(payload: dict, key: str, width: int) -> np.ndarray:
    dtype = np.int32 if key == "faces" else np.float32
    arr = np.asarray(payload.get(key, []), dtype=dtype)
    return np.empty((0, width), dtype=dtype) if arr.size == 0 else arr.reshape(-1, width)


def mesh_from_payload(payload: dict, name: str) -> trimesh.Trimesh:
    vertices = payload_array(payload, "vertices", 3)
    faces = payload_array(payload, "faces", 3).astype(np.int32)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"CardioSDF returned an empty {name} mesh.")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    return mesh


def build_full_meshes(model, cfg, contours, scale, centroid, snap: bool = True):
    """Full-resolution watertight endo/epi meshes straight from the SDF field.

    The webapp payload decimates faces and strips the base caps, which yields a
    broken (non-watertight) surface unsuitable for rendering.  Here we run
    marching cubes on the raw grid, optionally snap to the input contours, and
    un-normalize to millimetres, keeping the closed manifold intact.

    When ``snap`` is False the raw (un-snapped) SDF surface is returned; this is
    smooth and naturally LV-shaped, and is used only for visualisation.  The
    snapped surface (``snap=True``) honours the input rings exactly and is used
    for the quantitative slice-residual metric.
    """
    cont_t, mask_t = _build_contour_tensor(contours["xyz"], contours["tissue"], cfg, 0.0)
    z = model.encode(cont_t, mask_t)
    sdf_e, sdf_p, _dlt, lo, hi, voxel = _build_grid_and_query(
        z, model, contours["xyz"], cfg, GRID_RES
    )
    iso = cfg.get("iso_level", 0.0)
    endo_v, endo_f = _mc_field(sdf_e, lo, voxel, iso)
    epi_v, epi_f = _mc_field(sdf_p, lo, voxel, iso)
    if snap:
        if len(endo_v):
            endo_v = _snap_mesh_to_contours(endo_v, contours["xyz"], contours["tissue"], surface="endo")
        if len(epi_v):
            epi_v = _snap_mesh_to_contours(epi_v, contours["xyz"], contours["tissue"], surface="epi")
    flip = np.array([1.0, 1.0, -1.0 if FLIP_Z else 1.0], dtype=np.float32)
    endo_mm = (endo_v * flip) * scale + centroid if len(endo_v) else endo_v
    epi_mm = (epi_v * flip) * scale + centroid if len(epi_v) else epi_v
    endo_mesh = trimesh.Trimesh(vertices=endo_mm, faces=endo_f.astype(np.int32), process=False)
    epi_mesh = trimesh.Trimesh(vertices=epi_mm, faces=epi_f.astype(np.int32), process=False)
    endo_mesh = _clean_surface(endo_mesh)
    epi_mesh = _clean_surface(epi_mesh)
    return endo_mesh, epi_mesh


def _clean_surface(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Keep the largest connected component, fill holes, fix normals.

    Marching cubes can leave tiny disconnected islands and a small opening at
    the valve plane; removing the islands and filling the residual holes yields
    a single closed (watertight) surface that renders as a solid body.
    """
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            mesh = max(components, key=lambda c: len(c.faces))
    except Exception:
        pass
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    return mesh


def voxel_to_world(indices: np.ndarray, spacing_mm: tuple) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.float32)
    pts = np.empty((len(indices), 3), dtype=np.float32)
    pts[:, 0] = -indices[:, 1]
    pts[:, 1] = -indices[:, 0]
    pts[:, 2] = indices[:, 2] * spacing_mm[2]
    return pts


def voxelize_mesh_to_grid(mesh: trimesh.Trimesh, grid_shape: tuple) -> np.ndarray:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    idx = verts.astype(np.int32)
    for d in range(3):
        idx[:, d] = np.clip(idx[:, d], 0, grid_shape[d] - 1)
    mask = np.zeros(grid_shape, dtype=bool)
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    struct = generate_binary_structure(3, 1)
    mask = binary_dilation(mask, struct, iterations=2)
    mask = binary_fill_holes(mask)
    return mask.astype(bool)


def orient_normals(endo_mesh: trimesh.Trimesh, epi_vertices: np.ndarray) -> np.ndarray:
    points = np.asarray(endo_mesh.vertices, dtype=np.float32)
    normals = np.asarray(endo_mesh.vertex_normals, dtype=np.float32).copy()
    _, nn = cKDTree(epi_vertices).query(points, workers=-1)
    dots = np.sum(normals * (epi_vertices[nn] - points), axis=1)
    if float(np.nanmedian(dots)) < 0:
        normals *= -1.0
    return normals


def sample_field(field: np.ndarray, vertices: np.ndarray, myo_idx: np.ndarray, myo_tree: cKDTree) -> np.ndarray:
    _, nn = myo_tree.query(vertices, workers=-1)
    idx = myo_idx[nn]
    return np.asarray(field[idx[:, 0], idx[:, 1], idx[:, 2]], dtype=np.float32)


def fill_invalid(points: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    valid = np.isfinite(out)
    if valid.all() or not valid.any():
        return out
    _, nn = cKDTree(points[valid]).query(points[~valid], workers=-1)
    out[~valid] = out[valid][nn]
    return out


def evenly_spaced_indices(n: int, limit: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, limit, dtype=np.int64))


# ── Wall-thickness methods ────────────────────────────────────────
def method_edt_boundary_sum(lv_mask, epi_mask, myo_mask, volume_spacing, P, myo_idx, myo_tree):
    d_endo = distance_transform_edt(~lv_mask, sampling=volume_spacing)
    d_epi = distance_transform_edt(epi_mask, sampling=volume_spacing)
    thickness_vol = np.full(lv_mask.shape, np.nan, dtype=np.float32)
    thickness_vol[myo_mask] = (d_endo + d_epi)[myo_mask].astype(np.float32)
    return sample_field(thickness_vol, P, myo_idx, myo_tree)


def method_laplace_field(lv_mask, epi_mask, myo_mask, volume_spacing, P, myo_idx, myo_tree,
                         tol=1e-5, maxiter=3000):
    idx = np.argwhere(myo_mask).astype(np.int64)
    n = len(idx)
    local = -np.ones(lv_mask.shape, dtype=np.int64)
    local[myo_mask] = np.arange(n, dtype=np.int64)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(n, dtype=np.float64)
    axis_steps = [
        (np.array([1, 0, 0]), volume_spacing[0]), (np.array([-1, 0, 0]), volume_spacing[0]),
        (np.array([0, 1, 0]), volume_spacing[1]), (np.array([0, -1, 0]), volume_spacing[1]),
        (np.array([0, 0, 1]), volume_spacing[2]), (np.array([0, 0, -1]), volume_spacing[2]),
    ]
    shape = np.asarray(lv_mask.shape)
    for row, voxel in enumerate(idx):
        diag = 0.0
        for delta, h in axis_steps:
            nb = voxel + delta
            w = 1.0 / (float(h) ** 2)
            if np.any(nb < 0) or np.any(nb >= shape):
                diag += w
                b[row] += w * 1.0
                continue
            nb_t = tuple(int(v) for v in nb)
            if myo_mask[nb_t]:
                cols.append(int(local[nb_t]))
                rows.append(row)
                vals.append(-w)
                diag += w
            elif lv_mask[nb_t]:
                diag += w
            else:
                diag += w
                b[row] += w * 1.0
        rows.append(row)
        cols.append(row)
        vals.append(diag)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    try:
        psi_vec, info = spla.cg(A, b, rtol=tol, maxiter=maxiter)
    except TypeError:
        psi_vec, info = spla.cg(A, b, tol=tol, maxiter=maxiter)
    psi = np.full(lv_mask.shape, np.nan, dtype=np.float32)
    psi[myo_mask] = psi_vec.astype(np.float32)
    psi_filled = np.nan_to_num(psi, nan=0.0)
    grads = np.gradient(psi_filled, *volume_spacing, edge_order=1)
    grad_mag = np.sqrt(sum(g * g for g in grads))
    thickness_vol = np.full(lv_mask.shape, np.nan, dtype=np.float32)
    valid = myo_mask & np.isfinite(grad_mag) & (grad_mag > 1e-6)
    thickness_vol[valid] = (1.0 / grad_mag[valid]).astype(np.float32)
    values = sample_field(thickness_vol, P, myo_idx, myo_tree)
    return fill_invalid(P, values)


def method_yezzi_prince(lv_mask, epi_mask, myo_mask, volume_spacing, P, myo_idx, myo_tree):
    try:
        raw = compute_thickness_cardiac(endo=lv_mask.astype(bool), epi=epi_mask.astype(bool),
                                        sampling=volume_spacing)
    except TypeError:
        raw = compute_thickness_cardiac(endo=lv_mask.astype(bool), epi=epi_mask.astype(bool))
        raw = np.asarray(raw, dtype=np.float32) * float(np.mean(volume_spacing))
    thickness_vol = np.full(lv_mask.shape, np.nan, dtype=np.float32)
    thickness_vol[myo_mask] = np.asarray(raw, dtype=np.float32)[myo_mask]
    values = sample_field(thickness_vol, P, myo_idx, myo_tree)
    return fill_invalid(P, values)


def cone_rays(normal: np.ndarray, k: int = CONE_RAY_COUNT, alpha_deg: float = 30.0) -> np.ndarray:
    normal = np.asarray(normal, dtype=np.float64)
    normal /= max(np.linalg.norm(normal), 1e-12)
    alpha = math.radians(alpha_deg)
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(normal, u)
    phi = np.linspace(0.0, 2.0 * math.pi, k, endpoint=False)
    dirs = (normal[None, :] * math.cos(alpha)
            + np.cos(phi)[:, None] * u[None, :] * math.sin(alpha)
            + np.sin(phi)[:, None] * v[None, :] * math.sin(alpha))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True).clip(min=1e-12)
    return np.vstack([normal[None, :], dirs])


def ray_hits(origins, directions, epi_mesh, chunk_size=1024) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-12)
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(epi_mesh)
    distances = np.full(len(origins), np.nan, dtype=np.float32)
    for start in range(0, len(origins), chunk_size):
        stop = min(start + chunk_size, len(origins))
        locations, index_ray, _ = intersector.intersects_location(
            ray_origins=origins[start:stop], ray_directions=directions[start:stop],
            multiple_hits=False,
        )
        if not len(locations):
            continue
        d = np.linalg.norm(locations - origins[start:stop][index_ray], axis=1)
        for loc_dist, local_ray_idx in zip(d, index_ray):
            ray_idx = start + int(local_ray_idx)
            if loc_dist <= 1e-6:
                continue
            old = distances[ray_idx]
            if not np.isfinite(old) or loc_dist < old:
                distances[ray_idx] = loc_dist
    return distances


def method_sdf_cone_rays(P, Q, F, endo_normals, epi_mesh, k=CONE_RAY_COUNT, alpha_deg=30.0):
    sample_idx = evenly_spaced_indices(len(P), CONE_RAY_SAMPLE_LIMIT)
    all_origins, all_dirs, owner = [], [], []
    for local_i, point_idx in enumerate(sample_idx):
        dirs = cone_rays(endo_normals[point_idx], k=k, alpha_deg=alpha_deg)
        all_origins.append(np.repeat(P[point_idx][None, :], len(dirs), axis=0))
        all_dirs.append(dirs)
        owner.extend([local_i] * len(dirs))
    origins = np.vstack(all_origins)
    dirs = np.vstack(all_dirs)
    owner = np.asarray(owner, dtype=np.int64)
    hit = ray_hits(origins, dirs, epi_mesh)
    sample_values = np.full(len(sample_idx), np.nan, dtype=np.float32)
    for local_i in range(len(sample_idx)):
        vals = hit[owner == local_i]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            sample_values[local_i] = float(np.median(vals))
    values = np.full(len(P), np.nan, dtype=np.float32)
    valid = np.isfinite(sample_values)
    if valid.any():
        values[sample_idx[valid]] = sample_values[valid]
        values = fill_invalid(P, values)
    else:
        fallback, _ = cKDTree(Q).query(P, workers=-1)
        values = np.asarray(fallback, dtype=np.float32)
    return values


# ── AHA-17 ────────────────────────────────────────────────────────
def detect_base_apex(seg_volume, spacing_mm):
    labels = np.rint(seg_volume).astype(np.int16)
    lv = labels == LBL_LV
    epi = (labels == LBL_LV) | (labels == LBL_MYO)
    support = np.flatnonzero(epi.any(axis=(0, 1)))
    lv_area = lv.sum(axis=(0, 1)).astype(float)
    epi_area = epi.sum(axis=(0, 1)).astype(float)
    first, last = int(support[0]), int(support[-1])
    first_score = float(lv_area[first] + 0.25 * epi_area[first])
    last_score = float(lv_area[last] + 0.25 * epi_area[last])
    if first_score >= last_score:
        base_slice, apex_slice = first, last
    else:
        base_slice, apex_slice = last, first
    return base_slice * spacing_mm[2], apex_slice * spacing_mm[2]


def assign_aha17(vertices, seg_volume, spacing_mm, apical_end=0.85):
    base_z, apex_z = detect_base_apex(seg_volume, spacing_mm)
    labels = np.rint(seg_volume).astype(np.int16)
    vertices = np.asarray(vertices, dtype=np.float32)
    denom = apex_z - base_z
    z = np.clip((vertices[:, 2] - base_z) / denom, 0.0, 1.0)
    cx, cy = float(vertices[:, 0].mean()), float(vertices[:, 1].mean())
    anterior_angle = 0.0
    rv_mask = labels == LBL_RV
    if rv_mask.any():
        rv_pts = voxel_to_world(np.argwhere(rv_mask), spacing_mm)
        rv_xy = rv_pts[:, :2].mean(axis=0)
        septal_angle = math.atan2(float(rv_xy[1] - cy), float(rv_xy[0] - cx))
        anterior_angle = septal_angle - math.pi / 2.0
    angles = np.degrees(np.arctan2(vertices[:, 1] - cy, vertices[:, 0] - cx) - anterior_angle) % 360.0
    ids = np.full(len(vertices), 17, dtype=np.int16)
    basal = z < (1 / 3)
    mid = (z >= 1 / 3) & (z < 2 / 3)
    apical = (z >= 2 / 3) & (z < apical_end)
    seg6 = (angles / 60.0).astype(np.int16) % 6
    seg4 = (angles / 90.0).astype(np.int16) % 4
    ids[basal] = 1 + seg6[basal]
    ids[mid] = 7 + seg6[mid]
    ids[apical] = 13 + seg4[apical]
    return ids


# ── Per-patient pipeline ──────────────────────────────────────────
def process_patient(patient_id: str, model, cfg) -> dict | None:
    case_dir = DATA_ROOT / patient_id
    info = {}
    for line in (case_dir / "Info.cfg").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    ed_frame = int(info["ED"])
    group = info.get("Group", "?")
    seg_path = case_dir / f"{patient_id}_frame{ed_frame:02d}_gt.nii"
    try:
        nif = load_nifti(seg_path)
    except Exception as exc:
        print(f"  SKIP {patient_id}: cannot load seg ({exc})")
        return None
    seg = np.rint(nif["data"]).astype(np.int16)
    spacing = tuple(float(v) for v in nif["zooms"])
    affine = nif["affine"]
    if not (seg == LBL_LV).any() or not (seg == LBL_MYO).any():
        print(f"  SKIP {patient_id}: missing LV/MYO")
        return None

    ref_mean = _reference_wall_thickness_from_segmentation(seg, spacing)
    if ref_mean is None:
        print(f"  SKIP {patient_id}: reference failed")
        return None

    contours = extract_contours(seg, affine, spacing[2])
    scale = contours["scale"]
    centroid = contours["centroid"]
    endo_mesh, epi_mesh = build_full_meshes(model, cfg, contours, scale, centroid)
    P = np.asarray(endo_mesh.vertices, dtype=np.float32)
    Q = np.asarray(epi_mesh.vertices, dtype=np.float32)
    F = np.asarray(endo_mesh.faces, dtype=np.int32)

    inv_affine = np.linalg.inv(affine)

    def world_to_voxel(pts_mm):
        ones = np.ones((len(pts_mm), 1), dtype=np.float64)
        return (inv_affine @ np.hstack([pts_mm.astype(np.float64), ones]).T).T[:, :3]

    endo_vox = trimesh.Trimesh(vertices=world_to_voxel(P), faces=F, process=False)
    epi_vox = trimesh.Trimesh(vertices=world_to_voxel(Q),
                              faces=np.asarray(epi_mesh.faces, dtype=np.int32), process=False)
    lv_mask = voxelize_mesh_to_grid(endo_vox, seg.shape)
    epi_mask = voxelize_mesh_to_grid(epi_vox, seg.shape)
    myo_mask = epi_mask & ~lv_mask
    if not myo_mask.any():
        print(f"  SKIP {patient_id}: empty model myocardium")
        return None

    myo_idx = np.argwhere(myo_mask).astype(np.int64)
    myo_tree = cKDTree(voxel_to_world(myo_idx, spacing))
    endo_normals = orient_normals(endo_mesh, Q)
    kd_fallback, _ = cKDTree(Q).query(P, workers=-1)
    kd_fallback = np.asarray(kd_fallback, dtype=np.float32)

    fields = {}
    fields["Laplace field"] = method_laplace_field(lv_mask, epi_mask, myo_mask, spacing, P, myo_idx, myo_tree)
    fields["Yezzi-Prince"] = method_yezzi_prince(lv_mask, epi_mask, myo_mask, spacing, P, myo_idx, myo_tree)
    fields["SDF cone rays"] = method_sdf_cone_rays(P, Q, F, endo_normals, epi_mesh)
    fields["EDT boundary sum"] = method_edt_boundary_sum(lv_mask, epi_mask, myo_mask, spacing, P, myo_idx, myo_tree)

    # Clean invalid/out-of-range values.
    for name in METHODS:
        v = np.asarray(fields[name], dtype=np.float32)
        bad = ~np.isfinite(v) | (v < 0) | (v > THICKNESS_MAX_MM)
        if bad.any():
            v[bad] = np.nan
            if np.isfinite(v).any():
                v = fill_invalid(P, v)
            else:
                v = kd_fallback.copy()
        fields[name] = v

    aha_ids = assign_aha17(P, seg, spacing)

    # Reconstruction quality (vs derived segmentation surfaces).
    lv_ref = seg == LBL_LV
    epi_ref = lv_ref | (seg == LBL_MYO)
    endo_watertight = bool(endo_mesh.is_watertight)
    epi_watertight = bool(epi_mesh.is_watertight)

    # Slice residual: distance of endo mesh vertices to nearest input endo contour point.
    endo_contour = contours["xyz"][contours["tissue"] == 0]
    if len(endo_contour):
        sr, _ = cKDTree(endo_contour).query(P, workers=-1)
        slice_residual = float(np.mean(np.asarray(sr) * scale))
    else:
        slice_residual = float("nan")

    # Volume ratios (model vs segmentation) in voxel counts * voxel volume.
    voxvol = float(spacing[0] * spacing[1] * spacing[2])
    lv_vol_model = float(lv_mask.sum()) * voxvol
    lv_vol_seg = float((seg == LBL_LV).sum()) * voxvol
    epi_vol_model = float(epi_mask.sum()) * voxvol
    epi_vol_seg = float(epi_ref.sum()) * voxvol
    vol_ratio_endo = lv_vol_model / lv_vol_seg if lv_vol_seg else float("nan")
    vol_ratio_epi = epi_vol_model / epi_vol_seg if epi_vol_seg else float("nan")

    row = {
        "patient": patient_id,
        "group": group,
        "ref_mean": float(ref_mean),
        "slice_residual_mm": slice_residual,
        "vol_ratio_endo": vol_ratio_endo,
        "vol_ratio_epi": vol_ratio_epi,
        "endo_watertight": endo_watertight,
        "epi_watertight": epi_watertight,
        "n_vertices": int(len(P)),
    }

    payload = {
        "row": row,
        "fields": fields,
        "aha_ids": aha_ids,
        "P": P, "Q": Q, "F": F,
        "epi_faces": np.asarray(epi_mesh.faces, dtype=np.int32),
        "contours_xyz": contours["xyz"],
        "contours_tissue": contours["tissue"],
        "scale": float(scale),
        "centroid": np.asarray(centroid, dtype=np.float32),
        "ref_mean": float(ref_mean),
        "spacing": spacing,
    }

    # Smooth, un-snapped visualisation meshes for the representative patient
    # (used only by the figure scripts, never by the quantitative metrics).
    if patient_id == REPRESENTATIVE:
        viz_endo, viz_epi = build_full_meshes(model, cfg, contours, scale, centroid, snap=False)
        payload["viz_endo_v"] = np.asarray(viz_endo.vertices, dtype=np.float32)
        payload["viz_endo_f"] = np.asarray(viz_endo.faces, dtype=np.int32)
        payload["viz_epi_v"] = np.asarray(viz_epi.vertices, dtype=np.float32)
        payload["viz_epi_f"] = np.asarray(viz_epi.faces, dtype=np.int32)

    return payload


def _flush(quality_rows, method_rows, aha_rows):
    """Write CSVs after every patient so a crash never loses completed work."""
    pd.DataFrame(quality_rows).to_csv(OUT_DIR / "cohort_reconstruction_quality.csv", index=False)
    pd.DataFrame(method_rows).to_csv(OUT_DIR / "cohort_method_summary.csv", index=False)
    pd.DataFrame(aha_rows).to_csv(OUT_DIR / "cohort_aha17.csv", index=False)


def main():
    warnings.filterwarnings("ignore")
    print(f"Loading model: {MODEL_PATH}")
    model, cfg = load_model(MODEL_PATH)
    patients = sorted(p.name for p in DATA_ROOT.iterdir()
                      if p.is_dir() and p.name.startswith("patient"))
    # Optional subset: RESULTS_PATIENTS=patient001,patient002 (validate before full run).
    subset = os.environ.get("RESULTS_PATIENTS", "").strip()
    if subset:
        wanted = {s.strip() for s in subset.split(",") if s.strip()}
        patients = [p for p in patients if p in wanted]
    print(f"Threads={_THREADS}  Grid={GRID_RES}  Patients ({len(patients)}): {patients}\n")

    quality_rows = []
    method_rows = []
    aha_rows = []
    corr_records = {}  # per-vertex fields of representative patient

    for pid in patients:
        t0 = time.perf_counter()
        try:
            payload = process_patient(pid, model, cfg)
        except MemoryError:
            print(f"  ERROR {pid}: out of memory — skipping (lower CARDIOSDF_GRID_RES)")
            gc.collect()
            continue
        except Exception as exc:
            print(f"  ERROR {pid}: {exc}")
            gc.collect()
            continue
        if payload is None:
            gc.collect()
            continue
        row = payload["row"]
        fields = payload["fields"]
        aha_ids = payload["aha_ids"]
        ref_mean = payload["ref_mean"]
        quality_rows.append(row)

        for name in METHODS:
            v = fields[name]
            fin = v[np.isfinite(v)]
            raw_mean = float(fin.mean())
            factor = float(np.clip(ref_mean / raw_mean, 0.3, 4.0)) if raw_mean > 0.1 else 1.0
            cal = v * factor
            cfin = cal[np.isfinite(cal)]
            method_rows.append({
                "patient": pid, "method": name, "ref_mean_mm": ref_mean,
                "raw_mean_mm": raw_mean,
                "raw_std_mm": float(fin.std()),
                "raw_p5_mm": float(np.percentile(fin, 5)),
                "raw_p95_mm": float(np.percentile(fin, 95)),
                "raw_bias_mm": raw_mean - ref_mean,
                "cal_factor": factor,
                "cal_mean_mm": float(cfin.mean()),
                "cal_std_mm": float(cfin.std()),
                "cal_p5_mm": float(np.percentile(cfin, 5)),
                "cal_p95_mm": float(np.percentile(cfin, 95)),
            })
            for seg_id, seg_name in enumerate(AHA_17_NAMES, start=1):
                sv = v[aha_ids == seg_id]
                sv = sv[np.isfinite(sv)]
                aha_rows.append({
                    "patient": pid, "method": name, "segment_id": seg_id,
                    "segment": seg_name,
                    "mean_mm": float(sv.mean()) if sv.size else np.nan,
                    "p95_mm": float(np.percentile(sv, 95)) if sv.size else np.nan,
                })

        if pid == REPRESENTATIVE:
            np.savez_compressed(
                OUT_DIR / "representative_patient.npz",
                P=payload["P"], Q=payload["Q"], F=payload["F"],
                epi_faces=payload["epi_faces"],
                contours_xyz=payload["contours_xyz"],
                contours_tissue=payload["contours_tissue"],
                aha_ids=aha_ids,
                ref_mean=ref_mean,
                scale=payload["scale"],
                centroid=payload["centroid"],
                viz_endo_v=payload["viz_endo_v"], viz_endo_f=payload["viz_endo_f"],
                viz_epi_v=payload["viz_epi_v"], viz_epi_f=payload["viz_epi_f"],
                **{f"field_{n.replace(' ', '_')}": fields[n] for n in METHODS},
            )
            corr_records = {n: fields[n] for n in METHODS}
            print(f"  saved representative NPZ ({REPRESENTATIVE})")

        print(f"  {pid}: ref={ref_mean:.2f} mm  "
              + "  ".join(f"{n.split()[0]}={float(np.nanmean(fields[n])):.2f}" for n in METHODS)
              + f"  ({time.perf_counter()-t0:.1f}s)")

        # Persist after every patient and release memory before the next one.
        _flush(quality_rows, method_rows, aha_rows)
        del payload, fields, aha_ids
        gc.collect()

    quality_df = pd.DataFrame(quality_rows)
    method_df = pd.DataFrame(method_rows)
    aha_df = pd.DataFrame(aha_rows)
    quality_df.to_csv(OUT_DIR / "cohort_reconstruction_quality.csv", index=False)
    method_df.to_csv(OUT_DIR / "cohort_method_summary.csv", index=False)
    aha_df.to_csv(OUT_DIR / "cohort_aha17.csv", index=False)

    if corr_records:
        stacked = np.column_stack([corr_records[n] for n in METHODS])
        np.savez_compressed(OUT_DIR / "representative_pervertex.npz",
                            methods=np.array(METHODS), values=stacked)

    print(f"\nSaved cohort CSVs to {OUT_DIR}")
    print(f"  patients processed: {len(quality_rows)}")


if __name__ == "__main__":
    main()
