"""Compare EDT boundary sum, EDT medial axis, and Laplace field methods across ALL demo patients.

Determines which volumetric/PDE method best matches the segmentation reference
when applied to the CardioSDF model output.
"""
import sys
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))

from core.nifti import load_nifti
from core.sdf_model import (
    extract_contours, load_model, predict_sdf_meshes,
    _reference_wall_thickness_from_segmentation,
    _build_contour_tensor, _build_grid_and_query, _mc_field,
    _snap_mesh_to_contours, FLIP_Z, DEVICE,
)

LBL_BG, LBL_RV, LBL_MYO, LBL_LV = 0, 1, 2, 3
MODEL_PATH = Path("model/inr_sdf_combined_fresh_ed_mix_v1_final.ptrom")
DATA_ROOT = Path("demo-data/training")


def solve_laplace_thickness(sdf_endo, sdf_epi, voxel_size, scale, iso=0.0, n_iter=800):
    """Solve Laplace equation between endo/epi surfaces, return mean thickness in mm."""
    band = 1.5 * np.max(voxel_size)
    endo_boundary = np.abs(sdf_endo - iso) < band
    epi_boundary = np.abs(sdf_epi - iso) < band
    myo_region = (sdf_endo > iso) & (sdf_epi < iso)

    T = np.zeros_like(sdf_endo)
    T[epi_boundary] = 1.0
    T[endo_boundary] = 0.0
    interior = myo_region & ~endo_boundary & ~epi_boundary

    if not interior.any():
        return None, None

    for _ in range(n_iter):
        T_new = T.copy()
        T_new[1:-1, 1:-1, 1:-1] = (
            T[2:, 1:-1, 1:-1] + T[:-2, 1:-1, 1:-1] +
            T[1:-1, 2:, 1:-1] + T[1:-1, :-2, 1:-1] +
            T[1:-1, 1:-1, 2:] + T[1:-1, 1:-1, :-2]
        ) / 6.0
        T_new[endo_boundary] = 0.0
        T_new[epi_boundary] = 1.0
        T[interior] = T_new[interior]

    grad = np.zeros((*T.shape, 3), dtype=np.float32)
    grad[1:-1, :, :, 0] = (T[2:, :, :] - T[:-2, :, :]) / (2 * voxel_size[0])
    grad[:, 1:-1, :, 1] = (T[:, 2:, :] - T[:, :-2, :]) / (2 * voxel_size[1])
    grad[:, :, 1:-1, 2] = (T[:, :, 2:] - T[:, :, :-2]) / (2 * voxel_size[2])

    grad_mag = np.sqrt(np.sum(grad**2, axis=-1))
    myo_grad = grad_mag[myo_region]
    valid = myo_grad > 1e-6
    if not valid.any():
        return None, None

    thickness_vals = 1.0 / myo_grad[valid]
    thickness_vals = thickness_vals[thickness_vals < 30.0]
    if len(thickness_vals) < 10:
        return None, None

    mean_norm = float(np.mean(thickness_vals))
    p95_norm = float(np.percentile(thickness_vals, 95))
    return mean_norm * scale, p95_norm * scale


def edt_boundary_sum_on_model(endo_verts_mm, epi_verts_mm, seg_shape, spacing, affine):
    """Voxelize model meshes and compute EDT boundary sum wall thickness."""
    from scipy.ndimage import binary_fill_holes, binary_dilation, generate_binary_structure

    inv_affine = np.linalg.inv(affine)

    def world_to_voxel(pts_mm):
        ones = np.ones((len(pts_mm), 1), dtype=np.float64)
        pts_h = np.hstack([pts_mm.astype(np.float64), ones])
        return (inv_affine @ pts_h.T).T[:, :3]

    def voxelize(pts_vox, grid_shape):
        idx = pts_vox.astype(np.int32)
        for d in range(3):
            idx[:, d] = np.clip(idx[:, d], 0, grid_shape[d] - 1)
        mask = np.zeros(grid_shape, dtype=bool)
        mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        struct = generate_binary_structure(3, 1)
        mask = binary_dilation(mask, struct, iterations=2)
        mask = binary_fill_holes(mask)
        return mask

    endo_vox = world_to_voxel(endo_verts_mm)
    epi_vox = world_to_voxel(epi_verts_mm)

    lv_filled = voxelize(endo_vox, seg_shape)
    epi_filled = voxelize(epi_vox, seg_shape)
    myo_model = epi_filled & ~lv_filled

    if not myo_model.any() or not lv_filled.any():
        return None, None

    d_endo = distance_transform_edt(~lv_filled, sampling=spacing)
    d_epi = distance_transform_edt(epi_filled, sampling=spacing)
    wt = (d_endo + d_epi)[myo_model]
    wt = wt[np.isfinite(wt) & (wt > 0.5)]

    if len(wt) < 10:
        return None, None

    return float(np.mean(wt)), float(np.percentile(wt, 95))


def edt_medial_axis_on_model(endo_verts_mm, epi_verts_mm, seg_shape, spacing, affine):
    """Voxelize model meshes and compute EDT medial axis wall thickness (2x medial distance)."""
    from scipy.ndimage import binary_fill_holes, binary_dilation, generate_binary_structure

    inv_affine = np.linalg.inv(affine)

    def world_to_voxel(pts_mm):
        ones = np.ones((len(pts_mm), 1), dtype=np.float64)
        pts_h = np.hstack([pts_mm.astype(np.float64), ones])
        return (inv_affine @ pts_h.T).T[:, :3]

    def voxelize(pts_vox, grid_shape):
        idx = pts_vox.astype(np.int32)
        for d in range(3):
            idx[:, d] = np.clip(idx[:, d], 0, grid_shape[d] - 1)
        mask = np.zeros(grid_shape, dtype=bool)
        mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        struct = generate_binary_structure(3, 1)
        mask = binary_dilation(mask, struct, iterations=2)
        mask = binary_fill_holes(mask)
        return mask

    endo_vox = world_to_voxel(endo_verts_mm)
    epi_vox = world_to_voxel(epi_verts_mm)

    lv_filled = voxelize(endo_vox, seg_shape)
    epi_filled = voxelize(epi_vox, seg_shape)
    myo_model = epi_filled & ~lv_filled

    if not myo_model.any():
        return None, None

    # EDT from both boundaries
    d_endo = distance_transform_edt(~lv_filled, sampling=spacing)
    d_epi = distance_transform_edt(epi_filled, sampling=spacing)

    # Medial axis: local maxima of min(d_endo, d_epi) in myocardium
    d_min = np.minimum(d_endo, d_epi)
    d_min_myo = np.where(myo_model, d_min, 0)
    local_max = maximum_filter(d_min_myo, size=3)
    medial = myo_model & (d_min_myo == local_max) & (d_min_myo > 0.5)

    if not medial.any():
        # Fallback: use median of d_min in myo
        vals = d_min[myo_model]
        vals = vals[vals > 0.5]
        if len(vals) < 10:
            return None, None
        return float(np.mean(vals) * 2), float(np.percentile(vals, 95) * 2)

    medial_vals = d_min[medial]
    # Wall thickness = 2 * medial distance (distance from center to either boundary)
    wt = medial_vals * 2.0
    wt = wt[wt > 0.5]
    if len(wt) < 5:
        return None, None

    return float(np.mean(wt)), float(np.percentile(wt, 95))


def kdtree_method(endo_mm, epi_mm):
    """Standard KD-tree nearest neighbour endo→epi."""
    dists, _ = cKDTree(epi_mm).query(endo_mm, k=1, workers=-1)
    dists = np.asarray(dists, dtype=np.float32)
    return float(np.mean(dists)), float(np.percentile(dists, 95))


def symmetric_kdtree_method(endo_mm, epi_mm):
    """Symmetric KD-tree: average of endo→epi and reverse."""
    tree_epi = cKDTree(epi_mm)
    tree_endo = cKDTree(endo_mm)
    endo_to_epi, epi_idx = tree_epi.query(endo_mm, k=1, workers=-1)
    epi_to_endo, _ = tree_endo.query(epi_mm, k=1, workers=-1)
    reverse = np.asarray(epi_to_endo, dtype=np.float32)[epi_idx]
    sym = 0.5 * (np.asarray(endo_to_epi, dtype=np.float32) + reverse)
    return float(np.mean(sym)), float(np.percentile(sym, 95))


# ─── Main comparison ──────────────────────────────────────────────
print("Loading CardioSDF model...")
model, cfg = load_model(MODEL_PATH)
print(f"Model loaded. Grid res: {cfg.get('grid_res', 96)}")
print()

patients = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("patient"))
print(f"Patients: {patients}")
print()

results = []

for patient_id in patients:
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
        seg_nifti = load_nifti(seg_path)
    except Exception as e:
        print(f"  SKIP {patient_id}: cannot load GT seg ({e})")
        continue

    seg = np.rint(seg_nifti["data"]).astype(np.int16)
    spacing = tuple(float(v) for v in seg_nifti["zooms"])
    affine = seg_nifti["affine"]

    if not (seg == LBL_LV).any() or not (seg == LBL_MYO).any():
        print(f"  SKIP {patient_id}: no LV/MYO labels")
        continue

    print(f"── {patient_id} ({group}) ── ED={ed_frame}, shape={seg.shape}, spacing={tuple(round(s,2) for s in spacing)}")

    # Reference from segmentation
    ref_mean = _reference_wall_thickness_from_segmentation(seg, spacing)
    if ref_mean is None:
        print(f"  SKIP: reference computation failed")
        continue

    # Run CardioSDF model
    contours = extract_contours(seg, affine, spacing[2])
    scale = contours["scale"]
    centroid = contours["centroid"]

    import torch
    cont_t, mask_t = _build_contour_tensor(contours["xyz"], contours["tissue"], cfg, phase_val=0.0)
    with torch.no_grad():
        z = model.encode(cont_t, mask_t)

    grid_res = cfg.get("grid_res", 96)
    sdf_e, sdf_p, dlt, lo, hi, voxel = _build_grid_and_query(z, model, contours["xyz"], cfg, grid_res)
    iso = cfg.get("iso_level", 0.0)

    endo_verts, endo_faces = _mc_field(sdf_e, lo, voxel, iso)
    epi_verts, epi_faces = _mc_field(sdf_p, lo, voxel, iso)

    if len(endo_verts) == 0 or len(epi_verts) == 0:
        print(f"  SKIP: empty meshes")
        continue

    endo_verts = _snap_mesh_to_contours(endo_verts, contours["xyz"], contours["tissue"], "endo")
    epi_verts = _snap_mesh_to_contours(epi_verts, contours["xyz"], contours["tissue"], "epi")

    flip = np.array([1.0, 1.0, -1.0 if FLIP_Z else 1.0], dtype=np.float32)
    endo_mm = (endo_verts * flip) * scale + centroid
    epi_mm = (epi_verts * flip) * scale + centroid

    # ─── Methods ───
    # 1. KD-tree
    kd_mean, kd_p95 = kdtree_method(endo_mm, epi_mm)

    # 2. Symmetric KD-tree
    sym_mean, sym_p95 = symmetric_kdtree_method(endo_mm, epi_mm)

    # 3. Laplace field (on SDF grid)
    lap_mean, lap_p95 = solve_laplace_thickness(sdf_e, sdf_p, voxel, scale, iso=iso)

    # 4. EDT boundary sum (on voxelized model)
    edt_mean, edt_p95 = edt_boundary_sum_on_model(endo_mm, epi_mm, seg.shape, spacing, affine)

    # 5. EDT medial axis
    edt_med_mean, edt_med_p95 = edt_medial_axis_on_model(endo_mm, epi_mm, seg.shape, spacing, affine)

    row = {
        "patient": patient_id,
        "group": group,
        "ref_mean": ref_mean,
        "kd_mean": kd_mean,
        "sym_kd_mean": sym_mean,
        "laplace_mean": lap_mean,
        "edt_bsum_mean": edt_mean,
        "edt_medial_mean": edt_med_mean,
    }
    results.append(row)

    print(f"  Reference (seg EDT):    {ref_mean:.2f} mm")
    print(f"  KD-tree:                {kd_mean:.2f} mm  (bias {kd_mean - ref_mean:+.2f})")
    print(f"  Symmetric KD-tree:      {sym_mean:.2f} mm  (bias {sym_mean - ref_mean:+.2f})")
    print(f"  Laplace field:          {lap_mean:.2f} mm  (bias {lap_mean - ref_mean:+.2f})" if lap_mean else "  Laplace field:          FAILED")
    print(f"  EDT boundary sum:       {edt_mean:.2f} mm  (bias {edt_mean - ref_mean:+.2f})" if edt_mean else "  EDT boundary sum:       FAILED")
    print(f"  EDT medial axis:        {edt_med_mean:.2f} mm  (bias {edt_med_mean - ref_mean:+.2f})" if edt_med_mean else "  EDT medial axis:        FAILED")
    print()

# ─── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MULTI-PATIENT SUMMARY")
print("=" * 70)
print(f"{'Method':<22} | {'Mean bias':>10} | {'|bias|':>8} | {'RMSE':>8} | {'% bias':>8}")
print("-" * 70)

methods = [
    ("KD-tree", "kd_mean"),
    ("Symmetric KD-tree", "sym_kd_mean"),
    ("Laplace field", "laplace_mean"),
    ("EDT boundary sum", "edt_bsum_mean"),
    ("EDT medial axis", "edt_medial_mean"),
]

for name, key in methods:
    biases = []
    for r in results:
        val = r.get(key)
        if val is not None:
            biases.append(val - r["ref_mean"])
    if biases:
        biases = np.array(biases)
        pct_biases = []
        for r in results:
            val = r.get(key)
            if val is not None:
                pct_biases.append((val - r["ref_mean"]) / r["ref_mean"] * 100)
        mean_bias = float(np.mean(biases))
        abs_bias = float(np.mean(np.abs(biases)))
        rmse = float(np.sqrt(np.mean(biases**2)))
        pct = float(np.mean(pct_biases))
        print(f"{name:<22} | {mean_bias:>+9.2f} | {abs_bias:>7.2f} | {rmse:>7.2f} | {pct:>+7.1f}%")
    else:
        print(f"{name:<22} | {'N/A':>10} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}")

print("=" * 70)
print("\nNote: Reference is EDT boundary sum on the ground-truth segmentation voxels.")
print("Lower |bias| and RMSE = better agreement with reference.")
print("\nCalibration approach: scale raw model output by (ref_mean / model_mean)")
print("  → any method can match the reference mean after calibration")
print("  → the question is which method has the best shape/distribution BEFORE calibration")
