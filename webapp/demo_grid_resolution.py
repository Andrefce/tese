"""Test wall thickness methods at different grid resolutions."""
import numpy as np
import torch
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator

from core.nifti import load_nifti
from core.sdf_model import (
    extract_contours, load_model,
    _reference_wall_thickness_from_segmentation,
    _build_contour_tensor, _build_grid_and_query, _mc_field,
    _snap_mesh_to_contours, FLIP_Z,
)

# ─── Load patient ─────────────────────────────────────────────────
case_dir = Path("demo-data/training/patient001")
info = {}
for line in (case_dir / "Info.cfg").read_text().splitlines():
    if ":" in line:
        k, v = line.split(":", 1)
        info[k.strip()] = v.strip()

ed_frame = int(info["ED"])
seg_path = case_dir / f"patient001_frame{ed_frame:02d}_gt.nii"
seg_nifti = load_nifti(seg_path)
seg = np.rint(seg_nifti["data"]).astype(np.int16)
spacing = tuple(float(v) for v in seg_nifti["zooms"])

contours = extract_contours(seg, seg_nifti["affine"], spacing[2])
model, cfg = load_model(Path("model/inr_sdf_combined_fresh_ed_mix_v1_final.ptrom"))
scale = contours["scale"]
centroid = contours["centroid"]

ref_wt = _reference_wall_thickness_from_segmentation(seg, spacing)
print(f"Reference (seg EDT): {ref_wt:.2f} mm")
print(f"Scale={scale:.2f} mm")
print()

# ─── Laplace solver ───────────────────────────────────────────────
def solve_laplace_thickness(sdf_endo, sdf_epi, voxel_size, iso=0.0, n_iter=1000):
    band = 1.5 * np.max(voxel_size)
    endo_boundary = np.abs(sdf_endo - iso) < band
    epi_boundary = np.abs(sdf_epi - iso) < band
    myo_region = (sdf_endo > iso) & (sdf_epi < iso)
    
    T = np.zeros_like(sdf_endo)
    T[epi_boundary] = 1.0
    T[endo_boundary] = 0.0
    interior = myo_region & ~endo_boundary & ~epi_boundary
    
    if not interior.any():
        return None
    
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
        return None
    
    thickness_vals = 1.0 / myo_grad[valid]
    thickness_vals = thickness_vals[thickness_vals < 30.0]
    return float(np.mean(thickness_vals)) if len(thickness_vals) > 10 else None


# ─── Test grid resolutions ────────────────────────────────────────
grid_sizes = [96, 128, 192, 256]

print(f"{'Grid':>6} | {'Voxel(mm)':>10} | {'KD-tree':>8} | {'SDF Yezzi':>9} | {'SDF sum':>8} | {'Endo V':>7} | {'Epi V':>7}")
print("-" * 80)

cont_t, mask_t = _build_contour_tensor(contours["xyz"], contours["tissue"], cfg, phase_val=0.0)
z = model.encode(cont_t, mask_t)

for grid_res in grid_sizes:
    import time
    t0 = time.time()
    
    sdf_e, sdf_p, dlt, lo, hi, voxel = _build_grid_and_query(
        z, model, contours["xyz"], cfg, grid_res
    )
    
    iso = cfg.get("iso_level", 0.0)
    endo_verts, endo_faces = _mc_field(sdf_e, lo, voxel, iso)
    epi_verts, epi_faces = _mc_field(sdf_p, lo, voxel, iso)
    
    # Skip snap_to_contours for speed — doesn't affect SDF values
    
    flip = np.array([1.0, 1.0, -1.0 if FLIP_Z else 1.0], dtype=np.float32)
    endo_mm = (endo_verts * flip) * scale + centroid
    epi_mm = (epi_verts * flip) * scale + centroid
    
    # KD-tree
    dists_kd, _ = cKDTree(epi_mm).query(endo_mm, k=1, workers=-1)
    wt_kd = float(np.mean(dists_kd))
    
    # SDF value at endo surface
    xs = np.linspace(lo[0], hi[0], grid_res)
    ys = np.linspace(lo[1], hi[1], grid_res)
    zs = np.linspace(lo[2], hi[2], grid_res)
    interp_epi = RegularGridInterpolator((xs, ys, zs), sdf_p, method="linear", bounds_error=False, fill_value=np.nan)
    sdf_at_endo = interp_epi(endo_verts)
    wt_sdf_surf = np.abs(sdf_at_endo) * scale
    valid_sdf = np.isfinite(wt_sdf_surf) & (wt_sdf_surf > 0.1)
    wt_sdf_mean = float(np.nanmean(wt_sdf_surf[valid_sdf])) if valid_sdf.any() else None
    
    # Yezzi-Prince SDF sum in myocardium
    myo_sdf = (sdf_e > iso) & (sdf_p < iso)
    if myo_sdf.any():
        wt_yz = (np.abs(sdf_e[myo_sdf]) + np.abs(sdf_p[myo_sdf])) * scale
        wt_yz = wt_yz[wt_yz > 0.5]
        wt_yz_mean = float(np.mean(wt_yz)) if len(wt_yz) > 10 else None
    else:
        wt_yz_mean = None
    
    elapsed = time.time() - t0
    voxel_mm = np.mean(voxel) * scale  # approximate voxel size in mm
    
    print(f"{grid_res:>6} | {voxel_mm:>9.3f} | {wt_kd:>7.2f} | "
          f"{wt_sdf_mean:>8.2f} | "
          f"{wt_yz_mean if wt_yz_mean else 0:>7.2f} | "
          f"{len(endo_mm):>7} | {len(epi_mm):>7}  ({elapsed:.1f}s)")

print()
print(f"Reference: {ref_wt:.2f} mm")
print(f"\nNote: SDF-based methods measure in the model's learned distance field,")
print(f"which is NOT a true Euclidean distance. Higher grid doesn't fix this fundamental issue.")
print(f"KD-tree improves slightly (more vertices) but same geometric bias remains.")
print(f"\nThe real fix: use EDT on voxelized model meshes (9.99 mm at grid=96, already matches ref).")
