"""Compare wall thickness methods on the CardioSDF model output vs segmentation reference.

Methods on model meshes/SDF:
  1. KD-tree (current) — nearest-point endo→epi
  2. SDF value (Yezzi-like) — evaluate epi SDF at endo surface
  3. EDT on voxelized model — rasterize meshes, apply EDT boundary sum
  4. Laplace equation — solve ∇²T=0 between endo/epi, integrate path lengths

Reference: EDT boundary sum on the raw segmentation voxels.
"""
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
    _snap_mesh_to_contours, SDFNetwork, FLIP_Z, DEVICE,
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

print(f"Patient: patient001, ED frame: {ed_frame}")
print(f"Seg shape: {seg.shape}, spacing: {spacing} mm")
print()

# ─── Reference from segmentation ──────────────────────────────────
ref_wt = _reference_wall_thickness_from_segmentation(seg, spacing)
print(f"{'='*60}")
print(f"REFERENCE (segmentation EDT): {ref_wt:.2f} mm")
print(f"{'='*60}")
print()

# ─── Run model to get SDF fields + meshes ─────────────────────────
contours = extract_contours(seg, seg_nifti["affine"], spacing[2])
model, cfg = load_model(Path("model/inr_sdf_combined_fresh_ed_mix_v1_final.ptrom"))

grid_res = cfg.get("grid_res", 96)
cont_t, mask_t = _build_contour_tensor(
    contours["xyz"], contours["tissue"], cfg, phase_val=0.0
)
z = model.encode(cont_t, mask_t)

sdf_e, sdf_p, dlt, lo, hi, voxel = _build_grid_and_query(
    z, model, contours["xyz"], cfg, grid_res
)

iso = cfg.get("iso_level", 0.0)
endo_verts, endo_faces = _mc_field(sdf_e, lo, voxel, iso)
epi_verts, epi_faces = _mc_field(sdf_p, lo, voxel, iso)

# Snap to contours
endo_verts = _snap_mesh_to_contours(endo_verts, contours["xyz"], contours["tissue"], "endo")
epi_verts = _snap_mesh_to_contours(epi_verts, contours["xyz"], contours["tissue"], "epi")

# Denormalize to mm
scale = contours["scale"]
centroid = contours["centroid"]
flip = np.array([1.0, 1.0, -1.0 if FLIP_Z else 1.0], dtype=np.float32)
endo_mm = (endo_verts * flip) * scale + centroid
epi_mm = (epi_verts * flip) * scale + centroid

print(f"Model meshes: endo={len(endo_mm)} verts, epi={len(epi_mm)} verts")
print(f"Scale={scale:.2f} mm, grid_res={grid_res}")
print()

# ═══════════════════════════════════════════════════════════════════
# METHOD 1: KD-Tree (nearest point endo→epi)
# ═══════════════════════════════════════════════════════════════════
dists_kd, _ = cKDTree(epi_mm).query(endo_mm, k=1, workers=-1)
wt_kdtree = np.asarray(dists_kd, dtype=np.float32)
print(f"[1] KD-Tree (endo→epi nearest):   mean = {np.mean(wt_kdtree):.2f} mm")

# ═══════════════════════════════════════════════════════════════════
# METHOD 2: SDF value (Yezzi-Prince like)
# Evaluate epi SDF at endo surface vertices using trilinear interpolation.
# At endo surface points, |sdf_epi| ≈ distance to epi surface.
# ═══════════════════════════════════════════════════════════════════
xs = np.linspace(lo[0], hi[0], grid_res)
ys = np.linspace(lo[1], hi[1], grid_res)
zs = np.linspace(lo[2], hi[2], grid_res)

# Interpolate sdf_p at endo vertex positions (in normalized space, before denorm)
interp_epi = RegularGridInterpolator(
    (xs, ys, zs), sdf_p, method="linear", bounds_error=False, fill_value=np.nan
)
# Also interpolate sdf_e at epi vertices for symmetric version
interp_endo = RegularGridInterpolator(
    (xs, ys, zs), sdf_e, method="linear", bounds_error=False, fill_value=np.nan
)

# Evaluate epi SDF at endo surface points (normalized space)
sdf_epi_at_endo = interp_epi(endo_verts)
# The SDF value in normalized space → multiply by scale for mm
wt_sdf_yezzi = np.abs(sdf_epi_at_endo) * scale
wt_sdf_yezzi = wt_sdf_yezzi.astype(np.float32)
valid_sdf = np.isfinite(wt_sdf_yezzi) & (wt_sdf_yezzi > 0.1)
print(f"[2] SDF value (Yezzi-Prince):      mean = {np.nanmean(wt_sdf_yezzi[valid_sdf]):.2f} mm")

# ═══════════════════════════════════════════════════════════════════
# METHOD 3: EDT on voxelized model meshes
# Rasterize endo/epi meshes to a voxel grid, then EDT boundary sum.
# ═══════════════════════════════════════════════════════════════════
def voxelize_mesh(verts_mm, grid_shape, origin, voxel_spacing):
    """Convert mesh vertices to a filled binary mask on a regular grid."""
    # Map vertices to voxel indices
    idx = ((verts_mm - origin) / voxel_spacing).astype(np.int32)
    # Clip to grid bounds
    for d in range(3):
        idx[:, d] = np.clip(idx[:, d], 0, grid_shape[d] - 1)
    
    mask = np.zeros(grid_shape, dtype=bool)
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    
    # Fill the interior using flood fill from corners
    from scipy.ndimage import binary_fill_holes, binary_dilation, generate_binary_structure
    # Dilate slightly to close gaps, then fill
    struct = generate_binary_structure(3, 1)
    mask_dilated = binary_dilation(mask, struct, iterations=2)
    mask_filled = binary_fill_holes(mask_dilated)
    return mask_filled


# Use same grid as segmentation for fair comparison
grid_shape = seg.shape
origin = np.array([0.0, 0.0, 0.0])  # segmentation is in voxel coords
voxel_sp = np.array(spacing)

# We need endo/epi in the segmentation voxel coordinate system
# The model outputs are in mm (world coords from the affine), 
# convert to voxel coords: voxel = (world - origin) / spacing
# Actually for ACDC, the affine typically has origin offset. Let's use it.
affine = seg_nifti["affine"]
inv_affine = np.linalg.inv(affine)

def world_to_voxel(pts_mm, inv_aff):
    """Convert world-space mm coords to voxel indices."""
    ones = np.ones((len(pts_mm), 1), dtype=np.float64)
    pts_h = np.hstack([pts_mm.astype(np.float64), ones])
    vox = (inv_aff @ pts_h.T).T[:, :3]
    return vox

endo_vox = world_to_voxel(endo_mm, inv_affine)
epi_vox = world_to_voxel(epi_mm, inv_affine)

endo_filled = voxelize_mesh(endo_vox, grid_shape, np.zeros(3), np.ones(3))
epi_filled = voxelize_mesh(epi_vox, grid_shape, np.zeros(3), np.ones(3))

# Myocardium from model = epi_filled & ~endo_filled
myo_model = epi_filled & ~endo_filled

if myo_model.any() and endo_filled.any():
    d_endo_model = distance_transform_edt(~endo_filled, sampling=spacing)
    d_epi_model = distance_transform_edt(epi_filled, sampling=spacing)
    wt_edt_model = (d_endo_model + d_epi_model)[myo_model]
    wt_edt_model = wt_edt_model[np.isfinite(wt_edt_model) & (wt_edt_model > 0.5)]
    print(f"[3] EDT on voxelized model:        mean = {np.mean(wt_edt_model):.2f} mm")
    print(f"    (model myo voxels: {myo_model.sum()}, seg myo voxels: {(seg==2).sum()})")
else:
    print(f"[3] EDT on voxelized model:        FAILED (myo={myo_model.sum()}, endo={endo_filled.sum()})")

# ═══════════════════════════════════════════════════════════════════
# METHOD 4: Laplace equation between endo/epi on model SDF grid
# Solve ∇²T = 0 with T=0 on endo (sdf_e≈0), T=1 on epi (sdf_p≈0).
# Thickness ≈ 1/|∇T| averaged in the myocardium.
# ═══════════════════════════════════════════════════════════════════
def solve_laplace_thickness(sdf_endo, sdf_epi, voxel_size, iso=0.0, n_iter=500):
    """Solve Laplace equation between endo and epi iso-surfaces.
    
    Returns mean wall thickness from path integration approximation.
    """
    # Define boundary masks using narrow bands around iso-surfaces
    band = 1.5 * np.max(voxel_size)  # ~1.5 voxel band
    endo_boundary = np.abs(sdf_endo - iso) < band
    epi_boundary = np.abs(sdf_epi - iso) < band
    
    # Myocardium region: inside epi (sdf_p < 0) and outside endo (sdf_e > 0)
    myo_region = (sdf_endo > iso) & (sdf_epi < iso)
    
    # Solve Laplace: T=0 on endo, T=1 on epi, ∇²T=0 in between
    T = np.zeros_like(sdf_endo)
    T[epi_boundary] = 1.0
    T[endo_boundary] = 0.0
    
    # Interior mask (where we iterate)
    interior = myo_region & ~endo_boundary & ~epi_boundary
    
    if not interior.any():
        return None
    
    # Jacobi iteration
    for _ in range(n_iter):
        T_new = T.copy()
        # 6-connected average
        T_new[1:-1, 1:-1, 1:-1] = (
            T[2:, 1:-1, 1:-1] + T[:-2, 1:-1, 1:-1] +
            T[1:-1, 2:, 1:-1] + T[1:-1, :-2, 1:-1] +
            T[1:-1, 1:-1, 2:] + T[1:-1, 1:-1, :-2]
        ) / 6.0
        # Enforce boundary conditions
        T_new[endo_boundary] = 0.0
        T_new[epi_boundary] = 1.0
        # Only update interior
        T[interior] = T_new[interior]
    
    # Compute |∇T| in the myocardium using central differences
    grad = np.zeros((*T.shape, 3), dtype=np.float32)
    grad[1:-1, :, :, 0] = (T[2:, :, :] - T[:-2, :, :]) / (2 * voxel_size[0])
    grad[:, 1:-1, :, 1] = (T[:, 2:, :] - T[:, :-2, :]) / (2 * voxel_size[1])
    grad[:, :, 1:-1, 2] = (T[:, :, 2:] - T[:, :, :-2]) / (2 * voxel_size[2])
    
    grad_mag = np.sqrt(np.sum(grad**2, axis=-1))
    
    # Thickness = 1/|∇T| in the myocardium (where gradient is well-defined)
    myo_grad = grad_mag[myo_region]
    valid = myo_grad > 1e-6
    if not valid.any():
        return None
    
    # Mean thickness from Laplace
    thickness_vals = 1.0 / myo_grad[valid]
    # Filter outliers (> 30mm is unreasonable)
    thickness_vals = thickness_vals[thickness_vals < 30.0]
    
    return float(np.mean(thickness_vals)) if len(thickness_vals) > 10 else None


# The SDF grid is in normalized space; voxel size in normalized space
voxel_norm = voxel  # from _build_grid_and_query
wt_laplace_norm = solve_laplace_thickness(sdf_e, sdf_p, voxel_norm, iso=iso, n_iter=800)

if wt_laplace_norm is not None:
    # Convert from normalized units to mm
    wt_laplace_mm = wt_laplace_norm * scale
    print(f"[4] Laplace equation (model SDF):  mean = {wt_laplace_mm:.2f} mm")
else:
    print(f"[4] Laplace equation (model SDF):  FAILED")

# ═══════════════════════════════════════════════════════════════════
# METHOD 5: Yezzi-Prince full (symmetric SDF sum at midwall)
# t(x) ≈ |φ_endo(x)| + |φ_epi(x)| evaluated in the myocardium
# This is the SDF analogue of EDT boundary sum.
# ═══════════════════════════════════════════════════════════════════
myo_sdf_region = (sdf_e > iso) & (sdf_p < iso)  # between endo and epi

if myo_sdf_region.any():
    # In the myocardium: |sdf_e| = distance to endo, |sdf_p| = distance to epi
    wt_yezzi_grid = np.abs(sdf_e[myo_sdf_region]) + np.abs(sdf_p[myo_sdf_region])
    # This is in normalized space → convert to mm
    wt_yezzi_vals = wt_yezzi_grid * scale
    wt_yezzi_vals = wt_yezzi_vals[wt_yezzi_vals > 0.5]
    wt_yezzi_mm = float(np.mean(wt_yezzi_vals)) if len(wt_yezzi_vals) > 10 else None
    if wt_yezzi_mm is not None:
        print(f"[5] Yezzi-Prince (SDF sum):        mean = {wt_yezzi_mm:.2f} mm")
    else:
        print(f"[5] Yezzi-Prince (SDF sum):        FAILED (too few valid)")
else:
    print(f"[5] Yezzi-Prince (SDF sum):        FAILED (no myocardium in SDF)")

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print()
print(f"{'='*60}")
print(f"SUMMARY — Wall Thickness Comparison (patient001 ED)")
print(f"{'='*60}")
print(f"  Reference (segmentation EDT):      {ref_wt:.2f} mm")
print(f"  ─────────────────────────────────────────────────")
print(f"  [1] KD-tree (current):             {np.mean(wt_kdtree):.2f} mm  ({np.mean(wt_kdtree)/ref_wt*100:.0f}% of ref)")

sdf_mean = np.nanmean(wt_sdf_yezzi[valid_sdf])
print(f"  [2] SDF value (Yezzi-at-surface):  {sdf_mean:.2f} mm  ({sdf_mean/ref_wt*100:.0f}% of ref)")

if myo_model.any() and endo_filled.any() and len(wt_edt_model) > 0:
    edt_model_mean = np.mean(wt_edt_model)
    print(f"  [3] EDT on voxelized model:        {edt_model_mean:.2f} mm  ({edt_model_mean/ref_wt*100:.0f}% of ref)")

if wt_laplace_norm is not None:
    print(f"  [4] Laplace (model SDF grid):      {wt_laplace_mm:.2f} mm  ({wt_laplace_mm/ref_wt*100:.0f}% of ref)")

if myo_sdf_region.any() and wt_yezzi_mm is not None:
    print(f"  [5] Yezzi-Prince (SDF sum):        {wt_yezzi_mm:.2f} mm  ({wt_yezzi_mm/ref_wt*100:.0f}% of ref)")

print(f"  ─────────────────────────────────────────────────")
print(f"  Expected healthy LV: 8–12 mm")
print(f"{'='*60}")
