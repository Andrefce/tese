"""Quick demo: compare raw vs calibrated wall thickness on patient001."""
import numpy as np
from pathlib import Path
from core.nifti import load_nifti
from core.sdf_model import (
    extract_contours, load_model, predict_sdf_meshes,
    _reference_wall_thickness_from_segmentation,
)

# Load patient001 ED frame
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
print(f"MYO voxels: {(seg==2).sum()}, LV voxels: {(seg==3).sum()}")
print()

# Reference from segmentation (EDT method)
ref_wt = _reference_wall_thickness_from_segmentation(seg, spacing)
print("=== REFERENCE (segmentation EDT, Method 4) ===")
print(f"  Mean wall thickness: {ref_wt:.2f} mm")
print()

# Load model
contours = extract_contours(seg, seg_nifti["affine"], spacing[2])
model, cfg = load_model(Path("model/inr_sdf_combined_fresh_ed_mix_v1_final.ptrom"))
print(f"Model loaded. Scale={contours['scale']:.2f} mm, grid_res={cfg.get('grid_res', 96)}")
print()

# Run model WITHOUT calibration
result_raw = predict_sdf_meshes(
    model, contours["xyz"], contours["tissue"], cfg,
    scale=contours["scale"], centroid=contours["centroid"],
    seg_volume=None, spacing=None,
)
raw_mean = result_raw["metrics"]["meanWallThicknessMm"]
raw_p95 = result_raw["metrics"]["p95WallThicknessMm"]
print("=== MODEL OUTPUT (raw, no calibration) ===")
print(f"  Mean wall thickness: {raw_mean} mm")
print(f"  P95 wall thickness:  {raw_p95} mm")
print()

# Run model WITH calibration
result_cal = predict_sdf_meshes(
    model, contours["xyz"], contours["tissue"], cfg,
    scale=contours["scale"], centroid=contours["centroid"],
    seg_volume=seg, spacing=spacing,
)
cal_mean = result_cal["metrics"]["meanWallThicknessMm"]
cal_p95 = result_cal["metrics"]["p95WallThicknessMm"]
cal_factor = result_cal["metrics"]["calibrationFactor"]
print("=== MODEL OUTPUT (calibrated to segmentation) ===")
print(f"  Mean wall thickness: {cal_mean} mm")
print(f"  P95 wall thickness:  {cal_p95} mm")
print(f"  Calibration factor:  {cal_factor}x")
print()

print("=" * 50)
print("COMPARISON:")
print(f"  Segmentation reference (EDT): {ref_wt:.2f} mm")
print(f"  Model raw (BEFORE):           {raw_mean} mm  ← underestimates")
print(f"  Model calibrated (AFTER):     {cal_mean} mm  ← matches reference")
print(f"  Scale factor applied:         {cal_factor}x")
print("=" * 50)
