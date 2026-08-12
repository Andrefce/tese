"""Smoke test: contour extraction + both geometries on the demo patient (ED only)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from cardiosdf_model import load_model, slice_residual_mm
from geometry import (build_model_geometry, build_voxel_geometry, extract_contours,
                      load_segmentation, read_info_cfg, long_axis_frame, enforce_nesting)

REPO = Path(__file__).resolve().parents[3]
pdir = REPO / "tese/notebooks/patient002"
info = read_info_cfg(pdir / "Info.cfg")
print("Info.cfg:", info)

seg = load_segmentation(pdir / f"patient002_frame{int(info['ED']):02d}_gt.nii")
print("seg", seg.labels.shape, seg.spacing, "LV", seg.lv.sum(), "MYO", seg.myo.sum())

c = extract_contours(seg)
print("contours", c["xyz"].shape, "scale", c["scale"], "centroid", c["centroid"],
      "slices", c["slices"])
print("norm bbox", c["xyz"].min(0), c["xyz"].max(0))

t0 = time.perf_counter()
net, cfg, meta = load_model(REPO / "tese/notebooks/inr_sdf_combined_fresh_ed_mix_v1_final.pt")
print(f"model loaded {time.perf_counter()-t0:.1f}s", meta)

t0 = time.perf_counter()
mg = build_model_geometry(net, cfg, c, grid_res=96, phase_val=0.0)
print(f"model geometry {time.perf_counter()-t0:.1f}s")
for r in mg["reports"]:
    print("  ", r)
print("  slice residual mm", slice_residual_mm(net, mg["latent"], c["xyz"], c["tissue"], c["scale"]))

t0 = time.perf_counter()
vg = build_voxel_geometry(seg, iso_pitch=0.75)
print(f"voxel geometry {time.perf_counter()-t0:.1f}s")
for r in vg["reports"]:
    print("  ", r)

for tag, g in (("model", mg), ("voxel", vg)):
    e, p = g["endo"], g["epi"]
    print(f"{tag}: endo bbox {np.round(e.bounds,1).tolist()}  epi bbox {np.round(p.bounds,1).tolist()}")
    print(f"{tag}: endo vol {abs(e.volume)/1000:.1f} mL  epi vol {abs(p.volume)/1000:.1f} mL")
    fixed, rep = enforce_nesting(e, p)
    print(f"{tag}: nesting {rep}")
    fr = long_axis_frame(fixed, seg)
    print(f"{tag}: base_z={fr['base_z']:.1f} apex_z={fr['apex_z']:.1f}")

lv_vox = seg.lv.sum() * np.prod(seg.spacing) / 1000
epi_vox = seg.epi.sum() * np.prod(seg.spacing) / 1000
print(f"segmentation LV {lv_vox:.1f} mL  epi {epi_vox:.1f} mL")
