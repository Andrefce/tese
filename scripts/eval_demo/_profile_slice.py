"""Throwaway profile of the slice-ablation inner loop (model build vs metrics)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

THESIS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(THESIS / "scripts"))
sys.path.insert(0, str(THESIS / "scripts" / "eval_demo"))
sys.path.insert(0, str(THESIS / "test-new-model"))

import numpy as np
import torch

from cardiosdf2.model import load_v2
from cardiosdf_model import DEVICE
from fig_baseline_rbf_ssm import build_rbf_geometry, build_ssm_geometry
from geometry import (build_model_geometry, extract_contours, load_segmentation,
                      read_info_cfg)
from recon_metrics import overlap_metrics, surface_metrics

print("device:", DEVICE, "threads:", torch.get_num_threads())

pdir = THESIS / "test-new-model" / "training" / "patient021"
info = read_info_cfg(pdir / "Info.cfg")
frame = int(info["ED"])
seg = load_segmentation(pdir / f"{pdir.name}_frame{frame:02d}_gt.nii")

t = time.perf_counter()
contours = extract_contours(seg)
print(f"extract_contours     {time.perf_counter() - t:6.2f}s")

net, cfg, meta = load_v2(THESIS / "test-new-model" / "runs" / "u1u2_e50"
                         / "cardiosdf_v2_best.pt", DEVICE)
xyz_mm = np.asarray(contours["xyz_mm"], np.float64)
tissue = np.asarray(contours["tissue"], np.float64)

for res in (64, 96):
    t = time.perf_counter()
    g = build_model_geometry(net, cfg, contours, grid_res=res, phase_val=0.0)
    print(f"model  grid={res:<3}       {time.perf_counter() - t:6.2f}s  "
          f"endo_v={len(g['endo'].vertices)}")
    if res == 96:
        model_geom = g

t = time.perf_counter()
rbf = build_rbf_geometry(xyz_mm, tissue)
print(f"rbf                  {time.perf_counter() - t:6.2f}s")

t = time.perf_counter()
ssm = build_ssm_geometry(xyz_mm, tissue, "ED")
print(f"ssm                  {time.perf_counter() - t:6.2f}s")

t = time.perf_counter()
sm = surface_metrics(model_geom["endo"], rbf["endo"])
print(f"surface_metrics      {time.perf_counter() - t:6.2f}s  chamfer={sm['chamfer_mm']:.3f}")

t = time.perf_counter()
om = overlap_metrics(model_geom["endo"], model_geom["epi"], rbf["endo"], rbf["epi"])
print(f"overlap_metrics      {time.perf_counter() - t:6.2f}s  myo_dice={om['myo_dice']:.3f}")
