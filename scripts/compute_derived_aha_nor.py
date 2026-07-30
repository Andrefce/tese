"""Derived-geometry ('no-model') AHA-17 wall thickness for the NOR cohort.

The main cohort CSV stores the CardioSDF-geometry AHA-17 thickness only. To
compare the ED->ES thickening of the model surfaces against the segmentation-
derived baseline at cohort level, this script computes the Laplace-field wall
thickness on the derived marching-cubes meshes (no neural model involved) for
every NOR patient at ED and ES, assigns AHA-17 segments, and writes per-segment
means.  Combined with the existing ``cohort_aha17_thickening.csv`` (CardioSDF),
this yields the full model-vs-derived regional thickening table.

Output -> <RESULTS_OUT_DIR>/derived_aha17_nor.csv

Run:
    RESULTS_OUT_DIR=.../cohort_single .venv/bin/python scripts/compute_derived_aha_nor.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("RESULTS_DATA_ROOT", str(ROOT / "notebooks" / "data"))

import compute_results_full_cohort as F  # noqa: E402
import compute_results_cohort as C  # noqa: E402
from core.nifti import load_nifti  # noqa: E402

DATA_ROOT = F.DATA_ROOT
OUT = Path(os.environ.get(
    "RESULTS_OUT_DIR",
    str(ROOT / "scripts" / "webapp" / "notebooks" / "outputs" / "cohort_single")))
N_SEG = 17
LBL_MYO, LBL_LV = 2, 3


def is_nor(pid: str) -> bool:
    try:
        return F.read_info(DATA_ROOT / pid).get("Group", "") == "NOR"
    except Exception:
        return False


def derived_laplace(seg: np.ndarray, spacing) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Laplace thickness per derived endo vertex, plus AHA ids, wall mask, ref."""
    lv = seg == LBL_LV
    epi = (seg == LBL_LV) | (seg == LBL_MYO)
    myo = epi & ~lv
    de_v, _de_f = F.derived_surface(lv, spacing)
    endo_mm = np.asarray(F.to_mm_frame(de_v, spacing), np.float32)
    # Laplace field only (the reference method) -- avoids the slower Yezzi/cone
    # estimators, since this cohort pass feeds only the regional thickening.
    P_method = F.to_method_frame(de_v, spacing)
    myo_idx = np.argwhere(myo).astype(np.int64)
    myo_tree = cKDTree(C.voxel_to_world(myo_idx, spacing))
    lap = np.asarray(
        C.method_laplace_field(lv, epi, myo, spacing, P_method, myo_idx, myo_tree),
        np.float32)
    bad = ~np.isfinite(lap) | (lap < 0) | (lap > C.THICKNESS_MAX_MM)
    lap[bad] = np.nan
    aha = F.assign_aha17(endo_mm, seg, spacing)
    wall = F.cavity_wall_mask(endo_mm, seg, spacing)
    ref = C.segmentation_reference_thickness(seg, spacing) if hasattr(C, "segmentation_reference_thickness") else None
    v = lap[wall]
    v = v[np.isfinite(v)]
    ref = float(v.mean()) if v.size else float("nan")
    return lap, aha, wall, ref


def seg_means(field: np.ndarray, aha: np.ndarray, wall: np.ndarray) -> np.ndarray:
    out = np.full(N_SEG, np.nan)
    for sid in range(1, N_SEG + 1):
        m = field[(aha == sid) & wall]
        m = m[np.isfinite(m)]
        if m.size:
            out[sid - 1] = float(m.mean())
    return out


def main() -> None:
    patients = sorted(p.name for p in DATA_ROOT.iterdir()
                      if p.is_dir() and p.name.startswith("patient") and is_nor(p.name))
    print(f"NOR patients: {len(patients)}")
    rows = []
    for i, pid in enumerate(patients, 1):
        info = F.read_info(DATA_ROOT / pid)
        for phase in ("ED", "ES"):
            if phase not in info:
                continue
            frame = int(info[phase])
            seg_path = DATA_ROOT / pid / f"{pid}_frame{frame:02d}_gt.nii"
            nif = load_nifti(seg_path)
            seg = np.rint(nif["data"]).astype(np.int16)
            if seg.ndim == 4:
                seg = seg[..., 0]
            spacing = tuple(float(v) for v in nif["zooms"])
            try:
                lap, aha, wall, _ref = derived_laplace(seg, spacing)
            except Exception as exc:
                print(f"  {pid} {phase} ERROR: {exc}")
                continue
            means = seg_means(lap, aha, wall)
            for sid in range(1, N_SEG + 1):
                rows.append({"patient": pid, "phase": phase, "segment_id": sid,
                             "segment": F.AHA_17_NAMES[sid - 1], "mean_mm": means[sid - 1]})
        print(f"  [{i}/{len(patients)}] {pid} done")
    df = pd.DataFrame(rows)
    out_path = OUT / "derived_aha17_nor.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
