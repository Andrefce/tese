"""AHA-17 wall thickness measured directly on the input segmentation.

Provides the third block of the regional table in the Results chapter: the
label-mask reference against which the CardioSDF and voxel-derived geometries
are both compared. Thickness is the Euclidean distance-transform boundary sum
evaluated on every myocardial voxel of the isotropically resampled mask, so it
involves no meshing and no analysis band.

Reads the cached voxel endocardial meshes written by ``run_cohort.py`` to build
the same long-axis frame used there; no model inference is performed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
import trimesh                                                       # noqa: E402
from scipy.ndimage import distance_transform_edt, zoom as ndi_zoom   # noqa: E402

from geometry import (                                               # noqa: E402
    AHA_17_NAMES, _clean_inside, _crop_bounds, assign_aha17, load_segmentation,
    long_axis_frame, read_info_cfg,
)
from run_cohort import DEFAULT_OUT, discover_patients, find_frame     # noqa: E402


def reference_segments(seg, endo: trimesh.Trimesh, pitch: float) -> tuple[np.ndarray, float]:
    """Per-AHA-segment mean thickness on the raw label mask, and the global mean."""
    spacing = np.asarray(seg.spacing, np.float64)
    start, stop = _crop_bounds(seg.epi, spacing, 8.0)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    factors = spacing / pitch

    def resample(mask: np.ndarray) -> np.ndarray:
        return ndi_zoom(mask[sl].astype(np.float32), factors, order=1,
                        prefilter=False) >= 0.5

    lv = _clean_inside(resample(seg.lv))
    epi = _clean_inside(resample(seg.epi) | lv)
    myo = epi & ~lv
    if not myo.any():
        return np.full(17, np.nan), float("nan")

    d_endo = distance_transform_edt(~lv, sampling=(pitch,) * 3)
    d_epi = distance_transform_edt(epi, sampling=(pitch,) * 3)
    values = (d_endo + d_epi)[myo]

    # isotropic index -> original voxel index -> millimetre world frame
    idx = np.argwhere(myo).astype(np.float64) / factors + start
    world = np.column_stack([-idx[:, 1] * spacing[0],
                             -idx[:, 0] * spacing[1],
                             idx[:, 2] * spacing[2]])

    ids = assign_aha17(world, long_axis_frame(endo, seg))
    means = np.array([values[ids == sid].mean() if np.any(ids == sid) else np.nan
                      for sid in range(1, 18)])
    return means, float(values.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--group", default="NOR")
    parser.add_argument("--pitch", type=float, default=1.0)
    args = parser.parse_args()

    cache = args.out / "cache"
    rows = []
    for patient_dir in discover_patients(args.data_root, args.group):
        patient_id = patient_dir.name
        info = read_info_cfg(patient_dir / "Info.cfg")
        for phase in ("ED", "ES"):
            mesh_path = cache / f"{patient_id}_{phase}_voxel_endo.ply"
            if not mesh_path.exists():
                print(f"skip {patient_id} {phase}: no cached mesh")
                continue
            seg = load_segmentation(find_frame(patient_dir, patient_id, int(info[phase])))
            endo = trimesh.load(mesh_path, process=False)
            means, overall = reference_segments(seg, endo, args.pitch)
            for sid, value in enumerate(means, start=1):
                rows.append({"patient": patient_id, "phase": phase, "segment_id": sid,
                             "segment": AHA_17_NAMES[sid - 1], "mean_mm": value})
            print(f"{patient_id} {phase}: overall {overall:.2f} mm")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "reference_aha17.csv", index=False)

    pivot = df.pivot_table(index=["patient", "segment_id"], columns="phase",
                           values="mean_mm").dropna()
    print("\n== segmentation reference, AHA-17 ==")
    for sid in range(1, 18):
        block = df[df.segment_id == sid].pivot_table(index="patient", columns="phase",
                                                     values="mean_mm").dropna()
        ed, es = block["ED"].mean(), block["ES"].mean()
        print(f"  {sid:2d} {AHA_17_NAMES[sid-1]:22s} {ed:5.2f}  {es:5.2f}  "
              f"{100*(es-ed)/ed:+6.1f}%")
    ed, es = pivot["ED"].mean(), pivot["ES"].mean()
    print(f"   overall {ed:5.2f}  {es:5.2f}  {100*(es-ed)/ed:+6.1f}%")


if __name__ == "__main__":
    main()
