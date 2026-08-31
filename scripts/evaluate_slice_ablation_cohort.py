"""Cohort-wide slice-decimation ablation for the three ED reconstruction methods.

``scripts/fig_slice_ablation.py`` runs the decimation study on a single
illustrative case and renders the figure. This script repeats the *quantitative*
part of that study for every patient of the Chapter 4 cohort so that
``tab:slice-degradation`` can be reported as a cohort result instead of a
single-case demonstration.

Design decisions, all forced by the data:

* Patients carry between 6 and 14 annotated SAX slices, so a fixed "ten-slice"
  reference does not exist cohort-wide. The reference for each patient is its
  own complete stack, and the decimated levels are 6, 4 and 3 retained slices.
* A level is skipped when the patient does not have strictly more slices than
  the level, because decimating to the number of slices already present is a
  no-op that would bias the level towards zero deviation.
* Deviation is self-consistency: every decimated reconstruction is scored
  against the surface the *same* method produced from the complete stack. No
  independent 3D reference exists.

The model is the v2 checkpoint, loaded exactly as ``run_cohort_v2.py`` loads it.
Contour extraction, marching cubes, watertight repair and the metric code are
the shared modules used by the published Results tables.

    python scripts/evaluate_slice_ablation_cohort.py \
        --data-root test-new-model/training \
        --model test-new-model/runs/u1u2_e50/cardiosdf_v2_best.pt \
        --out test-new-model/slice_ablation_cohort
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

THESIS = Path(__file__).resolve().parents[1]
EVAL_DIR = THESIS / "scripts" / "eval_demo"
sys.path.insert(0, str(THESIS / "scripts"))
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(THESIS / "test-new-model"))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

from fig_baseline_rbf_ssm import (                                   # noqa: E402
    build_rbf_geometry, build_ssm_geometry,
)
from geometry import (                                               # noqa: E402
    Segmentation, build_model_geometry, extract_contours,
    load_segmentation, read_info_cfg,
)
from recon_metrics import overlap_metrics, surface_metrics           # noqa: E402

METHODS = (("model", "Proposed model"),
           ("rbf", "RBF fit"),
           ("ssm", "SSM fit"))
DECIMATED_LEVELS = (6, 4, 3)
PHASE = "ED"

# The 30 ACDC patients of the frozen Chapter 4 cohort (10 HCM + 20 NOR).
COHORT = [f"patient{i:03d}" for i in range(21, 31)] + \
         [f"patient{i:03d}" for i in range(61, 81)]

_MODEL: dict = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_net(model_path: Path):
    if "net" not in _MODEL:
        import torch
        from cardiosdf2.model import load_v2
        from cardiosdf_model import DEVICE

        torch.set_num_threads(1)
        net, cfg, meta = load_v2(model_path, DEVICE)
        _MODEL.update(net=net, cfg=cfg, meta=meta)
        log(f"v2 checkpoint epoch={meta['epoch']} val_loss={meta['val_loss']:.4f}")
    return _MODEL["net"], _MODEL["cfg"]


def keep_indices(available: np.ndarray, count: int) -> np.ndarray:
    """Evenly decimated slice indices, always retaining base and apex."""
    take = np.unique(np.round(np.linspace(0, len(available) - 1, count)).astype(int))
    return available[take]


def restrict(seg: Segmentation, keep: np.ndarray) -> Segmentation:
    """Copy of the segmentation with every non-retained slice blanked."""
    labels = np.zeros_like(seg.labels)
    labels[:, :, keep] = seg.labels[:, :, keep]
    return Segmentation(labels, seg.spacing, seg.path)


def build(method: str, contours: dict, net, cfg, grid_res: int) -> dict:
    xyz_mm = np.asarray(contours["xyz_mm"], dtype=np.float64)
    tissue = np.asarray(contours["tissue"], dtype=np.float64)
    if method == "model":
        return build_model_geometry(net, cfg, contours, grid_res=grid_res,
                                    phase_val=0.0)
    if method == "rbf":
        return build_rbf_geometry(xyz_mm, tissue)
    return build_ssm_geometry(xyz_mm, tissue, PHASE)


def myocardial_volume(geometry: dict) -> float:
    return abs(geometry["epi"].volume) - abs(geometry["endo"].volume)


def evaluate_patient(patient_dir: Path, model_path: Path, grid_res: int) -> list[dict]:
    """Deviation of every decimated reconstruction from its own full-stack result."""
    info = read_info_cfg(patient_dir / "Info.cfg")
    frame = int(info[PHASE])
    seg = load_segmentation(patient_dir / f"{patient_dir.name}_frame{frame:02d}_gt.nii")
    full = extract_contours(seg)
    available = np.asarray(full["slices"], dtype=int)
    n_full = len(available)

    net, cfg = load_net(model_path)

    reference = {}
    for method, _ in METHODS:
        start = time.perf_counter()
        reference[method] = build(method, full, net, cfg, grid_res)
        log(f"  {patient_dir.name} {n_full:>2} slices {method:<5} "
            f"{time.perf_counter() - start:5.1f} s")

    rows: list[dict] = []
    for count in DECIMATED_LEVELS:
        if count >= n_full:
            log(f"  {patient_dir.name} skips the {count}-slice level "
                f"(stack has only {n_full} slices)")
            continue
        keep = keep_indices(available, count)
        contours = extract_contours(restrict(seg, keep))
        gaps = np.diff(np.sort(np.asarray(contours["xyz_mm"], np.float64)[:, 2]))
        for method, label in METHODS:
            start = time.perf_counter()
            candidate = build(method, contours, net, cfg, grid_res)
            endo = surface_metrics(candidate["endo"], reference[method]["endo"])
            epi = surface_metrics(candidate["epi"], reference[method]["epi"])
            overlap = overlap_metrics(candidate["endo"], candidate["epi"],
                                      reference[method]["endo"],
                                      reference[method]["epi"])
            ref_myo = myocardial_volume(reference[method])
            rows.append({
                "patient": patient_dir.name,
                "method": label,
                "slices": count,
                "n_full_slices": n_full,
                "max_gap_mm": float(gaps.max()) if len(gaps) else 0.0,
                "endo_chamfer_mm": endo["chamfer_mm"],
                "epi_chamfer_mm": epi["chamfer_mm"],
                "endo_hd95_mm": endo["hd95_mm"],
                "epi_hd95_mm": epi["hd95_mm"],
                "cavity_dice": overlap["endo_dice"],
                "myo_dice": overlap["myo_dice"],
                "myo_volume_change_pct":
                    100.0 * (myocardial_volume(candidate) - ref_myo) / ref_myo,
                "endo_watertight": bool(candidate["endo"].is_watertight),
                "epi_watertight": bool(candidate["epi"].is_watertight),
            })
            log(f"  {patient_dir.name} {count:>2} slices {method:<5} "
                f"{time.perf_counter() - start:5.1f} s  "
                f"endoCh={endo['chamfer_mm']:.2f} myoDice={overlap['myo_dice']:.3f}")
    return rows


def aggregate(frame: pd.DataFrame, out: Path) -> pd.DataFrame:
    metrics = ["endo_chamfer_mm", "epi_chamfer_mm", "endo_hd95_mm", "epi_hd95_mm",
               "cavity_dice", "myo_dice", "myo_volume_change_pct"]
    grouped = frame.groupby(["method", "slices"], sort=False)
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "n_patients", grouped.size())
    summary = summary.reset_index().sort_values(
        ["method", "slices"], ascending=[True, False])
    summary.to_csv(out / "slice_ablation_cohort_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=THESIS / "test-new-model" / "training")
    parser.add_argument("--model", type=Path,
                        default=THESIS / "test-new-model" / "runs" / "u1u2_e50"
                        / "cardiosdf_v2_best.pt")
    parser.add_argument("--out", type=Path,
                        default=THESIS / "test-new-model" / "slice_ablation_cohort")
    parser.add_argument("--patients", nargs="*", default=COHORT)
    parser.add_argument("--grid-res", type=int, default=96)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    cache = args.out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    for name in args.patients:
        payload = cache / f"{name}.json"
        if payload.exists() or args.aggregate_only:
            continue
        patient_dir = args.data_root / name
        if not patient_dir.is_dir():
            log(f"{name}: directory missing, skipped")
            continue
        started = time.perf_counter()
        try:
            rows = evaluate_patient(patient_dir, args.model, args.grid_res)
        except Exception as error:                       # keep the cohort running
            log(f"{name}: FAILED ({type(error).__name__}: {error})")
            continue
        payload.write_text(json.dumps(rows, indent=1) + "\n")
        log(f"{name}: done in {time.perf_counter() - started:.0f} s")

    rows: list[dict] = []
    for name in args.patients:
        payload = cache / f"{name}.json"
        if payload.exists():
            rows.extend(json.loads(payload.read_text()))
    if not rows:
        raise SystemExit("no per-patient results found")

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "slice_ablation_cohort.csv", index=False)
    summary = aggregate(frame, args.out)

    log(f"{frame['patient'].nunique()} patients, {len(frame)} rows")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {(args.out / 'slice_ablation_cohort.csv').relative_to(THESIS)}")


if __name__ == "__main__":
    main()
