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
import gc
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")


def volunteer_for_oom() -> None:
    """Make this worker the kernel's preferred OOM victim.

    VS Code runs with oom_score_adj=300 while a child of its terminal inherits
    200, so under global pressure the kernel kills the editor rather than this
    process. The journal shows exactly that: three of four kills were ``code``.
    Raising our own score to the maximum inverts the choice.
    """
    try:
        Path("/proc/self/oom_score_adj").write_text("1000\n")
    except OSError as error:
        log(f"could not raise oom_score_adj: {error}")


def cap_heap(gigabytes: float) -> None:
    """Fail this process with MemoryError instead of inviting the OOM killer.

    The desktop leaves under 2 GB free, so an unbounded worker gets reaped by
    the kernel, which picks the largest process and has repeatedly killed the
    editor rather than the run. RLIMIT_DATA bounds anonymous memory while
    ignoring the tens of gigabytes of address space that the CUDA driver
    reserves, which is why RLIMIT_AS cannot be used here.
    """
    limit = int(gigabytes * 1024 ** 3)
    soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))


def limit_threads(n_cpus: int) -> None:
    """Restrict the worker to a few cores with small thread stacks.

    ``cKDTree.query(workers=-1)`` in the shared metric code fans out over
    ``os.cpu_count()`` threads, which ignores the affinity mask, and each stack
    is anonymous memory charged against RLIMIT_DATA. At the 8 MB default that
    is ~96 MB of stack demanded at the exact moment the heap is fullest, which
    is what raises "can't start new thread". These are ordinary
    ``threading.Thread`` workers, so a smaller stack applies to them; 1 MB is
    ample for a kd-tree query and costs nothing numerically.
    """
    threading.stack_size(1024 * 1024)
    try:
        available = sorted(os.sched_getaffinity(0))
    except AttributeError:
        return
    os.sched_setaffinity(0, set(available[:n_cpus]))


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2

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

        torch.set_num_threads(2)
        net, cfg, meta = load_v2(model_path, DEVICE)
        _MODEL.update(net=net, cfg=cfg, meta=meta)
        log(f"v2 checkpoint epoch={meta['epoch']} val_loss={meta['val_loss']:.4f} "
            f"device={DEVICE}")
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


def build(method: str, contours: dict, net, cfg, grid_res: int,
          batch: int) -> dict:
    xyz_mm = np.asarray(contours["xyz_mm"], dtype=np.float64)
    tissue = np.asarray(contours["tissue"], dtype=np.float64)
    if method == "model":
        geometry = build_model_geometry(net, cfg, contours, grid_res=grid_res,
                                        phase_val=0.0, batch=batch)
    elif method == "rbf":
        geometry = build_rbf_geometry(xyz_mm, tissue)
    else:
        geometry = build_ssm_geometry(xyz_mm, tissue, PHASE)
    # marching_cubes_mesh returns an empty mesh when the field never crosses the
    # iso-level, which otherwise only surfaces much later inside the metrics.
    for wall in ("endo", "epi"):
        if len(geometry[wall].faces) == 0:
            raise RuntimeError(
                f"{method} produced an empty {wall} mesh from "
                f"{len(xyz_mm)} contour points")
    # Keep the surfaces only; the dense SDF grids are dead weight after this.
    return {"endo": geometry["endo"], "epi": geometry["epi"]}


def myocardial_volume(geometry: dict) -> float:
    return abs(geometry["epi"].volume) - abs(geometry["endo"].volume)


def evaluate_patient(patient_dir: Path, model_path: Path, grid_res: int,
                     batch: int, overlap_pitch: float = 1.0) -> list[dict]:
    """Deviation of every decimated reconstruction from its own full-stack result."""
    info = read_info_cfg(patient_dir / "Info.cfg")
    frame = int(info[PHASE])
    seg = load_segmentation(patient_dir / f"{patient_dir.name}_frame{frame:02d}_gt.nii")
    full = extract_contours(seg)
    available = np.asarray(full["slices"], dtype=int)
    n_full = len(available)

    levels = []
    for count in DECIMATED_LEVELS:
        if count >= n_full:
            log(f"  {patient_dir.name} skips the {count}-slice level "
                f"(stack has only {n_full} slices)")
            continue
        levels.append(count)

    # Contour rings are small; caching them avoids re-deriving per method.
    decimated = {count: extract_contours(restrict(seg, keep_indices(available, count)))
                 for count in levels}
    del seg
    gc.collect()

    net, cfg = load_net(model_path)

    rows: list[dict] = []
    for method, label in METHODS:
        start = time.perf_counter()
        reference = build(method, full, net, cfg, grid_res, batch)
        ref_myo = myocardial_volume(reference)
        log(f"  {patient_dir.name} {n_full:>2} slices {method:<5} "
            f"{time.perf_counter() - start:5.1f} s")

        for count in levels:
            contours = decimated[count]
            gaps = np.diff(np.sort(np.asarray(contours["xyz_mm"], np.float64)[:, 2]))
            start = time.perf_counter()
            candidate = build(method, contours, net, cfg, grid_res, batch)
            endo = surface_metrics(candidate["endo"], reference["endo"])
            epi = surface_metrics(candidate["epi"], reference["epi"])
            overlap = overlap_metrics(candidate["endo"], candidate["epi"],
                                      reference["endo"], reference["epi"],
                                      pitch=overlap_pitch)
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
                f"endoCh={endo['chamfer_mm']:.2f} myoDice={overlap['myo_dice']:.3f} "
                f"peak={rss_gb():.2f} GB")
            del candidate
            gc.collect()

        del reference
        gc.collect()
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


def run_in_child(name: str, args) -> None:
    """Evaluate one patient in a fresh interpreter.

    Peak memory is dominated by torch activations and the voxelisation grids.
    Running each patient in its own process returns that memory to the OS
    instead of accumulating fragmentation over the whole cohort, and an OOM
    kills a single patient rather than the entire run.
    """
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--worker", name,
               "--data-root", str(args.data_root),
               "--model", str(args.model),
               "--out", str(args.out),
               "--grid-res", str(args.grid_res),
               "--decode-batch", str(args.decode_batch),
               "--overlap-pitch", str(args.overlap_pitch),
               "--cpus", str(args.cpus),
               "--mem-cap-gb", str(args.mem_cap_gb)]
    result = subprocess.run(command, check=False)
    if result.returncode < 0:
        log(f"{name}: worker killed by signal {-result.returncode} "
            "(likely the kernel OOM killer)")
    elif result.returncode != 0:
        log(f"{name}: worker exited with code {result.returncode}")


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
    parser.add_argument("--decode-batch", type=int, default=16384)
    parser.add_argument("--overlap-pitch", type=float, default=1.0,
                        help="voxel pitch for the Dice grids, in millimetres")
    parser.add_argument("--mem-cap-gb", type=float, default=2.0,
                        help="hard address-space limit applied inside a worker")
    parser.add_argument("--cpus", type=int, default=4,
                        help="cores a worker may use; bounds scipy thread fan-out")
    parser.add_argument("--worker", default=None,
                        help="internal: evaluate this single patient and exit")
    parser.add_argument("--in-process", action="store_true",
                        help="evaluate without per-patient subprocesses")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    cache = args.out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    if args.worker:
        name = args.worker
        volunteer_for_oom()
        limit_threads(args.cpus)
        if args.mem_cap_gb > 0:
            cap_heap(args.mem_cap_gb)
        payload = cache / f"{name}.json"
        patient_dir = args.data_root / name
        if not patient_dir.is_dir():
            log(f"{name}: directory missing, skipped")
            return
        started = time.perf_counter()
        rows = evaluate_patient(patient_dir, args.model, args.grid_res,
                                args.decode_batch, args.overlap_pitch)
        payload.write_text(json.dumps(rows, indent=1) + "\n")
        log(f"{name}: done in {time.perf_counter() - started:.0f} s, "
            f"peak {rss_gb():.2f} GB")
        return

    for name in args.patients:
        payload = cache / f"{name}.json"
        if payload.exists() or args.aggregate_only:
            continue
        patient_dir = args.data_root / name
        if not patient_dir.is_dir():
            log(f"{name}: directory missing, skipped")
            continue
        started = time.perf_counter()
        if args.in_process:
            try:
                rows = evaluate_patient(patient_dir, args.model, args.grid_res,
                                        args.decode_batch, args.overlap_pitch)
            except Exception as error:                   # keep the cohort running
                log(f"{name}: FAILED ({type(error).__name__}: {error})")
                continue
            payload.write_text(json.dumps(rows, indent=1) + "\n")
        else:
            run_in_child(name, args)
        log(f"{name}: {time.perf_counter() - started:.0f} s")

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
