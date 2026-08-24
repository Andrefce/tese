"""Evaluate a matched linear contour-lofting reconstruction baseline.

The baseline uses the same ED SAX contours, 1 mm physical grid, watertight
repair, segmentation-derived comparator, and reconstruction metrics as the
reported model. Existing model and comparator meshes are read from the cohort
cache, so no neural inference is repeated.

Example:
    /home/C052246/tese/.venv/bin/python scripts/evaluate_contour_lofting.py \
        --data-root notebooks/data/training \
        --cohort test-new-model/cohort_full_nor_hcm10 \
        --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
import trimesh

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE / "eval_demo"
sys.path.insert(0, str(EVAL_DIR))

from geometry import (  # noqa: E402
    build_loft_geometry,
    extract_contours,
    load_segmentation,
    read_info_cfg,
)
from recon_metrics import reconstruction_quality  # noqa: E402

METRICS = (
    "endo_chamfer_mm",
    "epi_chamfer_mm",
    "endo_assd_mm",
    "epi_assd_mm",
    "endo_hd95_mm",
    "epi_hd95_mm",
    "endo_dice",
    "myo_dice",
    "vol_ratio_endo",
    "vol_ratio_epi",
)


def find_ed_segmentation(patient_dir: Path, patient_id: str) -> Path:
    info = read_info_cfg(patient_dir / "Info.cfg")
    frame = int(info["ED"])
    stem = f"{patient_id}_frame{frame:02d}_gt"
    for candidate in (f"{stem}.nii.gz", f"{stem}.nii", stem):
        path = patient_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"{stem} not found in {patient_dir}")


def evaluate_patient(patient_id: str, data_root: Path, cohort: Path,
                     force: bool) -> dict[str, float | str | bool]:
    patient_dir = data_root / patient_id
    segmentation = load_segmentation(find_ed_segmentation(patient_dir, patient_id))
    contours = extract_contours(segmentation)

    loft_cache = cohort / "loft_cache"
    loft_cache.mkdir(parents=True, exist_ok=True)
    loft_paths = {
        surface: loft_cache / f"{patient_id}_ED_loft_{surface}.ply"
        for surface in ("endo", "epi")
    }
    if not force and all(path.exists() for path in loft_paths.values()):
        loft = {
            surface: trimesh.load(path, process=False)
            for surface, path in loft_paths.items()
        }
    else:
        geometry = build_loft_geometry(contours)
        loft = {surface: geometry[surface] for surface in ("endo", "epi")}
        for surface, path in loft_paths.items():
            loft[surface].export(path)

    cohort_cache = cohort / "cache"
    reference = {
        surface: trimesh.load(
            cohort_cache / f"{patient_id}_ED_voxel_{surface}.ply", process=False
        )
        for surface in ("endo", "epi")
    }
    metrics = reconstruction_quality(loft, reference, pitch=1.0)
    result_path = cohort_cache / f"{patient_id}_result.json"
    group = json.loads(result_path.read_text()).get("group", "")
    return {
        "patient": patient_id,
        "group": group,
        "n_contour_slices": len(contours["slices"]),
        **metrics,
    }


def bootstrap_interval(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_summary(model: pd.DataFrame, loft: pd.DataFrame,
                  bootstrap_samples: int) -> pd.DataFrame:
    paired = model.merge(loft, on="patient", suffixes=("_model", "_loft"),
                         validate="one_to_one")
    rows: list[dict[str, float | str]] = []
    for metric_index, metric in enumerate(METRICS):
        model_values = paired[f"{metric}_model"].to_numpy(float)
        loft_values = paired[f"{metric}_loft"].to_numpy(float)
        differences = loft_values - model_values
        ci_low, ci_high = bootstrap_interval(
            differences, bootstrap_samples, seed=20260824 + metric_index
        )
        rows.append({
            "metric": metric,
            "model_mean": model_values.mean(),
            "model_sd": model_values.std(ddof=1) if len(model_values) > 1 else np.nan,
            "loft_mean": loft_values.mean(),
            "loft_sd": loft_values.std(ddof=1) if len(loft_values) > 1 else np.nan,
            "loft_minus_model": differences.mean(),
            "difference_ci_low": ci_low,
            "difference_ci_high": ci_high,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model = pd.read_csv(args.cohort / "recon_quality.csv")
    patient_ids = model["patient"].astype(str).tolist()
    if args.patients:
        requested = set(args.patients)
        patient_ids = [patient for patient in patient_ids if patient in requested]
    if not patient_ids:
        raise ValueError("No cohort patients selected.")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_patient, patient, args.data_root, args.cohort, args.force
            ): patient
            for patient in patient_ids
        }
        for future in as_completed(futures):
            patient = futures[future]
            row = future.result()
            rows.append(row)
            print(f"completed {patient}", flush=True)

    loft = pd.DataFrame(rows).sort_values("patient")
    loft.to_csv(args.cohort / "contour_lofting.csv", index=False)

    selected_model = model[model["patient"].isin(patient_ids)]
    summary = build_summary(selected_model, loft, args.bootstrap_samples)
    summary.to_csv(args.cohort / "contour_lofting_summary.csv", index=False)

    print(f"\npatients: {len(loft)}")
    for row in summary.itertuples(index=False):
        print(
            f"{row.metric:<22} model {row.model_mean:.3f} +/- {row.model_sd:.3f}  "
            f"loft {row.loft_mean:.3f} +/- {row.loft_sd:.3f}  "
            f"delta {row.loft_minus_model:+.3f} "
            f"[{row.difference_ci_low:+.3f}, {row.difference_ci_high:+.3f}]"
        )
    print(
        "watertight: "
        f"endo {100.0 * loft['endo_watertight'].mean():.0f}%, "
        f"epi {100.0 * loft['epi_watertight'].mean():.0f}%"
    )


if __name__ == "__main__":
    main()
