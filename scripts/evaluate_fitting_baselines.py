"""Evaluate the RBF implicit and statistical shape-model fitting baselines.

Both baselines consume the same ED SAX contour rings, the same 1 mm comparison
grid, the same watertight repair and the same reconstruction metrics as the
reported model, so their numbers line up with the contour-lofting row. Contours
come from the per-patient sample cache and comparator meshes from the cohort
cache, so no neural inference is repeated.

Example:
    C:/Python313/python.exe scripts/evaluate_fitting_baselines.py \
        --cohort test-new-model/cohort_full_nor_hcm10 \
        --samples test-new-model/cache --workers 2
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

HERE = Path(__file__).resolve().parent
THESIS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "eval_demo"))

from fig_baseline_rbf_ssm import build_rbf_geometry, build_ssm_geometry  # noqa: E402
from recon_metrics import reconstruction_quality  # noqa: E402

METHODS = ("rbf", "ssm")
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


def load_contours(samples: Path, patient: str) -> tuple[np.ndarray, np.ndarray]:
    """Input rings in world millimetres, undoing the cache normalisation."""
    with np.load(samples / f"{patient}_ED.npz") as data:
        points = np.asarray(data["contour_xyz"], dtype=np.float64)
        points = points * np.array([1.0, 1.0, -1.0]) * float(data["scale"])
        points = points + np.asarray(data["centroid"], dtype=np.float64)
        tissue = np.asarray(data["contour_tissue"], dtype=np.float64)
    return points, tissue


def evaluate_patient(patient: str, cohort: Path, samples: Path,
                     force: bool) -> list[dict]:
    contours, tissue = load_contours(samples, patient)
    fit_cache = cohort / "fit_cache"
    fit_cache.mkdir(parents=True, exist_ok=True)
    reference = {
        surface: trimesh.load(
            cohort / "cache" / f"{patient}_ED_voxel_{surface}.ply", process=False)
        for surface in ("endo", "epi")
    }

    rows: list[dict] = []
    for method in METHODS:
        paths = {surface: fit_cache / f"{patient}_ED_{method}_{surface}.ply"
                 for surface in ("endo", "epi")}
        try:
            if not force and all(path.exists() for path in paths.values()):
                geometry = {surface: trimesh.load(path, process=False)
                            for surface, path in paths.items()}
            else:
                built = (build_rbf_geometry(contours, tissue) if method == "rbf"
                         else build_ssm_geometry(contours, tissue, "ED"))
                geometry = {surface: built[surface] for surface in ("endo", "epi")}
                for surface, path in paths.items():
                    geometry[surface].export(path)
            metrics = reconstruction_quality(geometry, reference, pitch=1.0)
        except Exception as error:  # a failed fit must not abort the cohort
            print(f"  {patient} {method} failed: {error}", flush=True)
            metrics = {key: np.nan for key in METRICS}
            metrics.update(endo_watertight=False, epi_watertight=False)
        rows.append({"patient": patient, "method": method,
                     "n_contour_slices": len(np.unique(contours[:, 2])),
                     **metrics})
    return rows


def bootstrap_interval(values: np.ndarray, samples: int,
                       seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_summary(model: pd.DataFrame, fits: pd.DataFrame,
                  bootstrap_samples: int) -> pd.DataFrame:
    rows: list[dict] = []
    for method in METHODS:
        subset = fits[fits["method"] == method]
        paired = model.merge(subset, on="patient", suffixes=("_model", "_fit"),
                             validate="one_to_one")
        for index, metric in enumerate(METRICS):
            model_values = paired[f"{metric}_model"].to_numpy(float)
            fit_values = paired[f"{metric}_fit"].to_numpy(float)
            keep = np.isfinite(model_values) & np.isfinite(fit_values)
            differences = fit_values[keep] - model_values[keep]
            low, high = bootstrap_interval(differences, bootstrap_samples,
                                           seed=20260829 + index)
            rows.append({
                "method": method,
                "metric": metric,
                "n": int(keep.sum()),
                "model_mean": model_values[keep].mean(),
                "model_sd": model_values[keep].std(ddof=1),
                "fit_mean": fit_values[keep].mean(),
                "fit_sd": fit_values[keep].std(ddof=1),
                "fit_minus_model": differences.mean(),
                "difference_ci_low": low,
                "difference_ci_high": high,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path,
                        default=THESIS / "test-new-model" / "cohort_full_nor_hcm10")
    parser.add_argument("--samples", type=Path,
                        default=THESIS / "test-new-model" / "cache")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model = pd.read_csv(args.cohort / "recon_quality.csv")
    patients = model["patient"].astype(str).tolist()
    if args.patients:
        patients = [patient for patient in patients if patient in set(args.patients)]
    if not patients:
        raise ValueError("No cohort patients selected.")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_patient, patient, args.cohort,
                                   args.samples, args.force): patient
                   for patient in patients}
        for done, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            print(f"completed {futures[future]} ({done}/{len(patients)})", flush=True)

    fits = pd.DataFrame(rows).sort_values(["method", "patient"])
    fits.to_csv(args.cohort / "fitting_baselines.csv", index=False)

    selected = model[model["patient"].isin(patients)]
    summary = build_summary(selected, fits, args.bootstrap_samples)
    summary.to_csv(args.cohort / "fitting_baselines_summary.csv", index=False)

    print(f"\npatients: {len(patients)}")
    for method in METHODS:
        subset = fits[fits["method"] == method]
        print(f"\n{method.upper()}  watertight endo "
              f"{100.0 * subset['endo_watertight'].mean():.0f}% / epi "
              f"{100.0 * subset['epi_watertight'].mean():.0f}%")
        for row in summary[summary["method"] == method].itertuples(index=False):
            print(f"  {row.metric:<18} model {row.model_mean:7.3f} +/- {row.model_sd:5.3f}"
                  f"   fit {row.fit_mean:7.3f} +/- {row.fit_sd:5.3f}"
                  f"   delta {row.fit_minus_model:+7.3f} "
                  f"[{row.difference_ci_low:+.3f}, {row.difference_ci_high:+.3f}]")


if __name__ == "__main__":
    main()
