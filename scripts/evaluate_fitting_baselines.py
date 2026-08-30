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
import json
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
from eval_demo import thickness as tk  # noqa: E402
from eval_demo.geometry import enforce_nesting, outward_normals  # noqa: E402
from recon_metrics import reconstruction_quality  # noqa: E402

METHODS = ("rbf", "ssm")
THICKNESS_METHODS = (
    "Laplace field",
    "Yezzi-Prince",
    "SDF cone rays",
    "EDT boundary sum",
)
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


def thickness_stats(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean_mm": np.nan, "std_mm": np.nan,
                "p5_mm": np.nan, "p95_mm": np.nan}
    return {
        "mean_mm": float(finite.mean()),
        "std_mm": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "p5_mm": float(np.percentile(finite, 5)),
        "p95_mm": float(np.percentile(finite, 95)),
    }


def evaluate_thickness(patient: str, method: str, geometry: dict,
                       contours: np.ndarray, pitch: float = 1.0) -> list[dict]:
    endo, _ = enforce_nesting(geometry["endo"], geometry["epi"])
    epi = geometry["epi"]
    vertices = np.asarray(endo.vertices, dtype=np.float64)
    normals = outward_normals(endo, np.asarray(epi.vertices))

    z_min, z_max = np.min(contours[:, 2]), np.max(contours[:, 2])
    long_axis = (vertices[:, 2] - z_min) / (z_max - z_min)
    valid_band = (long_axis >= 0.04) & (long_axis <= 0.97)

    context = tk.build_volume_context(endo, epi, pitch)
    laplace, _ = tk.solve_laplace(context)
    values = {
        "Laplace field": tk.method_laplace_streamline(
            context, vertices, normals, laplace).values,
        "Yezzi-Prince": tk.method_yezzi_prince(
            context, vertices, normals, laplace).values,
        "SDF cone rays": tk.method_cone_rays(endo, epi, normals).values,
        "EDT boundary sum": tk.method_edt_boundary_sum(
            context, vertices, normals).values,
    }

    rows = []
    for thickness_method in THICKNESS_METHODS:
        selected = np.where(valid_band, values[thickness_method], np.nan)
        rows.append({
            "patient": patient,
            "phase": "ED",
            "geometry": method,
            "method": thickness_method,
            "valid_fraction": float(np.isfinite(selected).mean()),
            **thickness_stats(selected),
        })
    return rows


def evaluate_patient(patient: str, cohort: Path, samples: Path, force: bool,
                     evaluate_reconstruction: bool) -> tuple[list[dict], list[dict]]:
    contours, tissue = load_contours(samples, patient)
    fit_cache = cohort / "fit_cache"
    fit_cache.mkdir(parents=True, exist_ok=True)
    wall_cache = fit_cache / f"{patient}_ED_wall_methods.json"
    if wall_cache.exists() and not force:
        return [], json.loads(wall_cache.read_text())

    reference = None
    if evaluate_reconstruction:
        reference = {
            surface: trimesh.load(
                cohort / "cache" / f"{patient}_ED_voxel_{surface}.ply", process=False)
            for surface in ("endo", "epi")
        }

    rows: list[dict] = []
    thickness_rows: list[dict] = []
    for method in METHODS:
        geometry = None
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
            metrics = (reconstruction_quality(geometry, reference, pitch=1.0)
                       if evaluate_reconstruction else None)
        except Exception as error:  # a failed fit must not abort the cohort
            print(f"  {patient} {method} failed: {error}", flush=True)
            metrics = {key: np.nan for key in METRICS}
            metrics.update(endo_watertight=False, epi_watertight=False)
        if metrics is not None:
            rows.append({"patient": patient, "method": method,
                         "n_contour_slices": len(np.unique(contours[:, 2])),
                         **metrics})
        try:
            if geometry is None:
                raise ValueError("fitted geometry is unavailable")
            thickness_rows.extend(evaluate_thickness(
                patient, method, geometry, contours))
        except Exception as error:
            print(f"  {patient} {method} thickness failed: {error}", flush=True)
            for thickness_method in THICKNESS_METHODS:
                thickness_rows.append({
                    "patient": patient, "phase": "ED", "geometry": method,
                    "method": thickness_method, "valid_fraction": 0.0,
                    **thickness_stats(np.asarray([])),
                })
    wall_cache.write_text(json.dumps(thickness_rows))
    return rows, thickness_rows


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

    fits_path = args.cohort / "fitting_baselines.csv"
    evaluate_reconstruction = args.force or not fits_path.exists()
    rows: list[dict] = []
    thickness_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_patient, patient, args.cohort,
                                   args.samples, args.force,
                                   evaluate_reconstruction): patient
                   for patient in patients}
        for done, future in enumerate(as_completed(futures), start=1):
            patient_rows, patient_thickness = future.result()
            rows.extend(patient_rows)
            thickness_rows.extend(patient_thickness)
            print(f"completed {futures[future]} ({done}/{len(patients)})", flush=True)

    if evaluate_reconstruction:
        fits = pd.DataFrame(rows).sort_values(["method", "patient"])
        fits.to_csv(fits_path, index=False)
    else:
        fits = pd.read_csv(fits_path)
        fits = fits[fits["patient"].isin(patients)]

    selected = model[model["patient"].isin(patients)]
    summary = build_summary(selected, fits, args.bootstrap_samples)
    summary.to_csv(args.cohort / "fitting_baselines_summary.csv", index=False)

    thickness = pd.DataFrame(thickness_rows).sort_values(
        ["geometry", "patient", "method"])
    thickness.to_csv(args.cohort / "fitting_baseline_wall_methods.csv", index=False)
    thickness_summary = (thickness.groupby(["geometry", "method"], sort=False)
                         [["valid_fraction", "mean_mm", "std_mm", "p5_mm", "p95_mm"]]
                         .mean().reset_index())
    thickness_summary.to_csv(
        args.cohort / "fitting_baseline_wall_summary.csv", index=False)

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
        print("  wall thickness")
        for row in thickness_summary[
                thickness_summary["geometry"] == method].itertuples(index=False):
            print(f"    {row.method:<20} {row.mean_mm:6.2f} {row.std_mm:6.2f} "
                  f"{row.p5_mm:6.2f} {row.p95_mm:6.2f}")


if __name__ == "__main__":
    main()
