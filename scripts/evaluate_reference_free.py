"""Reference-free comparison of the three contour-driven reconstructions.

No independent 3D ground truth exists for these scans, so nothing here is
scored against a reference. Two quantities are reported instead:

``fidelity``   distance from each reconstructed surface to the SAX contour
               rings it was built from, which is the only measured anchor;
``agreement``  pairwise surface and volume agreement between the proposed
               model, the RBF implicit fit and the statistical shape-model fit.

Meshes are read from the cohort cache and the fitting-baseline cache, so no
inference and no surface fitting is repeated.

Example:
    C:/Python313/python.exe scripts/evaluate_reference_free.py --workers 2
"""
from __future__ import annotations

import argparse
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import trimesh

HERE = Path(__file__).resolve().parent
THESIS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "eval_demo"))

from evaluate_fitting_baselines import load_contours  # noqa: E402
from recon_metrics import reconstruction_quality  # noqa: E402

METHODS = ("model", "rbf", "ssm")
AGREEMENT_METRICS = (
    "endo_chamfer_mm",
    "epi_chamfer_mm",
    "endo_assd_mm",
    "epi_assd_mm",
    "endo_hd95_mm",
    "epi_hd95_mm",
    "endo_dice",
    "myo_dice",
)


def mesh_path(cohort: Path, patient: str, method: str, surface: str) -> Path:
    if method == "model":
        return cohort / "cache" / f"{patient}_ED_model_{surface}.ply"
    return cohort / "fit_cache" / f"{patient}_ED_{method}_{surface}.ply"


def as_polydata(mesh: trimesh.Trimesh) -> pv.PolyData:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    cells = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float32), cells)


def contour_fidelity(mesh: trimesh.Trimesh, contours: np.ndarray,
                     tissue: np.ndarray, surface: str) -> tuple[float, float]:
    """Exact distance from the input rings to the reconstructed surface."""
    label = 0.0 if surface == "endo" else 1.0
    rings = contours[np.abs(tissue - label) < 0.5]
    _, closest = as_polydata(mesh).find_closest_cell(rings, return_closest_point=True)
    distances = np.linalg.norm(rings - closest, axis=1)
    return float(np.mean(distances)), float(np.percentile(distances, 95))


def evaluate_patient(patient: str, cohort: Path, samples: Path) -> dict:
    contours, tissue = load_contours(samples, patient)
    geometry = {
        method: {surface: trimesh.load(mesh_path(cohort, patient, method, surface),
                                       process=False)
                 for surface in ("endo", "epi")}
        for method in METHODS
    }

    fidelity: list[dict] = []
    for method, surfaces in geometry.items():
        for surface, mesh in surfaces.items():
            mean, p95 = contour_fidelity(mesh, contours, tissue, surface)
            fidelity.append({
                "patient": patient, "method": method, "surface": surface,
                "contour_mean_mm": mean, "contour_p95_mm": p95,
                "watertight": bool(mesh.is_watertight),
                "volume_ml": float(mesh.volume) / 1000.0,
            })

    agreement: list[dict] = []
    for left, right in itertools.combinations(METHODS, 2):
        metrics = reconstruction_quality(geometry[left], geometry[right], pitch=1.0)
        agreement.append({"patient": patient, "pair": f"{left}-{right}",
                          **{key: metrics[key] for key in AGREEMENT_METRICS}})
    return {"fidelity": fidelity, "agreement": agreement}


def bootstrap_interval(values: np.ndarray, samples: int,
                       seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, len(values)), replace=True)
    return (float(np.quantile(draws.mean(axis=1), 0.025)),
            float(np.quantile(draws.mean(axis=1), 0.975)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path,
                        default=THESIS / "test-new-model" / "cohort_full_nor_hcm10")
    parser.add_argument("--samples", type=Path,
                        default=THESIS / "test-new-model" / "cache")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--patients", nargs="*", default=None)
    args = parser.parse_args()

    patients = pd.read_csv(args.cohort / "recon_quality.csv")["patient"].astype(str)
    patients = [p for p in patients
                if not args.patients or p in set(args.patients)]

    fidelity: list[dict] = []
    agreement: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_patient, patient, args.cohort,
                                   args.samples): patient for patient in patients}
        for done, future in enumerate(as_completed(futures), start=1):
            payload = future.result()
            fidelity.extend(payload["fidelity"])
            agreement.extend(payload["agreement"])
            print(f"completed {futures[future]} ({done}/{len(patients)})", flush=True)

    fidelity_frame = pd.DataFrame(fidelity).sort_values(["method", "surface", "patient"])
    agreement_frame = pd.DataFrame(agreement).sort_values(["pair", "patient"])
    fidelity_frame.to_csv(args.cohort / "reference_free_fidelity.csv", index=False)
    agreement_frame.to_csv(args.cohort / "reference_free_agreement.csv", index=False)

    print(f"\ncontour fidelity over {len(patients)} patients (mm)")
    for method in METHODS:
        for surface in ("endo", "epi"):
            subset = fidelity_frame[(fidelity_frame.method == method)
                                    & (fidelity_frame.surface == surface)]
            mean = subset["contour_mean_mm"].to_numpy(float)
            low, high = bootstrap_interval(mean, args.bootstrap_samples, seed=7)
            print(f"  {method:<5} {surface:<4} mean {mean.mean():5.2f} +/- "
                  f"{mean.std(ddof=1):4.2f}  [{low:.2f}, {high:.2f}]   "
                  f"p95 {subset['contour_p95_mm'].mean():5.2f}   "
                  f"volume {subset['volume_ml'].mean():7.1f} ml   "
                  f"watertight {100.0 * subset['watertight'].mean():.0f}%")

    print("\npairwise agreement")
    for pair in agreement_frame["pair"].unique():
        subset = agreement_frame[agreement_frame.pair == pair]
        print(f"  {pair}")
        for metric in AGREEMENT_METRICS:
            values = subset[metric].to_numpy(float)
            print(f"    {metric:<16} {values.mean():6.3f} +/- {values.std(ddof=1):5.3f}")


if __name__ == "__main__":
    main()
