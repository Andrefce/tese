"""Select the reference wall-thickness estimator objectively.

The clinical geometry has no ground truth, so choosing a reference method by
convention is arbitrary. This script ranks the four methods retained in the
methodology on the analytic phantoms, where the true transmural thickness is
known in closed form, and cross-checks the ranking against their reproducibility
between the model and voxel geometries on the real cohort.

    python select_reference_method.py [--pitch 1.0 0.75]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

import phantoms as ph                                                # noqa: E402
import thickness as tk                                               # noqa: E402
from geometry import outward_normals                                 # noqa: E402
from run_cohort import DEFAULT_OUT, METHODS                          # noqa: E402

LV_LIKE = "Tapered LV-like shell"


def run_methods(phantom, pitch: float) -> dict:
    endo, epi = phantom.endo, phantom.epi
    verts = np.asarray(endo.vertices)
    normals = outward_normals(endo, np.asarray(epi.vertices))
    ctx = tk.build_volume_context(endo, epi, pitch)
    phi, _ = tk.solve_laplace(ctx)
    return {
        "Laplace field": tk.method_laplace_streamline(ctx, verts, normals, phi),
        "Yezzi-Prince": tk.method_yezzi_prince(ctx, verts, normals, phi),
        "SDF cone rays": tk.method_cone_rays(endo, epi, normals),
        "EDT boundary sum": tk.method_edt_boundary_sum(ctx, verts, normals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitch", type=float, nargs="+", default=[1.0, 0.75])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    rows = []
    for pitch in args.pitch:
        for factory in ph.ALL_PHANTOMS:
            phantom = factory()
            for name, res in run_methods(phantom, pitch).items():
                metrics = ph.error_metrics(res.values, phantom.true_thickness,
                                           res.runtime_s, phantom.valid)
                rows.append({"pitch_mm": pitch, "phantom": phantom.name,
                             "method": name,
                             "true_mean_mm": round(float(
                                 phantom.true_thickness[phantom.valid].mean()), 3),
                             **metrics})
                print(f"  [{pitch} mm] {phantom.name:24s} {name:18s} "
                      f"bias {metrics.get('bias_mm'):+6.3f}  "
                      f"MAE {metrics.get('mae_mm'):6.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "phantom_selection.csv", index=False)

    print("\n== ranking on the phantoms (lower is better) ==")
    print("  method              mean MAE   max MAE   mean |bias|   LV-like MAE")
    ranking = []
    for method in METHODS:
        sub = df[df.method == method]
        lv = sub[sub.phantom == LV_LIKE]
        ranking.append((method, sub.mae_mm.mean(), sub.mae_mm.max(),
                        sub.bias_mm.abs().mean(), lv.mae_mm.mean()))
    ranking.sort(key=lambda r: r[1])
    for method, mean_mae, max_mae, bias, lv_mae in ranking:
        print(f"  {method:18s} {mean_mae:8.3f}  {max_mae:8.3f}  {bias:10.3f}   "
              f"{lv_mae:10.3f}")

    # Only the tapered shell has spatially varying thickness, so it is the only
    # phantom that tests whether a method can resolve a map rather than a mean.
    print("\n== spatial fidelity on the LV-like shell (varying thickness) ==")
    print("  method              r vs truth   slope   MAE")
    fidelity = {}
    for pitch in args.pitch:
        phantom = ph.tapered_shell()
        truth = phantom.true_thickness
        for name, res in run_methods(phantom, pitch).items():
            ok = np.isfinite(res.values) & np.asarray(phantom.valid, bool)
            est, tru = np.asarray(res.values, float)[ok], truth[ok]
            r = float(np.corrcoef(est, tru)[0, 1])
            slope = float(np.polyfit(tru, est, 1)[0])
            fidelity.setdefault(name, []).append(r)
            print(f"  [{pitch} mm] {name:18s} {r:8.3f}  {slope:7.3f}  "
                  f"{np.abs(est - tru).mean():6.3f}")

    wall = pd.read_csv(args.out / "wall_methods.csv")
    wall = wall[wall.phase == "ED"]
    print("\n== reproducibility on the real cohort (ED, model vs voxel) ==")
    print("  method              bias    r")
    for method in METHODS:
        pivot = wall[wall.method == method].pivot(index="patient", columns="geometry",
                                                  values="mean_mm").dropna()
        print(f"  {method:18s} {(pivot['model'] - pivot['voxel']).mean():+6.2f}  "
              f"{np.corrcoef(pivot['model'], pivot['voxel'])[0, 1]:.2f}")

    print(f"\nbest mean MAE over all phantoms: {ranking[0][0]}")
    print(f"best MAE on the LV-like phantom: {min(ranking, key=lambda r: r[4])[0]}")
    print("best spatial fidelity (r vs truth): "
          f"{max(fidelity, key=lambda k: np.mean(fidelity[k]))}")


if __name__ == "__main__":
    main()
