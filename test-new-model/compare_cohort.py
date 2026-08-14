"""Compare two ``run_cohort.py`` output directories on their shared patients.

The field-domain evaluation in ``cardiosdf2/evaluate.py`` measures what the
network predicts. This measures what the *thesis pipeline* reports after
marching cubes, watertight repair and the four thickness estimators — the
numbers that actually appear in Chapter 4. Only patients present in both
directories are used, so the comparison stays paired.

    python compare_cohort.py ../scripts/cohort_nor cohort_nor_v2 \
        --patients patient064 patient074 patient079
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

RECON = {
    "endo_dice": +1, "myo_dice": +1,
    "endo_chamfer_mm": -1, "epi_chamfer_mm": -1,
    "endo_hd95_mm": -1, "epi_hd95_mm": -1,
    "normal_consistency": +1, "fscore_2mm": +1,
    "vol_ratio_endo": 0, "vol_ratio_epi": 0, "vol_ratio_myo": 0,
}


def load(out_dir: Path) -> dict[str, dict]:
    return {p.name.split("_result")[0]: json.loads(p.read_text())
            for p in sorted((out_dir / "cache").glob("*_result.json"))}


def _wall(payload: dict, phase: str, geometry: str, method: str) -> float:
    for r in payload["wall"]:
        if (r["phase"] == phase and r["geometry"] == geometry
                and r["method"] == method):
            return float(r["mean_mm"])
    return float("nan")


def _paired_line(name: str, a: list[float], b: list[float], direction: int,
                 target: float | None = None) -> None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return
    a, b = a[ok], b[ok]
    if direction == 0:                      # ratio metrics: closeness to 1.0
        score_a, score_b = np.abs(a - 1.0), np.abs(b - 1.0)
        better = "v2" if score_b.mean() < score_a.mean() else "base"
    else:
        better = "v2" if (b.mean() - a.mean()) * direction > 0 else "base"
    try:
        p = float(stats.wilcoxon(a, b).pvalue) if ok.sum() >= 5 else float("nan")
    except ValueError:
        p = 1.0
    p_s = f"{p:9.4f}" if np.isfinite(p) else "        -"
    tgt = f"{target:9.2f}" if target is not None and np.isfinite(target) else "         "
    print(f"  {name:<26}{a.mean():10.3f}{b.mean():10.3f}{b.mean() - a.mean():+10.3f}"
          f"{better:>7}{p_s}{tgt}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--patients", nargs="*", default=None,
                    help="restrict to these (e.g. the v2 held-out set)")
    args = ap.parse_args()

    base, cand = load(args.baseline), load(args.candidate)
    shared = sorted(set(base) & set(cand))
    if args.patients:
        shared = [p for p in shared if p in set(args.patients)]
    if not shared:
        raise SystemExit("no shared patients between the two output directories")
    print(f"{len(shared)} shared patients: {', '.join(shared)}\n")

    print("reconstruction quality (model mesh vs voxel geometry)")
    print(f"  {'metric':<26}{'baseline':>10}{'v2':>10}{'delta':>10}"
          f"{'better':>7}{'p':>9}{'ideal':>9}")
    print("  " + "-" * 72)
    for key, direction in RECON.items():
        _paired_line(key, [base[p]["recon"][key] for p in shared],
                     [cand[p]["recon"][key] for p in shared], direction,
                     1.0 if direction == 0 else None)

    for phase in ("ED", "ES"):
        print(f"\nwall thickness, {phase} — model geometry vs the same patient's "
              f"voxel geometry")
        print(f"  {'method':<26}{'baseline':>10}{'v2':>10}{'delta':>10}"
              f"{'better':>7}{'p':>9}{'voxel':>9}")
        print("  " + "-" * 72)
        for method in ("Laplace field", "Yezzi-Prince", "EDT boundary sum",
                       "SDF cone rays"):
            vox = [_wall(base[p], phase, "voxel", method) for p in shared]
            a = [_wall(base[p], phase, "model", method) for p in shared]
            b = [_wall(cand[p], phase, "model", method) for p in shared]
            bias_a = float(np.nanmean(np.subtract(a, vox)))
            bias_b = float(np.nanmean(np.subtract(b, vox)))
            _paired_line(method, a, b,
                         +1 if abs(bias_b) < abs(bias_a) else -1,
                         float(np.nanmean(vox)))
            print(f"    {'bias vs voxel':<24}{bias_a:10.3f}{bias_b:10.3f}"
                  f"{abs(bias_b) - abs(bias_a):+10.3f}"
                  f"{'v2' if abs(bias_b) < abs(bias_a) else 'base':>7}")

    print("\nsegmentation-anchored reference (no meshing, identical for both runs)")
    for phase in ("ED", "ES"):
        ref = [base[p]["reference_mm"][phase] for p in shared]
        a = [_wall(base[p], phase, "model", "Laplace field") for p in shared]
        b = [_wall(cand[p], phase, "model", "Laplace field") for p in shared]
        print(f"  {phase}: anchor {np.nanmean(ref):6.2f} mm | "
              f"baseline {np.nanmean(a):6.2f} ({np.nanmean(a) - np.nanmean(ref):+.2f}) | "
              f"v2 {np.nanmean(b):6.2f} ({np.nanmean(b) - np.nanmean(ref):+.2f})")


if __name__ == "__main__":
    main()
