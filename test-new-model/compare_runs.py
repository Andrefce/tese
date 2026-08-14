"""Paired baseline-vs-v2 comparison with significance tests.

``cardiosdf2/evaluate.py`` reports cohort means; a mean difference over 30
paired samples is not evidence on its own. This reads the per-sample rows that
``evaluate.py`` already writes and answers the only question that matters:
is the difference larger than the spread between patients?

Both columns are measured on the *same* samples at the *same* points, so every
metric is paired and the Wilcoxon signed-rank test applies directly.

    python compare_runs.py runs/u1u2_e50/eval_mesh.json
    python compare_runs.py runs/*/eval*.json --gates
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

# metric -> (direction, pretty name). +1 = higher is better, -1 = lower is better.
METRICS = {
    "residual_mm": (-1, "slice residual (mm)"),
    "sdf_mae_mm": (-1, "SDF MAE (mm)"),
    "wall_mae_mm": (-1, "wall MAE (mm)"),
    "abs_wall_bias_mm": (-1, "|wall bias| (mm)"),
    "seg_mae_mm": (-1, "segment MAE (mm)"),
    "seg_r": (+1, "segment r"),
    "abs_endo_vol_err": (-1, "|endo vol ratio - 1|"),
    "abs_epi_vol_err": (-1, "|epi vol ratio - 1|"),
}


def _augment(rows: list[dict]) -> list[dict]:
    for r in rows:
        r["abs_wall_bias_mm"] = abs(r["wall_bias_mm"])
        for tag in ("endo", "epi"):
            key = f"{tag}_volume_ratio"
            if key in r:
                r[f"abs_{tag}_vol_err"] = abs(r[key] - 1.0)
    return rows


def paired(rows_b: list[dict], rows_v: list[dict], key: str) -> dict | None:
    """Paired stats for one metric over the samples where both are finite."""
    b = np.array([r.get(key, np.nan) for r in rows_b], float)
    v = np.array([r.get(key, np.nan) for r in rows_v], float)
    ok = np.isfinite(b) & np.isfinite(v)
    if ok.sum() < 5:
        return None
    b, v, d = b[ok], v[ok], (v - b)[ok]
    try:
        p = float(stats.wilcoxon(b, v).pvalue)
    except ValueError:                       # all differences are zero
        p = 1.0
    return {"n": int(ok.sum()), "base": float(b.mean()), "v2": float(v.mean()),
            "delta": float(d.mean()), "p": p,
            "win_frac": float((d * METRICS[key][0] > 0).mean())}


def report(rows_b, rows_v, title: str, subset=None) -> None:
    if subset is not None:
        idx = [i for i, r in enumerate(rows_b) if subset(r)]
        rows_b = [rows_b[i] for i in idx]
        rows_v = [rows_v[i] for i in idx]
    if len(rows_b) < 5:
        return
    print(f"\n{title}  (n = {len(rows_b)})")
    print(f"  {'metric':<24}{'baseline':>10}{'v2':>10}{'delta':>10}"
          f"{'better':>9}{'p':>10}{'v2 wins':>10}")
    print("  " + "-" * 73)
    for key, (direction, name) in METRICS.items():
        st = paired(rows_b, rows_v, key)
        if st is None:
            continue
        improved = st["delta"] * direction > 0
        verdict = "v2" if improved else "base"
        star = "*" if st["p"] < 0.05 else " "
        print(f"  {name:<24}{st['base']:10.3f}{st['v2']:10.3f}{st['delta']:+10.3f}"
              f"{verdict:>8}{star}{st['p']:10.4f}{st['win_frac'] * 100:9.0f}%")


def gates(rows_b, rows_v) -> None:
    """Acceptance gates from the upgrade plan, §3, on whatever is available."""
    def sub(rows, group=None, phase=None):
        return [r for r in rows if (group is None or r["group"] == group)
                and (phase is None or r["phase"] == phase)]

    def mean(rows, key):
        vals = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    print("\nacceptance gates (upgrade plan §3, held-out subset)")
    print(f"  {'gate':<38}{'baseline':>11}{'v2':>11}{'pass':>7}")
    print("  " + "-" * 67)
    checks = []
    for rows, tag in ((rows_b, "base"), (rows_v, "v2")):
        nor_ed = sub(rows, "NOR", "ED")
        hcm = sub(rows, "HCM")
        checks.append({
            "endo watertight = 1.00": (mean(rows, "endo_watertight"), lambda x: x >= 1.0),
            "epi watertight = 1.00": (mean(rows, "epi_watertight"), lambda x: x >= 1.0),
            "NOR-ED endo vol ratio in [.95,1.05]":
                (mean(nor_ed, "endo_volume_ratio"), lambda x: 0.95 <= x <= 1.05),
            "NOR-ED epi vol ratio in [.95,1.05]":
                (mean(nor_ed, "epi_volume_ratio"), lambda x: 0.95 <= x <= 1.05),
            "slice residual <= 0.30 mm":
                (mean(rows, "residual_mm"), lambda x: x <= 0.30),
            "HCM mean max segment >= 14.5 mm":
                (mean(hcm, "max_seg_pred_mm"), lambda x: x >= 14.5),
            "HCM detection rate (>15 mm)":
                (float(np.mean([r["max_seg_pred_mm"] > 15 for r in hcm])) if hcm else np.nan,
                 lambda x: x >= 0.6),
        })
    hcm = sub(rows_b, "HCM")
    ref = float(np.mean([r["max_seg_gt_mm"] > 15 for r in hcm])) if hcm else np.nan
    for name in checks[0]:
        b, (v, test) = checks[0][name][0], checks[1][name]
        mark = "yes" if np.isfinite(v) and test(v) else "no"
        print(f"  {name:<38}{b:11.3f}{v:11.3f}{mark:>7}")
    if np.isfinite(ref):
        print(f"  {'  (voxel geometry reference)':<38}{'':>11}{ref:11.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--gates", action="store_true")
    args = ap.parse_args()

    for f in args.files:
        d = json.loads(f.read_text())
        rows_b = _augment(d["rows_baseline"])
        rows_v = _augment(d["rows_v2"])
        print("\n" + "=" * 79)
        print(f"{f}   ({len(rows_b)} paired samples, "
              f"{len({r['patient'] for r in rows_b})} patients)")
        print("=" * 79)
        report(rows_b, rows_v, "ALL")
        report(rows_b, rows_v, "end-diastole", lambda r: r["phase"] == "ED")
        report(rows_b, rows_v, "end-systole", lambda r: r["phase"] == "ES")
        report(rows_b, rows_v, "HCM only", lambda r: r["group"] == "HCM")
        report(rows_b, rows_v, "NOR only", lambda r: r["group"] == "NOR")
        if args.gates:
            gates(rows_b, rows_v)


if __name__ == "__main__":
    main()
