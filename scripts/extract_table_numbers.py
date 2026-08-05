"""Extract all numbers needed for the Results tables.

Computes statistics from the cohort_full CSVs, comparing:
 - CardioSDF model geometry vs segmentation-derived geometry (the voxel-based approach)
 - Split by cohort groups (NOR, MINF, ALL)

This script prints everything needed to update the LaTeX tables.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(r"c:\Users\André\Documents\tese-codigo\tese\scripts\webapp\notebooks\outputs\cohort_full")
SINGLE = Path(r"c:\Users\André\Documents\tese-codigo\tese\scripts\webapp\notebooks\outputs\cohort_single")

METHODS = ["Laplace field", "Yezzi-Prince", "SDF cone rays", "EDT boundary sum"]


def mean_sd(series: pd.Series) -> str:
    return f"{series.mean():.2f} ± {series.std(ddof=1):.2f}"


def icc21(a: np.ndarray, b: np.ndarray) -> float:
    """Two-way random effects, single measurement, absolute agreement ICC."""
    matrix = np.column_stack([a, b])
    n, k = matrix.shape
    grand = matrix.mean()
    ms_rows = k * ((matrix.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((matrix.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0) + grand
    ms_err = (residual ** 2).sum() / ((n - 1) * (k - 1))
    return float((ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n))


# =========================================================================
# TABLE 1: Reconstruction Quality
# =========================================================================
print("=" * 80)
print("TABLE 1: RECONSTRUCTION QUALITY")
print("=" * 80)
df = pd.read_csv(DATA / "cohort_recon_quality.csv")
print(f"\nTotal patients: {len(df)}, groups: {df['group'].value_counts().to_dict()}")

for group_label, sub in [("ALL", df), ("NOR", df[df["group"] == "NOR"]), ("MINF", df[df["group"] == "MINF"])]:
    print(f"\n--- {group_label} (n={len(sub)}) ---")
    rows = [
        ("Endo Chamfer (mm)", "endo_chamfer_mm"),
        ("Epi Chamfer (mm)", "epi_chamfer_mm"),
        ("Endo ASSD (mm)", "endo_assd_mm"),
        ("Epi ASSD (mm)", "epi_assd_mm"),
        ("Endo HD95 (mm)", "endo_hd95_mm"),
        ("Epi HD95 (mm)", "epi_hd95_mm"),
        ("Endo Dice", "endo_dice"),
        ("Myo Dice", "myo_dice"),
        ("Endo IoU", "endo_iou"),
        ("Myo IoU", "myo_iou"),
        ("Normal consistency", "normal_consistency"),
        ("F-score@1mm", "fscore_1mm"),
        ("F-score@2mm", "fscore_2mm"),
        ("Endo vol ratio", "vol_ratio_endo"),
        ("Epi vol ratio", "vol_ratio_epi"),
    ]
    for label, col in rows:
        print(f"  {label:30s}  {mean_sd(sub[col])}")
    for surface in ("endo", "epi"):
        rate = 100.0 * sub[f"{surface}_watertight"].mean()
        print(f"  Watertight ({surface})              {rate:.0f}%")


# =========================================================================
# TABLE 2: Wall-Thickness Methods - CardioSDF geometry
# =========================================================================
print("\n" + "=" * 80)
print("TABLE 2: WALL THICKNESS (CardioSDF geometry)")
print("=" * 80)
wm = pd.read_csv(DATA / "cohort_wall_methods.csv")
model = wm[wm["geometry"] == "cardiosdf"]
derived = wm[wm["geometry"] == "derived"]

print(f"\nSegmentation ref (all): {model.groupby('patient')['ref_mean_mm'].first().mean():.2f} mm")

for group_label, patients in [("ALL", model['patient'].unique()),
                               ("NOR", df[df["group"] == "NOR"]["patient"].values),
                               ("MINF", df[df["group"] == "MINF"]["patient"].values)]:
    print(f"\n--- {group_label} (n={len(patients)}) ---")
    print("  CardioSDF geometry:")
    for method in METHODS:
        sub = model[(model["method"] == method) & (model["patient"].isin(patients))]
        if sub.empty:
            continue
        print(f"    {method:20s} Mean={sub['raw_mean_mm'].mean():.2f}  "
              f"Std={sub['raw_std_mm'].mean():.2f}  "
              f"p5={sub['raw_p5_mm'].mean():.2f}  "
              f"p95={sub['raw_p95_mm'].mean():.2f}")

    print("  Derived (voxel) geometry:")
    for method in METHODS:
        sub = derived[(derived["method"] == method) & (derived["patient"].isin(patients))]
        if sub.empty:
            continue
        print(f"    {method:20s} Mean={sub['raw_mean_mm'].mean():.2f}  "
              f"Std={sub['raw_std_mm'].mean():.2f}  "
              f"p5={sub['raw_p5_mm'].mean():.2f}  "
              f"p95={sub['raw_p95_mm'].mean():.2f}")


# =========================================================================
# TABLE 3: Point-by-point agreement
# =========================================================================
print("\n" + "=" * 80)
print("TABLE 3: POINT-BY-POINT AGREEMENT (vs Laplace reference)")
print("=" * 80)
agr = pd.read_csv(DATA / "cohort_agreement.csv")
print(f"\nTotal paired points: {len(agr)}")

for method in agr["method"].unique():
    sub = agr[agr["method"] == method]
    value = sub["value"].to_numpy(float)
    reference = sub["reference"].to_numpy(float)
    ok = np.isfinite(value) & np.isfinite(reference)
    value, reference = value[ok], reference[ok]
    diff = value - reference
    bias, sd = diff.mean(), diff.std(ddof=1)
    r = np.corrcoef(value, reference)[0, 1]
    mae = np.abs(diff).mean()
    rmse = np.sqrt((diff ** 2).mean())
    icc = icc21(value, reference)
    print(f"  {method:20s}  r={r:.2f}  MAE={mae:.2f}  RMSE={rmse:.2f}  "
          f"Bias={bias:+.2f} (±{1.96*sd:.2f})  ICC={icc:.2f}  [n={len(value)}]")


# =========================================================================
# TABLE 4: AHA-17 Thickening (model geometry)
# =========================================================================
print("\n" + "=" * 80)
print("TABLE 4: AHA-17 THICKENING (CardioSDF geometry)")
print("=" * 80)
aha = pd.read_csv(DATA / "cohort_aha17_thickening.csv")
print(f"\nPatients: {aha['patient'].nunique()}, phases: {sorted(aha['phase'].unique())}")
print(f"Groups: {aha.groupby('patient')['group'].first().value_counts().to_dict()}")

ed = aha[aha["phase"] == "ED"].groupby(["segment_id", "segment"])["mean_mm"].mean()
es = aha[aha["phase"] == "ES"].groupby(["segment_id", "segment"])["mean_mm"].mean()

for (seg_id, seg_name), ed_val in ed.items():
    es_val = es.loc[(seg_id, seg_name)]
    delta = es_val - ed_val
    pct = 100.0 * delta / ed_val
    print(f"  {seg_id:2d}  {seg_name:22s}  ED={ed_val:.2f}  ES={es_val:.2f}  D={delta:+.2f}  {pct:+.1f}%")

ed_all = aha[aha["phase"] == "ED"]["mean_mm"].mean()
es_all = aha[aha["phase"] == "ES"]["mean_mm"].mean()
print(f"\n  Cohort average: ED={ed_all:.2f}  ES={es_all:.2f}  thickening={100*(es_all-ed_all)/ed_all:+.1f}%")


# =========================================================================
# TABLE 5: Derived AHA-17 (NOR only)
# =========================================================================
print("\n" + "=" * 80)
print("TABLE 5: DERIVED (VOXEL) AHA-17 THICKENING (NOR only)")
print("=" * 80)
try:
    d_aha = pd.read_csv(SINGLE / "derived_aha17_nor.csv")
    print(f"\nPatients: {d_aha['patient'].nunique()}")
    d_ed = d_aha[d_aha["phase"] == "ED"].groupby(["segment_id", "segment"])["mean_mm"].mean()
    d_es = d_aha[d_aha["phase"] == "ES"].groupby(["segment_id", "segment"])["mean_mm"].mean()
    for (seg_id, seg_name), ed_val in d_ed.items():
        es_val = d_es.loc[(seg_id, seg_name)]
        delta = es_val - ed_val
        pct = 100.0 * delta / ed_val
        print(f"  {seg_id:2d}  {seg_name:22s}  ED={ed_val:.2f}  ES={es_val:.2f}  Δ={delta:+.2f}  {pct:+.1f}%")
    d_ed_all = d_aha[d_aha["phase"] == "ED"]["mean_mm"].mean()
    d_es_all = d_aha[d_aha["phase"] == "ES"]["mean_mm"].mean()
    print(f"\n  Derived NOR average: ED={d_ed_all:.2f}  ES={d_es_all:.2f}  thickening={100*(d_es_all-d_ed_all)/d_ed_all:+.1f}%")
except Exception as exc:
    print(f"  Could not load derived AHA17: {exc}")

# Also compare model vs derived NOR:
print("\n--- MODEL (CardioSDF) NOR AHA-17 ---")
nor_pats = df[df["group"] == "NOR"]["patient"].values
aha_nor = aha[aha["patient"].isin(nor_pats)]
m_ed = aha_nor[aha_nor["phase"] == "ED"].groupby(["segment_id", "segment"])["mean_mm"].mean()
m_es = aha_nor[aha_nor["phase"] == "ES"].groupby(["segment_id", "segment"])["mean_mm"].mean()
for (seg_id, seg_name), ed_val in m_ed.items():
    es_val = m_es.loc[(seg_id, seg_name)]
    delta = es_val - ed_val
    pct = 100.0 * delta / ed_val
    print(f"  {seg_id:2d}  {seg_name:22s}  ED={ed_val:.2f}  ES={es_val:.2f}  Δ={delta:+.2f}  {pct:+.1f}%")


# =========================================================================
# TABLE 6: Model vs Derived per-patient agreement
# =========================================================================
print("\n" + "=" * 80)
print("TABLE 6: MODEL vs DERIVED (voxel) GEOMETRY AGREEMENT")
print("=" * 80)
sota = pd.read_csv(DATA / "cohort_sota_agreement.csv")
for method in METHODS:
    sub = sota[sota["method"] == method]
    if sub.empty:
        continue
    diff = sub["cardiosdf_raw_mm"] - sub["derived_raw_mm"]
    r = np.corrcoef(sub["cardiosdf_raw_mm"], sub["derived_raw_mm"])[0, 1]
    print(f"  {method:20s}  model={sub['cardiosdf_raw_mm'].mean():.2f}  "
          f"derived={sub['derived_raw_mm'].mean():.2f}  "
          f"bias={diff.mean():+.2f}  MAE={diff.abs().mean():.2f}  r={r:.2f}")

# =========================================================================
# Reconstruction quality split NOR vs MINF for comparison
# =========================================================================
print("\n" + "=" * 80)
print("RECONSTRUCTION QUALITY SPLIT")
print("=" * 80)
for group in ["NOR", "MINF"]:
    sub = df[df["group"] == group]
    print(f"\n--- {group} (n={len(sub)}) ---")
    for col in ["endo_chamfer_mm", "epi_chamfer_mm", "endo_dice", "myo_dice"]:
        print(f"  {col:25s}  {mean_sd(sub[col])}")
