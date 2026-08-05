"""Extract NOR-only numbers for all Results tables.
Compare CardioSDF vs voxel-derived geometry for NOR cohort only.
"""
from __future__ import annotations
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(r"c:\Users\André\Documents\tese-codigo\tese\scripts\webapp\notebooks\outputs\cohort_full")
SINGLE = Path(r"c:\Users\André\Documents\tese-codigo\tese\scripts\webapp\notebooks\outputs\cohort_single")
METHODS = ["Laplace field", "Yezzi-Prince", "SDF cone rays", "EDT boundary sum"]

def icc21(a, b):
    matrix = np.column_stack([a, b])
    n, k = matrix.shape
    grand = matrix.mean()
    ms_rows = k * ((matrix.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((matrix.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0) + grand
    ms_err = (residual ** 2).sum() / ((n - 1) * (k - 1))
    return float((ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n))

# Get NOR patients
rq = pd.read_csv(DATA / "cohort_recon_quality.csv")
nor_pats = rq[rq["group"] == "NOR"]["patient"].values
print(f"NOR patients: {len(nor_pats)}")

# =====================================================================
# TABLE: Reconstruction Quality (NOR only)
# =====================================================================
print("\n" + "=" * 70)
print("TABLE: RECONSTRUCTION QUALITY (NOR only, n=20)")
print("=" * 70)
nor_rq = rq[rq["group"] == "NOR"]
rows = [
    ("Endocardium Chamfer distance (mm)", "endo_chamfer_mm"),
    ("Epicardium Chamfer distance (mm)", "epi_chamfer_mm"),
    ("Endocardium ASSD (mm)", "endo_assd_mm"),
    ("Epicardium ASSD (mm)", "epi_assd_mm"),
    ("Endocardium HD95 (mm)", "endo_hd95_mm"),
    ("Epicardium HD95 (mm)", "epi_hd95_mm"),
    ("Endocardium Dice", "endo_dice"),
    ("Myocardium Dice", "myo_dice"),
    ("Endocardium IoU", "endo_iou"),
    ("Myocardium IoU", "myo_iou"),
    ("Normal consistency", "normal_consistency"),
    ("F-score @ 1mm", "fscore_1mm"),
    ("F-score @ 2mm", "fscore_2mm"),
    ("Endocardium volume ratio", "vol_ratio_endo"),
    ("Epicardium volume ratio", "vol_ratio_epi"),
]
for label, col in rows:
    m = nor_rq[col].mean()
    s = nor_rq[col].std(ddof=1)
    print(f"  {label:40s}  ${m:.2f} \\pm {s:.2f}$")
for surface in ("endo", "epi"):
    rate = 100.0 * nor_rq[f"{surface}_watertight"].mean()
    print(f"  Watertight ({surface})                         {rate:.0f}%")

# =====================================================================
# TABLE: Wall Thickness - CardioSDF vs Derived (NOR only)
# =====================================================================
print("\n" + "=" * 70)
print("TABLE: WALL THICKNESS CardioSDF vs Derived (NOR, n=20)")
print("=" * 70)
wm = pd.read_csv(DATA / "cohort_wall_methods.csv")
model_nor = wm[(wm["geometry"] == "cardiosdf") & (wm["patient"].isin(nor_pats))]
derived_nor = wm[(wm["geometry"] == "derived") & (wm["patient"].isin(nor_pats))]

ref_mean = model_nor.groupby('patient')['ref_mean_mm'].first().mean()
print(f"  Segmentation reference mean = {ref_mean:.2f} mm\n")

print("  METHOD               | CardioSDF                          | Voxel-derived")
print("                       | Mean   Std    p5     p95           | Mean   Std    p5     p95")
for method in METHODS:
    mc = model_nor[model_nor["method"] == method]
    dc = derived_nor[derived_nor["method"] == method]
    print(f"  {method:20s} | {mc['raw_mean_mm'].mean():5.2f}  {mc['raw_std_mm'].mean():5.2f}  "
          f"{mc['raw_p5_mm'].mean():5.2f}  {mc['raw_p95_mm'].mean():5.2f}  "
          f"| {dc['raw_mean_mm'].mean():5.2f}  {dc['raw_std_mm'].mean():5.2f}  "
          f"{dc['raw_p5_mm'].mean():5.2f}  {dc['raw_p95_mm'].mean():5.2f}")

# =====================================================================
# TABLE: Agreement CardioSDF vs Derived (NOR only) 
# =====================================================================
print("\n" + "=" * 70)
print("TABLE: MODEL vs DERIVED GEOMETRY AGREEMENT (NOR only)")
print("=" * 70)
sota = pd.read_csv(DATA / "cohort_sota_agreement.csv")
sota_nor = sota[sota["patient"].isin(nor_pats)]
for method in METHODS:
    sub = sota_nor[sota_nor["method"] == method]
    if sub.empty:
        continue
    diff = sub["cardiosdf_raw_mm"] - sub["derived_raw_mm"]
    r = np.corrcoef(sub["cardiosdf_raw_mm"], sub["derived_raw_mm"])[0, 1]
    print(f"  {method:20s}  model={sub['cardiosdf_raw_mm'].mean():.2f}  "
          f"derived={sub['derived_raw_mm'].mean():.2f}  "
          f"bias={diff.mean():+.2f}  MAE={diff.abs().mean():.2f}  r={r:.2f}")

# =====================================================================
# TABLE: AHA-17 Thickening - CardioSDF vs Derived (NOR only)
# =====================================================================
print("\n" + "=" * 70)
print("TABLE: AHA-17 THICKENING COMPARISON NOR (n=20)")
print("Model (CardioSDF) vs Derived (voxel-based)")
print("=" * 70)

# Model AHA-17 NOR
aha = pd.read_csv(DATA / "cohort_aha17_thickening.csv")
aha_nor = aha[aha["patient"].isin(nor_pats)]
m_ed = aha_nor[aha_nor["phase"] == "ED"].groupby(["segment_id", "segment"])["mean_mm"].mean()
m_es = aha_nor[aha_nor["phase"] == "ES"].groupby(["segment_id", "segment"])["mean_mm"].mean()

# Derived AHA-17 NOR
d_aha = pd.read_csv(SINGLE / "derived_aha17_nor.csv")
d_ed = d_aha[d_aha["phase"] == "ED"].groupby(["segment_id", "segment"])["mean_mm"].mean()
d_es = d_aha[d_aha["phase"] == "ES"].groupby(["segment_id", "segment"])["mean_mm"].mean()

print(f"\n  {'#':>2s}  {'Segment':22s}  {'--- CardioSDF ---':^28s}  {'--- Voxel-derived ---':^28s}")
print(f"  {'':2s}  {'':22s}  {'ED':>6s}  {'ES':>6s}  {'D':>6s}  {'%':>6s}  {'ED':>6s}  {'ES':>6s}  {'D':>6s}  {'%':>6s}")

for (seg_id, seg_name), m_ed_val in m_ed.items():
    m_es_val = m_es.loc[(seg_id, seg_name)]
    m_delta = m_es_val - m_ed_val
    m_pct = 100.0 * m_delta / m_ed_val
    
    d_ed_val = d_ed.loc[(seg_id, seg_name)]
    d_es_val = d_es.loc[(seg_id, seg_name)]
    d_delta = d_es_val - d_ed_val
    d_pct = 100.0 * d_delta / d_ed_val
    
    print(f"  {seg_id:2d}  {seg_name:22s}  {m_ed_val:6.2f}  {m_es_val:6.2f}  {m_delta:+6.2f}  {m_pct:+6.1f}  "
          f"{d_ed_val:6.2f}  {d_es_val:6.2f}  {d_delta:+6.2f}  {d_pct:+6.1f}")

# Cohort average
m_ed_all = aha_nor[aha_nor["phase"] == "ED"]["mean_mm"].mean()
m_es_all = aha_nor[aha_nor["phase"] == "ES"]["mean_mm"].mean()
d_ed_all = d_aha[d_aha["phase"] == "ED"]["mean_mm"].mean()
d_es_all = d_aha[d_aha["phase"] == "ES"]["mean_mm"].mean()
print(f"\n  Model NOR avg: ED={m_ed_all:.2f}  ES={m_es_all:.2f}  thickening={100*(m_es_all-m_ed_all)/m_ed_all:+.1f}%")
print(f"  Derived NOR avg: ED={d_ed_all:.2f}  ES={d_es_all:.2f}  thickening={100*(d_es_all-d_ed_all)/d_ed_all:+.1f}%")
