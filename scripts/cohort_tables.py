"""Summarise the CardioSDF cohort CSVs into the numbers used in the Results tables.

Reads the full-cohort outputs written by the evaluation pipeline
(``scripts/webapp/notebooks/outputs/cohort_full``) and prints, for each Results
table, the cohort-level statistics and the corresponding LaTeX rows.

Run:
    C:/Python313/python.exe scripts/cohort_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "scripts" / "webapp" / "notebooks" / "outputs" / "cohort_full"

METHODS = ["Laplace field", "Yezzi-Prince", "SDF cone rays", "EDT boundary sum"]


def mean_sd(series: pd.Series) -> str:
    return f"{series.mean():.2f} $\\pm$ {series.std(ddof=1):.2f}"


def icc21(a: np.ndarray, b: np.ndarray) -> float:
    """Two-way random effects, single measurement, absolute agreement ICC."""
    matrix = np.column_stack([a, b])
    n, k = matrix.shape
    grand = matrix.mean()
    ms_rows = k * ((matrix.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((matrix.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0) + grand
    ms_err = (residual ** 2).sum() / ((n - 1) * (k - 1))
    return float((ms_rows - ms_err)
                 / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n))


def reconstruction_quality() -> None:
    df = pd.read_csv(DATA / "cohort_recon_quality.csv")
    print(f"\n=== Reconstruction quality (n={len(df)} patients, "
          f"phase={sorted(df['phase'].unique())}, groups={df['group'].value_counts().to_dict()})")
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
        ("F-score @ 1\\,mm", "fscore_1mm"),
        ("F-score @ 2\\,mm", "fscore_2mm"),
        ("Endocardium volume ratio", "vol_ratio_endo"),
        ("Epicardium volume ratio", "vol_ratio_epi"),
    ]
    for label, column in rows:
        print(f"    {label:38s} & {mean_sd(df[column])} \\\\")
    for surface in ("endo", "epi"):
        rate = 100.0 * df[f"{surface}_watertight"].mean()
        print(f"    Watertight rate ({surface}) & {rate:.0f}\\% \\\\")


def wall_methods() -> None:
    df = pd.read_csv(DATA / "cohort_wall_methods.csv")
    model = df[df["geometry"] == "cardiosdf"]
    print(f"\n=== Wall thickness, CardioSDF geometry (n={model['patient'].nunique()} patients)")
    print(f"    segmentation reference mean = "
          f"{model.groupby('patient')['ref_mean_mm'].first().mean():.2f} mm")
    for method in METHODS:
        sub = model[model["method"] == method]
        if sub.empty:
            continue
        print(f"    {method:18s} & {sub['raw_mean_mm'].mean():.2f} & "
              f"{sub['raw_std_mm'].mean():.2f} & {sub['raw_p5_mm'].mean():.2f} & "
              f"{sub['raw_p95_mm'].mean():.2f} \\\\")
    derived = df[df["geometry"] == "derived"]
    print("    -- same methods on the segmentation-derived geometry --")
    for method in METHODS:
        sub = derived[derived["method"] == method]
        if not sub.empty:
            print(f"    {method:18s}   mean={sub['raw_mean_mm'].mean():.2f} mm")


def agreement() -> None:
    df = pd.read_csv(DATA / "cohort_agreement.csv")
    print(f"\n=== Point-wise agreement with the Laplace reference "
          f"(n={df['patient'].nunique()} patients, {len(df)} paired points)")
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        value = sub["value"].to_numpy(float)
        reference = sub["reference"].to_numpy(float)
        ok = np.isfinite(value) & np.isfinite(reference)
        value, reference = value[ok], reference[ok]
        difference = value - reference
        bias, sd = difference.mean(), difference.std(ddof=1)
        print(f"    {method:18s} & {np.corrcoef(value, reference)[0, 1]:.2f} & "
              f"{np.abs(difference).mean():.2f} & "
              f"{np.sqrt((difference ** 2).mean()):.2f} & "
              f"${bias:+.2f}\\ (\\pm {1.96 * sd:.2f})$ & "
              f"{icc21(value, reference):.2f} \\\\   [n={len(value)}]")


def aha17() -> None:
    df = pd.read_csv(DATA / "cohort_aha17_thickening.csv")
    ed = df[df["phase"] == "ED"].groupby(["segment_id", "segment"])["mean_mm"].mean()
    es = df[df["phase"] == "ES"].groupby(["segment_id", "segment"])["mean_mm"].mean()
    print(f"\n=== AHA-17 (n={df['patient'].nunique()} patients, "
          f"groups={df.groupby('patient')['group'].first().value_counts().to_dict()})")
    for (segment_id, segment), ed_value in ed.items():
        es_value = es.loc[(segment_id, segment)]
        delta = es_value - ed_value
        print(f"    {segment_id:2d} & {segment:20s} & {ed_value:.2f} & {es_value:.2f} & "
              f"${delta:+.2f}$ & ${100.0 * delta / ed_value:+.1f}$ \\\\")
    ed_all = df[df["phase"] == "ED"]["mean_mm"].mean()
    es_all = df[df["phase"] == "ES"]["mean_mm"].mean()
    print(f"    cohort mean: ED={ed_all:.2f} mm  ES={es_all:.2f} mm  "
          f"thickening={100.0 * (es_all - ed_all) / ed_all:+.1f}%")


def model_versus_derived() -> None:
    df = pd.read_csv(DATA / "cohort_sota_agreement.csv")
    print(f"\n=== CardioSDF vs segmentation-derived geometry "
          f"(n={df['patient'].nunique()} patients), raw means in mm")
    for method in METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        difference = sub["cardiosdf_raw_mm"] - sub["derived_raw_mm"]
        print(f"    {method:18s} model={sub['cardiosdf_raw_mm'].mean():.2f} "
              f"derived={sub['derived_raw_mm'].mean():.2f} "
              f"bias={difference.mean():+.2f} MAE={difference.abs().mean():.2f} "
              f"r={np.corrcoef(sub['cardiosdf_raw_mm'], sub['derived_raw_mm'])[0, 1]:.2f}")


if __name__ == "__main__":
    reconstruction_quality()
    wall_methods()
    agreement()
    aha17()
    model_versus_derived()
