"""Agreement statistics used to compare the two geometries.

Per-vertex correspondence does not exist between the CardioSDF surface and the
voxel-derived surface (different meshes), so agreement is evaluated on the
paired AHA-17 regional means, which is also how the thesis reports thickness.
"""
from __future__ import annotations

import numpy as np


def bland_altman(a: np.ndarray, b: np.ndarray) -> dict:
    """Agreement of ``a`` (test) against ``b`` (reference)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return {"n": int(ok.sum())}
    diff = a[ok] - b[ok]
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    return {
        "n": int(ok.sum()),
        "bias_mm": round(bias, 3),
        "sd_diff_mm": round(sd, 3),
        "loa_lower_mm": round(bias - 1.96 * sd, 3),
        "loa_upper_mm": round(bias + 1.96 * sd, 3),
        "mae_mm": round(float(np.abs(diff).mean()), 3),
        "rmse_mm": round(float(np.sqrt((diff ** 2).mean())), 3),
    }


def icc_two_one(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    x = np.column_stack([a[ok], b[ok]])
    n, k = x.shape
    grand = x.mean()
    ms_rows = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    ss_total = ((x - grand) ** 2).sum()
    ss_err = ss_total - ms_rows * (n - 1) - ms_cols * (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denom) if abs(denom) > 1e-12 else float("nan")


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    out = bland_altman(a, b)
    out["icc_2_1"] = round(icc_two_one(a, b), 4)
    out["pearson_r"] = round(pearson_r(a, b), 4)
    return out


def segment_means(values: np.ndarray, aha_ids: np.ndarray,
                  valid: np.ndarray | None = None) -> np.ndarray:
    """Mean thickness per AHA-17 segment (index 0 -> segment 1)."""
    values = np.asarray(values, np.float64)
    keep = np.isfinite(values) if valid is None else (np.isfinite(values) & valid)
    out = np.full(17, np.nan)
    for seg in range(1, 18):
        sel = keep & (aha_ids == seg)
        if sel.any():
            out[seg - 1] = float(values[sel].mean())
    return out
