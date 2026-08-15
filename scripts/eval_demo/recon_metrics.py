"""Reconstruction-quality metrics: CardioSDF surfaces vs voxel-derived surfaces.

Only the quantities reported in the Results chapter are computed: Chamfer
distance, ASSD, HD95, Dice/IoU (cavity and myocardium), normal consistency,
F-score at 1 and 2 mm, volume ratios and the watertight flag.

Chamfer is a vertex-to-vertex symmetric mean; ASSD, HD95 and the F-scores use
dense uniform surface samples, so the two are not trivially proportional.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from geometry import isotropic_grid, voxelise_surface

N_SURFACE_SAMPLES = 60000


def _sample(mesh: trimesh.Trimesh, n: int, seed: int = 0):
    points, face_idx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(points, np.float64), np.asarray(mesh.face_normals)[face_idx]


def _directed(a: np.ndarray, tree_b: cKDTree) -> np.ndarray:
    return tree_b.query(a, workers=-1)[0]


def surface_metrics(pred: trimesh.Trimesh, ref: trimesh.Trimesh,
                    n_samples: int = N_SURFACE_SAMPLES) -> dict:
    """Symmetric surface agreement between a predicted and a reference mesh."""
    pred_pts, pred_nrm = _sample(pred, n_samples, seed=0)
    ref_pts, ref_nrm = _sample(ref, n_samples, seed=1)
    tree_pred, tree_ref = cKDTree(pred_pts), cKDTree(ref_pts)

    d_pred = _directed(pred_pts, tree_ref)
    d_ref = _directed(ref_pts, tree_pred)

    pv = np.asarray(pred.vertices, np.float64)
    rv = np.asarray(ref.vertices, np.float64)
    chamfer = 0.5 * (float(cKDTree(rv).query(pv, workers=-1)[0].mean())
                     + float(cKDTree(pv).query(rv, workers=-1)[0].mean()))

    _, nn_ref = tree_ref.query(pred_pts, workers=-1)
    _, nn_pred = tree_pred.query(ref_pts, workers=-1)
    cos_a = np.abs(np.sum(pred_nrm * ref_nrm[nn_ref], axis=1))
    cos_b = np.abs(np.sum(ref_nrm * pred_nrm[nn_pred], axis=1))

    out = {
        "chamfer_mm": chamfer,
        "assd_mm": 0.5 * (float(d_pred.mean()) + float(d_ref.mean())),
        "hd95_mm": float(max(np.percentile(d_pred, 95), np.percentile(d_ref, 95))),
        "normal_consistency": 0.5 * (float(cos_a.mean()) + float(cos_b.mean())),
    }
    for tau in (1.0, 2.0):
        precision = float((d_pred <= tau).mean())
        recall = float((d_ref <= tau).mean())
        denom = precision + recall
        out[f"fscore_{int(tau)}mm"] = 0.0 if denom == 0 else 2 * precision * recall / denom
    return out


def overlap_metrics(pred_endo, pred_epi, ref_endo, ref_epi, pitch: float = 1.0) -> dict:
    """Dice / IoU of the cavity and of the myocardial shell on a shared grid."""
    origin, shape = isotropic_grid([pred_endo, pred_epi, ref_endo, ref_epi],
                                   pitch, pad_mm=3.0)
    masks = {name: voxelise_surface(mesh, origin, pitch, shape) for name, mesh in
             (("pred_endo", pred_endo), ("pred_epi", pred_epi),
              ("ref_endo", ref_endo), ("ref_epi", ref_epi))}
    masks["pred_epi"] |= masks["pred_endo"]
    masks["ref_epi"] |= masks["ref_endo"]
    pred_myo = masks["pred_epi"] & ~masks["pred_endo"]
    ref_myo = masks["ref_epi"] & ~masks["ref_endo"]

    out = {}
    for tag, a, b in (("endo", masks["pred_endo"], masks["ref_endo"]),
                      ("myo", pred_myo, ref_myo)):
        inter = float(np.count_nonzero(a & b))
        union = float(np.count_nonzero(a | b))
        total = float(np.count_nonzero(a) + np.count_nonzero(b))
        out[f"{tag}_dice"] = 0.0 if total == 0 else 2.0 * inter / total
        out[f"{tag}_iou"] = 0.0 if union == 0 else inter / union
    return out


def volume_ratios(pred_endo, pred_epi, ref_endo, ref_epi) -> dict:
    """Reconstructed cavity and wall volume divided by the reference value."""
    pe, pp = abs(float(pred_endo.volume)), abs(float(pred_epi.volume))
    re, rp = abs(float(ref_endo.volume)), abs(float(ref_epi.volume))
    return {
        "vol_ratio_endo": pe / re if re > 0 else np.nan,
        "vol_ratio_epi": pp / rp if rp > 0 else np.nan,
        "vol_ratio_myo": (pp - pe) / (rp - re) if (rp - re) > 0 else np.nan,
        "vol_endo_ml": pe, "vol_epi_ml": pp,
        "ref_vol_endo_ml": re, "ref_vol_epi_ml": rp,
    }


def reconstruction_quality(model: dict, voxel: dict, pitch: float = 1.0) -> dict:
    """All Results-chapter reconstruction metrics for one patient and phase."""
    row: dict = {}
    for surface in ("endo", "epi"):
        for key, value in surface_metrics(model[surface], voxel[surface]).items():
            row[f"{surface}_{key}" if key != "normal_consistency"
                else f"{surface}_normal_consistency"] = value
    row["normal_consistency"] = 0.5 * (row.pop("endo_normal_consistency")
                                       + row.pop("epi_normal_consistency"))
    for tau in (1, 2):
        row[f"fscore_{tau}mm"] = 0.5 * (row.pop(f"endo_fscore_{tau}mm")
                                        + row.pop(f"epi_fscore_{tau}mm"))
    row.update(overlap_metrics(model["endo"], model["epi"],
                               voxel["endo"], voxel["epi"], pitch))
    row.update(volume_ratios(model["endo"], model["epi"],
                             voxel["endo"], voxel["epi"]))
    row["endo_watertight"] = bool(model["endo"].is_watertight)
    row["epi_watertight"] = bool(model["epi"].is_watertight)
    return row
