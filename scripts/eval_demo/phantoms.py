"""Analytic phantoms with known wall thickness.

The clinical geometry has no true thickness reference, so each estimator is
first characterised on shells whose thickness is known in closed form. This is
what allows the demo to state which method is accurate rather than merely which
methods agree with each other.

    concentric_sphere    t = R_epi - R_endo                      (constant)
    concentric_cylinder  t = R_epi - R_endo                      (constant)
    ellipsoidal_shell    normal offset shell, t = offset          (constant)
    tapered_shell        LV-like prolate shell, t varies apex->base
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class Phantom:
    name: str
    endo: trimesh.Trimesh
    epi: trimesh.Trimesh
    true_thickness: np.ndarray      # per endocardial vertex, mm
    description: str
    valid: np.ndarray | None = None  # vertices where the analytic thickness holds

    def __post_init__(self):
        if self.valid is None:
            self.valid = np.ones(len(self.endo.vertices), dtype=bool)


def _icosphere(subdivisions: int, radius: float) -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def concentric_sphere(r_endo: float = 22.0, t: float = 8.0,
                      subdivisions: int = 4) -> Phantom:
    endo = _icosphere(subdivisions, r_endo)
    epi = _icosphere(subdivisions, r_endo + t)
    return Phantom("Concentric sphere", endo, epi,
                   np.full(len(endo.vertices), t, np.float64),
                   f"R_endo={r_endo} mm, t={t} mm")


def concentric_cylinder(r_endo: float = 20.0, t: float = 7.0, height: float = 60.0,
                        sections: int = 96) -> Phantom:
    """Radial wall of constant thickness ``t``.

    The epicardial cylinder is extended axially by ``t`` at both ends so the
    endocardium is strictly enclosed. Only the lateral wall has an analytic
    transmural thickness, so the flat end caps are excluded from scoring.
    """
    endo = trimesh.creation.cylinder(radius=r_endo, height=height, sections=sections)
    epi = trimesh.creation.cylinder(radius=r_endo + t, height=height + 2.0 * t,
                                    sections=sections)
    z = np.asarray(endo.vertices)[:, 2]
    lateral = np.abs(np.abs(z) - height / 2.0) < 1e-6
    radius = np.linalg.norm(np.asarray(endo.vertices)[:, :2], axis=1)
    lateral &= radius > r_endo - 1e-6
    return Phantom("Concentric cylinder", endo, epi,
                   np.full(len(endo.vertices), t, np.float64),
                   f"R_endo={r_endo} mm, t={t} mm, h={height} mm (lateral wall only)",
                   valid=lateral)


def _offset_surface(mesh: trimesh.Trimesh, offset: np.ndarray | float) -> trimesh.Trimesh:
    normals = np.array(mesh.vertex_normals, dtype=np.float64, copy=True)
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)
    off = np.asarray(offset, np.float64).reshape(-1, 1) if np.ndim(offset) else offset
    out = trimesh.Trimesh(vertices=np.asarray(mesh.vertices) + normals * off,
                          faces=mesh.faces, process=False)
    trimesh.repair.fix_normals(out)
    return out


def ellipsoidal_shell(a: float = 24.0, b: float = 22.0, c: float = 38.0,
                      t: float = 9.0, subdivisions: int = 4) -> Phantom:
    """Normal-offset shell: the offset distance *is* the true transmural thickness."""
    sphere = _icosphere(subdivisions, 1.0)
    endo = trimesh.Trimesh(vertices=np.asarray(sphere.vertices) * np.array([a, b, c]),
                           faces=sphere.faces, process=False)
    trimesh.repair.fix_normals(endo)
    epi = _offset_surface(endo, t)
    return Phantom("Ellipsoidal shell", endo, epi,
                   np.full(len(endo.vertices), t, np.float64),
                   f"semi-axes ({a}, {b}, {c}) mm, normal offset t={t} mm")


def tapered_shell(a: float = 23.0, b: float = 21.0, c: float = 40.0,
                  t_apex: float = 5.0, t_base: float = 12.0,
                  subdivisions: int = 4) -> Phantom:
    """Prolate LV-like shell whose wall thickens linearly from apex to base."""
    sphere = _icosphere(subdivisions, 1.0)
    verts = np.asarray(sphere.vertices) * np.array([a, b, c])
    endo = trimesh.Trimesh(vertices=verts, faces=sphere.faces, process=False)
    trimesh.repair.fix_normals(endo)
    z = np.asarray(endo.vertices)[:, 2]
    frac = (z - z.min()) / (z.max() - z.min())
    thickness = t_apex + (t_base - t_apex) * frac
    epi = _offset_surface(endo, thickness)
    return Phantom("Tapered LV-like shell", endo, epi, thickness,
                   f"semi-axes ({a}, {b}, {c}) mm, t {t_apex}->{t_base} mm")


ALL_PHANTOMS = [concentric_sphere, concentric_cylinder, ellipsoidal_shell, tapered_shell]


def error_metrics(estimate: np.ndarray, truth: np.ndarray, runtime_s: float,
                  valid: np.ndarray | None = None) -> dict:
    est = np.asarray(estimate, np.float64)
    tru = np.asarray(truth, np.float64)
    ok = np.isfinite(est) & np.isfinite(tru)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    if ok.sum() == 0:
        return {"n": 0, "invalid_fraction": 1.0, "runtime_s": round(runtime_s, 2)}
    err = est[ok] - tru[ok]
    scored = np.ones(len(est), dtype=bool) if valid is None else np.asarray(valid, bool)
    return {
        "n": int(ok.sum()),
        "invalid_fraction": round(float((~ok & scored).sum() / max(scored.sum(), 1)), 4),
        "bias_mm": round(float(err.mean()), 3),
        "mae_mm": round(float(np.abs(err).mean()), 3),
        "rmse_mm": round(float(np.sqrt((err ** 2).mean())), 3),
        "p95_abs_err_mm": round(float(np.percentile(np.abs(err), 95)), 3),
        "runtime_s": round(runtime_s, 2),
    }
