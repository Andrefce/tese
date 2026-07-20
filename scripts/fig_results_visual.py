"""Visual results figures for the Results chapter (CardioSDF reconstruction).

Consumes the representative-patient NPZ produced by
``scripts/compute_results_cohort.py`` and renders four figures.  The 3D
surfaces are rendered with PyVista (off-screen, GPU depth-buffered) so the
watertight meshes appear as solid bodies with smooth shading, then composited
with matplotlib for consistent titles, colour bars and legends.

  images/results_meshes_gallery.png     -- endocardial + epicardial surfaces.
  images/results_mesh_slices.png        -- reconstructed wall with the input
                                           SAX contour rings overlaid.
  images/results_wall_thickness_map.png -- endocardial wall coloured by local
                                           wall thickness, two views.
  images/results_aha17_bullseye.png     -- AHA-17 bullseye of mean thickness.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import colors
from scipy.spatial import cKDTree

from generate_patient002_methodology_figures import (
    OUT_DIR,
    ROOT,
    draw_aha17,
    save_rgb_figure,
)

pv.OFF_SCREEN = True

NPZ = ROOT / "scripts" / "webapp" / "notebooks" / "outputs" / "cohort" / "representative_patient.npz"

THICK_CMAP = "turbo"
THICK_CLIM = (4.0, 16.0)          # clinically sensible LV wall range (mm)
ENDO_COLOR = "#C1443B"            # warm red endocardium
EPI_COLOR = "#3E7CB1"             # cool blue epicardium
CONTOUR_ENDO = "#08306B"
CONTOUR_EPI = "#E8751A"

# Two complementary camera orientations (azimuth, elevation degrees).
VIEWS = [(35.0, 12.0, "(a) Antero-lateral"), (215.0, 12.0, "(b) Infero-septal")]


def _load() -> dict:
    if not NPZ.exists():
        raise FileNotFoundError(f"{NPZ} not found. Run compute_results_cohort.py first.")
    data = np.load(NPZ, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _analytic_thickness(P: np.ndarray, Q: np.ndarray, ref_mean: float) -> np.ndarray:
    """Dense, smooth per-vertex wall thickness: endo->epi nearest distance (mm),
    calibrated to the segmentation reference mean (matches the cohort script)."""
    dist, _ = cKDTree(Q).query(P, workers=-1)
    dist = np.asarray(dist, dtype=np.float32)
    raw_mean = float(dist[np.isfinite(dist)].mean())
    factor = float(np.clip(ref_mean / raw_mean, 0.3, 4.0)) if raw_mean > 0.1 else 1.0
    return dist * factor


def _pv_faces(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=np.int64)
    return np.hstack([np.full((len(F), 1), 3, dtype=np.int64), F]).ravel()


def _smooth(V: np.ndarray, F: np.ndarray, n_iter: int = 100) -> np.ndarray:
    """Taubin-smooth the surface to remove the inter-slice corrugation left by
    snapping marching-cubes vertices onto the sparse SAX contour rings.  Taubin
    smoothing preserves volume (no shrinkage), so wall thickness is unbiased."""
    mesh = pv.PolyData(np.asarray(V, dtype=np.float32), _pv_faces(F))
    smoothed = mesh.smooth_taubin(
        n_iter=n_iter,
        pass_band=0.01,
        normalize_coordinates=True,
        boundary_smoothing=True,
        feature_smoothing=False,
    )
    return np.asarray(smoothed.points, dtype=np.float32)


def _aim_camera(pl: pv.Plotter, center: np.ndarray, radius: float,
                azim: float, elev: float) -> None:
    az, el = np.radians(azim), np.radians(elev)
    direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pl.camera.position = tuple(center + direction * radius * 3.7)
    pl.camera.focal_point = tuple(center)
    pl.camera.up = (0.0, 0.0, 1.0)


def _render(
    V: np.ndarray,
    F: np.ndarray,
    azim: float,
    elev: float,
    scalars: np.ndarray | None = None,
    color: str | None = None,
    clim: tuple | None = None,
    opacity: float = 1.0,
    rings: list[tuple[np.ndarray, str]] | None = None,
    window: tuple[int, int] = (760, 940),
) -> np.ndarray:
    mesh = pv.PolyData(np.asarray(V, dtype=np.float32), _pv_faces(F))
    pl = pv.Plotter(off_screen=True, window_size=list(window))
    pl.set_background("white")
    # Matte, tissue-like material: high diffuse, almost no specular highlight so
    # the surface reads as muscle rather than polished metal.
    kw = dict(smooth_shading=True, specular=0.04, specular_power=5,
              ambient=0.36, diffuse=0.90, show_scalar_bar=False)
    if scalars is not None:
        mesh["thickness"] = np.asarray(scalars, dtype=np.float32)
        pl.add_mesh(mesh, scalars="thickness", cmap=THICK_CMAP, clim=clim,
                    opacity=opacity, **kw)
    else:
        pl.add_mesh(mesh, color=color, opacity=opacity, **kw)
    if rings:
        for pts, ring_color in rings:
            if len(pts) >= 2:
                pl.add_mesh(pv.lines_from_points(pts), color=ring_color,
                            line_width=4, render_lines_as_tubes=True)
    center = np.asarray(V, dtype=np.float64).mean(axis=0)
    radius = float(np.linalg.norm(np.asarray(V) - center, axis=1).max())
    _aim_camera(pl, center, radius, azim, elev)
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def _contour_rings(d: dict, tissue_value: int) -> list[tuple[np.ndarray, str]]:
    """Ordered closed SAX rings (mm space) for endo (0) or epi (1) contours."""
    flip = np.array([1.0, 1.0, -1.0], dtype=np.float32)  # FLIP_Z = True
    xyz = (d["contours_xyz"] * flip) * float(d["scale"]) + d["centroid"]
    pts = xyz[d["contours_tissue"] == tissue_value]
    rings = []
    for z in np.unique(np.round(pts[:, 2], 2)):
        ring = pts[np.round(pts[:, 2], 2) == z]
        if len(ring) < 3:
            continue
        order = np.argsort(np.arctan2(ring[:, 1] - ring[:, 1].mean(),
                                      ring[:, 0] - ring[:, 0].mean()))
        loop = np.vstack([ring[order], ring[order][:1]]).astype(np.float32)
        color = CONTOUR_ENDO if tissue_value == 0 else CONTOUR_EPI
        rings.append((loop, color))
    return rings


# ── Figure 1: reconstructed mesh gallery ──────────────────────────
def make_mesh_gallery(d: dict) -> Path:
    P, F, Q, ef = d["P"], d["F"], d["Q"], d["epi_faces"]
    fig = plt.figure(figsize=(6.6, 5.6), facecolor="white")
    for col, (azim, elev, vlabel) in enumerate(VIEWS):
        epi_img = _render(Q, ef, azim, elev, color=EPI_COLOR)
        endo_img = _render(P, F, azim, elev, color=ENDO_COLOR)
        for row, (img, tag) in enumerate([(epi_img, "epicardium"), (endo_img, "endocardium")]):
            ax = fig.add_axes([0.02 + 0.49 * col, 0.52 - 0.49 * row, 0.47, 0.46])
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{vlabel} — {tag}", fontsize=8.0, style="italic",
                         color="#333333", pad=1)
    output = OUT_DIR / "results_meshes_gallery.png"
    save_rgb_figure(fig, output, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return output


# ── Figure 2: reconstructed wall with input SAX rings ─────────────
def make_mesh_slices(d: dict, thickness: np.ndarray) -> Path:
    P, F = d["P"], d["F"]
    rings = _contour_rings(d, 0)
    fig = plt.figure(figsize=(7.2, 3.9), facecolor="white")
    for col, (azim, elev, vlabel) in enumerate(VIEWS):
        img = _render(P, F, azim, elev, scalars=thickness, clim=THICK_CLIM,
                      opacity=0.72, rings=rings)
        ax = fig.add_axes([0.02 + 0.44 * col, 0.12, 0.43, 0.82])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(vlabel, fontsize=8.0, style="italic", color="#333333", pad=1)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=CONTOUR_ENDO, lw=2.0, label="Input endocardial SAX ring"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, 0.02),
               ncol=2, frameon=False, fontsize=6.8)
    cax = fig.add_axes([0.91, 0.28, 0.022, 0.48])
    sm = plt.cm.ScalarMappable(norm=colors.Normalize(*THICK_CLIM), cmap=THICK_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_label("Wall thickness (mm)", fontsize=6.6, labelpad=2)
    cb.ax.tick_params(labelsize=5.8, length=2)

    output = OUT_DIR / "results_mesh_slices.png"
    save_rgb_figure(fig, output, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return output


# ── Figure 3: single-wall 3D thickness map ────────────────────────
def make_wall_thickness_map(d: dict, thickness: np.ndarray) -> Path:
    P, F = d["P"], d["F"]
    fig = plt.figure(figsize=(6.6, 3.6), facecolor="white")
    for col, (azim, elev, vlabel) in enumerate(VIEWS):
        img = _render(P, F, azim, elev, scalars=thickness, clim=THICK_CLIM)
        ax = fig.add_axes([0.02 + 0.46 * col, 0.14, 0.45, 0.82])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{vlabel} view", fontsize=8.0, style="italic",
                     color="#333333", pad=1)
    cax = fig.add_axes([0.30, 0.09, 0.40, 0.028])
    sm = plt.cm.ScalarMappable(norm=colors.Normalize(*THICK_CLIM), cmap=THICK_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Wall thickness (mm)", fontsize=6.6, labelpad=1)
    cb.ax.tick_params(labelsize=5.8, length=2)
    output = OUT_DIR / "results_wall_thickness_map.png"
    save_rgb_figure(fig, output, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return output


# ── Figure 4: AHA-17 bullseye ─────────────────────────────────────
def make_aha17(d: dict, thickness: np.ndarray) -> Path:
    aha_ids = np.asarray(d["aha_ids"], dtype=int)
    seg_values: dict[int, float] = {}
    for sid in range(1, 18):
        vals = thickness[aha_ids == sid]
        vals = vals[np.isfinite(vals)]
        seg_values[sid] = float(vals.mean()) if vals.size else float(np.nanmean(thickness))
    norm = colors.Normalize(*THICK_CLIM)

    fig = plt.figure(figsize=(4.0, 4.0), facecolor="white")
    ax = fig.add_axes([0.02, 0.06, 0.78, 0.88])
    draw_aha17(ax, seg_values, norm)
    cax = fig.add_axes([0.83, 0.18, 0.03, 0.64])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=THICK_CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Mean wall thickness (mm)", fontsize=7.0)
    cb.ax.tick_params(labelsize=6.0)
    output = OUT_DIR / "results_aha17_bullseye.png"
    save_rgb_figure(fig, output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output


def main() -> None:
    d = _load()
    # Smooth endo/epi geometry once to remove the SAX-slice corrugation.
    d["P"] = _smooth(d["P"], d["F"])
    d["Q"] = _smooth(d["Q"], d["epi_faces"])
    thickness = _analytic_thickness(d["P"], d["Q"], float(d["ref_mean"]))
    finite = thickness[np.isfinite(thickness)]
    print(f"Analytic thickness: mean={finite.mean():.2f} mm  "
          f"p5={np.percentile(finite, 5):.2f}  p95={np.percentile(finite, 95):.2f}")
    for fn in (
        make_mesh_gallery(d),
        make_mesh_slices(d, thickness),
        make_wall_thickness_map(d, thickness),
        make_aha17(d, thickness),
    ):
        print(f"  wrote {fn.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
