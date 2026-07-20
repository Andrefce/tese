"""
Render publication-quality images of the CardioSDF reconstruction.

Loads the trained CardioSDF checkpoint, reconstructs the left ventricle of
ACDC patient002 from its short-axis (SAX) segmentation contours, and renders
smooth, shaded, high-resolution figures using PyVista (VTK) for the meshes and
matplotlib for the contour panel:

  images/recon_input_contours.pdf   sparse SAX contour rings (model input)
  images/recon_meshes_3d.png        endo/epi surfaces, three viewpoints
  images/recon_wall_thickness.png   endocardium coloured by analytic thickness
  images/recon_overview.png         contours -> surfaces -> thickness strip

Geometry and wall thickness come directly from the trained model; nothing is
invented. Run from the repository root with the project virtualenv active:

    python scripts/fig_cardiosdf_reconstruction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "webapp"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pyvista as pv  # noqa: E402
import nibabel as nib  # noqa: E402
import torch  # noqa: E402
from skimage.measure import marching_cubes  # noqa: E402
from core.sdf_model import (  # noqa: E402
    load_model,
    extract_contours,
    DEVICE,
    FLIP_Z,
)

pv.OFF_SCREEN = True
pv.global_theme.background = "white"
pv.global_theme.smooth_shading = True
pv.global_theme.anti_aliasing = "ssaa"

MODEL_PATH = ROOT / "notebooks" / "inr_sdf_combined_fresh_ed_mix_v1_final.pt"
SEG_PATH = (
    ROOT / "notebooks" / "patient002" / "patient002_frame01_gt.nii"
    / "DCM04-OH-AL_V2_1.nii"
)
OUT_DIR = ROOT / "images"

COL_ENDO = "#e63946"    # endocardium (warm red)
COL_EPI = "#b8c6d6"     # epicardium (cool grey-blue shell)
COL_ENDO_PT = "#e63946"
COL_EPI_PT = "#457b9d"
THICKNESS_CMAP = "viridis"


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


# ══════════════════════════════════════════════════════════════════════
# Reconstruction
# ══════════════════════════════════════════════════════════════════════
def reconstruct():
    print("Loading model ...")
    model, cfg = load_model(MODEL_PATH)

    print("Reading segmentation ...")
    img = nib.as_closest_canonical(nib.load(str(SEG_PATH)))
    seg = np.asarray(img.dataobj)
    affine = img.affine
    dz = float(abs(affine[2, 2])) or 1.0

    print("Extracting SAX contours ...")
    contours = extract_contours(seg, affine, dz)
    xyz_n = contours["xyz"]
    tissue = contours["tissue"]
    scale = contours["scale"]
    centroid = contours["centroid"]

    print("Running CardioSDF inference (clean marching cubes) ...")
    endo_v, endo_f, epi_v, epi_f, thickness = _infer_meshes(
        model, cfg, xyz_n, tissue, scale
    )

    # Un-normalise vertices back to millimetre space.
    flip = np.array([1.0, 1.0, -1.0 if FLIP_Z else 1.0], dtype=np.float32)
    endo_mm = endo_v * flip * scale + centroid
    epi_mm = epi_v * flip * scale + centroid
    cont_mm = xyz_n * flip * scale + centroid

    return {
        "contours_mm": cont_mm,
        "tissue": tissue,
        "endo": {"vertices": endo_mm, "faces": endo_f, "values": thickness},
        "epi": {"vertices": epi_mm, "faces": epi_f, "values": None},
        "metrics": {
            "endoVertices": len(endo_mm),
            "epiVertices": len(epi_mm),
            "meanWallThicknessMm": round(float(np.nanmean(thickness)), 2),
        },
    }


def _infer_meshes(model, cfg, contour_xyz, tissue_labels, scale,
                  grid_res=128, batch=200000):
    """Encode contours, sample the SDF grid, and marching-cubes both surfaces.

    Returns endo/epi vertices+faces (normalised space) and per-endo-vertex
    analytic wall thickness in millimetres. No contour snapping or mesh repair
    is applied; the surfaces are the raw zero-level sets of the decoder.
    """
    input_dim = cfg.get("input_dim", 5)
    cont = np.column_stack([contour_xyz, tissue_labels]).astype(np.float32)
    if input_dim == 5:
        cont = np.column_stack([cont, np.zeros((len(cont), 1), np.float32)])
    cont_t = torch.from_numpy(cont).unsqueeze(0).to(DEVICE)
    mask_t = torch.ones(1, len(cont), dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        z = model.encode(cont_t, mask_t)

    pad = cfg.get("bbox_pad", 0.3)
    lo = contour_xyz.min(0) - pad
    hi = contour_xyz.max(0) + pad
    xs = np.linspace(lo[0], hi[0], grid_res)
    ys = np.linspace(lo[1], hi[1], grid_res)
    zs = np.linspace(lo[2], hi[2], grid_res)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1).astype(np.float32)

    sdf_e = np.empty(len(grid), np.float32)
    sdf_p = np.empty(len(grid), np.float32)
    with torch.no_grad():
        for s in range(0, len(grid), batch):
            chunk = torch.from_numpy(grid[s:s + batch]).unsqueeze(0).to(DEVICE)
            fe, fp, _ = model.decode(z, chunk)
            sdf_e[s:s + batch] = fe[0].float().cpu().numpy()
            sdf_p[s:s + batch] = fp[0].float().cpu().numpy()

    shape = (grid_res, grid_res, grid_res)
    voxel = (hi - lo) / (grid_res - 1)
    iso = cfg.get("iso_level", 0.0)

    def mc(field):
        f = field.reshape(shape)
        if f.min() > iso or f.max() < iso:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.int64)
        v, faces, _, _ = marching_cubes(f, level=iso, spacing=tuple(voxel))
        return (v + lo).astype(np.float32), faces.astype(np.int64)

    endo_v, endo_f = mc(sdf_e)
    epi_v, epi_f = mc(sdf_p)

    # Analytic thickness at endo vertices: delta(x) * scale (mm).
    thickness = np.full(len(endo_v), np.nan, np.float32)
    if len(endo_v):
        with torch.no_grad():
            out = np.empty(len(endo_v), np.float32)
            for s in range(0, len(endo_v), batch):
                chunk = torch.from_numpy(endo_v[s:s + batch]).unsqueeze(0).to(DEVICE)
                _, _, dl = model.decode(z, chunk)
                out[s:s + batch] = dl[0].float().cpu().numpy()
        thickness = (out * scale).astype(np.float32)

    return endo_v, endo_f, epi_v, epi_f, thickness


# ══════════════════════════════════════════════════════════════════════
# Mesh helpers
# ══════════════════════════════════════════════════════════════════════
def _to_polydata(payload):
    v = np.asarray(payload["vertices"], dtype=np.float32).reshape(-1, 3)
    f = np.asarray(payload["faces"], dtype=np.int64).reshape(-1, 3)
    faces = np.hstack([np.full((len(f), 1), 3, dtype=np.int64), f]).ravel()
    mesh = pv.PolyData(v, faces)
    vals = payload.get("values")
    if vals is not None and len(vals) == len(v):
        mesh["thickness"] = np.asarray(vals, dtype=np.float32)
    return mesh


def _clean(mesh, smooth=60):
    m = mesh.clean().extract_largest().triangulate()
    if smooth:
        m = m.smooth_taubin(n_iter=smooth, pass_band=0.02)
    m = m.compute_normals(auto_orient_normals=True, feature_angle=60.0)
    return m


def _frame_camera(pl, mesh, azim, elev, zoom=1.3):
    """Position an isometric-style camera around the mesh centre."""
    c = np.array(mesh.center)
    r = float(np.linalg.norm(np.ptp(np.array(mesh.bounds).reshape(3, 2), axis=1))) or 1.0
    az = np.deg2rad(azim)
    el = np.deg2rad(elev)
    direction = np.array([
        np.cos(el) * np.sin(az),
        np.cos(el) * np.cos(az),
        np.sin(el),
    ])
    pos = c + direction * r * 2.2
    pl.camera_position = [tuple(pos), tuple(c), (0, 0, 1)]
    pl.camera.zoom(zoom)


# ══════════════════════════════════════════════════════════════════════
# Figure 1 — input contour rings (matplotlib, vector PDF)
# ══════════════════════════════════════════════════════════════════════
def fig_input_contours(res):
    cont = res["contours_mm"]
    tissue = res["tissue"]
    fig = plt.figure(figsize=(4.2, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    for lbl, col, name in [(0.0, COL_ENDO_PT, "Endocardium"),
                           (1.0, COL_EPI_PT, "Epicardium")]:
        pts = cont[np.abs(tissue - lbl) < 0.5]
        for zc in np.unique(np.round(pts[:, 2], 3)):
            ring = pts[np.abs(pts[:, 2] - zc) < 1e-3]
            if len(ring) < 3:
                continue
            ctr = ring[:, :2].mean(0)
            ang = np.arctan2(ring[:, 1] - ctr[1], ring[:, 0] - ctr[0])
            ring = ring[np.argsort(ang)]
            ring = np.vstack([ring, ring[0]])
            ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=col, lw=1.4, alpha=0.9)
        ax.plot([], [], color=col, label=name)
    lo, hi = cont.min(0), cont.max(0)
    ctr = (lo + hi) / 2
    r = float((hi - lo).max()) / 2 * 1.05
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=16, azim=-60)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title("Input: sparse SAX contour rings")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recon_input_contours.pdf")
    plt.close(fig)
    print("  wrote recon_input_contours.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 2 — reconstructed meshes, three viewpoints (PyVista)
# ══════════════════════════════════════════════════════════════════════
def fig_meshes_3d(endo, epi):
    views = [(0, 12, "Anterior-oblique"), (120, 12, "Posterior-oblique"),
             (0, 88, "Basal (top)")]
    pl = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(3 * 850, 950),
                    border=False)
    for i, (azim, elev, title) in enumerate(views):
        pl.subplot(0, i)
        pl.add_mesh(epi, color=COL_EPI, opacity=0.30, smooth_shading=True,
                    specular=0.2)
        pl.add_mesh(endo, color=COL_ENDO, smooth_shading=True,
                    specular=0.35, specular_power=18)
        _frame_camera(pl, epi, azim, elev)
        pl.add_text(title, position="lower_edge", font_size=13, color="black")
    pl.screenshot(str(OUT_DIR / "recon_meshes_3d.png"))
    pl.close()
    print("  wrote recon_meshes_3d.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 3 — endocardium coloured by analytic wall thickness (PyVista)
# ══════════════════════════════════════════════════════════════════════
def fig_wall_thickness(endo):
    if "thickness" not in endo.point_data:
        print("  no per-vertex thickness; skipping thickness figure")
        return
    t = endo["thickness"]
    finite = t[np.isfinite(t)]
    clim = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))

    views = [(0, 12, "Septal view"), (180, 12, "Lateral view")]
    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2 * 900, 1000),
                    border=False)
    sargs = dict(title="Wall thickness (mm)", vertical=False,
                 title_font_size=24, label_font_size=18, color="black",
                 position_x=0.2, position_y=0.04, width=0.6, height=0.06)
    for i, (azim, elev, title) in enumerate(views):
        pl.subplot(0, i)
        pl.add_mesh(endo, scalars="thickness", cmap=THICKNESS_CMAP, clim=clim,
                    smooth_shading=True, specular=0.25,
                    scalar_bar_args=sargs, show_scalar_bar=(i == 0))
        _frame_camera(pl, endo, azim, elev)
        pl.add_text(title, position="upper_edge", font_size=14, color="black")
    pl.screenshot(str(OUT_DIR / "recon_wall_thickness.png"))
    pl.close()
    print("  wrote recon_wall_thickness.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 4 — overview strip (PyVista, three panels)
# ══════════════════════════════════════════════════════════════════════
def fig_overview(res, endo, epi):
    cont = res["contours_mm"]
    tissue = res["tissue"]

    pl = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(3 * 820, 950),
                    border=False)

    # Panel (a): contour rings as tubes
    pl.subplot(0, 0)
    for lbl, col in [(0.0, COL_ENDO_PT), (1.0, COL_EPI_PT)]:
        pts = cont[np.abs(tissue - lbl) < 0.5]
        for zc in np.unique(np.round(pts[:, 2], 3)):
            ring = pts[np.abs(pts[:, 2] - zc) < 1e-3]
            if len(ring) < 3:
                continue
            ctr = ring[:, :2].mean(0)
            ang = np.arctan2(ring[:, 1] - ctr[1], ring[:, 0] - ctr[0])
            ring = ring[np.argsort(ang)]
            ring = np.vstack([ring, ring[0]])
            poly = pv.lines_from_points(ring)
            pl.add_mesh(poly.tube(radius=0.7), color=col, smooth_shading=True)
    _frame_camera(pl, epi, 0, 14)
    pl.add_text("(a) Sparse SAX contours", position="lower_edge",
                font_size=13, color="black")

    # Panel (b): reconstructed surfaces
    pl.subplot(0, 1)
    pl.add_mesh(epi, color=COL_EPI, opacity=0.30, smooth_shading=True)
    pl.add_mesh(endo, color=COL_ENDO, smooth_shading=True, specular=0.35,
                specular_power=18)
    _frame_camera(pl, epi, 0, 14)
    pl.add_text("(b) Reconstructed surfaces", position="lower_edge",
                font_size=13, color="black")

    # Panel (c): wall thickness
    pl.subplot(0, 2)
    if "thickness" in endo.point_data:
        t = endo["thickness"]
        finite = t[np.isfinite(t)]
        clim = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
        sargs = dict(title="mm", vertical=True, title_font_size=20,
                     label_font_size=16, color="black", position_x=0.85,
                     position_y=0.25, width=0.09, height=0.5)
        pl.add_mesh(endo, scalars="thickness", cmap=THICKNESS_CMAP, clim=clim,
                    smooth_shading=True, scalar_bar_args=sargs)
    else:
        pl.add_mesh(endo, color=COL_ENDO, smooth_shading=True)
    _frame_camera(pl, endo, 0, 14)
    pl.add_text("(c) Analytic wall thickness", position="lower_edge",
                font_size=13, color="black")

    pl.screenshot(str(OUT_DIR / "recon_overview.png"))
    pl.close()
    print("  wrote recon_overview.png")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    res = reconstruct()
    m = res["metrics"]
    print(f"Reconstruction: endo={m.get('endoVertices')} verts, "
          f"epi={m.get('epiVertices')} verts, mean WT={m.get('meanWallThicknessMm')} mm")

    endo_raw = _to_polydata(res["endo"])
    endo = _clean(endo_raw)
    epi = _clean(_to_polydata(res["epi"]))
    # Re-attach thickness lost during cleaning/smoothing by nearest resample.
    if "thickness" in endo_raw.point_data:
        endo = endo.interpolate(endo_raw, radius=3.0, sharpness=4.0,
                                strategy="closest_point")

    print("Rendering figures ...")
    fig_input_contours(res)
    fig_meshes_3d(endo, epi)
    fig_wall_thickness(endo)
    fig_overview(res, endo, epi)
    print("All figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
