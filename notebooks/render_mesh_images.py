"""Render CardioSDF left-ventricle mesh images using the production pipeline.

Standalone, inference-only script that renders the *same* meshes the webapp
serves to the browser. Rather than re-implementing marching cubes and mesh
cleanup, it calls ``predict_sdf_meshes`` from the webapp's own
``core.sdf_model`` module — the exact Path B pipeline documented in
"CardioSDF — The 3D Model Pipeline" (contour extraction, PointNet encoding,
INR decoding with ``f_epi = f_endo - delta``, marching cubes, snap-to-contours,
``_remove_planar_z_caps`` and ``_reduce_mesh``) — and renders the returned
vertex/face/value payloads with PyVista.

The phase channel selects end-diastole (0.0) or end-systole (1.0) when the
checkpoint is phase-conditioned (input_dim == 5).

Usage (repository root, project virtualenv active):

    python notebooks/render_mesh_images.py

Outputs (images/):
    mesh_ed_es.png          ED and ES surfaces side by side
    mesh_wall_thickness.png endocardium coloured by wall thickness
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "webapp"))

import pyvista as pv  # noqa: E402
import nibabel as nib  # noqa: E402
from core.sdf_model import (  # noqa: E402
    load_model,
    extract_contours,
    predict_sdf_meshes,
)

pv.OFF_SCREEN = True
pv.global_theme.background = "white"
pv.global_theme.smooth_shading = True
pv.global_theme.anti_aliasing = "ssaa"

MODEL_PATH = ROOT / "notebooks" / "inr_sdf_combined_fresh_ed_mix_v1_final.pt"
SEG_ED = (
    ROOT / "notebooks" / "patient002" / "patient002_frame01_gt.nii"
    / "DCM04-OH-AL_V2_1.nii"
)
SEG_ES = (
    ROOT / "notebooks" / "patient002" / "patient002_frame12_gt.nii"
    / "DCM04-OH-AL_V2_12.nii"
)
OUT_DIR = ROOT / "images"

COL_ENDO = "#e63946"      # endocardium (warm red)
COL_EPI = "#b8c6d6"       # epicardium (cool grey-blue shell)
THICKNESS_CMAP = "viridis"


# ──────────────────────────────────────────────────────────────────────
# Payload -> PyVista
# ──────────────────────────────────────────────────────────────────────
def payload_to_polydata(mesh: dict) -> pv.PolyData:
    """Convert a webapp mesh payload (flat vertex/face/value lists) to PolyData."""
    verts = np.asarray(mesh["vertices"], np.float32).reshape(-1, 3)
    faces_idx = np.asarray(mesh["faces"], np.int64).reshape(-1, 3)
    faces = np.hstack(
        [np.full((len(faces_idx), 1), 3, np.int64), faces_idx]
    ).ravel()
    pd = pv.PolyData(verts, faces)
    vals = mesh.get("values")
    if vals is not None and len(vals) == len(verts):
        pd["thickness"] = np.asarray(vals, np.float32)
    return pd.compute_normals(auto_orient_normals=True, feature_angle=60.0)


# ──────────────────────────────────────────────────────────────────────
# Production reconstruction (webapp Path B)
# ──────────────────────────────────────────────────────────────────────
def reconstruct(model, cfg, seg_path: Path, phase: float, label: str) -> dict:
    img = nib.as_closest_canonical(nib.load(str(seg_path)))
    seg = np.asarray(img.dataobj)
    affine = img.affine
    spacing = tuple(float(abs(affine[i, i])) or 1.0 for i in range(3))
    dz = spacing[2]

    contours = extract_contours(seg, affine, dz)
    result = predict_sdf_meshes(
        model,
        contours["xyz"],
        contours["tissue"],
        cfg,
        phase_val=phase,
        scale=contours["scale"],
        centroid=contours["centroid"],
        seg_volume=seg,
        spacing=spacing,
    )
    m = result["metrics"]
    print(f"  {label}: endo={m['endoVertices']} verts, epi={m['epiVertices']} verts, "
          f"mean WT={m['meanWallThicknessMm']} mm")
    return result


# ──────────────────────────────────────────────────────────────────────
# Cameras + rendering
# ──────────────────────────────────────────────────────────────────────
def frame_camera(pl, mesh, azim, elev, zoom=1.3):
    c = np.array(mesh.center)
    r = float(np.linalg.norm(np.ptp(np.array(mesh.bounds).reshape(3, 2), 1))) or 1.0
    az, el = np.deg2rad(azim), np.deg2rad(elev)
    direction = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az),
                          np.sin(el)])
    pl.camera_position = [tuple(c + direction * r * 2.2), tuple(c), (0, 0, 1)]
    pl.camera.zoom(zoom)


def render_ed_es(panels, out_path):
    pl = pv.Plotter(off_screen=True, shape=(1, len(panels)),
                    window_size=(len(panels) * 900, 1000), border=False)
    for i, (result, title) in enumerate(panels):
        pl.subplot(0, i)
        epi_pd = payload_to_polydata(result["meshes"]["epi"])
        endo_pd = payload_to_polydata(result["meshes"]["endo"])
        pl.add_mesh(epi_pd, color=COL_EPI, opacity=0.30, smooth_shading=True,
                    specular=0.2)
        pl.add_mesh(endo_pd, color=COL_ENDO, smooth_shading=True,
                    specular=0.35, specular_power=18)
        frame_camera(pl, epi_pd, 20, 22)
        pl.add_text(title, position="lower_edge", font_size=14, color="black")
    pl.screenshot(str(out_path))
    pl.close()
    print("wrote", out_path)


def render_wall_thickness(result, out_path):
    mesh = payload_to_polydata(result["meshes"]["endo"])
    if "thickness" not in mesh.point_data:
        print("  no per-vertex thickness; skipping thickness render")
        return
    t = mesh["thickness"]
    finite = t[np.isfinite(t)]
    clim = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
    views = [(0, 12, "Septal view"), (180, 12, "Lateral view")]
    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1800, 1000),
                    border=False)
    sargs = dict(title="Wall thickness (mm)", vertical=False,
                 title_font_size=22, label_font_size=16, color="black",
                 position_x=0.2, position_y=0.04, width=0.6, height=0.06)
    for i, (azim, elev, title) in enumerate(views):
        pl.subplot(0, i)
        pl.add_mesh(mesh, scalars="thickness", cmap=THICKNESS_CMAP, clim=clim,
                    smooth_shading=True, specular=0.25,
                    scalar_bar_args=sargs, show_scalar_bar=(i == 0))
        frame_camera(pl, mesh, azim, elev)
        pl.add_text(title, position="upper_edge", font_size=14, color="black")
    pl.screenshot(str(out_path))
    pl.close()
    print("wrote", out_path)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    model, cfg = load_model(MODEL_PATH)

    print("Reconstructing ED (frame01) ...")
    result_ed = reconstruct(model, cfg, SEG_ED, 0.0, "ED")
    print("Reconstructing ES (frame12) ...")
    result_es = reconstruct(model, cfg, SEG_ES, 1.0, "ES")

    render_ed_es(
        [(result_ed, "End-diastole (ED)"), (result_es, "End-systole (ES)")],
        OUT_DIR / "mesh_ed_es.png",
    )
    render_wall_thickness(result_ed, OUT_DIR / "mesh_wall_thickness.png")


if __name__ == "__main__":
    main()
