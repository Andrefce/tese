"""Render watertight ED and ES CardioSDF reconstructions side by side.

Reconstructs the left ventricle of ACDC patient002 at end-diastole (ED,
frame01) and end-systole (ES, frame12) from the SAX segmentation contours,
using the phase-conditioned CardioSDF decoder (phase channel 0 for ED, 1 for
ES). The endo/epi surfaces are the zero-level sets of the signed-distance
field, then cleaned and hole-filled into closed (watertight) manifolds using
the same post-processing as the cohort evaluation. Watertight status is
printed.

Output: images/recon_ed_es_meshes.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "webapp"))

import pyvista as pv  # noqa: E402
import nibabel as nib  # noqa: E402
import trimesh  # noqa: E402
from core.sdf_model import (  # noqa: E402
    load_model,
    extract_contours,
    FLIP_Z,
    _build_contour_tensor,
    _build_grid_and_query,
    _mc_field,
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
GRID_RES = 96

COL_ENDO = "#e63946"
COL_EPI = "#b8c6d6"


def _clean_surface(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Largest component, fill holes, fix normals -> closed manifold."""
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            mesh = max(components, key=lambda c: len(c.faces))
    except Exception:
        pass
    for _ in range(3):
        if mesh.is_watertight:
            break
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            break
    trimesh.repair.fix_normals(mesh)
    return mesh


def reconstruct(model, cfg, seg_path: Path, phase: float, label: str):
    img = nib.as_closest_canonical(nib.load(str(seg_path)))
    seg = np.asarray(img.dataobj)
    affine = img.affine
    dz = float(abs(affine[2, 2])) or 1.0
    contours = extract_contours(seg, affine, dz)
    xyz_n = contours["xyz"]
    scale = contours["scale"]
    centroid = contours["centroid"]

    cont_t, mask_t = _build_contour_tensor(
        xyz_n, contours["tissue"], cfg, phase)
    z = model.encode(cont_t, mask_t)
    sdf_e, sdf_p, _dlt, lo, _hi, voxel = _build_grid_and_query(
        z, model, xyz_n, cfg, GRID_RES)
    iso = cfg.get("iso_level", 0.0)
    endo_v, endo_f = _mc_field(sdf_e, lo, voxel, iso)
    epi_v, epi_f = _mc_field(sdf_p, lo, voxel, iso)

    flip = np.array([1.0, 1.0, -1.0 if FLIP_Z else 1.0], dtype=np.float32)
    endo_mm = endo_v * flip * scale + centroid
    epi_mm = epi_v * flip * scale + centroid

    endo = _clean_surface(trimesh.Trimesh(endo_mm, endo_f.astype(np.int32),
                                          process=False))
    epi = _clean_surface(trimesh.Trimesh(epi_mm, epi_f.astype(np.int32),
                                         process=False))
    print(f"  {label} endo: {len(endo.vertices)} verts, "
          f"watertight={endo.is_watertight}")
    print(f"  {label} epi:  {len(epi.vertices)} verts, "
          f"watertight={epi.is_watertight}")
    return endo, epi


def to_polydata(m: trimesh.Trimesh) -> pv.PolyData:
    faces = np.hstack(
        [np.full((len(m.faces), 1), 3, np.int64), m.faces.astype(np.int64)]
    ).ravel()
    mesh = pv.PolyData(np.asarray(m.vertices, np.float32), faces)
    return mesh.smooth_taubin(n_iter=150, pass_band=0.01).compute_normals(
        auto_orient_normals=True, feature_angle=60.0)


def frame_camera(pl, mesh, azim, elev):
    c = np.array(mesh.center)
    r = float(np.linalg.norm(np.ptp(np.array(mesh.bounds).reshape(3, 2), 1))) or 1.0
    az, el = np.deg2rad(azim), np.deg2rad(elev)
    direction = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az),
                          np.sin(el)])
    pl.camera_position = [tuple(c + direction * r * 2.2), tuple(c), (0, 0, 1)]
    pl.camera.zoom(1.3)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    model, cfg = load_model(MODEL_PATH)

    print("Reconstructing ED (frame01) ...")
    endo_ed, epi_ed = reconstruct(model, cfg, SEG_ED, 0.0, "ED")
    print("Reconstructing ES (frame12) ...")
    endo_es, epi_es = reconstruct(model, cfg, SEG_ES, 1.0, "ES")

    panels = [(to_polydata(endo_ed), to_polydata(epi_ed), "End-diastole (ED)"),
              (to_polydata(endo_es), to_polydata(epi_es), "End-systole (ES)")]
    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2 * 900, 1000),
                    border=False)
    for i, (endo, epi, title) in enumerate(panels):
        pl.subplot(0, i)
        pl.add_mesh(epi, color=COL_EPI, opacity=0.30, smooth_shading=True,
                    specular=0.2)
        pl.add_mesh(endo, color=COL_ENDO, smooth_shading=True,
                    specular=0.35, specular_power=18)
        frame_camera(pl, epi, 20, 22)
        pl.add_text(title, position="lower_edge", font_size=14, color="black")
    pl.screenshot(str(OUT_DIR / "recon_ed_es_meshes.png"))
    pl.close()
    print("wrote", OUT_DIR / "recon_ed_es_meshes.png")


if __name__ == "__main__":
    main()
