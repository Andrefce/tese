"""Render watertight ED and ES CardioSDF reconstructions side by side.

Reconstructs the left ventricle of ACDC patient002 at end-diastole (ED,
frame01) and end-systole (ES, frame12) from the SAX segmentation contours,
using the phase-conditioned CardioSDF decoder (phase channel 0 for ED, 1 for
ES). The endo/epi surfaces are the zero-level sets of the signed-distance
field, then cleaned, hole-filled, and (if the field was truncated at the valve
plane) base-capped into closed manifolds. The long axis is flipped for display
via ``FLIP_LONG_AXIS_FOR_DISPLAY`` so the apex points down. Watertight status is
printed; verify it is True before using the figure.

NOTE: orientation, watertightness, and physical calibration should ultimately be
fixed upstream in ``core.sdf_model``; the controls here are render-side fallbacks.

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

# Long-axis (Z) display orientation. The raw reconstructions currently render
# "upside down" (the basal valve plane points up); when True the long axis is
# flipped so the apex points down and the base up, matching the clinical view.
# This is a DISPLAY-ONLY correction until the orientation is fixed upstream in
# core.sdf_model — verify visually after the first render and flip if needed.
FLIP_LONG_AXIS_FOR_DISPLAY = True

COL_ENDO = "#e63946"
COL_EPI = "#b8c6d6"


def _cap_open_boundaries(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Close open boundary loops with a centroid fan to make the mesh watertight.

    Marching cubes truncated at the top of the grid leaves the LV open at the
    valve plane, and ``trimesh.repair.fill_holes`` only closes small holes. Each
    remaining open boundary loop (edges used by a single face) is closed by
    adding its centroid as a new vertex and fan-triangulating the loop to it.
    This is a geometric fallback for visualisation; the proper fix belongs in the
    upstream field extraction.
    """
    try:
        edges = mesh.edges_sorted
        uniq, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = uniq[counts == 1]
        if len(boundary) == 0:
            return mesh
        adj: dict[int, list[int]] = {}
        for a, b in boundary.tolist():
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        verts = list(np.asarray(mesh.vertices, dtype=np.float64))
        faces = list(np.asarray(mesh.faces, dtype=np.int64))
        used: set[tuple[int, int]] = set()

        def key(a: int, b: int) -> tuple[int, int]:
            return (a, b) if a < b else (b, a)

        for a0, nbrs in adj.items():
            for b0 in nbrs:
                if key(a0, b0) in used:
                    continue
                loop = [a0]
                used.add(key(a0, b0))
                prev, cur = a0, b0
                loop.append(cur)
                steps = 0
                while cur != a0 and steps < len(boundary) + 2:
                    steps += 1
                    nxt = None
                    for cand in adj.get(cur, ()):
                        if cand != prev and key(cur, cand) not in used:
                            nxt = cand
                            break
                    if nxt is None:
                        break
                    used.add(key(cur, nxt))
                    prev, cur = cur, nxt
                    if cur == a0:
                        break
                    loop.append(cur)
                if len(loop) < 3:
                    continue
                centre = np.mean([verts[i] for i in loop], axis=0)
                ci = len(verts)
                verts.append(centre)
                for i in range(len(loop)):
                    faces.append([loop[i], loop[(i + 1) % len(loop)], ci])
        return trimesh.Trimesh(vertices=np.asarray(verts),
                               faces=np.asarray(faces, dtype=np.int64),
                               process=False)
    except Exception:
        return mesh


def _clean_surface(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Merge vertices, keep largest component, fill holes, cap, fix normals."""
    try:
        mesh.merge_vertices()
    except Exception:
        pass
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
    if not mesh.is_watertight:
        mesh = _cap_open_boundaries(mesh)
    trimesh.repair.fix_normals(mesh)
    return mesh


def reconstruct(model, cfg, seg_path: Path, phase: float, label: str):
    raw = nib.load(str(seg_path))
    img = nib.as_closest_canonical(raw)
    seg = np.asarray(img.dataobj)
    affine = img.affine
    # Robust through-plane spacing. Some ACDC/M&Ms NIfTIs store the real voxel
    # spacing only in the header pixdim while their affine is identity, which
    # would collapse the 10 mm SAX slice spacing to 1 mm and flatten the LV
    # long axis by ~10x. Express the slice spacing in the affine's in-plane unit
    # so the long-axis aspect ratio is correct whether or not the affine is
    # calibrated (for a well-formed affine this reduces to |affine[2, 2]|).
    zooms = np.abs(np.asarray(raw.header.get_zooms()[:3], dtype=float))
    aff_inplane = float(np.linalg.norm(affine[:3, 0])) or 1.0
    true_inplane = float(min(zooms[0], zooms[1])) or 1.0
    true_slice = float(zooms[2]) or 1.0
    dz = true_slice * (aff_inplane / true_inplane)
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

    if FLIP_LONG_AXIS_FOR_DISPLAY:
        # Display-only long-axis correction (apex down, base up). Both surfaces
        # are flipped identically so their relative geometry is preserved; the
        # camera reframes on the mesh centre so the absolute offset is irrelevant.
        endo_mm[:, 2] *= -1.0
        epi_mm[:, 2] *= -1.0

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
