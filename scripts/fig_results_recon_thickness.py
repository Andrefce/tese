"""Results-chapter counterparts of the methodology wall-thickness mesh figures.

Mirrors the two SSM-based methodology figures (``fig_lv_wall_thickness_3d`` and
``fig_aha17_cut``) but computes everything on the CardioSDF reconstruction of
ACDC patient002 instead of on the statistical-shape-model mean mesh: the
endocardial and epicardial surfaces are the zero-level sets of the predicted
signed-distance field, and the calibrated analytic offset is the decoder field
``delta`` read directly at each endocardial vertex. It is not a geometric
wall-thickness measurement.

  images/results_recon_thickness_3d.png -- reconstructed endocardium coloured by
      the calibrated analytic offset, ED and ES, two opposing views each.
  images/results_recon_aha17.png        -- the same reconstructed surface cut
      into the AHA-17 segments beside the unrolled bullseye, both coloured by
    the mean calibrated analytic offset.

The reconstruction is cached in ``CACHE`` so re-styling the figures does not
re-run the decoder; pass ``--refresh`` to recompute it.

Run:
    C:/Python313/python.exe scripts/fig_results_recon_thickness.py [--refresh]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "webapp"))

from generate_patient002_methodology_figures import (  # noqa: E402
    OUT_DIR,
    draw_aha17,
    save_rgb_figure,
    set_mesh_axes,
)

CACHE = (ROOT / "scripts" / "webapp" / "notebooks" / "outputs"
         / "patient002_recon_thickness.npz")
CMAP = "turbo"
VIEWS = [(20.0, -60.0, "Antero-lateral"), (20.0, 120.0, "Infero-septal")]
FIELDS = ("endo_v", "endo_f", "thickness", "epi_v", "epi_f")
COL_ENDO = "#c1443b"
COL_EPI = "#b8c6d6"
# Larger than the inference default: the epicardium extends further below the
# last contour than the endocardium and its apex is otherwise clipped by the
# grid floor, which leaves a frayed open boundary.
GRID_RES = 112
BBOX_PAD = 0.55
# Voxel pitch of the Laplace solve, matching the cohort pipeline.
LAPLACE_PITCH_MM = 1.0
# Completed-cohort meshes, used by --cohort when the patient002 cache is absent.
COHORT_CACHE = ROOT / "test-new-model" / "cohort_full_nor_hcm10" / "cache"
COHORT_PATIENT = "patient071"
DATA_ROOT = ROOT / "notebooks" / "data" / "training"


def load_cases(refresh: bool = False) -> tuple[dict, dict]:
    """Return the ED and ES reconstructions as dicts of arrays.

    Reads the cached reconstruction when available; otherwise runs CardioSDF on
    both phases and stores the result.
    """
    if CACHE.exists() and not refresh:
        data = np.load(CACHE)
        if all(f"ed_{field}" in data for field in FIELDS):
            print(f"using cached meshes from {CACHE.relative_to(ROOT)}")
            return tuple({field: data[f"{phase}_{field}"] for field in FIELDS}
                         for phase in ("ed", "es"))
        print("cache predates the epicardial surfaces; recomputing")

    from fig_ed_es_meshes import MODEL_PATH, SEG_ED, SEG_ES  # noqa: PLC0415
    from core.sdf_model import load_model  # noqa: PLC0415

    model, cfg = load_model(MODEL_PATH)
    print("Reconstructing ED (frame01) ...")
    ed = reconstruct_with_thickness(model, cfg, SEG_ED, 0.0)
    print("Reconstructing ES (frame12) ...")
    es = reconstruct_with_thickness(model, cfg, SEG_ES, 1.0)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        **{f"{phase}_{field}": case[field]
           for phase, case in (("ed", ed), ("es", es))
           for field in FIELDS},
    )
    print(f"cached meshes to {CACHE.relative_to(ROOT)}")
    return ed, es


def reconstruct_with_thickness(model, cfg, seg_path: Path, phase: float):
    """Reconstruct one phase and return (endo vertices, faces, thickness in mm).

    The thickness is the decoder's monotone-epi offset evaluated at the
    endocardial vertices, converted to millimetres with the contour scale.
    """
    import nibabel as nib  # noqa: PLC0415
    import trimesh  # noqa: PLC0415
    from core.sdf_model import (  # noqa: PLC0415
        FLIP_Z,
        _build_contour_tensor,
        _build_grid_and_query,
        _mc_field,
        _reference_wall_thickness_from_segmentation,
        extract_contours,
    )
    from fig_ed_es_meshes import (  # noqa: PLC0415
        FLIP_LONG_AXIS_FOR_DISPLAY,
        _clean_surface,
    )

    raw = nib.load(str(seg_path))
    img = nib.as_closest_canonical(raw)
    seg = np.asarray(img.dataobj)
    affine = img.affine
    # Same slice-spacing calibration as fig_ed_es_meshes.reconstruct: these
    # NIfTIs keep the real spacing only in the header pixdim, so deriving the
    # through-plane step from the affine alone flattens the long axis ~10x.
    zooms = np.abs(np.asarray(raw.header.get_zooms()[:3], dtype=float))
    aff_inplane = float(np.linalg.norm(affine[:3, 0])) or 1.0
    true_inplane = float(min(zooms[0], zooms[1])) or 1.0
    dz = (float(zooms[2]) or 1.0) * (aff_inplane / true_inplane)

    contours = extract_contours(seg, affine, dz)
    xyz_n = contours["xyz"]
    scale = contours["scale"]
    centroid = contours["centroid"]

    cont_t, mask_t = _build_contour_tensor(xyz_n, contours["tissue"], cfg, phase)
    z = model.encode(cont_t, mask_t)
    sdf_e, sdf_p, _dlt, lo, _hi, voxel = _build_grid_and_query(
        z, model, xyz_n, dict(cfg, bbox_pad=BBOX_PAD), GRID_RES)
    iso = cfg.get("iso_level", 0.0)

    flip = np.array([1.0, 1.0, -1.0 if FLIP_Z else 1.0], dtype=np.float32)
    ring_z_mm = np.unique(xyz_n[:, 2]) * flip[2] * scale + centroid[2]
    if FLIP_LONG_AXIS_FOR_DISPLAY:
        ring_z_mm = -ring_z_mm
    z_cut = float(ring_z_mm.max())

    surfaces = {}
    for name, field in (("endo", sdf_e), ("epi", sdf_p)):
        verts, faces = _mc_field(field, lo, voxel, iso)
        verts_mm = verts * flip * scale + centroid
        if FLIP_LONG_AXIS_FOR_DISPLAY:
            verts_mm[:, 2] *= -1.0
        mesh = _clean_surface(trimesh.Trimesh(verts_mm, faces.astype(np.int32),
                                              process=False))
        verts_mm, faces = _taubin(np.asarray(mesh.vertices, dtype=np.float32),
                                  np.asarray(mesh.faces, dtype=np.int64))
        verts_mm, faces = _trim_above(verts_mm, faces, z_cut)
        surfaces[name] = _drop_slivers(verts_mm, faces)

    thickness = _analytic_thickness(model, z, surfaces["endo"][0], centroid,
                                    scale, flip)

    # Absolute scale from the segmentation, spatial pattern from the model: the
    # decoder offset is only defined up to the contour normalisation, so it is
    # calibrated to the EDT reference of this phase exactly as the cohort
    # pipeline does (fig_results_nor_data.cal_factor, predict_sdf_meshes).
    reference = _reference_wall_thickness_from_segmentation(seg, tuple(zooms))
    factor = 1.0
    if reference is not None and float(np.nanmean(thickness)) > 0.1:
        factor = float(np.clip(reference / float(np.nanmean(thickness)), 0.3, 4.0))
        thickness = thickness * factor
    print(f"  segmentation reference={reference:.2f} mm  "
          f"calibration factor={factor:.3f}")
    xyz_mm = xyz_n * flip * scale + centroid
    if FLIP_LONG_AXIS_FOR_DISPLAY:
        xyz_mm[:, 2] *= -1.0
        
    return {
        "endo_v": surfaces["endo"][0], "endo_f": surfaces["endo"][1],
        "epi_v": surfaces["epi"][0], "epi_f": surfaces["epi"][1],
        "thickness": thickness,
        "contours_v": xyz_mm,
    }


def _trim_above(vertices: np.ndarray, faces: np.ndarray, z_cut: float):
    """Cut the surface at the most basal input ring, leaving the valve plane open.

    Above the last observed SAX ring the zero-level set is pure extrapolation and
    closes into a dome, so the reconstruction is trimmed there: the base ends in
    an open rim like the SSM reference mesh while the apex stays closed.
    Triangles straddling the plane are clipped rather than dropped, which keeps
    the rim straight instead of leaving a sawtooth edge.
    """
    import trimesh  # noqa: PLC0415

    verts = [np.asarray(v, dtype=np.float64) for v in vertices]
    cuts: dict[tuple[int, int], int] = {}

    def crossing(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        if key not in cuts:
            a, b = verts[key[0]], verts[key[1]]
            t = (z_cut - a[2]) / (b[2] - a[2])
            cuts[key] = len(verts)
            verts.append(a + t * (b - a))
        return cuts[key]

    kept: list[list[int]] = []
    above = vertices[:, 2] > z_cut
    for tri in faces:
        flags = above[tri]
        n_above = int(flags.sum())
        if n_above == 0:
            kept.append(list(tri))
        elif n_above == 1:
            # Rotate so the single vertex above the plane comes last.
            r = int(np.argmax(flags))
            a, b, c = tri[(r + 1) % 3], tri[(r + 2) % 3], tri[r]
            p, q = crossing(b, c), crossing(c, a)
            kept.append([a, b, p])
            kept.append([a, p, q])
        elif n_above == 2:
            # Rotate so the single vertex below the plane comes first.
            r = int(np.argmin(flags))
            a, b, c = tri[r], tri[(r + 1) % 3], tri[(r + 2) % 3]
            kept.append([a, crossing(a, b), crossing(c, a)])

    if not kept:
        return vertices, faces
    mesh = trimesh.Trimesh(np.asarray(verts), np.asarray(kept, dtype=np.int64),
                           process=False)
    mesh.remove_unreferenced_vertices()
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda c: len(c.faces))
    return (np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int64))


def _drop_slivers(vertices: np.ndarray, faces: np.ndarray, ratio: float = 4.0):
    """Remove the stretched triangles that marching cubes leaves as apical spikes."""
    import trimesh  # noqa: PLC0415

    triangles = vertices[faces]
    longest = np.linalg.norm(triangles - triangles[:, [1, 2, 0]], axis=2).max(axis=1)
    keep = longest <= ratio * float(np.median(longest))
    if keep.all():
        return vertices, faces
    mesh = trimesh.Trimesh(vertices, faces[keep], process=False)
    mesh.remove_unreferenced_vertices()
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda c: len(c.faces))
    return (np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int64))


def _taubin(vertices: np.ndarray, faces: np.ndarray, n_iter: int = 600):
    """Taubin-smooth the marching-cubes surface.

    Attenuates the inter-slice corrugation left by reconstructing from rings
    7.3 mm apart and the sub-voxel staircase of marching cubes; Taubin smoothing
    is volume preserving, so the wall thickness read from the field at the
    resulting vertices is unbiased. Quadric decimation was tried first and is
    deliberately not used: it collapses the sub-voxel slivers but introduces
    spikes and detached fragments.
    """
    import pyvista as pv  # noqa: PLC0415

    pv.OFF_SCREEN = True
    pv_faces = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    mesh = pv.PolyData(vertices, pv_faces).smooth_taubin(
        n_iter=n_iter, pass_band=0.003, normalize_coordinates=True)
    smoothed = np.asarray(mesh.points, dtype=np.float32)
    if len(smoothed) != len(vertices):
        return vertices, faces
    return smoothed, faces


def _analytic_thickness(model, z, vertices_mm, centroid, scale, flip) -> np.ndarray:
    """Decoder wall offset at the given millimetre vertices, returned in mm."""
    import torch  # noqa: PLC0415
    from core.sdf_model import DEVICE  # noqa: PLC0415
    from fig_ed_es_meshes import FLIP_LONG_AXIS_FOR_DISPLAY  # noqa: PLC0415

    query = vertices_mm.astype(np.float32).copy()
    if FLIP_LONG_AXIS_FOR_DISPLAY:
        query[:, 2] *= -1.0
    query = ((query - centroid) / scale) * flip
    values = np.empty(len(query), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(query), 131072):
            chunk = torch.from_numpy(query[start:start + 131072]).unsqueeze(0)
            _fe, _fp, delta = model.decode(z, chunk.to(DEVICE))
            values[start:start + 131072] = delta[0].float().cpu().numpy()
    return values * float(scale)


def _centred(vertices: np.ndarray) -> np.ndarray:
    """Centre the surface in-plane so the AHA circumferential angle is defined."""
    centred = vertices - vertices.mean(axis=0)
    return centred.astype(np.float32)


def _shade(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face light factor from averaged vertex normals.

    Lighting each triangle by its own normal, as the methodology helper does,
    turns the sub-millimetre roughness of a marching-cubes surface into visible
    terracing; averaging the normals at the shared vertices removes that shading
    noise without moving any geometry. The absolute value makes the lighting
    two-sided, so the interior seen through the open base is not black.
    """
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0],
                            triangles[:, 2] - triangles[:, 0])
    face_normals /= np.maximum(
        np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-8)
    vertex_normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], face_normals)
    vertex_normals /= np.maximum(
        np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-8)
    shading_normals = vertex_normals[faces].mean(axis=1)
    shading_normals /= np.maximum(
        np.linalg.norm(shading_normals, axis=1, keepdims=True), 1e-8)

    light_direction = np.asarray([-0.25, -0.50, 0.83])
    light_direction /= np.linalg.norm(light_direction)
    return 0.50 + 0.50 * np.abs(shading_normals @ light_direction)


def _draw_surface(ax, layers, value_norm=None, black_faces=None):
    """Draw one or more surfaces as a single depth-sorted collection.

    Merging the layers into one ``Poly3DCollection`` is what makes a translucent
    epicardium composite correctly over the endocardium: matplotlib sorts faces
    by depth only within a collection.
    Each layer is ``(vertices, faces, colour, alpha)`` where ``colour`` is either
    a per-vertex value array or a colour string.
    """
    triangles, facecolors, drawn = [], [], []
    for vertices, faces, colour, alpha in layers:
        light = _shade(vertices, faces)
        if isinstance(colour, str):
            rgba = np.tile(np.asarray(colors.to_rgba(colour)), (len(faces), 1))
        else:
            rgba = plt.get_cmap(CMAP)(value_norm(colour[faces].mean(axis=1)))
        rgba[:, :3] = np.clip(rgba[:, :3] * (0.68 + 0.32 * light[:, None]), 0.0, 1.0)
        rgba[:, 3] = alpha
        # Painted into the same collection so the lines depth-sort with the mesh.
        if black_faces is not None and not isinstance(colour, str):
            rgba[black_faces] = (0.0, 0.0, 0.0, alpha)
        triangles.append(vertices[faces])
        facecolors.append(rgba)
        drawn.append(vertices)

    ax.add_collection3d(Poly3DCollection(
        np.concatenate(triangles), facecolors=np.concatenate(facecolors),
        edgecolors="none", linewidths=0.0))
    set_mesh_axes(ax, np.concatenate(drawn))


def aha_segment_ids(vertices: np.ndarray) -> np.ndarray:
    """Assign each vertex to an AHA-17 segment (same rule as aha_segment_values)."""
    z_norm = (vertices[:, 2] - vertices[:, 2].min()) / max(np.ptp(vertices[:, 2]), 1e-8)
    angles = (np.arctan2(vertices[:, 1], vertices[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)
    ids = np.full(len(vertices), 17, dtype=int)
    segment_id = 1
    for lower, upper, segment_count in [(0.67, 1.01, 6), (0.34, 0.67, 6), (0.10, 0.34, 4)]:
        ring_mask = (z_norm >= lower) & (z_norm < upper)
        for segment_index in range(segment_count):
            angle_lower = 2.0 * np.pi * segment_index / segment_count
            angle_upper = 2.0 * np.pi * (segment_index + 1) / segment_count
            ids[ring_mask & (angles >= angle_lower) & (angles < angle_upper)] = segment_id
            segment_id += 1
    ids[z_norm < 0.10] = 17
    return ids


def make_thickness_3d(cases, norm) -> Path:
    """ED and ES reconstructions coloured by the calibrated analytic offset."""
    fig = plt.figure(figsize=(6.6, 5.8), facecolor="white")
    for row, (label, case) in enumerate(cases):
        for col, (elev, azim, view) in enumerate(VIEWS):
            ax = fig.add_axes([0.02 + 0.46 * col, 0.54 - 0.45 * row, 0.44, 0.44],
                              projection="3d")
            _draw_surface(ax, [(_centred(case["endo_v"]), case["endo_f"],
                                case["thickness"], 0.98)], value_norm=norm)
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect((1.0, 1.0, 1.0), zoom=1.18)
            ax.set_title(f"({'ab'[row]}{col + 1}) {label} — {view} view",
                         fontsize=8.0, style="italic", color="#333333",
                         pad=0, y=1.0)

    colorbar_axis = fig.add_axes([0.30, 0.07, 0.40, 0.024])
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    colorbar = fig.colorbar(mappable, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Calibrated analytic offset (mm)", fontsize=6.6, labelpad=1)
    colorbar.ax.tick_params(labelsize=5.8, length=2)

    output = OUT_DIR / "results_recon_thickness_3d.png"
    save_rgb_figure(fig, output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output


def _smooth_over_neighbours(values: np.ndarray, faces: np.ndarray,
                            iterations: int = 25) -> np.ndarray:
    """Average each vertex with its mesh neighbours.

    Per-vertex streamline thickness is noisy enough to speckle a continuous
    colour map. This is applied to the drawn field only; the reported segment
    statistics stay on the raw values.
    """
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    degree = np.bincount(edges[:, 0], minlength=len(values)).astype(np.float64)
    smoothed = np.asarray(values, np.float64).copy()
    for _ in range(iterations):
        total = np.bincount(edges[:, 0], weights=smoothed[edges[:, 1]],
                            minlength=len(smoothed))
        smoothed = np.where(degree > 0, total / np.maximum(degree, 1.0), smoothed)
    return smoothed


def _boundary_faces(faces: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Faces spanning two AHA segments, painted black to separate the regions."""
    labels = ids[faces]
    return (labels[:, 0] != labels[:, 1]) | (labels[:, 1] != labels[:, 2])


def _laplace_thickness(case) -> np.ndarray:
    """Per-vertex Laplace-field wall thickness between the two reconstructed surfaces.

    This is the same estimator the cohort pipeline reports, applied here to the
    endocardial and epicardial surfaces of this case, so the figure shows a
    measured transmural thickness rather than the decoder's wall offset.
    """
    sys.path.insert(0, str(ROOT / "scripts" / "eval_demo"))
    import trimesh  # noqa: PLC0415
    import thickness as tk  # noqa: PLC0415
    from geometry import enforce_nesting, outward_normals  # noqa: PLC0415

    endo = trimesh.Trimesh(np.asarray(case["endo_v"], np.float64),
                           np.asarray(case["endo_f"], np.int64), process=False)
    epi = trimesh.Trimesh(np.asarray(case["epi_v"], np.float64),
                          np.asarray(case["epi_f"], np.int64), process=False)
    endo, _ = enforce_nesting(endo, epi)
    vertices = np.asarray(endo.vertices, np.float64)
    normals = outward_normals(endo, np.asarray(epi.vertices, np.float64))
    context = tk.build_volume_context(endo, epi, LAPLACE_PITCH_MM)
    potential, _ = tk.solve_laplace(context)
    return np.asarray(
        tk.method_laplace_streamline(context, vertices, normals, potential).values,
        dtype=np.float64)




def load_cohort_case(patient: str = COHORT_PATIENT, phase: str = "ED") -> dict:
    """Endo/epi surfaces of one cohort case, read from the cached watertight PLYs."""
    import trimesh  # noqa: PLC0415

    surfaces = {
        name: trimesh.load(COHORT_CACHE / f"{patient}_{phase}_model_{name}.ply",
                           process=False)
        for name in ("endo", "epi")
    }
    return {
        "endo_v": np.asarray(surfaces["endo"].vertices, np.float64),
        "endo_f": np.asarray(surfaces["endo"].faces, np.int64),
        "epi_v": np.asarray(surfaces["epi"].vertices, np.float64),
        "epi_f": np.asarray(surfaces["epi"].faces, np.int64),
    }


def cohort_frame(case: dict, patient: str = COHORT_PATIENT,
                 phase: str = "ED") -> tuple[np.ndarray, np.ndarray]:
    """AHA ids and the valid long-axis band, as ``run_cohort`` computes them.

    Base/apex ordering and the septal reference come from the label mask, and the
    band drops the basal and apical rims where the cut planes thin the wall, so
    the segment means reproduce the stored cohort table.
    """
    sys.path.insert(0, str(ROOT / "scripts" / "eval_demo"))
    import trimesh  # noqa: PLC0415
    from geometry import (  # noqa: PLC0415
        assign_aha17, load_segmentation, long_axis_frame, read_info_cfg,
    )
    from run_cohort import VALID_LONG_AXIS_BAND, find_frame  # noqa: PLC0415

    patient_dir = DATA_ROOT / patient
    info = read_info_cfg(patient_dir / "Info.cfg")
    seg = load_segmentation(find_frame(patient_dir, patient, int(info[phase])))
    endo = trimesh.Trimesh(case["endo_v"], case["endo_f"], process=False)
    frame = long_axis_frame(endo, seg)
    vertices = np.asarray(case["endo_v"], np.float64)
    along = np.clip((vertices[:, 2] - frame["base_z"]) /
                    (frame["apex_z"] - frame["base_z"]), 0.0, 1.0)
    band = ((along >= VALID_LONG_AXIS_BAND[0]) & (along <= VALID_LONG_AXIS_BAND[1]))
    return assign_aha17(vertices, frame), band


def _autocrop(image: np.ndarray, pad: int = 8) -> np.ndarray:
    """Trim the white margin left by the offscreen renderer."""
    mask = (image[:, :, :3] < 248).any(axis=2)
    if not mask.any():
        return image
    rows, cols = np.where(mask)
    return image[max(0, rows.min() - pad):rows.max() + pad,
                 max(0, cols.min() - pad):cols.max() + pad]


def _open_basal_cap(surface, flatness: float = 0.80, top_fraction: float = 0.12):
    """Drop the flat lid the watertight repair leaves at the valve plane."""
    oriented = surface.compute_normals(cell_normals=True, point_normals=False,
                                       auto_orient_normals=True)
    normals = np.asarray(oriented.cell_normals)
    centres = np.asarray(oriented.cell_centers().points)
    height = float(np.ptp(np.asarray(surface.points)[:, 2]))
    ceiling = float(np.asarray(surface.points)[:, 2].max()) - top_fraction * height
    lid = (np.abs(normals[:, 2]) > flatness) & (centres[:, 2] > ceiling)
    if not lid.any():
        return surface
    return oriented.extract_cells(np.flatnonzero(~lid)).extract_surface()


def _render_segmented_surface(vertices: np.ndarray, faces: np.ndarray,
                              values: np.ndarray, ids: np.ndarray,
                              norm: colors.Normalize) -> np.ndarray:
    """Render the segmented surface with VTK.

    Matplotlib's 3D axes have no depth buffer, so the mesh reads flat and the
    segment boundaries bleed through from the far side. VTK z-buffers both, and
    ambient occlusion gives the wall its shape.
    """
    import pyvista as pv  # noqa: PLC0415

    pv.OFF_SCREEN = True
    cells = np.hstack([np.full((len(faces), 1), 3, np.int64),
                       np.asarray(faces, np.int64)]).ravel()
    surface = pv.PolyData(np.asarray(vertices, np.float32), cells)
    surface["thickness"] = np.asarray(values, np.float32)
    surface["segment"] = np.asarray(ids, np.int32)
    # Volume-preserving smoothing: takes the marching-cubes staircase off both the
    # surface and the segment boundaries that follow its triangle edges.
    smoothed = surface.smooth_taubin(n_iter=40, pass_band=0.05,
                                     normalize_coordinates=True)
    if smoothed.n_points == surface.n_points:
        surface = smoothed
    surface = _open_basal_cap(surface)
    points = np.asarray(surface.points, np.float32)
    segment = np.asarray(surface["segment"], np.int64)
    triangles = surface.faces.reshape(-1, 4)[:, 1:]

    plotter = pv.Plotter(off_screen=True, window_size=(1500, 1600))
    plotter.set_background("white")
    plotter.add_mesh(surface, scalars="thickness", cmap=CMAP,
                     clim=(norm.vmin, norm.vmax), smooth_shading=True,
                     specular=0.25, specular_power=16, ambient=0.30, diffuse=0.90,
                     show_scalar_bar=False)

    boundary = {(min(int(a), int(b)), max(int(a), int(b)))
                for triangle in triangles
                for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]),
                             (triangle[2], triangle[0]))
                if segment[a] != segment[b]}
    if boundary:
        edges = np.asarray(sorted(boundary), np.int64)
        lines = np.hstack([np.full((len(edges), 1), 2, np.int64), edges]).ravel()
        plotter.add_mesh(pv.PolyData(points, lines=lines),
                         color="black", line_width=2, render_lines_as_tubes=True)

    plotter.view_vector((1.1, -1.9, 0.55), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(0.92)
    for enable in (lambda: plotter.enable_ssao(radius=6.0),
                   lambda: plotter.enable_anti_aliasing("ssaa")):
        try:
            enable()
        except Exception:                                    # noqa: BLE001
            pass
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return _autocrop(np.asarray(image))


def make_aha17(case, ids: np.ndarray | None = None,
               band: np.ndarray | None = None) -> tuple[Path, dict[int, float]]:
    """Reconstructed surface cut into AHA-17 segments beside its bullseye."""
    vertices, faces = case["endo_v"], case["endo_f"]
    thickness = _laplace_thickness(case)
    if band is not None:
        thickness = np.where(band, thickness, np.nan)
    centred = _centred(vertices)
    if ids is None:
        ids = aha_segment_ids(centred)
    # The cached meshes may hold the apex at either end of z; draw it downwards.
    if centred[ids == 17, 2].mean() > centred[np.isin(ids, range(1, 7)), 2].mean():
        centred = centred.copy()
        centred[:, 2] *= -1.0
    finite = np.isfinite(thickness)
    segment_values = {
        sid: float(thickness[(ids == sid) & finite].mean())
        if np.any((ids == sid) & finite) else float(thickness[finite].mean())
        for sid in range(1, 18)
    }
    # Panel (a) shows the continuous per-vertex field, so thickness fades across a
    # border instead of stepping at it; the black lines only delimit the regions.
    # Rim vertices outside the valid band fall back to their segment mean.
    fallback = np.array([segment_values[int(sid)] for sid in ids], dtype=np.float32)
    per_vertex = _smooth_over_neighbours(np.where(finite, thickness, fallback),
                                         np.asarray(faces, np.int64)).astype(np.float32)
    # Range from the field itself, clipped to the 5th-95th percentile so a few
    # extreme vertices do not flatten the gradient everywhere else.
    norm = colors.Normalize(
        vmin=float(np.floor(np.percentile(thickness[finite], 5))),
        vmax=float(np.ceil(np.percentile(thickness[finite], 95))))

    fig = plt.figure(figsize=(7.2, 3.9), facecolor="white")
    ax0 = fig.add_axes([0.045, 0.20, 0.31, 0.65])
    ax0.imshow(_render_segmented_surface(centred, faces, per_vertex, ids, norm))
    ax0.set_axis_off()

    ax1 = fig.add_axes([0.44, 0.10, 0.44, 0.86])
    draw_aha17(ax1, segment_values, norm)
    for x_pos, text in [(0.20, "(a) Reconstructed surface, local wall thickness"),
                        (0.66, "(b) Unrolled AHA-17 bullseye")]:
        fig.text(x_pos, 0.955, text, ha="center", va="center", fontsize=8.0,
                 style="italic", color="#333333")
    for angle_deg, name in [
        (90.0, "Anterior"), (150.0, "Antero-\nseptal"), (210.0, "Infero-\nseptal"),
        (270.0, "Inferior"), (330.0, "Infero-\nlateral"), (30.0, "Antero-\nlateral"),
    ]:
        angle = np.deg2rad(angle_deg)
        ax1.text(1.22 * np.cos(angle), 1.22 * np.sin(angle), name,
                 ha="center", va="center", fontsize=6.2, color="#243447")
    ax1.set_xlim(-1.45, 1.45)
    ax1.set_ylim(-1.35, 1.35)

    fig.add_artist(patches.FancyArrowPatch(
        (0.395, 0.52), (0.455, 0.52), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=13.0, linewidth=1.1, color="#555555"))

    colorbar_axis = fig.add_axes([0.90, 0.24, 0.022, 0.56])
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    colorbar = fig.colorbar(mappable, cax=colorbar_axis, orientation="vertical")
    colorbar.set_label("Mean wall thickness, Laplace field (mm)", fontsize=6.6, labelpad=2)
    colorbar.ax.tick_params(labelsize=5.8, length=2)

    output = OUT_DIR / "results_recon_aha17.png"
    save_rgb_figure(fig, output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output, segment_values


def make_ed_es_meshes(cases) -> Path:
    """Endocardial and epicardial surfaces at both phases, same style as above."""
    fig = plt.figure(figsize=(6.8, 3.6), facecolor="white")
    for col, (label, case) in enumerate(cases):
        centre = case["epi_v"].mean(axis=0)
        ax = fig.add_axes([0.01 + 0.49 * col, 0.06, 0.47, 0.88], projection="3d")
        _draw_surface(ax, [
            (case["epi_v"] - centre, case["epi_f"], COL_EPI, 0.30),
            (case["endo_v"] - centre, case["endo_f"], COL_ENDO, 1.00),
        ])
        ax.view_init(elev=18.0, azim=-62.0)
        ax.set_box_aspect((1.0, 1.0, 1.0), zoom=1.15)
        ax.set_title(f"({'ab'[col]}) {label}", fontsize=8.0, style="italic",
                     color="#333333", pad=0, y=1.0)

        contours_c = case["contours_v"] - centre
        ax.scatter(contours_c[:, 0], contours_c[:, 1], contours_c[:, 2],
                   color="#333333", s=0.8, alpha=0.3, zorder=10)

    handles = [
        patches.Patch(facecolor=COL_ENDO, edgecolor="none", label="Endocardium"),
        patches.Patch(facecolor=COL_EPI, edgecolor="none", label="Epicardium"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333",
                   markersize=3, label="Input SAX slices"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=3, frameon=False, fontsize=7.0)

    output = OUT_DIR / "results_recon_ed_es.png"
    save_rgb_figure(fig, output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    if "--cohort" in sys.argv[1:]:
        case = load_cohort_case()
        ids, band = cohort_frame(case)
        output, segment_values = make_aha17(case, ids=ids, band=band)
        for sid in range(1, 18):
            print(f"  AHA {sid:2d}: {segment_values[sid]:.2f} mm")
        print("wrote", output.relative_to(ROOT), f"from {COHORT_PATIENT}")
        return
    ed, es = load_cases(refresh="--refresh" in sys.argv[1:])
    for label, case in (("ED", ed), ("ES", es)):
        thickness = case["thickness"]
        print(f"  {label}: endo {len(case['endo_v'])} verts, "
              f"epi {len(case['epi_v'])} verts  "
              f"thickness mean={thickness.mean():.2f} mm "
              f"p5={np.percentile(thickness, 5):.2f} "
              f"p95={np.percentile(thickness, 95):.2f}")

    pooled = np.concatenate([ed["thickness"], es["thickness"]])
    norm = colors.Normalize(vmin=float(np.floor(np.percentile(pooled, 2))),
                            vmax=float(np.ceil(np.percentile(pooled, 98))))
    print(f"Colour scale: {norm.vmin:.0f}-{norm.vmax:.0f} mm")

    cases = [("End-diastole (ED)", ed), ("End-systole (ES)", es)]
    outputs = [make_ed_es_meshes(cases), make_thickness_3d(cases, norm)]
    aha_output, segment_values = make_aha17(ed)
    outputs.append(aha_output)
    for sid in range(1, 18):
        print(f"  AHA {sid:2d}: {segment_values[sid]:.2f} mm")
    for output in outputs:
        print("wrote", output.relative_to(ROOT))


if __name__ == "__main__":
    main()
