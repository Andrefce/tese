"""Methodology overview figure: two-stage training, the model, per-phase outputs.

Panels:
  (a) Stage 1, pre-training: one synthetic ED shape sampled from the UK Digital
      Heart SSM with its synthetic epicardium, and the 10 SAX rings sliced from
      it.
  (b) Stage 2, fine-tuning: an ACDC SAX slice with its expert labels and the
      RBF surfaces fitted to the label contours at ED and ES.
  (c) The model shared by both stages, ending in a long-axis section of the
      coupled surfaces it reconstructs for the ED case.
  (d)-(g) Inference on ACDC patient002 with the v2 checkpoint, at ED and ES:
      observed SAX rings, reconstructed surfaces and Laplace-field wall
      thickness on the endocardium; (g) merges both phases into the systolic
      thickening (ES minus ED), shown on the ES endocardium and as AHA-17
      segment differences.

The reconstruction, thickness and AHA steps call the cohort pipeline of
``scripts/eval_demo`` with the v2 loader, as ``run_cohort_v2.py`` does, so the
values follow the protocol of the Results chapter. They are cached in
``CASE_CACHE`` and the renders in ``RENDER_CACHE``; pass ``--refresh`` to
recompute everything or ``--rerender`` to redo only the renders. Display
conventions match the Results renders: one display-only Taubin pass per
surface, model surfaces trimmed at the most basal input ring, and per-vertex
thickness smoothed over mesh neighbours for display only (segment means use
the raw values). The per-vertex thickening pairs each ES vertex with the ED
vertex at the nearest normalised long-axis position and circumferential angle,
as in ``fig_intro_overview.py``.

Outputs: images/fig_methodology_pipeline.pdf and images/fig_methodology_pipeline.png

Run:
    C:/Python313/python.exe scripts/fig_methodology_pipeline.py [--refresh | --rerender]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
import trimesh  # noqa: E402
from matplotlib import colors, patches, patheffects, ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

THESIS = Path(__file__).resolve().parents[1]
SCRIPTS = THESIS / "scripts"
EVAL_DEMO = SCRIPTS / "eval_demo"
V2_DIR = THESIS / "test-new-model"
OUTPUTS = EVAL_DEMO / "outputs"
MODEL_PATH = V2_DIR / "runs" / "u1u2_e50" / "cardiosdf_v2_best.pt"
PATIENT = "patient002"
PATIENT_DIR = THESIS / "notebooks" / PATIENT
CASE_CACHE = OUTPUTS / f"methodology_pipeline_v2_{PATIENT}.npz"
SSM_CACHE = OUTPUTS / "methodology_pipeline_ssm_sample.npz"
RENDER_CACHE = OUTPUTS / "methodology_pipeline_renders.npz"
RBF_CACHE = OUTPUTS / "recon_ed_es_baselines.npz"
OUT_STEM = THESIS / "images" / "fig_methodology_pipeline"

PHASES = ("ED", "ES")
GRID_RES = 96
PITCH_MM = 1.0
POINTS_PER_RING = 60
SSM_ACTIVE_MODES = 32
SSM_SEED = 11
N_SAX_LEVELS = 10
TAUBIN_ITERATIONS = 600
TAUBIN_PASS_BAND = 0.003
THICKNESS_SMOOTHING = 40
# Depth of the display cut below the lowest basal rim of a surface pair, which
# opens the flat lid that the watertight repair or the RBF trim leaves there.
BASE_CUT_MM = 3.0

# Figure geometry in inches; the thesis text block is 16 cm wide.
FIG_W, FIG_H = 6.30, 7.30
RENDER_SIZE = (900, 1100)

INK = "#1F2933"
GREY = "#5F6B7A"
RULE = "#C9D1DB"
LANE = "#F2F4F7"
NAVY = "#243447"
BOX = "#E6ECF3"
ENDO = "#0072B2"
EPI = "#D55E00"
ENDO_SURFACE = "#2B83C6"
EPI_SURFACE = "#EFA273"
PLANE = "#B9C2CC"
THICKNESS_CMAP = "turbo"
DELTA_CMAP = "RdBu_r"

VIEW_DIR = np.array([1.0, -1.75, 0.55])
VIEW_DIR /= np.linalg.norm(VIEW_DIR)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.0,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.cal": "cmsy10",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
})
pv.OFF_SCREEN = True


# ──────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────
def prepare_case(refresh: bool = False) -> dict:
    """v2 reconstruction, Laplace thickness and AHA-17 means of PATIENT, ED and ES."""
    if CASE_CACHE.exists() and not refresh:
        with np.load(CASE_CACHE) as data:
            print(f"using cached case: {CASE_CACHE.relative_to(THESIS)}")
            return {key: data[key] for key in data.files}

    sys.path[:0] = [str(V2_DIR), str(EVAL_DEMO)]
    warnings.filterwarnings("ignore")
    import thickness as tk  # noqa: PLC0415
    from cardiosdf2.model import load_v2  # noqa: PLC0415
    from cardiosdf_model import DEVICE  # noqa: PLC0415
    from geometry import (  # noqa: PLC0415
        assign_aha17, build_model_geometry, enforce_nesting, extract_contours,
        load_segmentation, long_axis_frame, outward_normals, read_info_cfg,
    )
    from run_cohort import VALID_LONG_AXIS_BAND, find_frame, segment_means  # noqa: PLC0415

    net, cfg, meta = load_v2(MODEL_PATH, DEVICE)
    print(f"v2 checkpoint {MODEL_PATH.relative_to(THESIS)} (epoch {meta['epoch']})")
    info = read_info_cfg(PATIENT_DIR / "Info.cfg")
    out: dict[str, np.ndarray] = {"checkpoint_epoch": np.int64(meta["epoch"])}

    for phase, phase_val in (("ED", 0.0), ("ES", 1.0)):
        print(f"  {phase}: reconstructing ...")
        seg = load_segmentation(find_frame(PATIENT_DIR, PATIENT, int(info[phase])))
        contours = extract_contours(seg)
        geom = build_model_geometry(net, cfg, contours, grid_res=GRID_RES,
                                    phase_val=phase_val)
        epi = geom["epi"]
        endo, _ = enforce_nesting(geom["endo"], epi)
        verts = np.asarray(endo.vertices, np.float64)
        normals = outward_normals(endo, np.asarray(epi.vertices))

        frame = long_axis_frame(endo, seg)
        t_axis = np.clip((verts[:, 2] - frame["base_z"]) /
                         (frame["apex_z"] - frame["base_z"]), 0.0, 1.0)
        band = ((t_axis >= VALID_LONG_AXIS_BAND[0]) &
                (t_axis <= VALID_LONG_AXIS_BAND[1]))

        print(f"  {phase}: Laplace thickness ...")
        context = tk.build_volume_context(endo, epi, PITCH_MM)
        potential, _ = tk.solve_laplace(context)
        laplace = tk.method_laplace_streamline(context, verts, normals, potential).values
        laplace = np.where(band, laplace, np.nan).astype(np.float64)
        aha = assign_aha17(verts, frame)

        out.update({
            f"{phase}_endo_v": verts.astype(np.float32),
            f"{phase}_endo_f": np.asarray(endo.faces, np.int64),
            f"{phase}_epi_v": np.asarray(epi.vertices, np.float32),
            f"{phase}_epi_f": np.asarray(epi.faces, np.int64),
            f"{phase}_laplace": laplace,
            f"{phase}_aha": aha.astype(np.int16),
            f"{phase}_segment_means": np.asarray(segment_means(laplace, aha)),
            f"{phase}_contours_mm": contours["xyz_mm"].astype(np.float32),
            f"{phase}_tissue": contours["tissue"].astype(np.float32),
            f"{phase}_base_z": np.float64(frame["base_z"]),
            f"{phase}_apex_z": np.float64(frame["apex_z"]),
        })
        print(f"  {phase}: endo {len(verts)} v, epi {len(epi.vertices)} v, "
              f"Laplace mean {np.nanmean(laplace):.2f} mm "
              f"(valid {np.isfinite(laplace).mean():.0%})")

    CASE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CASE_CACHE, **out)
    print(f"cached case to {CASE_CACHE.relative_to(THESIS)}")
    return out


def prepare_ssm_sample(refresh: bool = False) -> dict:
    """One accepted SSM shape with the synthetic epicardium of the SSM figure."""
    if SSM_CACHE.exists() and not refresh:
        with np.load(SSM_CACHE) as data:
            return {key: data[key] for key in data.files}

    from scipy.stats import chi2  # noqa: PLC0415

    sys.path.insert(0, str(SCRIPTS))
    from fig_ssm_explanation import load_ssm, make_epicardium  # noqa: PLC0415

    mean_pts, pc, sigmas, faces = load_ssm()
    rng = np.random.default_rng(SSM_SEED)
    threshold = chi2.ppf(0.99, SSM_ACTIVE_MODES)
    while True:
        b = np.zeros(pc.shape[1])
        b[:SSM_ACTIVE_MODES] = rng.normal(0.0, sigmas[:SSM_ACTIVE_MODES])
        ratio = b[:SSM_ACTIVE_MODES] / sigmas[:SSM_ACTIVE_MODES]
        if np.all(np.abs(ratio) <= 3.0) and np.sum(ratio ** 2) <= threshold:
            break
    endo = (mean_pts.flatten() + pc @ b).reshape(-1, 3)
    epi = make_epicardium(endo, faces)
    sample = {"endo_v": endo.astype(np.float32), "epi_v": epi.astype(np.float32),
              "faces": np.asarray(faces, np.int64)}
    np.savez_compressed(SSM_CACHE, **sample)
    return sample


def load_rbf_targets() -> dict:
    with np.load(RBF_CACHE) as data:
        return {(phase, wall): (data[f"rbf_{phase}_{wall}_v"], data[f"rbf_{phase}_{wall}_f"])
                for phase in PHASES for wall in ("endo", "epi")}


def load_sax_slice() -> tuple[np.ndarray, np.ndarray, float]:
    """Mid-ventricular ED image and labels of PATIENT, cropped around the LV."""
    sys.path.insert(0, str(EVAL_DEMO))
    from geometry import resolve_nifti  # noqa: PLC0415

    image_nii = nib.load(str(resolve_nifti(PATIENT_DIR / f"{PATIENT}_frame01.nii")))
    label_nii = nib.load(str(resolve_nifti(PATIENT_DIR / f"{PATIENT}_frame01_gt.nii")))
    image = np.asarray(image_nii.dataobj, np.float32)
    labels = np.rint(np.asarray(label_nii.dataobj)).astype(np.int16)
    lv_slices = np.flatnonzero((labels == 3).any(axis=(0, 1)))
    s = int(lv_slices[len(lv_slices) // 2 - 1])
    rows, cols = np.nonzero(np.isin(labels[:, :, s], (1, 2, 3)))
    margin = 16
    side = max(rows.max() - rows.min(), cols.max() - cols.min()) + 2 * margin
    r0 = max((rows.min() + rows.max()) // 2 - side // 2, 0)
    c0 = max((cols.min() + cols.max()) // 2 - side // 2, 0)
    crop = (slice(r0, r0 + side), slice(c0, c0 + side), s)
    spacing = float(image_nii.header.get_zooms()[0])
    return np.rot90(image[crop]), np.rot90(labels[crop]), spacing


# ──────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────
def apex_up(vertices: np.ndarray) -> bool:
    """True when the narrow (apical) end of an open or capped ventricle is at +z."""
    z = vertices[:, 2]
    span = np.ptp(z)
    centre = vertices[:, :2].mean(axis=0)
    radius = np.linalg.norm(vertices[:, :2] - centre, axis=1)
    top = radius[z > z.max() - 0.12 * span].mean()
    bottom = radius[z < z.min() + 0.12 * span].mean()
    return bool(top < bottom)


def display_transform(reference: np.ndarray, flip: bool | None = None):
    """Centre on ``reference`` and turn the apex down by a proper rotation."""
    centre = reference.mean(axis=0)
    flip = apex_up(reference) if flip is None else flip

    def apply(points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, np.float64) - centre
        return p * np.array([1.0, -1.0, -1.0]) if flip else p

    return apply


def polydata(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    cells = np.hstack([np.full((len(faces), 1), 3, np.int64),
                       np.asarray(faces, np.int64)]).ravel()
    return pv.PolyData(np.asarray(vertices, np.float32), cells)


def smoothed(mesh: pv.PolyData) -> pv.PolyData:
    """Display-only Taubin pass, as in the Results renders; keeps vertex order."""
    out = mesh.smooth_taubin(n_iter=TAUBIN_ITERATIONS, pass_band=TAUBIN_PASS_BAND,
                             normalize_coordinates=True)
    return out if out.n_points == mesh.n_points else mesh


def keep_below(mesh: pv.PolyData, z_cut: float) -> pv.PolyData:
    return mesh.clip(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, z_cut),
                     invert=True).extract_surface().triangulate()


def open_base(endo: pv.PolyData, epi: pv.PolyData, z_limit: float = np.inf):
    """Cut a surface pair just below its lowest basal rim (and below ``z_limit``)."""
    z_cut = min(z_limit, endo.bounds[5], epi.bounds[5]) - BASE_CUT_MM
    return keep_below(endo, z_cut), keep_below(epi, z_cut)


def wedge_cut(mesh: pv.PolyData) -> pv.PolyData:
    """Remove the 90-degree wedge that faces the camera, exposing the cavity."""
    horizontal = np.array([VIEW_DIR[0], VIEW_DIR[1], 0.0])
    horizontal /= np.linalg.norm(horizontal)
    angle = np.deg2rad(45.0)
    rot = lambda a: np.array([[np.cos(a), -np.sin(a), 0.0],  # noqa: E731
                              [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    a, b = rot(angle) @ horizontal, rot(-angle) @ horizontal
    points = np.asarray(mesh.points, np.float64)
    work = mesh.copy()
    work["wedge"] = np.minimum(points @ a, points @ b).astype(np.float32)
    return work.clip_scalar(scalars="wedge", value=0.0, invert=True).extract_surface()


def ring_list(points: np.ndarray, tissue: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Split contour rows into closed rings; rows come in blocks of POINTS_PER_RING."""
    return [(points[s:s + POINTS_PER_RING], float(tissue[s]))
            for s in range(0, len(points), POINTS_PER_RING)
            if len(points[s:s + POINTS_PER_RING]) >= 3]


def ssm_rings(endo: np.ndarray, epi: np.ndarray, faces: np.ndarray):
    """Planar sections of the SSM pair at N_SAX_LEVELS evenly spaced levels."""
    meshes = ((trimesh.Trimesh(endo, faces, process=False), 0.0),
              (trimesh.Trimesh(epi, faces, process=False), 1.0))
    z_min, z_max = endo[:, 2].min(), endo[:, 2].max()
    levels = np.linspace(z_min + 0.07 * (z_max - z_min), z_max - 0.07 * (z_max - z_min),
                         N_SAX_LEVELS)
    rings = []
    for z in levels:
        for mesh, tissue in meshes:
            section = mesh.section(plane_origin=[0.0, 0.0, z], plane_normal=[0.0, 0.0, 1.0])
            if section is not None:
                rings.append((np.asarray(max(section.discrete, key=len)), tissue))
    return rings


def smooth_over_neighbours(values: np.ndarray, faces: np.ndarray, iterations: int):
    """Display-only neighbour averaging of a per-vertex field."""
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    degree = np.bincount(edges[:, 0], minlength=len(values)).astype(np.float64)
    out = np.asarray(values, np.float64).copy()
    for _ in range(iterations):
        total = np.bincount(edges[:, 0], weights=out[edges[:, 1]], minlength=len(out))
        out = np.where(degree > 0, total / np.maximum(degree, 1.0), out)
    return out


def display_thickness(case: dict, phase: str) -> np.ndarray:
    """Laplace field with rim vertices filled by their segment mean, then smoothed."""
    values = case[f"{phase}_laplace"]
    ids = case[f"{phase}_aha"].astype(int)
    means = case[f"{phase}_segment_means"]
    fallback = np.where(np.isfinite(means[ids - 1]), means[ids - 1], np.nanmean(values))
    filled = np.where(np.isfinite(values), values, fallback)
    return smooth_over_neighbours(filled, case[f"{phase}_endo_f"], THICKNESS_SMOOTHING)


def thickness_norm(case: dict) -> colors.Normalize:
    pooled = np.concatenate([case[f"{p}_laplace"] for p in PHASES])
    pooled = pooled[np.isfinite(pooled)]
    return colors.Normalize(vmin=float(np.floor(np.percentile(pooled, 3))),
                            vmax=float(np.ceil(np.percentile(pooled, 97))))


def wall_coordinates(case: dict, phase: str) -> np.ndarray:
    """Normalised long-axis position and circumferential angle of the endo vertices.

    The angle is taken about a per-level centre so that it follows the
    ventricular axis rather than the global centroid.
    """
    verts = np.asarray(case[f"{phase}_endo_v"], np.float64)
    base_z, apex_z = float(case[f"{phase}_base_z"]), float(case[f"{phase}_apex_z"])
    t = np.clip((verts[:, 2] - base_z) / (apex_z - base_z), 0.0, 1.0)
    n_bins = 20
    bins = np.clip((t * n_bins).astype(int), 0, n_bins - 1)
    centres = np.array([verts[bins == b, :2].mean(axis=0) if np.any(bins == b)
                        else (np.nan, np.nan) for b in range(n_bins)])
    valid = np.isfinite(centres[:, 0])
    levels = (np.arange(n_bins) + 0.5) / n_bins
    axis_x = np.interp(t, levels[valid], centres[valid, 0])
    axis_y = np.interp(t, levels[valid], centres[valid, 1])
    angle = np.arctan2(verts[:, 1] - axis_y, verts[:, 0] - axis_x)
    return np.column_stack([np.cos(angle), np.sin(angle), 3.0 * t])


def thickening_values(case: dict) -> np.ndarray:
    """ES minus ED display thickness at matching wall positions, on the ES endo vertices."""
    _, index = cKDTree(wall_coordinates(case, "ED")).query(wall_coordinates(case, "ES"))
    delta = display_thickness(case, "ES") - display_thickness(case, "ED")[index]
    return smooth_over_neighbours(delta, case["ES_endo_f"], 15)


def segment_thickening(case: dict) -> np.ndarray:
    return case["ES_segment_means"] - case["ED_segment_means"]


def thickening_norm(case: dict) -> colors.Normalize:
    """Symmetric scale about zero covering the vertex and segment thickening."""
    delta = thickening_values(case)
    limit = max(np.percentile(np.abs(delta), 97),
                np.nanmax(np.abs(segment_thickening(case))))
    limit = float(np.ceil(limit))
    return colors.Normalize(vmin=-limit, vmax=limit)


def display_case(case: dict) -> dict:
    """Model surfaces in a shared apex-down frame, smoothed and trimmed at the base."""
    flip = bool(case["ED_apex_z"] > case["ED_base_z"])
    to_display = display_transform(case["ED_epi_v"], flip=flip)
    shown: dict = {"to_display": to_display}
    for phase in PHASES:
        rings = [(to_display(loop), t) for loop, t in
                 ring_list(case[f"{phase}_contours_mm"], case[f"{phase}_tissue"])]
        z_base = max(loop[:, 2].mean() for loop, _ in rings)
        endo = smoothed(polydata(to_display(case[f"{phase}_endo_v"]), case[f"{phase}_endo_f"]))
        endo["thickness"] = display_thickness(case, phase).astype(np.float32)
        if phase == "ES":
            endo["thickening"] = thickening_values(case).astype(np.float32)
        epi = smoothed(polydata(to_display(case[f"{phase}_epi_v"]), case[f"{phase}_epi_f"]))
        endo, epi = open_base(endo, epi, z_base)
        shown[phase] = {"rings": rings, "endo": endo, "epi": epi}
    return shown


def section_curve(mesh: pv.PolyData) -> np.ndarray:
    """Longest polyline of the mesh cut by the plane y = 0, as (x, z) pairs."""
    cut = mesh.slice(normal=(0.0, 1.0, 0.0), origin=(0.0, 0.0, 0.0)).strip(join=True)
    lines, points, polylines, i = cut.lines, np.asarray(cut.points), [], 0
    while i < len(lines):
        n = int(lines[i])
        polylines.append(points[lines[i + 1:i + 1 + n]])
        i += n + 1
    curve = max(polylines, key=len)[:, [0, 2]]
    return curve if curve[0, 0] < curve[-1, 0] else curve[::-1]


# ──────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────
SURFACE_STYLE = dict(smooth_shading=True, specular=0.20, specular_power=20,
                     ambient=0.30, diffuse=0.80)


def render(actors, half_height: float) -> np.ndarray:
    """Off-screen parallel-projection render with a transparent background."""
    plotter = pv.Plotter(off_screen=True, window_size=RENDER_SIZE)
    plotter.set_background("white")
    for mesh, kwargs in actors:
        plotter.add_mesh(mesh, **kwargs)
    plotter.enable_parallel_projection()
    camera = plotter.camera
    camera.focal_point = (0.0, 0.0, 0.0)
    camera.position = tuple(VIEW_DIR * 600.0)
    camera.up = (0.0, 0.0, 1.0)
    camera.parallel_scale = half_height
    plotter.enable_anti_aliasing("ssaa")
    image = plotter.screenshot(transparent_background=True, return_img=True)
    plotter.close()
    return np.asarray(image)


def crop_together(images: list[np.ndarray], pad: int = 10) -> list[np.ndarray]:
    """Crop a set of renders to the union of their opaque pixels."""
    mask = np.zeros(images[0].shape[:2], bool)
    for image in images:
        mask |= image[:, :, 3] > 8
    rows, cols = np.nonzero(mask)
    r0, r1 = max(rows.min() - pad, 0), rows.max() + pad
    c0, c1 = max(cols.min() - pad, 0), cols.max() + pad
    return [image[r0:r1, c0:c1] for image in images]


def half_height_for(points: np.ndarray, margin: float = 1.05) -> float:
    """Parallel scale that fits ``points`` in the portrait render viewport."""
    right = np.cross(VIEW_DIR, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    screen_up = np.cross(right, VIEW_DIR)
    aspect = RENDER_SIZE[1] / RENDER_SIZE[0]
    return margin * max(np.ptp(points @ screen_up) / 2.0,
                        np.ptp(points @ right) / 2.0 * aspect)


def pair_actors(endo: pv.PolyData, epi: pv.PolyData):
    return [(endo, dict(color=ENDO_SURFACE, **SURFACE_STYLE)),
            (wedge_cut(epi), dict(color=EPI_SURFACE, **SURFACE_STYLE))]


def ring_actors(rings, tube_radius: float, plane_pad: float = 4.0):
    """Tubes along each ring plus a faint outline of its SAX plane."""
    actors = []
    for value, colour in ((0.0, ENDO), (1.0, EPI)):
        tubes = [pv.lines_from_points(np.vstack([loop, loop[:1]])).tube(
            radius=tube_radius, n_sides=10) for loop, t in rings if t == value]
        if tubes:
            actors.append((pv.merge(tubes), dict(color=colour, smooth_shading=True,
                                                 ambient=0.35)))
    every = np.vstack([loop for loop, _ in rings])
    (x0, y0), (x1, y1) = every[:, :2].min(0) - plane_pad, every[:, :2].max(0) + plane_pad
    outlines = []
    for z in np.unique(np.round([loop[:, 2].mean() for loop, _ in rings], 1)):
        corners = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], [x0, y0, z]])
        outlines.append(pv.lines_from_points(corners).tube(radius=tube_radius * 0.35))
    actors.append((pv.merge(outlines), dict(color=PLANE, ambient=0.6)))
    return actors


def make_renders(case: dict, shown: dict, ssm: dict, rbf: dict) -> dict[str, np.ndarray]:
    renders: dict[str, np.ndarray] = {}

    # (a) SSM sample and its rings, on one scale; the pair is cut at its top ring.
    to_ssm = display_transform(ssm["endo_v"])
    ssm_endo, ssm_epi = to_ssm(ssm["endo_v"]), to_ssm(ssm["epi_v"])
    scale = half_height_for(ssm_epi)
    rings = ssm_rings(ssm_endo, ssm_epi, ssm["faces"])
    top_ring = max(loop[:, 2].mean() for loop, _ in rings)
    renders["ssm_pair"], renders["ssm_rings"] = crop_together([
        render(pair_actors(*open_base(polydata(ssm_endo, ssm["faces"]),
                                      polydata(ssm_epi, ssm["faces"]),
                                      top_ring + BASE_CUT_MM)), scale),
        render(ring_actors(rings, 0.65), scale),
    ])

    # (b) RBF targets at ED and ES on one scale, with the flat basal caps removed.
    to_rbf = display_transform(rbf[("ED", "epi")][0])
    targets = {}
    for phase in PHASES:
        endo = smoothed(polydata(to_rbf(rbf[(phase, "endo")][0]), rbf[(phase, "endo")][1]))
        epi = smoothed(polydata(to_rbf(rbf[(phase, "epi")][0]), rbf[(phase, "epi")][1]))
        targets[phase] = open_base(endo, epi)
    scale = half_height_for(np.vstack([np.asarray(targets[p][1].points) for p in PHASES]))
    renders["rbf_ED"], renders["rbf_ES"] = crop_together(
        [render(pair_actors(*targets[p]), scale) for p in PHASES])

    # (c) model input: the ED contour points.
    points = np.vstack([loop for loop, _ in shown["ED"]["rings"]])
    tissue = np.concatenate([np.full(len(loop), t) for loop, t in shown["ED"]["rings"]])
    cloud = [(pv.PolyData(points[tissue == value].astype(np.float32)),
              dict(color=colour, point_size=46, render_points_as_spheres=True))
             for value, colour in ((0.0, ENDO), (1.0, EPI))]
    renders["model_input"] = crop_together([render(cloud, half_height_for(points))])[0]

    # (d)-(f) one scale for every row-3 render so ED and ES compare directly.
    scale = half_height_for(np.vstack([np.asarray(shown[p]["epi"].points) for p in PHASES]),
                            margin=1.02)
    norm = thickness_norm(case)
    images = []
    for phase in PHASES:
        images.append(render(ring_actors(shown[phase]["rings"], 0.60), scale))
    for phase in PHASES:
        images.append(render(pair_actors(shown[phase]["endo"], shown[phase]["epi"]), scale))
    for phase in PHASES:
        images.append(render([(shown[phase]["endo"], dict(
            scalars="thickness", cmap=THICKNESS_CMAP, clim=(norm.vmin, norm.vmax),
            show_scalar_bar=False, **SURFACE_STYLE))], scale))
    delta_norm = thickening_norm(case)
    images.append(render([(shown["ES"]["endo"], dict(
        scalars="thickening", cmap=DELTA_CMAP, clim=(delta_norm.vmin, delta_norm.vmax),
        show_scalar_bar=False, **SURFACE_STYLE))], scale))
    images = crop_together(images)
    for k, phase in enumerate(PHASES):
        renders[f"contours_{phase}"] = images[k]
        renders[f"surfaces_{phase}"] = images[2 + k]
        renders[f"thickness_{phase}"] = images[4 + k]
    renders["thickening"] = images[6]
    return renders


# ──────────────────────────────────────────────────────────────────────────
# 2D drawing helpers (all coordinates in inches from the top-left corner)
# ──────────────────────────────────────────────────────────────────────────
class Canvas:
    def __init__(self, fig: plt.Figure):
        self.fig = fig
        self.back = self._layer(-10)
        self.front = self._layer(10)

    def _layer(self, zorder: int) -> plt.Axes:
        ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=zorder)
        ax.set_xlim(0.0, FIG_W)
        ax.set_ylim(FIG_H, 0.0)
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        return ax

    def axes(self, x: float, y: float, w: float, h: float) -> plt.Axes:
        return self.fig.add_axes([x / FIG_W, 1.0 - (y + h) / FIG_H, w / FIG_W, h / FIG_H])

    def image(self, img: np.ndarray, x: float, y: float, w: float, h: float,
              anchor: str = "C") -> plt.Axes:
        ax = self.axes(x, y, w, h)
        ax.imshow(img, interpolation="lanczos")
        ax.set_anchor(anchor)
        ax.set_axis_off()
        return ax

    def box(self, x, y, w, h, *, face=LANE, edge="none", lw=0.0, radius=0.06,
            layer="back", zorder=0):
        target = self.back if layer == "back" else self.front
        target.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder))

    def pill(self, x, y, text, w, h=0.20, size=7.0):
        self.box(x, y, w, h, face=NAVY, radius=0.05, layer="front", zorder=2)
        self.front.text(x + 0.08, y + h / 2 + 0.004, text, ha="left", va="center",
                        color="white", fontsize=size, fontweight="bold", zorder=3)

    def text(self, x, y, s, **kwargs):
        kwargs.setdefault("color", INK)
        kwargs.setdefault("zorder", 4)
        return self.front.text(x, y, s, **kwargs)

    def arrow(self, start, end, *, lw=0.9, scale=7.5, style="-|>", colour=INK):
        self.front.annotate("", xy=end, xytext=start, zorder=5, arrowprops=dict(
            arrowstyle=style, color=colour, lw=lw, mutation_scale=scale,
            shrinkA=0, shrinkB=0))


def frame_axes(ax: plt.Axes):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(0.5)


def draw_sax_panel(ax: plt.Axes, image: np.ndarray, labels: np.ndarray, spacing: float):
    lo, hi = np.percentile(image[image > 0], [1.0, 99.5])
    ax.imshow(image, cmap="gray", vmin=lo, vmax=hi, interpolation="lanczos")
    for mask, colour in ((np.isin(labels, (2, 3)), EPI), (labels == 3, ENDO)):
        contour = ax.contour(mask.astype(float), levels=[0.5], colors=[colour],
                             linewidths=1.0)
        contour.set_path_effects([patheffects.withStroke(linewidth=1.8, foreground="white")])
    frame_axes(ax)


def section_data(shown: dict, phase: str = "ED") -> dict[str, np.ndarray]:
    """Long-axis cut (plane y = 0) of the reconstructed pair and the ring crossings."""
    crossings = []
    for loop, _ in shown[phase]["rings"]:
        y, nxt = loop[:, 1], np.roll(np.arange(len(loop)), -1)
        for i in np.flatnonzero(np.sign(y) != np.sign(y[nxt])):
            t = y[i] / (y[i] - y[nxt[i]])
            crossings.append(loop[i] + t * (loop[nxt[i]] - loop[i]))
    return {"section_endo": section_curve(shown[phase]["endo"]),
            "section_epi": section_curve(shown[phase]["epi"]),
            "section_crossings": np.asarray(crossings)[:, [0, 2]]}


def draw_coupled_section(ax: plt.Axes, renders: dict):
    """Long-axis cut of the reconstructed pair: cavity, wall and ring crossings."""
    endo, epi = renders["section_endo"], renders["section_epi"]
    crossings = renders["section_crossings"]
    ax.fill(np.r_[epi[:, 0], endo[::-1, 0]], np.r_[epi[:, 1], endo[::-1, 1]],
            color="#F8DCCB", lw=0, zorder=1)
    ax.fill(endo[:, 0], endo[:, 1], color="#D9EAF6", lw=0, zorder=1)
    ax.plot(endo[:, 0], endo[:, 1], color=ENDO, lw=1.2, zorder=3)
    ax.plot(epi[:, 0], epi[:, 1], color=EPI, lw=1.2, zorder=3)
    ax.scatter(crossings[:, 0], crossings[:, 1], s=6, color=INK, zorder=4, linewidths=0)
    pad = 3.0
    ax.set_xlim(min(epi[:, 0].min(), crossings[:, 0].min()) - pad,
                max(epi[:, 0].max(), crossings[:, 0].max()) + pad)
    ax.set_ylim(epi[:, 1].min() - pad,
                max(epi[:, 1].max(), endo[:, 1].max(), crossings[:, 1].max()) + pad)
    ax.set_aspect("equal")
    ax.set_anchor("C")
    ax.set_axis_off()


def draw_bullseye(ax: plt.Axes, values: np.ndarray, norm: colors.Normalize,
                  cmap_name: str = THICKNESS_CMAP):
    """AHA-17 bullseye in the standard layout: anterior at the top, septum left."""
    cmap = plt.get_cmap(cmap_name)
    rings = [(0.70, 1.00, 6, 1, 60.0), (0.42, 0.70, 6, 7, 60.0), (0.16, 0.42, 4, 13, 45.0)]
    for inner, outer, count, first, start in rings:
        step = 360.0 / count
        for k in range(count):
            theta = start + k * step
            ax.add_patch(patches.Wedge((0.0, 0.0), outer, theta, theta + step,
                                       width=outer - inner, edgecolor="white", lw=0.8,
                                       facecolor=cmap(norm(values[first - 1 + k]))))
    ax.add_patch(patches.Circle((0.0, 0.0), 0.16, facecolor=cmap(norm(values[16])),
                                edgecolor="white", lw=0.8))
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-1.06, 1.06)
    ax.set_aspect("equal")
    ax.set_axis_off()


# ──────────────────────────────────────────────────────────────────────────
# Composition
# ──────────────────────────────────────────────────────────────────────────
def compose(case: dict, renders: dict, sax: tuple) -> plt.Figure:
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    c = Canvas(fig)
    norm = thickness_norm(case)
    margin = 0.05
    note = dict(fontsize=6.0, color=GREY)
    caption = dict(fontsize=6.4, ha="center", va="center", color=INK)

    # ── Tier 1: the two training stages ──────────────────────────────────
    lane_y, lane_w, lane_h, lane_gap = margin, 3.0, 1.80, 0.20
    lane1_x, lane2_x = margin, margin + lane_w + lane_gap
    for x in (lane1_x, lane2_x):
        c.box(x, lane_y, lane_w, lane_h)
    c.pill(lane1_x + 0.08, lane_y + 0.08, "(a)  Stage 1: pre-training", 1.80)
    c.text(lane1_x + 0.10, lane_y + 0.40, "Synthetic hearts from a statistical shape model",
           va="center", **note)
    c.pill(lane2_x + 0.08, lane_y + 0.08, "(b)  Stage 2: fine-tuning", 1.76)
    c.text(lane2_x + 0.10, lane_y + 0.40, "Real hearts from clinical MRI, ED and ES",
           va="center", **note)
    arrow_y = lane_y + 0.18
    c.arrow((lane1_x + 1.92, arrow_y), (lane2_x + 0.05, arrow_y), lw=1.0)
    c.text((lane1_x + 1.92 + lane2_x) / 2, arrow_y - 0.03, "weights", ha="center",
           va="bottom", **note)

    vis_y, vis_h = lane_y + 0.55, 1.00
    cap_y = vis_y + vis_h + 0.11
    c.image(renders["ssm_pair"], lane1_x + 0.12, vis_y, 1.25, vis_h)
    c.image(renders["ssm_rings"], lane1_x + 1.66, vis_y, 1.25, vis_h)
    c.arrow((lane1_x + 1.38, vis_y + vis_h / 2), (lane1_x + 1.64, vis_y + vis_h / 2))
    c.text(lane1_x + 0.745, cap_y, "synthetic surfaces", **caption)
    c.text(lane1_x + 2.285, cap_y, "sliced contours", **caption)

    draw_sax_panel(c.axes(lane2_x + 0.17, vis_y + 0.03, 0.94, 0.94), *sax)
    c.arrow((lane2_x + 1.20, vis_y + vis_h / 2), (lane2_x + 1.46, vis_y + vis_h / 2))
    for k, phase in enumerate(PHASES):
        x = lane2_x + 1.50 + k * 0.74
        c.image(renders[f"rbf_{phase}"], x, vis_y + 0.12, 0.72, vis_h - 0.12, anchor="S")
        c.text(x + 0.36, vis_y + 0.03, phase, fontsize=6.4, ha="center", va="center",
               fontweight="bold")
    c.text(lane2_x + 0.64, cap_y, "MRI contours", **caption)
    c.text(lane2_x + 2.24, cap_y, "fitted surfaces", **caption)

    # ── Tier 2: the model ────────────────────────────────────────────────
    band_x, band_y = margin, lane_y + lane_h + 0.30
    band_w, band_h = FIG_W - 2 * margin, 1.42
    c.box(band_x, band_y, band_w, band_h, face="white", edge=NAVY, lw=0.9)
    c.pill(band_x + 0.08, band_y + 0.08, "(c)  Reconstruction model", 1.86)
    for x, label in ((lane1_x + lane_w / 2, "pre-train"), (lane2_x + lane_w / 2, "fine-tune")):
        c.arrow((x, lane_y + lane_h + 0.01), (x, band_y - 0.01), lw=1.1)
        c.text(x + 0.06, lane_y + lane_h + 0.15, label, va="center", ha="left", **note)

    row_y, row_h = band_y + 0.36, 0.80
    mid_y = row_y + row_h / 2
    label_y = row_y + row_h + 0.13
    c.image(renders["model_input"], band_x + 0.20, row_y - 0.04, 0.95, row_h + 0.04)
    c.text(band_x + 0.675, label_y, "contours + cardiac phase", **caption)

    boxes = (("Encoder", "summarises the contours"),
             ("Implicit decoder", "predicts both walls at any point"))
    box_x, box_w, box_gap = band_x + 1.55, 1.42, 0.38
    c.arrow((band_x + 1.20, mid_y), (box_x - 0.03, mid_y))
    for k, (title, line) in enumerate(boxes):
        x = box_x + k * (box_w + box_gap)
        c.box(x, row_y + 0.08, box_w, row_h - 0.16, face=BOX, radius=0.05, layer="front",
              zorder=1)
        c.text(x + box_w / 2, mid_y - 0.10, title, fontsize=7.0, fontweight="bold",
               ha="center", va="center")
        c.text(x + box_w / 2, mid_y + 0.11, line, ha="center", va="center", **note)
        c.arrow((x + box_w + 0.03, mid_y), (x + box_w + box_gap - 0.03, mid_y))
    sec_x = box_x + 2 * (box_w + box_gap)
    sec_w = band_x + band_w - 0.12 - sec_x
    draw_coupled_section(c.axes(sec_x, row_y - 0.08, sec_w, row_h + 0.12), renders)
    c.text(sec_x + sec_w / 2, label_y, "endo and epi walls", **caption)

    # ── Tier 3: inference, one row per cardiac phase ─────────────────────
    inf_y = band_y + band_h + 0.30
    inf_h = FIG_H - inf_y - margin
    c.box(margin, inf_y, FIG_W - 2 * margin, inf_h)
    c.pill(margin + 0.08, inf_y + 0.08, "Inference and wall thickness, one patient", 2.80)
    c.arrow((1.80, band_y + band_h + 0.01), (1.80, inf_y - 0.01), lw=1.1)
    c.text(1.86, band_y + band_h + 0.15, "trained model", va="center", **note)
    handles = [Line2D([0], [0], color=ENDO, lw=1.6, label="endocardium"),
               Line2D([0], [0], color=EPI, lw=1.6, label="epicardium")]
    c.front.legend(handles=handles, loc="upper right", ncol=2, frameon=False,
                   fontsize=6.2, bbox_to_anchor=(FIG_W - margin - 0.06, inf_y + 0.08),
                   bbox_transform=c.front.transData, handlelength=1.6,
                   columnspacing=1.0, borderaxespad=0.0)

    col_d, col_e, col_f, col_g = 0.38, 1.92, 3.30, 4.92
    col_w, bull_w = 1.20, 0.96
    head_y = inf_y + 0.47
    for x, w, title in ((col_d, col_w, "(d)  Contours"), (col_e, col_w, "(e)  Reconstruction"),
                        (col_f, col_w, "(f)  Wall thickness"),
                        (col_g, col_w, "(g)  Thickening, ES \u2212 ED")):
        c.text(x + w / 2, head_y, title, fontsize=7.0, fontweight="bold", ha="center",
               va="center")

    rows_y = head_y + 0.14
    bar_strip = 0.34
    row_h = (inf_y + inf_h - 0.05 - bar_strip - rows_y) / 2.0
    mids = []
    for k, phase in enumerate(PHASES):
        y = rows_y + k * row_h
        if k:
            c.front.plot([margin + 0.12, col_f + col_w + 0.06], [y, y], color=RULE, lw=0.6,
                         zorder=1)
        c.text(0.20, y + row_h / 2, {"ED": "End-diastole", "ES": "End-systole"}[phase],
               rotation=90, fontsize=6.8, fontweight="bold", ha="center", va="center")
        for x, key in ((col_d, "contours"), (col_e, "surfaces"), (col_f, "thickness")):
            c.image(renders[f"{key}_{phase}"], x, y + 0.05, col_w, row_h - 0.10)
        mid = y + row_h / 2
        mids.append(mid)
        c.arrow((col_d + col_w + 0.02, mid), (col_e - 0.02, mid), lw=1.0)
        c.text((col_d + col_w + col_e) / 2, mid - 0.06, "model", ha="center", va="bottom",
               **note)
        c.arrow((col_e + col_w + 0.01, mid), (col_f - 0.01, mid), lw=1.0)

    # Both phases merge into one arrow that feeds the thickening column.
    merge_x = col_f + col_w + 0.16
    ed_mid, es_mid = mids
    c.front.plot([col_f + col_w + 0.01, merge_x, merge_x, col_f + col_w + 0.01],
                 [ed_mid, ed_mid, es_mid, es_mid], color=INK, lw=1.0, zorder=5,
                 solid_joinstyle="miter")
    c.arrow((merge_x, ed_mid), (col_g + 0.14, ed_mid), lw=1.0)

    delta_norm = thickening_norm(case)
    c.image(renders["thickening"], col_g, rows_y + 0.05, col_w, row_h - 0.10)
    bull_x = col_g + (col_w - bull_w) / 2
    draw_bullseye(c.axes(bull_x, rows_y + row_h + 0.12, bull_w, row_h - 0.20),
                  segment_thickening(case), delta_norm, DELTA_CMAP)
    c.arrow((col_g + col_w / 2, rows_y + row_h - 0.04),
            (col_g + col_w / 2, rows_y + row_h + 0.11), lw=1.0)
    c.text(col_g + col_w / 2 + 0.06, rows_y + row_h + 0.035, "AHA-17", ha="left",
           va="center", **note)

    bar_y = rows_y + 2 * row_h + 0.04
    for x, cmap_name, bar_norm, label in (
            (col_f, THICKNESS_CMAP, norm, "Wall thickness (mm)"),
            (col_g, DELTA_CMAP, delta_norm, "Thickening, ES \u2212 ED (mm)")):
        cbar_ax = c.axes(x + 0.12, bar_y, col_w - 0.24, 0.06)
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=bar_norm, cmap=cmap_name),
                           cax=cbar_ax, orientation="horizontal")
        bar.locator = ticker.MaxNLocator(nbins=5, integer=True)
        bar.formatter = ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212"))
        bar.update_ticks()
        bar.set_label(label, fontsize=6.0, labelpad=1.5)
        bar.ax.tick_params(labelsize=5.6, length=2, width=0.5, pad=1.5)
        bar.outline.set_linewidth(0.5)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true",
                        help="recompute the reconstruction, SSM sample and renders")
    parser.add_argument("--rerender", action="store_true",
                        help="recompute the renders from the cached data")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    case = prepare_case(args.refresh)
    ssm = prepare_ssm_sample(args.refresh)
    for phase in PHASES:
        means = case[f"{phase}_segment_means"]
        print(phase, "AHA-17 (mm):", " ".join(f"{v:.1f}" for v in means))
    if args.prepare_only:
        return

    if RENDER_CACHE.exists() and not (args.refresh or args.rerender):
        with np.load(RENDER_CACHE) as data:
            renders = {key: data[key] for key in data.files}
    else:
        shown = display_case(case)
        renders = make_renders(case, shown, ssm, load_rbf_targets())
        renders.update(section_data(shown))
        np.savez_compressed(RENDER_CACHE, **renders)
    fig = compose(case, renders, load_sax_slice())
    fig.savefig(OUT_STEM.with_suffix(".pdf"), dpi=450, facecolor="white")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)
    for suffix in (".pdf", ".png"):
        print("wrote", OUT_STEM.with_suffix(suffix).relative_to(THESIS))


if __name__ == "__main__":
    main()
