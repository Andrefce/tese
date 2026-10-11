"""Introduction overview figure: from short-axis cine MRI to regional thickening.

Panels, read top to bottom:

  (a) the ED and ES short-axis frames of ACDC ``patient002`` with the expert
      labels,
  (b) the segmented ED slices stacked at their physical positions, drawn with
      the same camera as the methodology stack figure,
  (c), (d) the reconstructed endocardium of each phase coloured by the Laplace
      wall thickness measured between the reconstructed surfaces,
  (e) the end-systolic surface coloured by the ED-to-ES thickness change at
      corresponding wall positions, with the AHA-17 segment boundaries,
  (f) the AHA-17 bullseye of the segment-mean thickness change.

All three-dimensional panels use the cached demo reconstruction of
``scripts/eval_demo`` (``demo_patient002_{ED,ES}.npz``), i.e. the same surfaces
and per-vertex Laplace thickness that the Results chapter reports; the bullseye
reads the segment means from ``aha17_thickening.csv`` so that it matches the
tabulated values. Nothing is fitted or re-computed here apart from the
vertex-wise correspondence used to subtract the two phases.

Output: ``images/intro_pipeline_overview.png``.

Run:
    C:/Python313/python.exe scripts/fig_intro_overview.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pyvista as pv
from matplotlib import colors, patches, patheffects
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from skimage import measure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_patient002_methodology_figures import (  # noqa: E402
    ED_MRI,
    ED_SEG,
    ES_MRI,
    ES_SEG,
    FLOW_BLUE,
    FLOW_GREEN,
    FLOW_ORANGE,
    OUT_DIR,
    PLATE_GREY,
    PLATE_INK,
    PLATE_RC,
    draw_contours,
    draw_scale_bar,
    label_legend_handles,
    representative_lv_slice,
    save_rgb_figure,
    style_viewport,
    union_crop,
    window_bounds,
)

DEMO = ROOT / "scripts" / "eval_demo" / "outputs"
OUTPUT = OUT_DIR / "intro_pipeline_overview.png"
THICKNESS_KEY = "model_wt_laplace_streamline_symmetric"
THICKNESS_METHOD = "Laplace streamline (symmetric)"

THICKNESS_CMAP = "viridis"
THICKNESS_NORM = colors.Normalize(vmin=5.0, vmax=13.0)
DELTA_CMAP = "RdBu_r"
DELTA_NORM = colors.Normalize(vmin=-4.0, vmax=4.0)
NAN_COLOUR = "#cfcfcf"
SLICE_ALPHA = 185

# One camera for every left-ventricular render: slightly above the valve
# plane, looking at the infero-lateral wall, apex pointing down.
LV_VIEW = (1.1, -1.9, -0.55)
LV_UP = (0.0, 0.0, -1.0)
# Same elevation and azimuth as the methodology stack (elev 24, azim -58).
STACK_VIEW = (0.484, -0.775, 0.407)
STACK_UP = (0.0, 0.0, 1.0)
RENDER_SIZE = (1400, 1500)

pv.OFF_SCREEN = True


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_phase(phase: str) -> dict:
    with np.load(DEMO / f"demo_{PATIENT}_{phase}.npz") as data:
        return {
            "endo_v": np.asarray(data["model_endo_v"], np.float64),
            "endo_f": np.asarray(data["model_endo_f"], np.int64),
            "epi_v": np.asarray(data["model_epi_v"], np.float64),
            "epi_f": np.asarray(data["model_epi_f"], np.int64),
            "aha": np.asarray(data["model_aha"], np.int64),
            "thickness": np.asarray(data[THICKNESS_KEY], np.float64),
        }


PATIENT = "patient002"


def segment_means() -> tuple[dict[int, float], dict[int, float]]:
    """ED and ES segment means of the model geometry, as tabulated."""
    ed: dict[int, float] = {}
    es: dict[int, float] = {}
    with open(DEMO / "aha17_thickening.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["geometry"] != "model" or row["method"] != THICKNESS_METHOD:
                continue
            segment = int(row["segment_id"])
            ed[segment] = float(row["ed_mm"])
            es[segment] = float(row["es_mm"])
    assert len(ed) == 17 and len(es) == 17
    return ed, es


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def polydata(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    cells = np.hstack([np.full((len(faces), 1), 3, np.int64),
                       np.asarray(faces, np.int64)]).ravel()
    return pv.PolyData(np.asarray(vertices, np.float32), cells)


def drop_basal_lid(surface: pv.PolyData, depth_mm: float = 3.0,
                   flatness: float = 0.80) -> pv.PolyData:
    """Remove the flat cap closing the surface at the valve plane (low z)."""
    oriented = surface.compute_normals(cell_normals=True, point_normals=False,
                                       auto_orient_normals=True)
    normals = np.asarray(oriented.cell_normals)
    centres = np.asarray(oriented.cell_centers().points)
    floor = float(np.asarray(surface.points)[:, 2].min()) + depth_mm
    lid = (np.abs(normals[:, 2]) > flatness) & (centres[:, 2] < floor)
    if not lid.any():
        return surface
    return oriented.extract_cells(np.flatnonzero(~lid)).extract_surface()


def display_surface(vertices: np.ndarray, faces: np.ndarray,
                    **point_arrays: np.ndarray) -> pv.PolyData:
    surface = polydata(vertices, faces)
    for name, values in point_arrays.items():
        surface[name] = np.asarray(values, np.float32)
    smoothed = surface.smooth_taubin(n_iter=40, pass_band=0.05,
                                     normalize_coordinates=True)
    if smoothed.n_points == surface.n_points:
        surface = smoothed
    return drop_basal_lid(surface)


def cylindrical_features(vertices: np.ndarray, z_weight: float = 3.0) -> np.ndarray:
    """Normalised long-axis position and circumferential angle about the local axis.

    The in-plane centre is taken per long-axis bin so that the angle follows the
    ventricular axis rather than the global centroid.
    """
    z = vertices[:, 2]
    z_norm = (z - z.min()) / max(np.ptp(z), 1e-8)
    n_bins = 20
    bins = np.clip((z_norm * n_bins).astype(int), 0, n_bins - 1)
    centres = np.full((n_bins, 2), np.nan)
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            centres[b] = vertices[mask, :2].mean(axis=0)
    valid = ~np.isnan(centres[:, 0])
    positions = (np.arange(n_bins) + 0.5) / n_bins
    axis_x = np.interp(z_norm, positions[valid], centres[valid, 0])
    axis_y = np.interp(z_norm, positions[valid], centres[valid, 1])
    angle = np.arctan2(vertices[:, 1] - axis_y, vertices[:, 0] - axis_x)
    return np.column_stack([np.cos(angle), np.sin(angle), z_weight * z_norm])


def thickness_change(ed: dict, es: dict) -> np.ndarray:
    """ES minus ED Laplace thickness at corresponding wall positions, on ES vertices."""
    ed_valid = ~np.isnan(ed["thickness"])
    tree = cKDTree(cylindrical_features(ed["endo_v"])[ed_valid])
    _, index = tree.query(cylindrical_features(es["endo_v"]))
    delta = es["thickness"] - ed["thickness"][ed_valid][index]
    return smooth_vertex_field(delta, es["endo_f"], iterations=15)


def smooth_vertex_field(values: np.ndarray, faces: np.ndarray,
                        iterations: int) -> np.ndarray:
    """Neighbour averaging of a vertex field, leaving undefined vertices undefined."""
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    defined = ~np.isnan(values)
    smoothed = np.where(defined, values, 0.0)
    for _ in range(iterations):
        weight = defined[edges[:, 1]].astype(np.float64)
        total = np.bincount(edges[:, 0], weights=smoothed[edges[:, 1]] * weight,
                            minlength=len(values))
        count = np.bincount(edges[:, 0], weights=weight, minlength=len(values))
        smoothed = np.where(count > 0, total / np.maximum(count, 1.0), smoothed)
    return np.where(defined, smoothed, np.nan)


def segment_boundary_lines(surface: pv.PolyData, ids: np.ndarray) -> pv.PolyData | None:
    triangles = surface.faces.reshape(-1, 4)[:, 1:]
    labels = ids[triangles]
    boundary = set()
    for a, b in ((0, 1), (1, 2), (2, 0)):
        differing = labels[:, a] != labels[:, b]
        pairs = np.sort(triangles[differing][:, [a, b]], axis=1)
        boundary.update(map(tuple, pairs.tolist()))
    if not boundary:
        return None
    edges = np.asarray(sorted(boundary), np.int64)
    lines = np.hstack([np.full((len(edges), 1), 2, np.int64), edges]).ravel()
    return pv.PolyData(np.asarray(surface.points, np.float32), lines=lines)


# --------------------------------------------------------------------------- #
# Renders
# --------------------------------------------------------------------------- #
def autocrop(image: np.ndarray, pad: int = 6) -> np.ndarray:
    mask = (image[:, :, :3] < 250).any(axis=2)
    if not mask.any():
        return image
    rows, cols = np.where(mask)
    return image[max(0, rows.min() - pad):rows.max() + pad + 1,
                 max(0, cols.min() - pad):cols.max() + pad + 1]


def new_plotter(size=RENDER_SIZE) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=size)
    plotter.set_background("white")
    return plotter


def finish(plotter: pv.Plotter, view, up, zoom: float, *, ssao: bool) -> np.ndarray:
    plotter.view_vector(view, viewup=up)
    plotter.camera.zoom(zoom)
    if ssao:
        try:
            plotter.enable_ssao(radius=6.0)
        except Exception:  # noqa: BLE001
            pass
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:  # noqa: BLE001
        pass
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return autocrop(np.asarray(image))


def render_thickness_surface(case: dict, values: np.ndarray, cmap: str,
                             norm: colors.Normalize, *, with_epi: bool,
                             boundaries: np.ndarray | None = None) -> np.ndarray:
    arrays = {"value": values}
    if boundaries is not None:
        arrays["segment"] = boundaries
    surface = display_surface(case["endo_v"], case["endo_f"], **arrays)
    plotter = new_plotter()
    plotter.add_mesh(surface, scalars="value", cmap=cmap,
                     clim=(norm.vmin, norm.vmax), nan_color=NAN_COLOUR,
                     smooth_shading=True, specular=0.25, specular_power=16,
                     ambient=0.30, diffuse=0.90, show_scalar_bar=False)
    if boundaries is not None:
        surface_ids = np.rint(np.asarray(surface["segment"])).astype(np.int64)
        lines = segment_boundary_lines(surface, surface_ids)
        if lines is not None:
            plotter.add_mesh(lines, color="black", line_width=2,
                             render_lines_as_tubes=True)
    if with_epi:
        epi = display_surface(case["epi_v"], case["epi_f"])
        plotter.add_mesh(epi, color="#f2f2f2", opacity=0.22,
                         smooth_shading=True, specular=0.0, ambient=0.5)
        plotter.enable_depth_peeling(number_of_peels=16, occlusion_ratio=0.0)
    return finish(plotter, LV_VIEW, LV_UP, 1.0, ssao=not with_epi)


def slice_rings(labels: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """Closed contours (row, col) of each clinical label in one slice."""
    rings = []
    strokes = [(np.isin(labels, (2, 3)), FLOW_ORANGE), (labels == 3, FLOW_BLUE),
               (labels == 1, FLOW_GREEN)]
    for binary, colour in strokes:
        if not binary.any():
            continue
        for contour in measure.find_contours(binary.astype(float), 0.5):
            if len(contour) > 8:
                rings.append((contour, colour))
    return rings


def ring_points_mm(ring: np.ndarray, spacing, z: float) -> np.ndarray:
    return np.column_stack([ring[:, 1] * spacing[0], ring[:, 0] * spacing[1],
                            np.full(len(ring), z)])


def add_slice_frame(plotter: pv.Plotter, width: float, height: float, z: float) -> None:
    corners = np.array([[0, 0, z], [width, 0, z], [width, height, z], [0, height, z]],
                       dtype=np.float32)
    plotter.add_mesh(pv.lines_from_points(corners, close=True), color="#bdbdbd",
                     line_width=1.2)


def render_stack(volume: np.ndarray, segmentation: np.ndarray, spacing,
                 crop) -> np.ndarray:
    """Every acquired slice at its physical position, base on top, as in the
    methodology stack figure."""
    rows, cols = crop
    n_rows, n_cols = rows.stop - rows.start, cols.stop - cols.start
    vmin, vmax = window_bounds(volume, crop=crop)
    x = np.arange(n_cols) * spacing[0]
    y = np.arange(n_rows) * spacing[1]
    grid_x, grid_y = np.meshgrid(x, y)

    plotter = new_plotter((1600, 1500))
    for k in range(volume.shape[2]):
        z = -float(k) * spacing[2]
        image = volume[rows, cols, k]
        labels = segmentation[rows, cols, k]
        grey = np.clip((image - vmin) / (vmax - vmin), 0.0, 1.0)
        rgba = np.empty((n_rows, n_cols, 4), np.uint8)
        rgba[..., :3] = (grey * 255.0).astype(np.uint8)[..., None]
        rgba[..., 3] = SLICE_ALPHA
        plane = pv.StructuredGrid(grid_x, grid_y, np.full_like(grid_x, z))
        plane["rgba"] = rgba.transpose(1, 0, 2).reshape(-1, 4)
        plotter.add_mesh(plane, scalars="rgba", rgba=True, lighting=False,
                         show_scalar_bar=False)
        add_slice_frame(plotter, x[-1], y[-1], z)
        for ring, colour in slice_rings(labels):
            points = ring_points_mm(ring, spacing, z + 0.15)
            plotter.add_mesh(pv.lines_from_points(points.astype(np.float32), close=True),
                             color=colour, line_width=6, render_lines_as_tubes=True)
    plotter.enable_depth_peeling(number_of_peels=24, occlusion_ratio=0.0)
    return finish(plotter, STACK_VIEW, STACK_UP, 1.0, ssao=False)


# --------------------------------------------------------------------------- #
# Matplotlib composition
# --------------------------------------------------------------------------- #
def show_image(ax: plt.Axes, image: np.ndarray) -> None:
    ax.imshow(image, interpolation="lanczos")
    ax.set_axis_off()


def header(fig, ax, tag: str, title: str, note: str | None = None,
           y: float | None = None) -> None:
    box = ax.get_position()
    y_title = box.y1 + 0.022 if y is None else y
    fig.text(box.x0, y_title, f"({tag})", ha="left", va="baseline",
             fontsize=7.2, fontweight="bold", color=PLATE_INK)
    fig.text(box.x0 + 0.030, y_title, title, ha="left", va="baseline",
             fontsize=7.2, fontweight="bold", color=PLATE_INK)
    if note:
        fig.text(box.x0, y_title - 0.015, note, ha="left", va="baseline",
                 fontsize=5.9, color=PLATE_GREY, style="italic")


def arrow(fig, start, end, **kwargs) -> None:
    style = dict(arrowstyle="-|>", mutation_scale=7.0, linewidth=0.75,
                 color=PLATE_INK, shrinkA=0, shrinkB=0, clip_on=False, zorder=40)
    style.update(kwargs)
    fig.add_artist(patches.FancyArrowPatch(start, end, transform=fig.transFigure,
                                           **style))


def route(fig, points, *, arrowhead: bool = True) -> None:
    segments = list(zip(points[:-1], points[1:]))
    tail = segments[:-1] if arrowhead else segments
    for start, end in tail:
        fig.add_artist(Line2D([start[0], end[0]], [start[1], end[1]],
                              transform=fig.transFigure, color=PLATE_INK, lw=0.75,
                              solid_capstyle="round", clip_on=False, zorder=40))
    if arrowhead:
        arrow(fig, *segments[-1])


def branch_label(fig, x, y, text, ha="center") -> None:
    fig.text(x, y, text, ha=ha, va="center", fontsize=6.2, color=PLATE_INK,
             bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                       edgecolor="none"), zorder=41)


def colourbar(fig, rect, cmap: str, norm: colors.Normalize, label: str,
              ticks=None) -> None:
    axis = fig.add_axes(rect)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(mappable, cax=axis, orientation="vertical", ticks=ticks)
    bar.set_label(label, fontsize=6.2, color=PLATE_INK, labelpad=3)
    bar.outline.set_linewidth(0.5)
    bar.outline.set_edgecolor(PLATE_INK)
    bar.ax.tick_params(labelsize=5.7, length=2, width=0.5, colors=PLATE_INK)


def draw_bullseye(ax: plt.Axes, values: dict[int, float], norm: colors.Normalize,
                  cmap: str) -> None:
    """Standard AHA-17 display: anterior at the top, septum on the left.

    Basal and mid segments are numbered counter-clockwise from the anterior
    wall; apical segments follow the same convention at 90-degree spacing.
    """
    colormap = plt.get_cmap(cmap)
    effects = [patheffects.withStroke(linewidth=1.3, foreground="black")]
    rings = [(0.72, 1.00, 6, 1, 60.0), (0.45, 0.72, 6, 7, 60.0),
             (0.20, 0.45, 4, 13, 45.0)]
    for inner, outer, count, first, start in rings:
        span = 360.0 / count
        for index in range(count):
            segment = first + index
            theta1 = start + index * span
            theta2 = theta1 + span
            ax.add_patch(patches.Wedge((0.0, 0.0), outer, theta1, theta2,
                                       width=outer - inner,
                                       facecolor=colormap(norm(values[segment])),
                                       edgecolor=PLATE_INK, linewidth=0.45))
            mid = np.deg2rad(0.5 * (theta1 + theta2))
            radius = 0.5 * (inner + outer)
            text = ax.text(radius * np.cos(mid), radius * np.sin(mid), str(segment),
                           ha="center", va="center", fontsize=5.4, color="white")
            text.set_path_effects(effects)
    ax.add_patch(patches.Circle((0.0, 0.0), 0.20,
                                facecolor=colormap(norm(values[17])),
                                edgecolor=PLATE_INK, linewidth=0.45))
    centre = ax.text(0.0, 0.0, "17", ha="center", va="center", fontsize=5.4,
                     color="white")
    centre.set_path_effects(effects)
    for label, (x, y, ha, va) in {
        "anterior": (0.0, 1.06, "center", "bottom"),
        "inferior": (0.0, -1.06, "center", "top"),
        "septal": (-1.06, 0.0, "right", "center"),
        "lateral": (1.06, 0.0, "left", "center"),
    }.items():
        ax.text(x, y, label, ha=ha, va=va, fontsize=5.8, color=PLATE_GREY)
    ax.set_aspect("equal")
    ax.set_xlim(-1.40, 1.40)
    ax.set_ylim(-1.22, 1.22)
    ax.axis("off")


def tight_box(ax: plt.Axes):
    """Axes box after the aspect lock, i.e. the area the image really occupies."""
    ax.apply_aspect()
    return ax.get_position()


def compose(renders: dict, slices: dict, bullseye: dict[int, float]) -> Path:
    with plt.rc_context(PLATE_RC):
        fig = plt.figure(figsize=(7.2, 8.6), facecolor="white")
        square = 0.150 * 7.2 / 8.6

        # Row 1: observation and slice stack.
        ed_axis = fig.add_axes([0.035, 0.958 - square, 0.150, square])
        es_axis = fig.add_axes([0.035, 0.943 - 2 * square, 0.150, square])
        stack_axis = fig.add_axes([0.255, 0.672, 0.470, 0.275])
        row1_header = 0.975

        # Row 2: per-phase thickness.
        ed_lv_axis = fig.add_axes([0.085, 0.375, 0.290, 0.235])
        es_lv_axis = fig.add_axes([0.545, 0.375, 0.290, 0.235])
        row2_header = 0.628

        # Row 3: thickening on the surface and in the bullseye.
        delta_axis = fig.add_axes([0.190, 0.035, 0.290, 0.265])
        bullseye_axis = fig.add_axes([0.545, 0.040, 0.290, 0.245])
        row3_header = 0.322

        for axis, phase in ((ed_axis, "ED"), (es_axis, "ES")):
            image, labels, spacing, title = slices[phase]
            axis.imshow(image, cmap="gray", origin="lower", vmin=slices["window"][0],
                        vmax=slices["window"][1], interpolation="nearest")
            draw_contours(axis, labels, plate=True)
            draw_scale_bar(axis, spacing[0], length_mm=20.0)
            style_viewport(axis)
            for spine in axis.spines.values():
                spine.set_color(PLATE_INK)
                spine.set_linewidth(0.6)
            axis.text(0.04, 0.96, title, transform=axis.transAxes, ha="left",
                      va="top", fontsize=6.0, color="white",
                      path_effects=[patheffects.withStroke(linewidth=1.4,
                                                           foreground="black")])

        show_image(stack_axis, renders["stack"])
        show_image(ed_lv_axis, renders["ED"])
        show_image(es_lv_axis, renders["ES"])
        show_image(delta_axis, renders["delta"])
        draw_bullseye(bullseye_axis, bullseye, DELTA_NORM, DELTA_CMAP)

        header(fig, ed_axis, "a", "Short-axis cine MRI",
               "ACDC frames with expert labels", y=row1_header)
        header(fig, stack_axis, "b", "Slice stack",
               "segmented slices at their physical positions", y=row1_header)
        header(fig, ed_lv_axis, "c", "End-diastolic reconstruction",
               "endocardium coloured by local wall thickness", y=row2_header)
        header(fig, es_lv_axis, "d", "End-systolic reconstruction",
               "endocardium coloured by local wall thickness", y=row2_header)
        header(fig, delta_axis, "e", "Systolic wall thickening",
               "ES minus ED thickness at corresponding positions", y=row3_header)
        header(fig, bullseye_axis, "f", "AHA-17 summary",
               "segment means of the thickening", y=row3_header)

        fig.legend(handles=label_legend_handles(), loc="center left",
                   bbox_to_anchor=(0.760, 0.820), frameon=False, ncol=1,
                   fontsize=5.9, labelcolor=PLATE_GREY, handlelength=1.6,
                   labelspacing=0.5, handletextpad=0.35, borderaxespad=0.0)

        colourbar(fig, [0.900, 0.410, 0.016, 0.180], THICKNESS_CMAP, THICKNESS_NORM,
                  "wall thickness (mm)")
        colourbar(fig, [0.900, 0.075, 0.016, 0.180], DELTA_CMAP, DELTA_NORM,
                  "thickness change, ES $-$ ED (mm)", ticks=[-4, -2, 0, 2, 4])

        # ---- flow ---------------------------------------------------------
        def mid_y(box):
            return 0.5 * (box.y0 + box.y1)

        def mid_x(box):
            return 0.5 * (box.x0 + box.x1)

        slice_box = ed_axis.get_position()
        es_slice_box = es_axis.get_position()
        stack_box = tight_box(stack_axis)
        ed_box = tight_box(ed_lv_axis)
        es_box = tight_box(es_lv_axis)
        delta_box = tight_box(delta_axis)
        bull_box = bullseye_axis.get_position()
        row1_mid = 0.5 * (slice_box.y1 + es_slice_box.y0)

        arrow(fig, (slice_box.x1 + 0.012, row1_mid), (stack_box.x0 - 0.006, row1_mid))

        # Bifurcation: the stack feeds the model once per phase.
        bus_y = 0.660
        route(fig, [(mid_x(stack_box), stack_box.y0 - 0.004), (mid_x(stack_box), bus_y)],
              arrowhead=False)
        route(fig, [(mid_x(stack_box), bus_y), (mid_x(ed_box), bus_y),
                    (mid_x(ed_box), row2_header + 0.012)])
        route(fig, [(mid_x(stack_box), bus_y), (mid_x(es_box), bus_y),
                    (mid_x(es_box), row2_header + 0.012)])
        branch_label(fig, mid_x(ed_box) + 0.012, bus_y - 0.015,
                     "proposed model, ED contours", ha="left")
        branch_label(fig, mid_x(es_box) + 0.012, bus_y - 0.015,
                     "proposed model, ES contours", ha="left")

        # Merge: both phases converge on the thickening map.
        merge_y = 0.5 * (min(ed_box.y0, es_box.y0) + row3_header + 0.012)
        route(fig, [(mid_x(ed_box), ed_box.y0 - 0.004), (mid_x(ed_box), merge_y),
                    (mid_x(delta_box), merge_y)], arrowhead=False)
        route(fig, [(mid_x(es_box), es_box.y0 - 0.004), (mid_x(es_box), merge_y),
                    (mid_x(delta_box), merge_y), (mid_x(delta_box), row3_header + 0.012)])
        arrow(fig, (delta_box.x1 + 0.006, mid_y(delta_box)),
              (bull_box.x0 - 0.004, mid_y(delta_box)))

        OUTPUT.parent.mkdir(exist_ok=True)
        save_rgb_figure(fig, OUTPUT, dpi=400, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    return OUTPUT


# --------------------------------------------------------------------------- #
def main() -> None:
    ed_volume = nib.load(str(ED_MRI)).get_fdata(dtype=np.float32)
    es_volume = nib.load(str(ES_MRI)).get_fdata(dtype=np.float32)
    ed_seg = np.asarray(nib.load(str(ED_SEG)).dataobj).astype(np.uint8)
    es_seg = np.asarray(nib.load(str(ES_SEG)).dataobj).astype(np.uint8)
    spacing = tuple(float(v) for v in nib.load(str(ED_MRI)).header.get_zooms()[:3])
    crop = union_crop(ed_seg, es_seg)
    window = window_bounds(ed_volume, es_volume, crop=crop)

    slices = {"window": window}
    for phase, volume, seg, title in (("ED", ed_volume, ed_seg, "ED, frame 1"),
                                      ("ES", es_volume, es_seg, "ES, frame 12")):
        k = representative_lv_slice(seg)
        slices[phase] = (volume[crop[0], crop[1], k], seg[crop[0], crop[1], k],
                         spacing, title)

    ed, es = load_phase("ED"), load_phase("ES")
    delta = thickness_change(ed, es)
    finite = delta[~np.isnan(delta)]
    print(f"thickening on ES vertices: median {np.median(finite):.2f} mm, "
          f"5-95 % {np.percentile(finite, 5):.2f}..{np.percentile(finite, 95):.2f} mm")

    ed_means, es_means = segment_means()
    bullseye = {s: es_means[s] - ed_means[s] for s in range(1, 18)}
    print("segment thickening (mm):",
          ", ".join(f"{s}:{bullseye[s]:.1f}" for s in range(1, 18)))

    print("rendering ...")
    renders = {
        "stack": render_stack(ed_volume, ed_seg, spacing, crop),
        "ED": render_thickness_surface(ed, ed["thickness"], THICKNESS_CMAP,
                                       THICKNESS_NORM, with_epi=False),
        "ES": render_thickness_surface(es, es["thickness"], THICKNESS_CMAP,
                                       THICKNESS_NORM, with_epi=False),
        "delta": render_thickness_surface(es, delta, DELTA_CMAP, DELTA_NORM,
                                          with_epi=False, boundaries=es["aha"]),
    }
    if "--dump" in sys.argv:
        from PIL import Image  # noqa: PLC0415
        dump = Path(sys.argv[sys.argv.index("--dump") + 1])
        dump.mkdir(parents=True, exist_ok=True)
        for name, image in renders.items():
            Image.fromarray(image).save(dump / f"panel_{name}.png")

    output = compose(renders, slices, bullseye)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
