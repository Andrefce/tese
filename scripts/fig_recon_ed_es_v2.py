"""Render the qualitative reconstruction comparison of the Results chapter.

Six panels: the proposed model, a smoothed RBF implicit fit, and a statistical
shape-model fit, each at end-diastole and end-systole. The three reconstructions
consume the same SAX contour rings of ``patient002``. The end-systolic
shape-model panel is intentionally empty because the public UK Digital Heart
Project left-ventricular model ships end-diastolic modes only.

Rendering conventions are identical in every panel: one linked camera, parallel
projection, a shared bounding frame, an apex-down orientation derived from the
geometry itself, and one display-only Taubin pass with the same parameters for
every surface. No mesh is snapped or deformed towards the input contours.

Output: ``images/results_recon_ed_es.png``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

THESIS = Path(__file__).resolve().parents[1]
DEMO_OUT = THESIS / "scripts" / "eval_demo" / "outputs"
CACHE = DEMO_OUT / "recon_ed_es_baselines.npz"
DISPLAY_CACHE = DEMO_OUT / "recon_ed_es_display.npz"
OUT = THESIS / "images" / "results_recon_ed_es.png"
PATIENT = "patient002"

sys.path.insert(0, str(THESIS / "scripts"))
sys.path.insert(0, str(THESIS / "scripts" / "webapp"))
from fig_baseline_rbf_ssm import (  # noqa: E402
    build_rbf_geometry,
    build_ssm_geometry,
)
from core.sdf_model import _snap_mesh_to_contours  # noqa: E402

ENDO_COLOUR = "#b3313c"
EPI_COLOUR = "#c8ccd1"
CONTOUR_COLOUR = "#22333b"
CONTOUR_WIDTH = 1.1

# Display-only Taubin parameters, applied identically to every rendered surface.
SMOOTH_ITERATIONS = 600
SMOOTH_PASS_BAND = 0.003
SCALE_BAR_MM = 20.0
# Axial support of the RBF grid; every surface is held to the same range.
AXIAL_MARGIN_MM = 1.5
ENDO_CUT_INSET_MM = 1.0
PANEL_PIXELS = (760, 1080)

pv.OFF_SCREEN = True
pv.global_theme.font.family = "times"


def _polydata(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    cells = np.hstack([
        np.full((len(faces), 1), 3, dtype=np.int64),
        np.asarray(faces, dtype=np.int64),
    ]).ravel()
    return pv.PolyData(np.asarray(vertices, dtype=np.float32), cells)


def load_demo() -> tuple[dict, dict, dict]:
    """Cached model surfaces, snapped to the rings by the inference pipeline."""
    contours: dict[str, np.ndarray] = {}
    tissue: dict[str, np.ndarray] = {}
    model: dict[tuple[str, str], trimesh.Trimesh] = {}
    for phase in ("ED", "ES"):
        with np.load(DEMO_OUT / f"demo_{PATIENT}_{phase}.npz") as data:
            contours[phase] = np.asarray(data["contours_xyz_mm"], dtype=np.float64)
            tissue[phase] = np.asarray(data["contours_tissue"], dtype=np.float64)
            for wall in ("endo", "epi"):
                vertices = _snap_mesh_to_contours(
                    np.asarray(data[f"model_{wall}_v"], dtype=np.float32),
                    contours[phase].astype(np.float32),
                    tissue[phase].astype(np.float32), surface=wall)
                model[(phase, wall)] = trimesh.Trimesh(
                    np.asarray(vertices, dtype=np.float64),
                    np.asarray(data[f"model_{wall}_f"], dtype=np.int64),
                    process=False)
    return contours, tissue, model


def trim_to_observed(mesh: trimesh.Trimesh, low: float, high: float,
                     inset: float = 0.0) -> trimesh.Trimesh:
    """Clip to the observed slice range, so every method has the same support.

    The endocardium is cut slightly inside the epicardium, otherwise the two
    caps are coplanar and z-fight in the render.
    """
    clipped = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=[0.0, 0.0, 1.0],
        plane_origin=[0.0, 0.0, low - AXIAL_MARGIN_MM + inset], cap=True)
    return trimesh.intersections.slice_mesh_plane(
        clipped, plane_normal=[0.0, 0.0, -1.0],
        plane_origin=[0.0, 0.0, high + AXIAL_MARGIN_MM - inset], cap=True)
    clipped.merge_vertices()
    if not clipped.is_watertight:
        trimesh.repair.fill_holes(clipped)
    trimesh.repair.fix_normals(clipped)
    return clipped


def _fit_baseline(method: str, phase: str, points: np.ndarray,
                  labels: np.ndarray) -> dict:
    return (build_rbf_geometry(points, labels) if method == "rbf"
            else build_ssm_geometry(points, labels, phase))


def build_baselines(contours: dict, tissue: dict,
                    refresh: bool) -> dict[tuple[str, str, str], trimesh.Trimesh]:
    """RBF fits for both phases and a shape-model fit at end-diastole."""
    wanted = (("rbf", "ED"), ("rbf", "ES"), ("ssm", "ED"))
    if CACHE.exists() and not refresh:
        with np.load(CACHE) as data:
            keys = {f"{method}_{phase}_{wall}_v"
                    for method, phase in wanted for wall in ("endo", "epi")}
            if keys.issubset(set(data.files)):
                print(f"using cached baselines: {CACHE.relative_to(THESIS)}")
                return {
                    (method, phase, wall): trimesh.Trimesh(
                        data[f"{method}_{phase}_{wall}_v"],
                        data[f"{method}_{phase}_{wall}_f"], process=False)
                    for method, phase in wanted for wall in ("endo", "epi")
                }

    meshes: dict[tuple[str, str, str], trimesh.Trimesh] = {}
    payload: dict[str, np.ndarray] = {}
    for method, phase in wanted:
        start = time.perf_counter()
        geometry = _fit_baseline(method, phase, contours[phase], tissue[phase])
        print(f"  {method.upper()} {phase} fitted in "
              f"{time.perf_counter() - start:.1f} s")
        for wall in ("endo", "epi"):
            meshes[(method, phase, wall)] = geometry[wall]
            payload[f"{method}_{phase}_{wall}_v"] = np.asarray(
                geometry[wall].vertices, np.float32)
            payload[f"{method}_{phase}_{wall}_f"] = np.asarray(
                geometry[wall].faces, np.int64)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **payload)
    print(f"cached baselines: {CACHE.relative_to(THESIS)}")
    return meshes


def apex_towards_positive_z(mesh: trimesh.Trimesh) -> bool:
    """True when the apex sits at the high-z end of the short-axis stack.

    The apex is the end at which the cross-section shrinks, so the two ends are
    compared by their mean in-plane radius.
    """
    points = np.asarray(mesh.vertices, dtype=np.float64)
    z = points[:, 2]

    def radius(subset: np.ndarray) -> float:
        plane = subset[:, :2] - subset[:, :2].mean(axis=0)
        return float(np.linalg.norm(plane, axis=1).mean())

    return radius(points[z > np.percentile(z, 75)]) < \
        radius(points[z < np.percentile(z, 25)])


def display_surfaces(meshes: dict, refresh: bool) -> dict[tuple, pv.PolyData]:
    """Taubin-smoothed copies used only for rendering, cached across runs."""
    signature = np.array([SMOOTH_ITERATIONS, SMOOTH_PASS_BAND], dtype=np.float64)
    if DISPLAY_CACHE.exists() and not refresh:
        with np.load(DISPLAY_CACHE) as data:
            names = {f"{method}_{phase}_{wall}"
                     for method, phase, wall in meshes}
            if ("signature" in data.files
                    and np.array_equal(data["signature"], signature)
                    and all(f"{name}_v" in data.files for name in names)
                    and all(len(data[f"{key[0]}_{key[1]}_{key[2]}_v"])
                            == len(mesh.vertices) for key, mesh in meshes.items())):
                print(f"using cached display meshes: "
                      f"{DISPLAY_CACHE.relative_to(THESIS)} "
                      f"(mean displacement {float(data['shift_mean']):.2f} mm, "
                      f"max {float(data['shift_max']):.2f} mm)")
                return {
                    key: _polydata(data[f"{key[0]}_{key[1]}_{key[2]}_v"], mesh.faces)
                    for key, mesh in meshes.items()
                }

    surfaces: dict[tuple, pv.PolyData] = {}
    shifts: list[float] = []
    payload: dict[str, np.ndarray] = {"signature": signature}
    for key, mesh in meshes.items():
        raw = _polydata(mesh.vertices, mesh.faces)
        smoothed = raw.smooth_taubin(n_iter=SMOOTH_ITERATIONS,
                                     pass_band=SMOOTH_PASS_BAND,
                                     normalize_coordinates=True)
        shifts.append(float(np.linalg.norm(smoothed.points - raw.points,
                                           axis=1).mean()))
        surfaces[key] = smoothed
        payload[f"{key[0]}_{key[1]}_{key[2]}_v"] = np.asarray(smoothed.points,
                                                              np.float32)

    payload["shift_mean"] = np.float64(np.mean(shifts))
    payload["shift_max"] = np.float64(np.max(shifts))
    np.savez_compressed(DISPLAY_CACHE, **payload)
    print(f"display smoothing: Taubin {SMOOTH_ITERATIONS} iterations, pass band "
          f"{SMOOTH_PASS_BAND}, mean vertex displacement {np.mean(shifts):.2f} mm "
          f"(max {np.max(shifts):.2f} mm)")
    return surfaces


def contour_rings(contours: np.ndarray, tissue: np.ndarray) -> pv.PolyData:
    """Input SAX rings as closed hairline polylines."""
    loops = []
    for label in (0.0, 1.0):
        points = contours[np.abs(tissue - label) < 0.5]
        for height in np.unique(points[:, 2]):
            ring = points[points[:, 2] == height]
            if len(ring) < 3:
                continue
            loops.append(pv.lines_from_points(np.vstack([ring, ring[:1]])))
    return pv.merge(loops)


def contour_distance(surface: pv.PolyData, contours: np.ndarray,
                     tissue: np.ndarray, wall: str) -> tuple[float, float]:
    """Exact point-to-surface distance from the input rings, via a cell locator."""
    label = 0.0 if wall == "endo" else 1.0
    rings = contours[np.abs(tissue - label) < 0.5]
    _, closest = surface.find_closest_cell(rings, return_closest_point=True)
    distances = np.linalg.norm(rings - closest, axis=1)
    return float(np.mean(distances)), float(np.percentile(distances, 95))


def render_panels(surfaces: dict, rings: dict, layout: list, up: tuple,
                  offset: np.ndarray) -> tuple[dict, float]:
    """One off-screen render per panel under a single shared camera.

    Returns the panel images and the resolution in pixels per millimetre, which
    is constant because the projection is parallel and the camera never moves.
    """
    bounds = np.array([surface.bounds for surface in surfaces.values()])
    frame = pv.Box(bounds=(bounds[:, 0].min(), bounds[:, 1].max(),
                           bounds[:, 2].min(), bounds[:, 3].max(),
                           bounds[:, 4].min(), bounds[:, 5].max()))

    plotter = pv.Plotter(off_screen=True, window_size=list(PANEL_PIXELS),
                         border=False, lighting="light_kit")
    plotter.set_background("white")
    plotter.add_mesh(frame, opacity=0.0, show_scalar_bar=False)
    plotter.enable_parallel_projection()
    plotter.view_vector(tuple(offset), viewup=up)
    plotter.reset_camera()
    plotter.camera.zoom(1.04)
    plotter.enable_anti_aliasing("ssaa")
    pixels_per_mm = PANEL_PIXELS[1] / (2.0 * plotter.camera.parallel_scale)

    images: dict[str, np.ndarray] = {}
    for key, source, _ in layout:
        actors = []
        if source is not None:
            method, phase = source
            actors.append(plotter.add_mesh(
                surfaces[(method, phase, "epi")], color=EPI_COLOUR, opacity=0.30,
                smooth_shading=True, specular=0.15, diffuse=0.9,
                reset_camera=False))
            actors.append(plotter.add_mesh(
                surfaces[(method, phase, "endo")], color=ENDO_COLOUR,
                smooth_shading=True, specular=0.25, specular_power=25,
                diffuse=0.9, ambient=0.18, reset_camera=False))
        phase = source[1] if source is not None else "ES"
        actors.append(plotter.add_mesh(rings[phase], color=CONTOUR_COLOUR,
                                       line_width=CONTOUR_WIDTH, opacity=0.85,
                                       reset_camera=False))
        images[key] = np.asarray(plotter.screenshot(return_img=True))
        for actor in actors:
            plotter.remove_actor(actor, reset_camera=False)

    renderer = next((line.split(":", 1)[1].strip()
                     for line in plotter.ren_win.ReportCapabilities().splitlines()
                     if line.startswith("OpenGL renderer string")), "unknown")
    plotter.close()
    print(f"rendered with OpenGL device: {renderer}")
    return images, pixels_per_mm


def common_crop(images: dict, pad: int = 12) -> dict:
    """Trim the shared white margin, keeping one crop window for every panel."""
    ink = np.zeros(next(iter(images.values())).shape[:2], dtype=bool)
    for image in images.values():
        ink |= image.min(axis=2) < 245
    rows, columns = np.where(ink)
    top, bottom = max(rows.min() - pad, 0), min(rows.max() + pad + 1, ink.shape[0])
    left, right = max(columns.min() - pad, 0), min(columns.max() + pad + 1, ink.shape[1])
    return {key: image[top:bottom, left:right] for key, image in images.items()}


def compose(images: dict, layout: list, pixels_per_mm: float) -> None:
    """Assemble the panels into the thesis figure with shared typography."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "savefig.facecolor": "white",
    })
    height, width = next(iter(images.values())).shape[:2]
    columns = ("Proposed model", "RBF implicit fit", "Shape-model fit")
    rows = ("End-diastole", "End-systole")

    figure, axes = plt.subplots(2, 3, figsize=(6.5, 6.9 * 2 * height /
                                               (3 * width)))
    for (key, source, letter), axis in zip(layout, axes.ravel()):
        axis.imshow(images[key], interpolation="lanczos")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.text(0.02, 0.98, letter, transform=axis.transAxes, va="top",
                  ha="left", fontsize=9)
        if source is None:
            axis.text(0.5, 0.06, "no end-systolic modes\nin the shape model",
                      transform=axis.transAxes, ha="center", va="bottom",
                      fontsize=8, color="#5c677d")

    for axis, title in zip(axes[0], columns):
        axis.set_title(title, fontsize=9.5, pad=6)
    for axis, label in zip(axes[:, 0], rows):
        axis.set_ylabel(label, fontsize=9.5, labelpad=6)

    bar = SCALE_BAR_MM * pixels_per_mm
    base = axes[1, 0]
    y = height * 0.94
    x = width * 0.06
    base.plot([x, x + bar], [y, y], color="black", linewidth=1.4,
              solid_capstyle="butt")
    base.text(x + bar / 2, y - height * 0.015, f"{SCALE_BAR_MM:.0f} mm",
              ha="center", va="bottom", fontsize=8)

    handles = [
        Patch(facecolor=ENDO_COLOUR, edgecolor="none", label="Endocardium"),
        Patch(facecolor=EPI_COLOUR, edgecolor="none", label="Epicardium"),
        Line2D([], [], color=CONTOUR_COLOUR, linewidth=1.1,
               label="Input SAX contours"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                  fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    figure.subplots_adjust(left=0.055, right=0.99, top=0.945, bottom=0.06,
                           wspace=0.10, hspace=0.13)
    figure.savefig(OUT, dpi=400)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="Recompute the cached RBF and shape-model fits.")
    args = parser.parse_args()

    contours, tissue, model = load_demo()
    meshes: dict[tuple[str, str, str], trimesh.Trimesh] = {
        ("model", phase, wall): mesh for (phase, wall), mesh in model.items()
    }
    meshes.update(build_baselines(contours, tissue, args.refresh))

    print(f"\n{PATIENT}: surface diagnostics before trimming and display smoothing")
    for key in sorted(meshes):
        method, phase, wall = key
        mesh = meshes[key]
        mean, p95 = contour_distance(_polydata(mesh.vertices, mesh.faces),
                                     contours[phase], tissue[phase], wall)
        print(f"  {method:<5} {phase} {wall:<4} "
              f"watertight={str(mesh.is_watertight):<5} "
              f"contour distance {mean:5.2f} mm mean / {p95:5.2f} mm p95")

    limits = {phase: (float(points[:, 2].min()), float(points[:, 2].max()))
              for phase, points in contours.items()}
    meshes = {
        key: trim_to_observed(
            mesh, *limits[key[1]],
            inset=ENDO_CUT_INSET_MM if key[2] == "endo" else 0.0)
        for key, mesh in meshes.items()
    }
    print(f"surfaces held to the observed slices +/- {AXIAL_MARGIN_MM} mm: "
          + ", ".join(f"{phase} {low:.0f}--{high:.0f} mm"
                      for phase, (low, high) in limits.items()))

    surfaces = display_surfaces(meshes, args.refresh)
    rings = {phase: contour_rings(contours[phase], tissue[phase])
             for phase in ("ED", "ES")}

    up = (0.0, 0.0, -1.0) if apex_towards_positive_z(model[("ED", "endo")]) \
        else (0.0, 0.0, 1.0)
    # Tilted off the slice planes so the input rings read as ellipses rather
    # than projecting edge-on to straight segments.
    offset = np.array([1.6, -1.6, 1.05 * up[2]], dtype=float)
    offset /= np.linalg.norm(offset)

    layout = [
        ("model_ED", ("model", "ED"), "(a)"),
        ("rbf_ED", ("rbf", "ED"), "(b)"),
        ("ssm_ED", ("ssm", "ED"), "(c)"),
        ("model_ES", ("model", "ES"), "(d)"),
        ("rbf_ES", ("rbf", "ES"), "(e)"),
        ("ssm_ES", None, "(f)"),
    ]

    images, pixels_per_mm = render_panels(surfaces, rings, layout, up, offset)
    images = common_crop(images)
    compose(images, layout, pixels_per_mm)

    print(f"\nwrote {OUT.relative_to(THESIS)} from {PATIENT}; "
          f"apex rendered downwards, {pixels_per_mm:.2f} px/mm")


if __name__ == "__main__":
    main()
