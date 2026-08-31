"""Slice-count ablation: model, RBF implicit fit and shape-model fit.

The end-diastolic SAX stack of ``patient002`` carries ten annotated slices. The
stack is progressively decimated, always keeping the most basal and the most
apical slice so that every reconstruction has the same axial support, and each
of the three reconstruction methods is re-run on the surviving rings alone.

Rendering follows the same template as ``fig_recon_ed_es_v2.py``: one shared
camera, a parallel projection, a common bounding frame, an apex-down
orientation derived from the geometry, and one display-only Taubin pass applied
identically to every surface. No mesh is snapped or deformed towards the rings.
The grid runs methods across and slice count down the page. Each panel is
annotated with the mean endocardial contour distance and the enclosed
endocardial volume, both measured on the untrimmed and unsmoothed mesh.

Output: ``images/results_slice_ablation.png``.
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
EVAL_DIR = THESIS / "scripts" / "eval_demo"
DEMO_OUT = EVAL_DIR / "outputs"
CACHE = DEMO_OUT / "slice_ablation.npz"
OUT = THESIS / "images" / "results_slice_ablation.png"
PATIENT_DIR = THESIS / "notebooks" / "patient002"
MODEL_PATH = THESIS / "notebooks" / "inr_sdf_combined_fresh_ed_mix_v1_final.pt"
PHASE = "ED"

sys.path.insert(0, str(THESIS / "scripts"))
sys.path.insert(0, str(EVAL_DIR))

from cardiosdf_model import load_model  # noqa: E402
from fig_baseline_rbf_ssm import (  # noqa: E402
    build_rbf_geometry,
    build_ssm_geometry,
)
from geometry import (  # noqa: E402
    Segmentation,
    build_model_geometry,
    extract_contours,
    load_segmentation,
    read_info_cfg,
)
from recon_metrics import overlap_metrics, surface_metrics  # noqa: E402

METHODS = (("model", "Proposed model"),
           ("rbf", "RBF fit"),
           ("ssm", "SSM fit"))
SLICE_COUNTS = (10, 6, 4, 3)

ENDO_COLOUR = "#b3313c"
EPI_COLOUR = "#c8ccd1"
CONTOUR_COLOUR = "#22333b"
CONTOUR_WIDTH = 1.1

SMOOTH_ITERATIONS = 600
SMOOTH_PASS_BAND = 0.003
SCALE_BAR_MM = 20.0
AXIAL_MARGIN_MM = 1.5
ENDO_CUT_INSET_MM = 1.0
PANEL_PIXELS = (700, 1000)
GRID_RES = 96

pv.OFF_SCREEN = True
pv.global_theme.font.family = "times"


def _polydata(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    cells = np.hstack([
        np.full((len(faces), 1), 3, dtype=np.int64),
        np.asarray(faces, dtype=np.int64),
    ]).ravel()
    return pv.PolyData(np.asarray(vertices, dtype=np.float32), cells)


def slice_subsets(available: np.ndarray) -> dict[int, np.ndarray]:
    """Evenly decimated slice indices, always retaining base and apex."""
    subsets: dict[int, np.ndarray] = {}
    for count in SLICE_COUNTS:
        take = np.unique(np.round(
            np.linspace(0, len(available) - 1, count)).astype(int))
        subsets[count] = available[take]
    return subsets


def restrict(seg: Segmentation, keep: np.ndarray) -> Segmentation:
    """Copy of the segmentation with every non-retained slice blanked."""
    labels = np.zeros_like(seg.labels)
    labels[:, :, keep] = seg.labels[:, :, keep]
    return Segmentation(labels, seg.spacing, seg.path)


def reconstruct(refresh: bool) -> tuple[dict, dict]:
    """Contour rings and the three reconstructions for every slice count."""
    info = read_info_cfg(PATIENT_DIR / "Info.cfg")
    frame = int(info[PHASE])
    seg = load_segmentation(
        PATIENT_DIR / f"{PATIENT_DIR.name}_frame{frame:02d}_gt.nii")
    full = extract_contours(seg)
    subsets = slice_subsets(full["slices"])

    if CACHE.exists() and not refresh:
        with np.load(CACHE) as data:
            wanted = {f"{method}_{count}_{wall}_v"
                      for method, _ in METHODS for count in SLICE_COUNTS
                      for wall in ("endo", "epi")}
            if wanted.issubset(set(data.files)):
                print(f"using cached reconstructions: {CACHE.relative_to(THESIS)}")
                rings = {count: (data[f"rings_{count}_xyz"],
                                 data[f"rings_{count}_tissue"])
                         for count in SLICE_COUNTS}
                meshes = {
                    (method, count, wall): trimesh.Trimesh(
                        data[f"{method}_{count}_{wall}_v"],
                        data[f"{method}_{count}_{wall}_f"], process=False)
                    for method, _ in METHODS for count in SLICE_COUNTS
                    for wall in ("endo", "epi")}
                return rings, meshes

    net, cfg, meta = load_model(MODEL_PATH)
    print(f"checkpoint epoch={meta['epoch']} val_loss={meta['val_loss']:.4f}")

    rings: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    meshes: dict[tuple[str, int, str], trimesh.Trimesh] = {}
    payload: dict[str, np.ndarray] = {}

    for count, keep in subsets.items():
        contours = extract_contours(restrict(seg, keep))
        xyz_mm = np.asarray(contours["xyz_mm"], dtype=np.float64)
        tissue = np.asarray(contours["tissue"], dtype=np.float64)
        rings[count] = (xyz_mm, tissue)
        payload[f"rings_{count}_xyz"] = xyz_mm.astype(np.float32)
        payload[f"rings_{count}_tissue"] = tissue.astype(np.float32)
        print(f"\n{count} slices (indices {list(keep)}): "
              f"{len(xyz_mm)} contour points")

        for method, _ in METHODS:
            start = time.perf_counter()
            if method == "model":
                geometry = build_model_geometry(net, cfg, contours,
                                                grid_res=GRID_RES,
                                                phase_val=0.0)
            elif method == "rbf":
                geometry = build_rbf_geometry(xyz_mm, tissue)
            else:
                geometry = build_ssm_geometry(xyz_mm, tissue, PHASE)
            print(f"  {method:<5} fitted in {time.perf_counter() - start:5.1f} s")
            for wall in ("endo", "epi"):
                mesh = geometry[wall]
                meshes[(method, count, wall)] = mesh
                payload[f"{method}_{count}_{wall}_v"] = np.asarray(
                    mesh.vertices, np.float32)
                payload[f"{method}_{count}_{wall}_f"] = np.asarray(
                    mesh.faces, np.int64)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **payload)
    print(f"\ncached reconstructions: {CACHE.relative_to(THESIS)}")
    return rings, meshes


def trim_to_observed(mesh: trimesh.Trimesh, low: float, high: float,
                     inset: float = 0.0) -> trimesh.Trimesh:
    """Clip to the full observed slice range, identical for every panel."""
    clipped = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=[0.0, 0.0, 1.0],
        plane_origin=[0.0, 0.0, low - AXIAL_MARGIN_MM + inset], cap=True)
    return trimesh.intersections.slice_mesh_plane(
        clipped, plane_normal=[0.0, 0.0, -1.0],
        plane_origin=[0.0, 0.0, high + AXIAL_MARGIN_MM - inset], cap=True)


def apex_towards_positive_z(mesh: trimesh.Trimesh) -> bool:
    """True when the cross-section shrinks at the high-z end of the stack."""
    points = np.asarray(mesh.vertices, dtype=np.float64)
    z = points[:, 2]

    def radius(subset: np.ndarray) -> float:
        plane = subset[:, :2] - subset[:, :2].mean(axis=0)
        return float(np.linalg.norm(plane, axis=1).mean())

    return radius(points[z > np.percentile(z, 75)]) < \
        radius(points[z < np.percentile(z, 25)])


def display_surfaces(meshes: dict) -> dict[tuple, pv.PolyData]:
    """Taubin-smoothed copies used only for rendering."""
    surfaces: dict[tuple, pv.PolyData] = {}
    shifts: list[float] = []
    for key, mesh in meshes.items():
        raw = _polydata(mesh.vertices, mesh.faces)
        smoothed = raw.smooth_taubin(n_iter=SMOOTH_ITERATIONS,
                                     pass_band=SMOOTH_PASS_BAND,
                                     normalize_coordinates=True)
        shifts.append(float(np.linalg.norm(smoothed.points - raw.points,
                                           axis=1).mean()))
        surfaces[key] = smoothed
    print(f"display smoothing: Taubin {SMOOTH_ITERATIONS} iterations, pass band "
          f"{SMOOTH_PASS_BAND}, mean vertex displacement {np.mean(shifts):.2f} mm "
          f"(max {np.max(shifts):.2f} mm)")
    return surfaces


def contour_rings(points: np.ndarray, tissue: np.ndarray) -> pv.PolyData:
    """Input SAX rings as closed hairline polylines."""
    loops = []
    for label in (0.0, 1.0):
        selected = points[np.abs(tissue - label) < 0.5]
        for height in np.unique(selected[:, 2]):
            ring = selected[selected[:, 2] == height]
            if len(ring) < 3:
                continue
            loops.append(pv.lines_from_points(np.vstack([ring, ring[:1]])))
    return pv.merge(loops)


def contour_distance(surface: pv.PolyData, points: np.ndarray,
                     tissue: np.ndarray, wall: str) -> tuple[float, float]:
    """Exact point-to-surface distance from the input rings."""
    label = 0.0 if wall == "endo" else 1.0
    selected = points[np.abs(tissue - label) < 0.5]
    _, closest = surface.find_closest_cell(selected, return_closest_point=True)
    distances = np.linalg.norm(selected - closest, axis=1)
    return float(np.mean(distances)), float(np.percentile(distances, 95))


def render_panels(surfaces: dict, rings: dict, up: tuple,
                  offset: np.ndarray) -> tuple[dict, float]:
    """One off-screen render per panel under a single shared camera."""
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

    images: dict[tuple[str, int], np.ndarray] = {}
    for method, _ in METHODS:
        for count in SLICE_COUNTS:
            actors = [
                plotter.add_mesh(surfaces[(method, count, "epi")],
                                 color=EPI_COLOUR, opacity=0.30,
                                 smooth_shading=True, specular=0.15,
                                 diffuse=0.9, reset_camera=False),
                plotter.add_mesh(surfaces[(method, count, "endo")],
                                 color=ENDO_COLOUR, smooth_shading=True,
                                 specular=0.25, specular_power=25, diffuse=0.9,
                                 ambient=0.18, reset_camera=False),
                plotter.add_mesh(rings[count], color=CONTOUR_COLOUR,
                                 line_width=CONTOUR_WIDTH, opacity=0.85,
                                 reset_camera=False),
            ]
            images[(method, count)] = np.asarray(
                plotter.screenshot(return_img=True))
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


def compose(images: dict, pixels_per_mm: float, annotations: dict) -> None:
    """Assemble the panels into a portrait grid, slice count down the page."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "savefig.facecolor": "white",
    })
    height, width = next(iter(images.values())).shape[:2]
    letters = iter("abcdefghijkl")

    figure_width = 5.2
    figure, axes = plt.subplots(
        len(SLICE_COUNTS), len(METHODS),
        figsize=(figure_width,
                 1.055 * figure_width * len(SLICE_COUNTS) * height
                 / (len(METHODS) * width)))
    for row, count in enumerate(SLICE_COUNTS):
        for column, (method, _) in enumerate(METHODS):
            axis = axes[row, column]
            axis.imshow(images[(method, count)], interpolation="lanczos")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_edgecolor("#9aa3ad")
                spine.set_linewidth(0.5)
            axis.text(0.035, 0.965, next(letters), transform=axis.transAxes,
                      va="top", ha="left", fontsize=8, fontweight="bold")
            distance, volume = annotations[(method, count)]
            axis.text(0.965, 0.035,
                      f"$d$ = {distance:.2f} mm\n$V$ = {volume:.0f} ml",
                      transform=axis.transAxes, va="bottom", ha="right",
                      fontsize=6.6, linespacing=1.25, color="#33393f")

    for axis, (_, label) in zip(axes[0], METHODS):
        axis.set_title(label, fontsize=8.5, pad=4)
    for axis, count in zip(axes[:, 0], SLICE_COUNTS):
        axis.set_ylabel(f"{count} slices", fontsize=8.5, labelpad=4)

    bar = SCALE_BAR_MM * pixels_per_mm
    base = axes[-1, 0]
    y = height * 0.955
    x = width * 0.05
    base.plot([x, x + bar], [y, y], color="black", linewidth=1.3,
              solid_capstyle="butt")
    base.text(x + bar / 2, y - height * 0.012, f"{SCALE_BAR_MM:.0f} mm",
              ha="center", va="bottom", fontsize=6.6)

    handles = [
        Patch(facecolor=ENDO_COLOUR, edgecolor="none", label="Endocardium"),
        Patch(facecolor=EPI_COLOUR, edgecolor="none", label="Epicardium"),
        Line2D([], [], color=CONTOUR_COLOUR, linewidth=1.1,
               label="Retained SAX contours"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                  fontsize=7.5, handlelength=1.6, columnspacing=1.6,
                  bbox_to_anchor=(0.5, 0.0))
    figure.subplots_adjust(left=0.062, right=0.995, top=0.972, bottom=0.030,
                           wspace=0.045, hspace=0.055)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=400)
    plt.close(figure)


def report(meshes: dict, rings: dict) -> dict[tuple[str, int], tuple[float, float]]:
    """Print the per-panel diagnostics and return the endocardial annotations."""
    print("\ncontour fidelity and enclosed volume, before trimming and smoothing")
    header = f"  {'method':<6}{'slices':>7}{'wall':>6}{'watertight':>12}" \
             f"{'dist (mm)':>11}{'p95 (mm)':>10}{'volume (ml)':>13}"
    print(header)
    annotations: dict[tuple[str, int], tuple[float, float]] = {}
    for method, _ in METHODS:
        for count in SLICE_COUNTS:
            for wall in ("endo", "epi"):
                mesh = meshes[(method, count, wall)]
                mean, p95 = contour_distance(
                    _polydata(mesh.vertices, mesh.faces), *rings[count], wall)
                volume = mesh.volume / 1000.0
                if wall == "endo":
                    annotations[(method, count)] = (mean, volume)
                print(f"  {method:<6}{count:>7}{wall:>6}"
                      f"{str(mesh.is_watertight):>12}{mean:>11.2f}{p95:>10.2f}"
                      f"{volume:>13.1f}")
    return annotations


def degradation(meshes: dict) -> None:
    """Score every decimated reconstruction against its own full-stack surface.

    No reference geometry exists, so degradation is measured as self-consistency:
    how far a method moves away from what it produced with the complete stack.
    """
    full = max(SLICE_COUNTS)
    rows: list[dict] = []
    for method, label in METHODS:
        reference = {wall: meshes[(method, full, wall)] for wall in ("endo", "epi")}
        reference_myo = abs(reference["epi"].volume) - abs(reference["endo"].volume)
        for count in SLICE_COUNTS:
            if count == full:
                continue
            candidate = {wall: meshes[(method, count, wall)]
                         for wall in ("endo", "epi")}
            endo = surface_metrics(candidate["endo"], reference["endo"])
            epi = surface_metrics(candidate["epi"], reference["epi"])
            overlap = overlap_metrics(candidate["endo"], candidate["epi"],
                                      reference["endo"], reference["epi"])
            myo = abs(candidate["epi"].volume) - abs(candidate["endo"].volume)
            rows.append({
                "method": label, "slices": count,
                "endo_chamfer_mm": endo["chamfer_mm"],
                "epi_chamfer_mm": epi["chamfer_mm"],
                "endo_hd95_mm": endo["hd95_mm"],
                "cavity_dice": overlap["endo_dice"],
                "myo_dice": overlap["myo_dice"],
                "myo_volume_change_pct": 100.0 * (myo - reference_myo) / reference_myo,
            })

    out = DEMO_OUT / "slice_ablation_degradation.csv"
    columns = list(rows[0])
    with out.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(
                f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                for c in columns) + "\n")

    print(f"\ndeviation from each method's own {full}-slice reconstruction")
    print(f"  {'method':<16}{'slices':>7}{'endo Ch':>9}{'epi Ch':>8}"
          f"{'endo HD95':>11}{'cav Dice':>10}{'myo Dice':>10}{'myo dV %':>10}")
    for row in rows:
        print(f"  {row['method']:<16}{row['slices']:>7}"
              f"{row['endo_chamfer_mm']:>9.2f}{row['epi_chamfer_mm']:>8.2f}"
              f"{row['endo_hd95_mm']:>11.2f}{row['cavity_dice']:>10.3f}"
              f"{row['myo_dice']:>10.3f}{row['myo_volume_change_pct']:>10.1f}")
    print(f"wrote {out.relative_to(THESIS)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="Recompute the cached reconstructions.")
    args = parser.parse_args()

    ring_data, meshes = reconstruct(args.refresh)
    rings_mm = {count: contour_rings(*ring_data[count]) for count in SLICE_COUNTS}
    annotations = report(meshes, ring_data)
    degradation(meshes)

    reference = ring_data[max(SLICE_COUNTS)][0]
    low, high = float(reference[:, 2].min()), float(reference[:, 2].max())
    print(f"\nsurfaces held to {low:.0f}--{high:.0f} mm "
          f"+/- {AXIAL_MARGIN_MM} mm in every panel")
    trimmed = {
        key: trim_to_observed(mesh, low, high,
                              inset=ENDO_CUT_INSET_MM if key[2] == "endo" else 0.0)
        for key, mesh in meshes.items()
    }

    surfaces = display_surfaces(trimmed)
    up = (0.0, 0.0, -1.0) if apex_towards_positive_z(
        meshes[("model", max(SLICE_COUNTS), "endo")]) else (0.0, 0.0, 1.0)
    offset = np.array([1.6, -1.6, 1.05 * up[2]], dtype=float)
    offset /= np.linalg.norm(offset)

    images, pixels_per_mm = render_panels(surfaces, rings_mm, up, offset)
    compose(common_crop(images), pixels_per_mm, annotations)
    print(f"wrote {OUT.relative_to(THESIS)}")


if __name__ == "__main__":
    main()
