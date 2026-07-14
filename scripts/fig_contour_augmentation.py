#!/usr/bin/env python3
"""Visualise contour-input augmentation on the ACDC patient002 ED stack.

The figure deliberately derives every panel from the same labelled SAX stack.
Panels (b)--(e) show separate representative augmentation draws; panel (f)
shows the derived 3D supervision target that remains fixed for all of them.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import trimesh
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours, marching_cubes


ROOT = Path(__file__).resolve().parents[1]
SEG_PATH = ROOT / "notebooks/patient002/patient002_frame01_gt.nii/DCM04-OH-AL_V2_1.nii"
OUT_PATH = ROOT / "images/fig_contour_augmentation.png"

C_ENDO = "#0072B2"
C_EPI = "#D55E00"
C_GREY = "#7A7A7A"
VISUAL_Z_BOOST = 8.0


def load_segmentation() -> tuple[np.ndarray, tuple[float, float, float]]:
    image = nib.as_closest_canonical(nib.load(str(SEG_PATH)))
    return image.get_fdata().astype(np.int16), tuple(float(v) for v in image.header.get_zooms()[:3])


def resample_closed_curve(curve: np.ndarray, n_points: int = 60) -> np.ndarray:
    """Uniformly resample a closed 2D curve by arc length."""
    curve = np.asarray(curve, dtype=np.float64)
    closed = np.vstack([curve, curve[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    sample_at = np.linspace(0.0, cumulative[-1], n_points, endpoint=False)
    out = np.empty((n_points, 2), dtype=np.float64)
    for dim in range(2):
        out[:, dim] = np.interp(sample_at, cumulative, closed[:, dim])
    return out


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    candidates = find_contours(mask.astype(np.float32), level=0.5)
    if not candidates:
        return None
    return max(candidates, key=len)


def contour_stack(seg: np.ndarray, spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return normalised xyz+tissue contour points and their slice ids."""
    epi = np.isin(seg, [2, 3])
    valid_z = np.flatnonzero(epi.any(axis=(0, 1)))
    picks = valid_z[np.linspace(0, len(valid_z) - 1, 10).round().astype(int)]
    points, slice_ids = [], []
    for sid, z in enumerate(picks):
        for tissue, mask in ((0, seg[:, :, z] == 3), (1, epi[:, :, z])):
            curve = largest_contour(mask)
            if curve is None:
                continue
            xy = resample_closed_curve(curve, n_points=60)
            xyz = np.column_stack([
                xy[:, 1] * spacing[1],
                xy[:, 0] * spacing[0],
                np.full(len(xy), z * spacing[2]),
            ])
            points.append(np.column_stack([xyz, np.full(len(xyz), tissue)]))
            slice_ids.append(np.full(len(xyz), sid, dtype=np.int16))

    contour = np.vstack(points).astype(np.float64)
    slice_ids = np.concatenate(slice_ids)
    contour[:, :3] -= contour[:, :3].mean(axis=0, keepdims=True)
    contour[:, 2] *= -1.0
    scale = np.max(np.ptp(contour[:, :3], axis=0))
    contour[:, :3] /= max(scale, 1e-8)
    return contour, slice_ids


def augment(contour: np.ndarray, slice_ids: np.ndarray, operation: str) -> np.ndarray:
    """Representative, fixed-seed draws of the training-time operations."""
    out = contour.copy()
    rng = np.random.default_rng(20260714)

    if operation == "translation_jitter":
        for sid in np.unique(slice_ids):
            mask = slice_ids == sid
            # Same translation for endo and epi on a given SAX slice.
            out[mask, :2] += rng.normal(0.0, 0.10, size=2)
        out[:, :2] += rng.normal(0.0, 0.025, size=(len(out), 2))
    elif operation == "slice_dropout":
        # Two dropped slice levels make the operation readily visible. This is
        # within the one-to-two levels sampled by the loader for a 10-level
        # contour stack and retains more than the training minimum of three.
        drop = np.isin(slice_ids, [2, 7])
        out = out[~drop]
    elif operation == "rotation_scale":
        theta = np.deg2rad(12.0)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        centre = out[:, :2].mean(axis=0, keepdims=True)
        out[:, :2] = (out[:, :2] - centre) @ rotation.T + centre
        out[:, :3] *= 1.06
    elif operation == "point_dropout":
        keep = rng.random(len(out)) > 0.30
        out = out[keep]
    else:
        raise ValueError(f"Unknown operation: {operation}")
    return out


def style_3d(ax: plt.Axes) -> None:
    ax.view_init(elev=17, azim=-61)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlim(-0.72, 0.72)
    ax.set_ylim(-0.72, 0.72)
    ax.set_zlim(-0.46, 0.46)
    ax.set_axis_off()
    ax.set_facecolor("white")


def draw_contours(ax: plt.Axes, contour: np.ndarray) -> None:
    contour = contour.copy()
    # The source stack has 1 mm slice spacing and a roughly 100 mm in-plane
    # field of view. Exaggerating only the display z coordinate makes the
    # clinically important slice sparsity legible without altering the data.
    contour[:, 2] *= VISUAL_Z_BOOST
    for tissue, colour in ((0, C_ENDO), (1, C_EPI)):
        points = contour[contour[:, 3] == tissue]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4.5,
                   c=colour, alpha=0.82, depthshade=False, linewidths=0)
    style_3d(ax)


def smoothed_mesh(mask: np.ndarray, spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    sigma = tuple(1.5 / value for value in spacing)
    field = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    vertices, faces, _, _ = marching_cubes(field, level=0.5, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.53, nu=-0.55, iterations=25)
    return mesh.vertices, mesh.faces


def draw_target(ax: plt.Axes, seg: np.ndarray, spacing: tuple[float, float, float]) -> None:
    all_vertices = []
    meshes = []
    for mask, colour, alpha in ((seg == 3, C_ENDO, 0.92), (np.isin(seg, [2, 3]), C_EPI, 0.23)):
        vertices, faces = smoothed_mesh(mask, spacing)
        all_vertices.append(vertices)
        meshes.append((vertices, faces, colour, alpha))
    centre = np.vstack(all_vertices).mean(axis=0)
    scale = max(np.ptp(np.vstack(all_vertices), axis=0))
    for vertices, faces, colour, alpha in meshes:
        vertices = (vertices - centre) / scale
        vertices[:, 2] *= -VISUAL_Z_BOOST
        face_colour = (*to_rgb(colour), alpha)
        ax.add_collection3d(Poly3DCollection(vertices[faces], facecolors=face_colour,
                                              edgecolors="none", linewidths=0.0))
    style_3d(ax)


def label_panel(ax: plt.Axes, label: str, title: str, subtitle: str) -> None:
    ax.set_title(f"{label} {title}", loc="left", fontsize=10.5, fontweight="bold", pad=5)
    ax.text2D(0.0, -0.11, subtitle, transform=ax.transAxes, fontsize=8.2,
              color="#454545", ha="left", va="top", linespacing=1.25)


def main() -> None:
    seg, spacing = load_segmentation()
    original, slice_ids = contour_stack(seg, spacing)
    panels = [
        ("(a)", "Original contour stack", "Sparse SAX observation", original),
        ("(b)", "Translation + jitter", "Independent in-plane shift per slice\nand point-wise in-plane noise", augment(original, slice_ids, "translation_jitter")),
        ("(c)", "Slice dropout", "Two SAX levels omitted\n(at least three levels are retained)", augment(original, slice_ids, "slice_dropout")),
        ("(d)", "Rotation + scale", "Long-axis rotation and\nglobal scale perturbation", augment(original, slice_ids, "rotation_scale")),
        ("(e)", "Point dropout", "Thirty percent of contour\npoints omitted", augment(original, slice_ids, "point_dropout")),
    ]

    fig = plt.figure(figsize=(16.0, 4.8), dpi=300, facecolor="white")
    grid = fig.add_gridspec(1, 6, left=0.035, right=0.99, bottom=0.20, top=0.78, wspace=0.16)
    for index, (label, title, subtitle, points) in enumerate(panels):
        ax = fig.add_subplot(grid[0, index], projection="3d")
        draw_contours(ax, points)
        label_panel(ax, label, title, subtitle)

    target_ax = fig.add_subplot(grid[0, 5], projection="3d")
    draw_target(target_ax, seg, spacing)
    label_panel(target_ax, "(f)", "Fixed 3D supervision", "Derived mesh, sampled surfaces,\nand query targets are unchanged")

    fig.text(0.035, 0.925, "Contour-input augmentation during CardioSDF training",
             fontsize=15, fontweight="bold", color="#202020", ha="left")
    fig.text(0.035, 0.875,
             "Every training stream receives a representative perturbation of its encoder observation; the paired target in panel (f) remains fixed.",
             fontsize=9.5, color="#454545", ha="left")
    fig.text(0.035, 0.055, "● Endocardium", color=C_ENDO, fontsize=9.5, fontweight="bold")
    fig.text(0.145, 0.055, "● Epicardium", color=C_EPI, fontsize=9.5, fontweight="bold")
    fig.text(0.292, 0.055, "Display z is exaggerated to make SAX slice spacing visible; all panels use the same patient002 ED case.",
             color=C_GREY, fontsize=8.8)

    fig.savefig(OUT_PATH, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
