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
# The NIfTI header reports 1 mm through-plane spacing, so the 10 acquired SAX
# levels span only ~9 mm and any mesh built from them collapses to a pancake.
# A realistic clinical SAX slice gap is applied consistently to both the
# contour stacks and the reconstructed mesh so every panel shows true LV
# long-axis proportions at the same scale.
SLICE_MM = 8.0


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
                np.full(len(xy), float(sid) * SLICE_MM),
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
    ax.set_box_aspect((1.0, 1.0, 1.0), zoom=0.86)
    ax.set_xlim(-0.28, 0.28)
    ax.set_ylim(-0.28, 0.28)
    ax.set_zlim(-0.28, 0.28)
    ax.set_axis_off()
    ax.set_facecolor("white")


def draw_contours(ax: plt.Axes, contour: np.ndarray) -> None:
    contour = contour.copy()
    contour[:, :3] *= 0.72
    for tissue, colour in ((0, C_ENDO), (1, C_EPI)):
        points = contour[contour[:, 3] == tissue]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=3.2,
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
    # Reconstruct the mesh with the same realistic slice spacing used for the
    # contour stacks so panel (f) matches panels (a)--(e) in scale and shape.
    mesh_spacing = (spacing[0], spacing[1], SLICE_MM)
    # Endocardium drawn as an opaque inner surface, epicardium as a translucent
    # outer shell so the myocardial wall is visible.
    for mask, colour, alpha in ((seg == 3, C_ENDO, 1.0), (np.isin(seg, [2, 3]), C_EPI, 0.20)):
        vertices, faces = smoothed_mesh(mask, mesh_spacing)
        all_vertices.append(vertices)
        meshes.append((vertices, faces, colour, alpha))
    stacked = np.vstack(all_vertices)
    centre = stacked.mean(axis=0)
    # Same normalisation as the contour panels: centre and divide by max extent.
    scale = max(np.ptp(stacked, axis=0))
    light = np.array([0.3, 0.4, 0.85])
    light = light / np.linalg.norm(light)
    for vertices, faces, colour, alpha in meshes:
        vertices = (vertices - centre) / max(scale, 1e-8)
        vertices *= 0.72
        # Match the contour treatment: flip the long axis only.
        vertices[:, 2] *= -1.0
        tris = vertices[faces]
        base_rgb = np.array(to_rgb(colour))
        if alpha >= 0.99:
            # Flat shading for the opaque endocardium to convey 3D curvature.
            normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.clip(norms, 1e-8, None)
            shade = np.clip(np.abs(normals @ light), 0.0, 1.0)
            shade = 0.45 + 0.55 * shade
            face_colours = np.clip(base_rgb[None, :] * shade[:, None], 0.0, 1.0)
            face_colours = np.column_stack([face_colours, np.full(len(tris), alpha)])
            collection = Poly3DCollection(tris, facecolors=face_colours,
                                          edgecolors="none", linewidths=0.0)
        else:
            face_colour = (*base_rgb, alpha)
            collection = Poly3DCollection(tris, facecolors=face_colour,
                                          edgecolors=(*base_rgb, 0.35), linewidths=0.15)
        collection.set_sort_zpos(0.0 if alpha >= 0.99 else 1.0)
        ax.add_collection3d(collection)
    style_3d(ax)


def label_panel(ax: plt.Axes, label: str) -> None:
    ax.text2D(0.02, 0.02, label, transform=ax.transAxes, fontsize=11,
              fontweight="bold", color="#202020", ha="left", va="bottom")


def main() -> None:
    seg, spacing = load_segmentation()
    original, slice_ids = contour_stack(seg, spacing)
    panels = [
        ("(a)", original),
        ("(b)", augment(original, slice_ids, "translation_jitter")),
        ("(c)", augment(original, slice_ids, "slice_dropout")),
        ("(d)", augment(original, slice_ids, "rotation_scale")),
        ("(e)", augment(original, slice_ids, "point_dropout")),
    ]

    fig = plt.figure(figsize=(8.6, 5.8), dpi=700, facecolor="white")
    grid = fig.add_gridspec(2, 3, left=0.04, right=0.98, bottom=0.04, top=0.97,
                            wspace=0.30, hspace=0.24)
    for index, (label, points) in enumerate(panels):
        ax = fig.add_subplot(grid[divmod(index, 3)], projection="3d")
        draw_contours(ax, points)
        label_panel(ax, label)

    target_ax = fig.add_subplot(grid[1, 2], projection="3d")
    draw_target(target_ax, seg, spacing)
    label_panel(target_ax, "(f)")

    fig.savefig(OUT_PATH, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
