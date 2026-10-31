#!/usr/bin/env python3
"""
Real-data RBF mesh pipeline figure with separate endocardial and epicardial
branches.

Layout (landscape):

                                          ┌─→ (c)  endo contours → (d) RBF extraction → (e) quality controlled
    (a) SAX + contours → (b) binary masks ┤
                                          └─→ (c') epi contours → (d') RBF extraction → (e') quality controlled

The shared inputs (a, b) sit on the left, vertically centred; the pipeline
branches horizontally into two colour-coded rows (endocardium in blue,
epicardium in orange). Panel labels are placed BELOW each axis.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import nibabel as nib
from scipy.interpolate import RBFInterpolator
import trimesh

THESIS = Path(__file__).resolve().parents[1]
EVAL_DIR = THESIS / "scripts" / "eval_demo"
sys.path.insert(0, str(EVAL_DIR))

from geometry import (  # noqa: E402
    _clean_inside,
    marching_cubes_mesh,
    repair_if_invalid,
    signed_distance_from_mask,
)

# One ACDC case throughout, so the MRI slice and the contours are the same volume.
CASE_ID = "patient002"
CACHE_PATH = THESIS / "test-new-model/cache/patient002_ED.npz"
MRI_PATH = THESIS / "notebooks/patient002/patient002_frame01.nii/DCM04Gate1.nii"
MRI_SEG_PATH = THESIS / "notebooks/patient002/patient002_frame01_gt.nii/DCM04-OH-AL_V2_1.nii"
RBF_CACHE_PATH = THESIS / "scripts/eval_demo/outputs/fig_real_mesh_pipeline_rbf.npz"
RBF_PARAMETERS = {
    "offset_mm": 2.5,
    "pitch_mm": 1.0,
    "rbf_regularisation": 0.5,
    "field_smoothing_mm": 1.2,
}

# Okabe-Ito colorblind-safe palette
C_ENDO = "#0072B2"   # blue
C_EPI  = "#D55E00"   # orange
C_RV   = "#009E73"   # green
C_TEXT = "#2B2B2B"
C_MUTED = "#6B7280"
C_CARD = "#F4F5F7"
C_CARD_EDGE = "#DADDE2"

ELEV, AZIM = 18, -65
_e, _a = np.radians(ELEV), np.radians(AZIM)
EYE = np.array([np.cos(_e) * np.cos(_a), np.cos(_e) * np.sin(_a), np.sin(_e)])
EYE_H = np.array([EYE[0], EYE[1], 0.0]) / np.linalg.norm(EYE[:2])

plt.rcParams.update({"font.family": "DejaVu Sans"})


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _lighten(hex_color, amount):
    base = np.array(to_rgb(hex_color))
    if amount < 0:
        return base * (1.0 + amount)
    return base + (1.0 - base) * amount


def shaded_faces(verts, faces, base_hex, lighten=0.22):
    """Blinn-Phong face colours with outward normals and a soft rim light."""
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    fn = _unit(np.asarray(mesh.vertex_normals)[faces].mean(axis=1))
    outward = verts[faces].mean(axis=1) - verts.mean(axis=0)
    if np.mean(np.sum(fn * outward, axis=1)) < 0:
        fn = -fn  # the display z-flip mirrors the winding
    key = _unit(EYE + np.array([-0.45, 0.25, 0.55]))
    fill = _unit(np.array([0.7, 0.5, -0.1]))
    half = _unit(key + EYE)
    lambert = np.clip(fn @ key, 0, 1)
    fill_l = np.clip(fn @ fill, 0, 1)
    spec = np.clip(fn @ half, 0, 1) ** 18
    rim = (1.0 - np.abs(fn @ EYE)) ** 3
    base = _lighten(base_hex, lighten)
    col = base[None] * (0.42 + 0.50 * lambert + 0.12 * fill_l)[:, None]
    col += 0.12 * spec[:, None] + 0.10 * rim[:, None]
    return np.clip(col, 0, 1), fn


def set_3d_view(ax, verts):
    ax.view_init(elev=ELEV, azim=AZIM, roll=0)
    r = [np.ptp(verts[:, i]) for i in range(3)]
    rmax = max(r)
    ax.set_box_aspect((r[0] / rmax, r[1] / rmax, r[2] / rmax))
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.axis("off")
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)


def panel_label(ax, text, is_3d=False):
    """Place label text BELOW the axis."""
    y = -0.10 if is_3d else -0.08
    kw = dict(ha="center", va="top", fontsize=9.5, fontweight="bold", color=C_TEXT)
    if is_3d:
        ax.text2D(0.5, y, text, transform=ax.transAxes, **kw)
    else:
        ax.annotate(text, xy=(0.5, y), xycoords="axes fraction", **kw)


def add_figspace_arrow(fig, posA, posB, color, lw=1.6, rad=0.0):
    fig.add_artist(FancyArrowPatch(
        posA, posB,
        transform=fig.transFigure,
        arrowstyle='-|>', mutation_scale=14,
        color=color, linewidth=lw,
        connectionstyle=f'arc3,rad={rad}',
        zorder=12, capstyle='round',
    ))


# ── Panel content ────────────────────────────────────────────

def load_canonical_nifti(path):
    image = nib.as_closest_canonical(nib.load(path))
    return image.get_fdata()


def panel_mri_with_contours(ax, volume, segmentation):
    counts = np.count_nonzero(np.isin(segmentation, [2, 3]), axis=(0, 1))
    levels = np.flatnonzero(counts > 25)
    z = int(levels[len(levels) // 2])
    mask = np.isin(segmentation[:, :, z], [2, 3])
    rows, columns = np.where(mask)
    padding = 38
    row_min = max(0, int(rows.min()) - padding)
    row_max = min(volume.shape[0], int(rows.max()) + padding)
    col_min = max(0, int(columns.min()) - padding)
    col_max = min(volume.shape[1], int(columns.max()) + padding)

    image = volume[:, :, z].T
    lo, hi = np.percentile(image[col_min:col_max, row_min:row_max], [1, 99.5])
    plane = segmentation[:, :, z].T
    # The epicardium bounds cavity + myocardium; the myocardium label alone is an annulus.
    boundaries = (
        (plane == 3, C_ENDO),
        (np.isin(plane, [2, 3]), C_EPI),
        (plane == 1, C_RV),
    )
    ax.imshow(image, cmap="gray", origin="lower", aspect="equal", vmin=lo, vmax=hi)
    for label_mask, color in boundaries:
        if label_mask.any():
            ax.contour(label_mask, levels=[0.5], colors=["white"], linewidths=3.2,
                       origin="lower", alpha=0.55)
            ax.contour(label_mask, levels=[0.5], colors=[color], linewidths=1.9,
                       origin="lower")
    ax.set_xlim(row_min, row_max)
    ax.set_ylim(col_min, col_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(C_CARD_EDGE)
        spine.set_linewidth(1.0)
    patches = [
        mpatches.Patch(color=C_ENDO, label="Endocardium"),
        mpatches.Patch(color=C_EPI, label="Epicardium"),
        mpatches.Patch(color=C_RV, label="RV"),
    ]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.24),
              ncol=3, fontsize=7.5, framealpha=0.0, handlelength=1.0,
              columnspacing=0.8, handletextpad=0.4, labelcolor=C_TEXT)


def _ring_at(points, z):
    ring = points[np.isclose(points[:, 2], z)]
    c = ring.mean(axis=0)
    return ring[np.argsort(np.arctan2(ring[:, 1] - c[1], ring[:, 0] - c[0]))]


def panel_stacked_labels(ax, surface_points):
    levels = np.unique(surface_points["endo"][:, 2])
    selected = levels[np.linspace(0, len(levels) - 1, 3).round().astype(int)]
    span = np.ptp(np.vstack(list(surface_points.values()))[:, 1])
    names = ("Basal", "Mid-cavity", "Apical")
    offsets = [(1 - i) * span * 1.35 for i in range(3)]

    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=C_CARD, edgecolor=C_CARD_EDGE,
                                linewidth=1.0, zorder=0, clip_on=False))
    for name, z, off in zip(names, selected, offsets):
        endo = _ring_at(surface_points["endo"], z)
        epi = _ring_at(surface_points["epi"], z)
        c = np.vstack([endo, epi])[:, :2].mean(axis=0)
        e_xy = endo[:, :2] - c
        p_xy = epi[:, :2] - c
        e_xy[:, 1] += off
        p_xy[:, 1] += off
        ax.fill(p_xy[:, 0], p_xy[:, 1], color=_lighten(C_EPI, 0.55), zorder=2,
                linewidth=0)
        ax.fill(e_xy[:, 0], e_xy[:, 1], color=_lighten(C_ENDO, 0.55), zorder=3,
                linewidth=0)
        ax.plot(*np.vstack([p_xy, p_xy[:1]]).T, color=C_EPI, lw=1.8, zorder=4)
        ax.plot(*np.vstack([e_xy, e_xy[:1]]).T, color=C_ENDO, lw=1.8, zorder=4)
        ax.text(0, off + 0.62 * span, name, color=C_MUTED, fontsize=8,
                fontweight="bold", ha="center", va="bottom", zorder=5)
    ax.set_xlim(-0.85 * span, 0.85 * span)
    ax.set_ylim(offsets[-1] - 0.85 * span, offsets[0] + 0.95 * span)
    ax.set_aspect("equal")
    ax.axis("off")


def _rings_of(points):
    return [points[np.isclose(points[:, 2], z)] for z in np.unique(points[:, 2])]


def _ring_normals(ring):
    tangent = np.roll(ring, -1, axis=0) - np.roll(ring, 1, axis=0)
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0], np.zeros(len(ring))])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    radial = ring - ring.mean(axis=0)
    radial[:, 2] = 0.0
    flip = np.sign(np.sum(normal * radial, axis=1))
    flip[flip == 0] = 1.0
    return normal * flip[:, None]


def build_rbf_stages(points, surface, offset_mm=2.5, pitch=1.0,
                     smoothing=0.5, field_smoothing_mm=1.2):
    rings = [ring for ring in _rings_of(points) if len(ring) >= 3]
    if len(rings) < 2:
        raise ValueError(f"RBF fitting needs at least two {surface} rings.")

    on_surface = np.vstack(rings)
    normals = np.vstack([_ring_normals(ring) for ring in rings])
    centres = np.vstack([
        on_surface,
        on_surface + offset_mm * normals,
        on_surface - offset_mm * normals,
    ])
    values = np.concatenate([
        np.zeros(len(on_surface)),
        np.full(len(on_surface), offset_mm),
        np.full(len(on_surface), -offset_mm),
    ])
    interpolator = RBFInterpolator(
        centres,
        values,
        kernel="thin_plate_spline",
        degree=1,
        smoothing=smoothing,
    )

    lower = on_surface.min(axis=0) - np.array([8.0, 8.0, 1.5])
    upper = on_surface.max(axis=0) + np.array([8.0, 8.0, 1.5])
    shape = tuple(
        int(np.ceil((upper[axis] - lower[axis]) / pitch)) + 1
        for axis in range(3)
    )
    axes = [lower[axis] + np.arange(shape[axis]) * pitch for axis in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    field = np.empty(len(grid), dtype=np.float64)
    for start in range(0, len(grid), 50_000):
        field[start:start + 50_000] = interpolator(grid[start:start + 50_000])
    field = field.reshape(shape)

    inside = _clean_inside(field <= 0.0)
    redistanced = signed_distance_from_mask(
        inside,
        np.full(3, pitch),
        smooth_sigma=field_smoothing_mm / pitch,
    )
    extracted = marching_cubes_mesh(redistanced, lower, np.full(3, pitch))
    final, _ = repair_if_invalid(extracted, f"figure-rbf-{surface}")
    return extracted, final


def load_or_build_rbf_stages(surface_points, refresh=False):
    metadata = {
        "case_id": CASE_ID,
        "source_mtime_ns": CACHE_PATH.stat().st_mtime_ns,
        **RBF_PARAMETERS,
    }
    if RBF_CACHE_PATH.exists() and not refresh:
        with np.load(RBF_CACHE_PATH, allow_pickle=False) as cached:
            stored = json.loads(str(cached["metadata"]))
            if stored == metadata:
                print(f"  Using cached RBF surfaces: {RBF_CACHE_PATH.relative_to(THESIS)}")
                return {
                    surface: (
                        trimesh.Trimesh(
                            vertices=cached[f"{surface}_extracted_vertices"],
                            faces=cached[f"{surface}_extracted_faces"],
                            process=False,
                        ),
                        trimesh.Trimesh(
                            vertices=cached[f"{surface}_final_vertices"],
                            faces=cached[f"{surface}_final_faces"],
                            process=False,
                        ),
                    )
                    for surface in ("endo", "epi")
                }

    print("  Computing RBF surfaces ...")
    stages = {
        surface: build_rbf_stages(
            points,
            surface,
            offset_mm=RBF_PARAMETERS["offset_mm"],
            pitch=RBF_PARAMETERS["pitch_mm"],
            smoothing=RBF_PARAMETERS["rbf_regularisation"],
            field_smoothing_mm=RBF_PARAMETERS["field_smoothing_mm"],
        )
        for surface, points in surface_points.items()
    }
    RBF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": np.asarray(json.dumps(metadata, sort_keys=True))}
    for surface, (extracted, final) in stages.items():
        payload.update({
            f"{surface}_extracted_vertices": np.asarray(extracted.vertices),
            f"{surface}_extracted_faces": np.asarray(extracted.faces),
            f"{surface}_final_vertices": np.asarray(final.vertices),
            f"{surface}_final_faces": np.asarray(final.faces),
        })
    np.savez_compressed(RBF_CACHE_PATH, **payload)
    print(f"  Cached RBF surfaces: {RBF_CACHE_PATH.relative_to(THESIS)}")
    return stages


def _display_points(points, centre):
    displayed = np.asarray(points, dtype=np.float64).copy() - centre
    displayed[:, 2] *= -1.0
    return displayed


def draw_rings(ax, points, centre, color, back_alpha=0.28, lw=1.1, dots=True,
               front_z=3, back_z=1):
    """Closed contour rings; the half facing away from the viewer is faded."""
    rgb = np.array(to_rgb(color))
    front_segs, back_segs, front_pts, back_pts = [], [], [], []
    displayed = _display_points(points, centre)
    for z in np.unique(displayed[:, 2]):
        ring = _ring_at(displayed, z)
        depth = (ring - ring.mean(axis=0)) @ EYE_H
        for i, j in enumerate(np.roll(np.arange(len(ring)), -1)):
            target = front_segs if depth[i] + depth[j] >= 0 else back_segs
            target.append([ring[i], ring[j]])
        front_pts.append(ring[depth >= 0])
        back_pts.append(ring[depth < 0])
    ax.add_collection3d(Line3DCollection(back_segs, colors=[(*rgb, back_alpha)],
                                         linewidths=lw * 0.8, zorder=back_z))
    ax.add_collection3d(Line3DCollection(front_segs, colors=[(*rgb, 1.0)],
                                         linewidths=lw, zorder=front_z))
    if dots:
        fp, bp = np.vstack(front_pts), np.vstack(back_pts)
        ax.scatter(*bp.T, s=1.6, color=rgb, alpha=back_alpha, depthshade=False,
                   linewidths=0, zorder=back_z)
        ax.scatter(*fp.T, s=3.2, color=rgb, alpha=1.0, depthshade=False,
                   edgecolors="white", linewidths=0.25, zorder=front_z + 0.1)


def panel_sax_stack(ax, points, centre, color):
    ax.set_proj_type('ortho')
    draw_rings(ax, points, centre, color)
    set_3d_view(ax, _display_points(points, centre))


def panel_mesh(ax, mesh, centre, color, translucent=False, contour_points=None):
    ax.set_proj_type('ortho')
    verts = _display_points(mesh.vertices, centre)
    faces = np.asarray(mesh.faces)
    fc, fn = shaded_faces(verts, faces, color)
    if translucent:
        keep = fn @ EYE > 0  # front shell only, so overlapping layers do not muddy
        poly = Poly3DCollection(verts[faces[keep]], facecolors=fc[keep],
                                edgecolors="none", alpha=0.42, zorder=2)
    else:
        poly = Poly3DCollection(verts[faces], facecolors=fc, edgecolors="none",
                                alpha=1.0, zorder=2)
    ax.add_collection3d(poly)
    if contour_points is not None:
        draw_rings(ax, contour_points, centre, _lighten(color, -0.35),
                   back_alpha=0.18, lw=0.9, dots=False)
    set_3d_view(ax, verts)


# ── Build figure ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--refresh", action="store_true",
                    help="Recompute the cached RBF surfaces.")
args = parser.parse_args()

print(f"Generating real-data RBF pipeline figure from {CASE_ID} …")

mri_volume = load_canonical_nifti(MRI_PATH)
mri_segmentation = load_canonical_nifti(MRI_SEG_PATH)
with np.load(CACHE_PATH) as cached:
    contour_norm = np.asarray(cached["contour_xyz"], dtype=np.float64)
    tissue = np.asarray(cached["contour_tissue"], dtype=np.float64)
    centroid = np.asarray(cached["centroid"], dtype=np.float64)
    scale = float(cached["scale"])
flip = np.array([1.0, 1.0, -1.0])
contour_mm = contour_norm * flip * scale + centroid
centre = contour_mm.mean(axis=0)
surface_points = {
    "endo": contour_mm[np.isclose(tissue, 0.0)],
    "epi": contour_mm[np.isclose(tissue, 1.0)],
}
rbf_stages = load_or_build_rbf_stages(surface_points, refresh=args.refresh)

fig = plt.figure(figsize=(16.5, 7.4), dpi=300)
fig.patch.set_facecolor("white")

# 2 rows × 5 cols. Cols 0-1 = shared inputs (span both rows).
gs = fig.add_gridspec(
    2, 5,
    width_ratios=[1.05, 0.72, 1.15, 1.15, 1.15],
    hspace=0.30, wspace=0.06,
    left=0.045, right=0.985, top=0.90, bottom=0.10,
)

# ── Shared inputs (vertically centred over both rows)
ax_mri   = fig.add_subplot(gs[:, 0])
ax_masks = fig.add_subplot(gs[:, 1])
panel_mri_with_contours(ax_mri, mri_volume, mri_segmentation)
panel_stacked_labels(ax_masks, surface_points)
panel_label(ax_mri,   "(a)  SAX MRI with contours")
panel_label(ax_masks, "(b)  Basal-to-apical labels")

# ── Endocardial flow (top row)
ax_ec = fig.add_subplot(gs[0, 2], projection='3d', computed_zorder=False)
ax_er = fig.add_subplot(gs[0, 3], projection='3d', computed_zorder=False)
ax_es = fig.add_subplot(gs[0, 4], projection='3d', computed_zorder=False)
panel_sax_stack(ax_ec, surface_points["endo"], centre, C_ENDO)
panel_mesh(ax_er, rbf_stages["endo"][0], centre, C_ENDO, translucent=True,
           contour_points=surface_points["endo"])
panel_mesh(ax_es, rbf_stages["endo"][1], centre, C_ENDO)
panel_label(ax_ec, "(c)\u2009 SAX contours",        is_3d=True)
panel_label(ax_er, "(d)\u2009 RBF fit to contours",  is_3d=True)
panel_label(ax_es, "(e)\u2009 Final RBF surface",    is_3d=True)

# ── Epicardial flow (bottom row)
ax_pc = fig.add_subplot(gs[1, 2], projection='3d', computed_zorder=False)
ax_pr = fig.add_subplot(gs[1, 3], projection='3d', computed_zorder=False)
ax_ps = fig.add_subplot(gs[1, 4], projection='3d', computed_zorder=False)
panel_sax_stack(ax_pc, surface_points["epi"], centre, C_EPI)
panel_mesh(ax_pr, rbf_stages["epi"][0], centre, C_EPI, translucent=True,
           contour_points=surface_points["epi"])
panel_mesh(ax_ps, rbf_stages["epi"][1], centre, C_EPI)
panel_label(ax_pc, "(c\u2019)\u2009 SAX contours",         is_3d=True)
panel_label(ax_pr, "(d\u2019)\u2009 RBF fit to contours",   is_3d=True)
panel_label(ax_ps, "(e\u2019)\u2009 Final RBF surface",     is_3d=True)

# Force a layout pass so get_position() returns final coordinates.
fig.canvas.draw()


def mid_right(ax):
    bb = ax.get_position(); return (bb.x1, (bb.y0 + bb.y1) / 2)

def mid_left(ax):
    bb = ax.get_position(); return (bb.x0, (bb.y0 + bb.y1) / 2)


# ── Subtle colour bands behind each pipeline row (aesthetic grouping)
def row_band(axes, color):
    x0 = min(a.get_position().x0 for a in axes) - 0.008
    x1 = max(a.get_position().x1 for a in axes) + 0.008
    y0 = min(a.get_position().y0 for a in axes) - 0.055
    y1 = max(a.get_position().y1 for a in axes) + 0.010
    rect = FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.002,rounding_size=0.012",
        transform=fig.transFigure, facecolor=_lighten(color, 0.93),
        edgecolor=_lighten(color, 0.70), linewidth=0.9, zorder=0,
    )
    fig.add_artist(rect)

row_band([ax_ec, ax_er, ax_es], C_ENDO)
row_band([ax_pc, ax_pr, ax_ps], C_EPI)

# ── Row titles (coloured) above the first pipeline panel of each row
def row_title(ax, text, color):
    bb = ax.get_position()
    fig.text(bb.x0, bb.y1 + 0.016, text, ha='left', va='bottom',
             fontsize=10.5, fontweight='bold', color=color)

row_title(ax_ec, "Endocardium", C_ENDO)
row_title(ax_pc, "Epicardium",  C_EPI)

# ── Within-row horizontal flow arrows
for src, dst, col in [
    (ax_ec, ax_er, C_ENDO), (ax_er, ax_es, C_ENDO),
    (ax_pc, ax_pr, C_EPI),  (ax_pr, ax_ps, C_EPI),
]:
    add_figspace_arrow(fig, mid_right(src), mid_left(dst), color=col, lw=1.6)

# ── Horizontal bifurcation from (b) into the two rows
bb_b = ax_masks.get_position()
split = (bb_b.x1 + 0.004, (bb_b.y0 + bb_b.y1) / 2)
add_figspace_arrow(fig, split, mid_left(ax_ec), color=C_ENDO, lw=1.9, rad=0.16)
add_figspace_arrow(fig, split, mid_left(ax_pc), color=C_EPI,  lw=1.9, rad=-0.16)

# ── Arrow (a) → (b)
add_figspace_arrow(fig, mid_right(ax_mri), mid_left(ax_masks),
                   color="#555555", lw=1.6)

plt.savefig(THESIS / "images/fig_real_mesh_steps.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("  Saved: images/fig_real_mesh_steps.png")
print("Done.")
