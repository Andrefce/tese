#!/usr/bin/env python3
"""
Real-data mesh pipeline figure — horizontal left-to-right layout with a
horizontal bifurcation into endocardial and epicardial flows.

Layout (landscape):

                                          ┌─→ (c)  endo SAX contours → (d)  raw → (e)  smoothed
    (a) SAX + contours → (b) binary masks ┤
                                          └─→ (c') epi  SAX contours → (d') raw → (e') smoothed

The shared inputs (a, b) sit on the left, vertically centred; the pipeline
branches horizontally into two colour-coded rows (endocardium in blue,
epicardium in orange). Panel labels are placed BELOW each axis.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
from skimage.measure import marching_cubes, find_contours
import trimesh

# Paths
ED_MRI = "notebooks/patient002/patient002_frame01.nii/DCM04Gate1.nii"
ED_SEG = "notebooks/patient002/patient002_frame01_gt.nii/DCM04-OH-AL_V2_1.nii"

# Okabe-Ito colorblind-safe palette
C_ENDO = "#0072B2"   # blue
C_EPI  = "#D55E00"   # orange
C_RV   = "#009E73"   # green
C_BG   = "#0e0e12"   # near-black panel background for masks


def canonical_reorient(path):
    nii = nib.load(path)
    nii_c = nib.as_closest_canonical(nii)
    return nii_c.get_fdata(), nii_c.affine, nii_c.header.get_zooms()


def phong_colors(verts, faces, base_hex, ambient=0.38, diffuse=0.62):
    base = np.array(to_rgb(base_hex))
    mesh_t = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    fn = mesh_t.face_normals
    key  = np.array([-0.4, -0.3,  0.9]); key  /= np.linalg.norm(key)
    fill = np.array([ 0.6,  0.4, -0.2]); fill /= np.linalg.norm(fill)
    light = ambient + diffuse * (0.7 * np.maximum(0, fn @ key)
                                + 0.3 * np.maximum(0, fn @ fill))
    return np.clip(base[None] * np.clip(light, 0, 1)[:, None], 0, 1)


def set_3d_view(ax, verts, z_boost=5.5):
    ax.view_init(elev=18, azim=-65, roll=0)
    r = [np.ptp(verts[:, i]) for i in range(3)]
    rmax = max(r) if max(r) > 0 else 1.0
    ax.set_box_aspect((r[0]/rmax, r[1]/rmax, r[2]/rmax * z_boost))
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.axis("off")
    ax.set_facecolor("white")


def panel_label(ax, text, is_3d=False):
    """Place label text BELOW the axis."""
    y = -0.10 if is_3d else -0.08
    if is_3d:
        ax.text2D(0.5, y, text, transform=ax.transAxes,
                  ha='center', va='top', fontsize=9.5, fontweight='bold',
                  color='#222222')
    else:
        ax.annotate(text, xy=(0.5, y), xycoords='axes fraction',
                    ha='center', va='top', fontsize=9.5, fontweight='bold',
                    color='#222222')


def add_figspace_arrow(fig, posA, posB, color, lw=1.6, rad=0.0):
    arr = FancyArrowPatch(
        posA, posB,
        transform=fig.transFigure,
        arrowstyle='-|>', mutation_scale=15,
        color=color, linewidth=lw,
        connectionstyle=f'arc3,rad={rad}',
        zorder=12, capstyle='round',
    )
    fig.add_artist(arr)


# ── Geometry helpers ─────────────────────────────────────────

def myo_slices(seg, n=3):
    """Return n slice indices (basal→apical) that contain myocardium."""
    counts = [(np.isin(seg[:, :, z], [2, 3])).sum() for z in range(seg.shape[2])]
    valid = [z for z, c in enumerate(counts) if c > 25]
    if len(valid) < n:
        return valid
    picks = np.linspace(0, len(valid) - 1, n).round().astype(int)
    return [valid[i] for i in picks]


def heart_bbox(seg, zs, pad):
    """Bounding box (axis0, axis1) of myocardium over the given slices."""
    m = np.zeros(seg.shape[:2], dtype=bool)
    for z in zs:
        m |= np.isin(seg[:, :, z], [2, 3])
    a0, a1 = np.where(m)
    return (a0.min() - pad, a0.max() + pad, a1.min() - pad, a1.max() + pad)


# ── Panel content ────────────────────────────────────────────

def panel_mri_with_contours(ax, vol, seg, zs):
    mid_z = zs[len(zs) // 2]
    r0, r1, c0, c1 = heart_bbox(seg, [mid_z], pad=42)
    r0 = max(0, r0); c0 = max(0, c0)
    r1 = min(vol.shape[0], r1); c1 = min(vol.shape[1], c1)

    ax.imshow(vol[:, :, mid_z].T, cmap="gray", origin="lower", aspect="equal")
    for label, color in [(3, C_ENDO), (2, C_EPI), (1, C_RV)]:
        m = (seg[:, :, mid_z].T == label)
        if m.sum() > 0:
            ax.contour(m, levels=[0.5], colors=[color], linewidths=2.0,
                       origin="lower")
    ax.set_xlim(r0, r1)
    ax.set_ylim(c0, c1)
    patches = [
        mpatches.Patch(color=C_ENDO, label='Endocardium'),
        mpatches.Patch(color=C_EPI,  label='Epicardium'),
        mpatches.Patch(color=C_RV,   label='RV'),
    ]
    ax.legend(handles=patches, loc='lower center',
              bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=7.5,
              framealpha=0.0, handlelength=1.0, borderpad=0.3,
              columnspacing=1.0, handletextpad=0.4)
    ax.axis("off")


def panel_stacked_masks(ax, seg, zs):
    """3 SAX levels stacked vertically, cropped to the heart, on dark bg."""
    r0, r1, c0, c1 = heart_bbox(seg, zs, pad=12)
    r0 = max(0, r0); c0 = max(0, c0)
    r1 = min(seg.shape[0], r1); c1 = min(seg.shape[1], c1)

    tiles = []
    for z in zs:                       # basal → apical
        sub = seg[r0:r1, c0:c1, z].T   # (W, H) display orientation
        rgb = np.tile(np.array(to_rgb(C_BG), np.float32),
                      (sub.shape[0], sub.shape[1], 1))
        rgb[sub == 2] = to_rgb(C_EPI)  # myocardial shell
        rgb[sub == 3] = to_rgb(C_ENDO) # cavity
        tiles.append(rgb)

    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    sep = 5
    canvas = np.ones((len(tiles) * h + (len(tiles) - 1) * sep, w, 3),
                     np.float32)
    for i, t in enumerate(tiles):
        y = i * (h + sep)
        canvas[y:y + t.shape[0], :t.shape[1]] = t

    ax.imshow(canvas, origin="upper", aspect="equal")
    names = ["Basal", "Mid", "Apical"][:len(tiles)]
    for i, name in enumerate(names):
        ax.text(3, i * (h + sep) + 3, name, ha="left", va="top",
                fontsize=7.5, color="white", fontweight="bold")
    ax.axis("off")


def panel_sax_stack(ax, seg, spacing, label_val, color):
    ax.set_proj_type('ortho')
    N = 10
    z_indices = np.linspace(0, seg.shape[2]-1, N, dtype=int)
    mask_vol = (seg == 3) if label_val == 3 else np.isin(seg, [2, 3])
    pts = []
    for z_idx in z_indices:
        m2d = mask_vol[:, :, z_idx]
        if m2d.sum() == 0:
            continue
        for cnt in find_contours(m2d.astype(float), 0.5):
            for pt in cnt:
                pts.append([pt[1]*spacing[1], pt[0]*spacing[0], z_idx*spacing[2]])
    if not pts:
        ax.axis("off"); return
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    pts[:, 2] = -pts[:, 2]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=1.0, alpha=0.6)
    set_3d_view(ax, pts, z_boost=5.5)


def _build_mesh(seg, spacing, label_val):
    mask = (seg == 3) if label_val == 3 else np.isin(seg, [2, 3])
    pad = 2
    padded = np.pad(mask.astype(float), pad, constant_values=0)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing)
    verts -= np.array([pad*spacing[0], pad*spacing[1], pad*spacing[2]])
    verts -= verts.mean(axis=0)
    verts[:, 2] = -verts[:, 2]
    return verts, faces


def panel_raw_mesh(ax, seg, spacing, label_val, color):
    ax.set_proj_type('ortho')
    verts, faces = _build_mesh(seg, spacing, label_val)
    fc = phong_colors(verts, faces, color)
    ax.add_collection3d(Poly3DCollection(verts[faces], facecolors=fc,
                                         edgecolors="none", alpha=0.95))
    set_3d_view(ax, verts, z_boost=5.5)


def panel_taubin_mesh(ax, seg, spacing, label_val, color):
    ax.set_proj_type('ortho')
    verts, faces = _build_mesh(seg, spacing, label_val)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=25)
    verts, faces = mesh.vertices.copy(), mesh.faces.copy()
    verts -= verts.mean(axis=0)
    fc = phong_colors(verts, faces, color)
    ax.add_collection3d(Poly3DCollection(verts[faces], facecolors=fc,
                                         edgecolors="none", alpha=0.95))
    set_3d_view(ax, verts, z_boost=5.5)


# ── Build figure ─────────────────────────────────────────────
print("Generating real-data mesh pipeline figure (horizontal bifurcation) …")

vol, _, _       = canonical_reorient(ED_MRI)
seg, _, spacing = canonical_reorient(ED_SEG)
zs = myo_slices(seg, n=3)

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
panel_mri_with_contours(ax_mri, vol, seg, zs)
panel_stacked_masks(ax_masks, seg, zs)
panel_label(ax_mri,   "(a)  SAX slice with contours")
panel_label(ax_masks, "(b)  Binary masks")

# ── Endocardial flow (top row)
ax_ec = fig.add_subplot(gs[0, 2], projection='3d')
ax_er = fig.add_subplot(gs[0, 3], projection='3d')
ax_es = fig.add_subplot(gs[0, 4], projection='3d')
panel_sax_stack(ax_ec,  seg, spacing, label_val=3, color=C_ENDO)
panel_raw_mesh( ax_er,  seg, spacing, label_val=3, color=C_ENDO)
panel_taubin_mesh(ax_es, seg, spacing, label_val=3, color=C_ENDO)
panel_label(ax_ec, "(c)\u2009 SAX contours",       is_3d=True)
panel_label(ax_er, "(d)\u2009 Raw surface",         is_3d=True)
panel_label(ax_es, "(e)\u2009 Taubin-smoothed",     is_3d=True)

# ── Epicardial flow (bottom row)
ax_pc = fig.add_subplot(gs[1, 2], projection='3d')
ax_pr = fig.add_subplot(gs[1, 3], projection='3d')
ax_ps = fig.add_subplot(gs[1, 4], projection='3d')
panel_sax_stack(ax_pc,  seg, spacing, label_val=2, color=C_EPI)
panel_raw_mesh( ax_pr,  seg, spacing, label_val=2, color=C_EPI)
panel_taubin_mesh(ax_ps, seg, spacing, label_val=2, color=C_EPI)
panel_label(ax_pc, "(c\u2019)\u2009 SAX contours",   is_3d=True)
panel_label(ax_pr, "(d\u2019)\u2009 Raw surface",     is_3d=True)
panel_label(ax_ps, "(e\u2019)\u2009 Taubin-smoothed", is_3d=True)

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
        transform=fig.transFigure, facecolor=color, edgecolor='none',
        alpha=0.06, zorder=0,
    )
    fig.add_artist(rect)

row_band([ax_ec, ax_er, ax_es], C_ENDO)
row_band([ax_pc, ax_pr, ax_ps], C_EPI)

# ── Row titles (coloured) above the first pipeline panel of each row
def row_title(ax, text, color):
    bb = ax.get_position()
    fig.text(bb.x0, bb.y1 + 0.015, text, ha='left', va='bottom',
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

plt.savefig("images/fig_real_mesh_steps.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("  Saved: images/fig_real_mesh_steps.png")
print("Done.")
