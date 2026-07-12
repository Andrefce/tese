"""
Generate a step-by-step figure illustrating real-data mesh construction
from a NIfTI cardiac segmentation.

Source: ACDC patient002 (ED frame 1) stored in notebooks/patient002/

Steps shown:
  (a) MRI slice with segmentation labels
  (b) Binary LV endo and epi masks (3 representative slices)
  (c) 3D SAX stack view (contours stacked in 3D)
  (d) Raw marching-cubes surface
  (e) Taubin-smoothed and decimated mesh
  (f) Final SAX contour rings extracted from the mesh

Output: images/fig_real_mesh_steps.png
Requires: numpy, matplotlib, nibabel, scikit-image, trimesh
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import colors as mcolors
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "images"
CASE_DIR = ROOT / "notebooks" / "patient002"
ED_MRI = CASE_DIR / "patient002_frame01.nii" / "DCM04Gate1.nii"
ED_SEG = CASE_DIR / "patient002_frame01_gt.nii" / "DCM04-OH-AL_V2_1.nii"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 7.5,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

C_ENDO  = "#0072B2"
C_EPI   = "#D55E00"
C_RV    = "#009E73"
DRAW_ORDER = (2, 3, 1)
LABEL_COLORS = {1: C_RV, 2: C_EPI, 3: C_ENDO}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_nifti(path: Path):
    import nibabel as nib
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), tuple(float(v) for v in img.header.get_zooms()[:3])


def canonical_reorient(path: Path):
    """Load, reorient to RAS+, return data + spacing."""
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(str(path)))
    return img.get_fdata(dtype=np.float32), tuple(float(v) for v in img.header.get_zooms()[:3])


def union_crop(seg, margin=20):
    mask = np.any(seg > 0, axis=2)
    rows, cols = np.where(mask)
    rs = max(int(rows.min()) - margin, 0)
    re = min(int(rows.max()) + margin + 1, mask.shape[0])
    cs = max(int(cols.min()) - margin, 0)
    ce = min(int(cols.max()) + margin + 1, mask.shape[1])
    return slice(rs, re), slice(cs, ce)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def style_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#555555"); sp.set_linewidth(0.6)


def draw_contours_on_ax(ax, seg_slice):
    for lbl in DRAW_ORDER:
        bin_ = (seg_slice == lbl).astype(float)
        if bin_.any():
            ax.contour(bin_, levels=[0.5], colors=[LABEL_COLORS[lbl]], linewidths=1.2)


# (a) Representative MRI slice with label overlay
def panel_mri_slice(ax, vol, seg, spacing, crop):
    sl = int(np.argmax(np.sum(seg == 3, axis=(0, 1))))
    v = vol[crop[0], crop[1], sl]
    s = seg[crop[0], crop[1], sl]
    p1, p99 = np.percentile(v[v > 0], [1, 99.4])
    ax.imshow(v, cmap="gray", origin="lower", vmin=p1, vmax=p99, interpolation="nearest")
    draw_contours_on_ax(ax, s)
    style_ax(ax)


# (b) Binary endo and epi masks for 3 representative slices
def panel_binary_masks(ax, seg, crop):
    lv_slices = np.where(np.any(seg == 3, axis=(0, 1)))[0]
    picks = [lv_slices[len(lv_slices) // 4],
             lv_slices[len(lv_slices) // 2],
             lv_slices[3 * len(lv_slices) // 4]]
    n = len(picks)
    block_h = 1.0 / n
    for row, sl in enumerate(picks):
        endo_m = (seg[crop[0], crop[1], sl] == 3).astype(float)
        epi_m  = np.isin(seg[crop[0], crop[1], sl], [2, 3]).astype(float)
        y0 = 1.0 - (row + 1) * block_h
        ax_e = ax.inset_axes([0.01, y0 + 0.01, 0.48, block_h - 0.03])
        ax_p = ax.inset_axes([0.51, y0 + 0.01, 0.48, block_h - 0.03])
        ax_e.imshow(endo_m, cmap="Blues", origin="lower", vmin=0, vmax=1)
        ax_p.imshow(epi_m, cmap="Oranges", origin="lower", vmin=0, vmax=1)
        for a in (ax_e, ax_p):
            a.set_xticks([]); a.set_yticks([])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# (c) 3D SAX stack with contour outlines
def panel_3d_stack(ax, seg, spacing, crop):
    row_c, col_c = crop
    H = row_c.stop - row_c.start
    W = col_c.stop - col_c.start
    lv_slices = np.where(np.any(seg == 3, axis=(0, 1)))[0]
    x = (np.arange(W) - W / 2) * spacing[0]
    y = (np.arange(H) - H / 2) * spacing[1]

    def contour_segs(bin_):
        if not bin_.any():
            return []
        fig2, a2 = plt.subplots(figsize=(1, 1))
        cs = a2.contour(bin_.astype(float), levels=[0.5])
        segs = [s.copy() for s in cs.allsegs[0] if len(s) > 2]
        plt.close(fig2)
        return segs

    z_values = (lv_slices.mean() - lv_slices) * spacing[2]
    for sl, z in zip(lv_slices, z_values):
        labels = seg[row_c, col_c, sl]
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                [z] * 5, color="#CCCCCC", linewidth=0.35, alpha=0.7)
        for lbl, col in [(3, C_ENDO), ([2, 3], C_EPI)]:
            bin_ = np.isin(labels, lbl) if isinstance(lbl, list) else (labels == lbl)
            for seg_ in contour_segs(bin_):
                xc = (seg_[:, 0] - W / 2) * spacing[0]
                yc = (seg_[:, 1] - H / 2) * spacing[1]
                ax.plot(xc, yc, np.full_like(xc, z), color=col, linewidth=1.0)

    ax.view_init(elev=22, azim=-52)
    ax.set_box_aspect((1, 1, 0.75))
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z_values.min() - 4, z_values.max() + 4)
    ax.set_axis_off()


# (d) Raw marching-cubes mesh
def panel_raw_mesh(ax, seg, spacing, label_set=(3,)):
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        ax.text(0.5, 0.5, "scikit-image\nnot available",
                ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_axis_off()
        return None, None

    mask = np.zeros(seg.shape, dtype=bool)
    for lbl in label_set:
        mask |= (seg == lbl)
    # pad to avoid open boundaries
    padded = np.pad(mask, 2, constant_values=False)
    verts, faces, *_ = marching_cubes(
        padded.astype(float), level=0.5,
        spacing=tuple(spacing[:3]),
    )
    verts -= verts.mean(0)

    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                  verts[faces[:, 2]] - verts[faces[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-8)
    light = np.array([-0.3, -0.5, 0.8]); light /= np.linalg.norm(light)
    lf = 0.45 + 0.55 * np.clip(fn @ light, 0, 1)
    base = np.array(mcolors.to_rgb(C_ENDO))
    fc = np.empty((len(faces), 4))
    fc[:, :3] = np.clip(base * lf[:, None] + 0.18 * (1 - lf[:, None]), 0, 1)
    fc[:, 3] = 0.88

    stride = max(1, len(faces) // 8000)
    ax.add_collection3d(Poly3DCollection(
        verts[faces[::stride]], facecolors=fc[::stride],
        edgecolors="none", linewidths=0,
    ))
    mn, mx = verts.min(0), verts.max(0)
    c, hr = (mn + mx) / 2, np.max(mx - mn) / 2 * 1.1
    ax.set_xlim(c[0]-hr, c[0]+hr); ax.set_ylim(c[1]-hr, c[1]+hr); ax.set_zlim(c[2]-hr, c[2]+hr)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-55)
    return verts, faces


# (e) Taubin-smoothed mesh
def panel_smoothed_mesh(ax, verts, faces, n_iter=30):
    if verts is None:
        ax.text(0.5, 0.5, "mesh\nnot available",
                ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_axis_off()
        return

    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=n_iter)
        sv, sf = np.array(mesh.vertices), np.array(mesh.faces)
    except ImportError:
        # fall back to Laplacian smoothing
        sv = verts.copy()
        adj = [[] for _ in range(len(verts))]
        for f in faces:
            for i in range(3):
                adj[f[i]].extend([f[(i+1)%3], f[(i+2)%3]])
        for _ in range(n_iter):
            new_v = sv.copy()
            for vi in range(len(sv)):
                nbrs = list(set(adj[vi]))
                if nbrs:
                    new_v[vi] = 0.5 * sv[vi] + 0.5 * sv[nbrs].mean(0)
            sv = new_v
        sf = faces

    sv -= sv.mean(0)
    fn = np.cross(sv[sf[:, 1]] - sv[sf[:, 0]], sv[sf[:, 2]] - sv[sf[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-8)
    light = np.array([-0.3, -0.5, 0.8]); light /= np.linalg.norm(light)
    lf = 0.45 + 0.55 * np.clip(fn @ light, 0, 1)
    base = np.array(mcolors.to_rgb(C_ENDO))
    fc = np.empty((len(sf), 4))
    fc[:, :3] = np.clip(base * lf[:, None] + 0.18 * (1-lf[:, None]), 0, 1)
    fc[:, 3] = 0.90

    stride = max(1, len(sf) // 8000)
    ax.add_collection3d(Poly3DCollection(
        sv[sf[::stride]], facecolors=fc[::stride], edgecolors="none", linewidths=0,
    ))
    mn, mx = sv.min(0), sv.max(0)
    c, hr = (mn+mx)/2, np.max(mx-mn)/2*1.1
    ax.set_xlim(c[0]-hr, c[0]+hr); ax.set_ylim(c[1]-hr, c[1]+hr); ax.set_zlim(c[2]-hr, c[2]+hr)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-55)


# (f) Contour rings from the smoothed mesh
def panel_contour_rings_from_mesh(ax, verts, faces, n_slices=8):
    if verts is None:
        ax.text(0.5, 0.5, "mesh\nnot available",
                ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_axis_off()
        return

    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=20)
        sv = np.array(mesh.vertices)
    except ImportError:
        sv = verts.copy()

    z_vals = np.linspace(sv[:, 2].min() * 0.90, sv[:, 2].max() * 0.90, n_slices)
    band = np.ptp(sv[:, 2]) / (n_slices * 1.8)

    for z in z_vals:
        ring_mask = np.abs(sv[:, 2] - z) < band
        ring = sv[ring_mask]
        if len(ring) < 5:
            continue
        angles = np.arctan2(ring[:, 1], ring[:, 0])
        order = np.argsort(angles)
        rx, ry = ring[order, 0], ring[order, 1]
        ax.plot(np.append(rx, rx[0]), np.append(ry, ry[0]),
                np.full(len(rx)+1, z), color=C_ENDO, linewidth=1.0, alpha=0.88)

    mn, mx = sv.min(0), sv.max(0)
    c, hr = (mn+mx)/2, np.max(mx-mn)/2*1.1
    ax.set_xlim(c[0]-hr, c[0]+hr); ax.set_ylim(c[1]-hr, c[1]+hr); ax.set_zlim(c[2]-hr, c[2]+hr)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-55)


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------

def fig_real_mesh_steps():
    vol, spacing = load_nifti(ED_MRI)
    seg, _ = canonical_reorient(ED_SEG)
    seg = seg.astype(np.uint8)
    crop = union_crop(seg)

    projections = ["rectilinear", "rectilinear", "3d", "3d", "3d", "3d"]
    titles = [
        "(a) MRI + labels",
        "(b) Binary masks\n(endo / epi)",
        "(c) 3D SAX stack\n(10 levels)",
        "(d) Marching cubes\n(raw surface)",
        "(e) Taubin smooth\n+ decimation",
        "(f) SAX contour\nextraction",
    ]

    fig = plt.figure(figsize=(11.4, 3.8), facecolor="white")
    positions = [
        [0.01,  0.10, 0.145, 0.80],
        [0.165, 0.10, 0.140, 0.80],
        [0.315, 0.10, 0.145, 0.80],
        [0.470, 0.10, 0.165, 0.80],
        [0.645, 0.10, 0.165, 0.80],
        [0.820, 0.10, 0.165, 0.80],
    ]
    axes = [fig.add_axes(p, projection=proj)
            for p, proj in zip(positions, projections)]

    # populate panels
    panel_mri_slice(axes[0], vol, seg, spacing, crop)
    panel_binary_masks(axes[1], seg, crop)
    panel_3d_stack(axes[2], seg, spacing, crop)
    verts, faces = panel_raw_mesh(axes[3], seg, spacing, label_set=[3])
    panel_smoothed_mesh(axes[4], verts, faces)
    panel_contour_rings_from_mesh(axes[5], verts, faces)

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=7, pad=3)

    # arrows
    arrow_kw = dict(
        transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=9,
        linewidth=0.9, color="#555555",
        shrinkA=0, shrinkB=0,
        clip_on=False, zorder=50,
    )
    import matplotlib.patches as mpatch
    for i in range(len(axes) - 1):
        b0, b1 = axes[i].get_position(), axes[i+1].get_position()
        mid_y = 0.5 * (b0.y0 + b0.y1)
        fig.add_artist(mpatch.FancyArrowPatch(
            (b0.x1 + 0.003, mid_y), (b1.x0 - 0.003, mid_y), **arrow_kw
        ))

    # legend
    handles = [
        Line2D([0], [0], color=C_ENDO, lw=1.3, label="LV endo"),
        Line2D([0], [0], color=C_EPI,  lw=1.3, label="LV epi / MYO"),
        Line2D([0], [0], color=C_RV,   lw=1.3, label="RV"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.50, 0.00), ncol=3,
               fontsize=6.5, frameon=False, handlelength=1.2, columnspacing=0.9)

    out = OUT_DIR / "fig_real_mesh_steps.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.06,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    print("Generating real-data mesh pipeline figure …")
    fig_real_mesh_steps()
    print("Done.")
