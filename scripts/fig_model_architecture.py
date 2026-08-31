"""Architecture figure for the proposed model.

Four stages read from top to bottom: the sparse SAX contour input, the two
encoder paths, the implicit decoder, and the coupled output fields with the
extracted surfaces.

The input point cloud and the output surfaces are rendered from the cached
inference of a real case (scripts/eval_demo/outputs/demo_patient002_ED.npz)
rather than drawn by hand. Layer widths, grid sizes and feature dimensions
follow test-new-model/cardiosdf2/model.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "images"
DEMO = ROOT / "scripts" / "eval_demo" / "outputs" / "demo_patient002_ED.npz"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 7.5,
    "mathtext.fontset": "dejavuserif",
    "figure.dpi": 160,
    "savefig.dpi": 450,
})

INK = "#111111"
GREY = "#555555"
FAINT = "#9AA0A6"
FILL = "#F0F1F3"
WHITE = "#FFFFFF"
C_ENDO = "#0072B2"
C_EPI = "#D55E00"

W, H = 100.0, 150.0
SPINE = 50.0          # every inter-stage connector runs down this column
LIGHT = np.array([-0.30, -0.45, 0.84])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


# ------------------------------------------------------------------ 2D basics

def rule(ax, x0, x1, y, color=INK, lw=0.9):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=4,
            solid_capstyle="butt")


def vrule(ax, x, y0, y1, color=INK, lw=0.9):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, zorder=4,
            solid_capstyle="butt")


def divider(ax, y):
    """Stage separator, broken so the central connector passes through."""
    rule(ax, 3.0, SPINE - 3.0, y, color=FAINT, lw=0.6)
    rule(ax, SPINE + 3.0, 97.0, y, color=FAINT, lw=0.6)


def arrow(ax, p, q, color=INK, lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=lw, color=color, shrinkA=0,
                                 shrinkB=0, zorder=5))


def stage_label(ax, y, tag, title):
    ax.text(3.0, y, tag, ha="left", va="center", fontsize=8.2,
            fontweight="bold", color=INK, zorder=6)
    ax.text(8.5, y, title, ha="left", va="center", fontsize=8.2,
            color=INK, zorder=6)


def caption(ax, x, y, text, fs=6.8, ha="center", color=GREY):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color,
            zorder=6, linespacing=1.45)


# ------------------------------------------------------------ schematic parts

def layer_block(ax, x0, x1, cy, widths, tallest=13.0, gap=0.30):
    """Row of layer bars, height following the log of the layer width."""
    step = (x1 - x0) / len(widths)
    ref = np.log2(max(widths))
    for i, width in enumerate(widths):
        cx = x0 + step * (i + 0.5)
        h = tallest * (0.34 + 0.66 * np.log2(width) / ref)
        ax.add_patch(Rectangle((cx - 0.5 * step * (1 - gap), cy - 0.5 * h),
                               step * (1 - gap), h, facecolor=FILL,
                               edgecolor=INK, linewidth=0.6, zorder=3))


def latent_bar(ax, cx, cy, w, h, n=8):
    ax.add_patch(Rectangle((cx - 0.5 * w, cy - 0.5 * h), w, h,
                           facecolor=WHITE, edgecolor=INK, linewidth=0.8,
                           zorder=3))
    for i in range(1, n):
        y = cy - 0.5 * h + h * i / n
        ax.plot([cx - 0.5 * w, cx + 0.5 * w], [y, y], color=FAINT, lw=0.4,
                zorder=4)


def voxel_grid(ax, x0, y0, size, n=4, skew=(0.34, 0.24)):
    """Isometric feature grid: front face plus top and right faces."""
    dx, dy = skew[0] * size, skew[1] * size
    x1, y1 = x0 + size, y0 + size
    ax.add_patch(Rectangle((x0, y0), size, size, facecolor=WHITE,
                           edgecolor=INK, linewidth=0.8, zorder=3))
    top = np.array([[x0, y1], [x1, y1], [x1 + dx, y1 + dy], [x0 + dx, y1 + dy]])
    right = np.array([[x1, y0], [x1 + dx, y0 + dy], [x1 + dx, y1 + dy],
                      [x1, y1]])
    for face in (top, right):
        ax.fill(face[:, 0], face[:, 1], facecolor=FILL, edgecolor=INK,
                linewidth=0.8, zorder=3)
    for i in range(1, n):
        t = size * i / n
        ax.plot([x0 + t, x0 + t], [y0, y1], color=FAINT, lw=0.4, zorder=4)
        ax.plot([x0, x1], [y0 + t, y0 + t], color=FAINT, lw=0.4, zorder=4)
        ax.plot([x0 + t, x0 + t + dx], [y1, y1 + dy], color=FAINT, lw=0.4,
                zorder=4)
        ax.plot([x1, x1 + dx], [y0 + t, y0 + t + dy], color=FAINT, lw=0.4,
                zorder=4)


def field_box(ax, cx, cy, w, h, text, fs=8.2):
    ax.add_patch(Rectangle((cx - 0.5 * w, cy - 0.5 * h), w, h,
                           facecolor=WHITE, edgecolor=INK, linewidth=0.8,
                           zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK,
            zorder=4)


def skip_arc(ax, x0, x1, y_base, rise=5.0):
    xs = np.linspace(x0, x1, 90)
    ys = y_base + rise * np.sin(np.pi * (xs - x0) / (x1 - x0))
    ax.plot(xs, ys, color=INK, lw=0.7, zorder=4)
    arrow(ax, (xs[-4], ys[-4]), (xs[-1], ys[-1]), lw=0.7)


# ------------------------------------------------------------- 3D renderings

def upright(points):
    """Cached geometry stores the apex at high z; flip so the apex points down."""
    flipped = points.copy()
    flipped[:, 2] = -flipped[:, 2]
    return flipped


def smooth_mesh(vertices, faces, iterations=12):
    mesh = trimesh.Trimesh(vertices=vertices.astype(np.float64),
                           faces=faces, process=False)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.55, nu=-0.58,
                                    iterations=iterations)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)


def cut_away(vertices, faces, azim):
    """Drop the faces nearest the camera so the cavity stays visible."""
    angle = np.radians(azim)
    towards_camera = np.array([np.cos(angle), np.sin(angle), 0.0])
    offset = vertices[faces].mean(axis=1) - vertices.mean(axis=0)
    return faces[offset @ towards_camera < 0.0]


def shade(vertices, faces, base_colour):
    """Per-face colour from averaged vertex normals, so faceting stays hidden."""
    tri = vertices[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vert_n = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(vert_n, faces[:, k], face_n)
    vert_n /= np.clip(np.linalg.norm(vert_n, axis=1, keepdims=True), 1e-9, None)
    normals = vert_n[faces].mean(axis=1)
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9,
                       None)
    lit = np.clip(np.abs(normals @ LIGHT), 0.0, 1.0)
    rgb = np.array(to_rgb(base_colour))
    return np.clip(rgb[None, :] * (0.52 + 0.48 * lit[:, None]), 0.0, 1.0)


def new_3d(fig, rect, azim=-64.0):
    ax = fig.add_axes(rect, projection="3d")
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    ax.view_init(elev=14.0, azim=azim)
    return ax


def fit_axes(ax, points, zoom=1.0):
    centre = points.mean(axis=0)
    reach = 0.5 * np.ptp(points, axis=0).max() / zoom
    ax.set_xlim(centre[0] - reach, centre[0] + reach)
    ax.set_ylim(centre[1] - reach, centre[1] + reach)
    ax.set_zlim(centre[2] - reach, centre[2] + reach)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def draw_contours_3d(fig, rect, xyz, tissue):
    ax = new_3d(fig, rect)
    pts = upright(xyz)
    endo = tissue < 0.5
    ax.scatter(pts[endo, 0], pts[endo, 1], pts[endo, 2], s=1.0, c=C_ENDO,
               depthshade=False, linewidths=0.0)
    ax.scatter(pts[~endo, 0], pts[~endo, 1], pts[~endo, 2], s=1.0, c=C_EPI,
               depthshade=False, linewidths=0.0)
    fit_axes(ax, pts, zoom=1.15)


def draw_surfaces_3d(fig, rect, endo_v, endo_f, epi_v, epi_f):
    """Endocardium inside a cut epicardial shell, so the wall stays visible."""
    azim = -58.0
    ax = new_3d(fig, rect, azim=azim)
    endo_v, endo_f = smooth_mesh(upright(endo_v), endo_f)
    epi_v, epi_f = smooth_mesh(upright(epi_v), epi_f)
    cut_f = cut_away(epi_v, epi_f, azim)
    ax.add_collection3d(Poly3DCollection(
        endo_v[endo_f], facecolors=shade(endo_v, endo_f, C_ENDO),
        edgecolors="none", linewidths=0.0, rasterized=True))
    ax.add_collection3d(Poly3DCollection(
        epi_v[cut_f], facecolors=shade(epi_v, cut_f, C_EPI),
        edgecolors="none", linewidths=0.0, rasterized=True))
    fit_axes(ax, epi_v, zoom=1.06)


# ------------------------------------------------------------------- assembly

def draw_input(ax, fig, data):
    stage_label(ax, 146.0, "(a)", "Sparse contour input")
    draw_contours_3d(fig, [4.0 / W, 111.0 / H, 42.0 / W, 31.0 / H],
                     data["contours_xyz_mm"], data["contours_tissue"])
    ax.text(50.0, 137.0, "endocardial and epicardial rings",
            ha="left", va="center", fontsize=7.8, color=INK)
    ax.text(50.0, 132.5, "on the acquired SAX planes",
            ha="left", va="center", fontsize=7.8, color=INK)
    caption(ax, 50.0, 126.0,
            "each point carries $(x,y,z)$, a tissue\n"
            "label and the phase (ED or ES)", ha="left")
    rule(ax, 50.0, 53.5, 118.0, color=C_ENDO, lw=1.4)
    caption(ax, 55.0, 118.0, "endocardium", ha="left")
    rule(ax, 74.0, 77.5, 118.0, color=C_EPI, lw=1.4)
    caption(ax, 79.0, 118.0, "epicardium", ha="left")
    arrow(ax, (SPINE, 112.0), (SPINE, 108.5))


def draw_encoder(ax):
    stage_label(ax, 106.0, "(b)", "Contour encoder")
    # the shared observation splits into the two paths
    vrule(ax, SPINE, 108.5, 102.0)
    rule(ax, 22.0, 66.0, 102.0)
    arrow(ax, (22.0, 102.0), (22.0, 99.5))
    arrow(ax, (66.0, 102.0), (66.0, 101.5))
    # global path
    layer_block(ax, 8.0, 36.0, 92.0, [5, 64, 128, 256])
    arrow(ax, (37.0, 92.0), (40.0, 92.0))
    latent_bar(ax, 42.0, 92.0, 3.2, 12.0)
    caption(ax, 24.0, 80.0, "shared point network\n$5-64-128-256$")
    caption(ax, 44.0, 80.0, "max-pool\n$z \\in \\mathbb{R}^{256}$")
    # local path
    voxel_grid(ax, 58.0, 85.0, 13.0)
    arrow(ax, (77.0, 92.0), (81.0, 92.0))
    ax.text(86.5, 92.0, "$v(x)$", ha="center", va="center", fontsize=8.0,
            color=INK, zorder=6)
    caption(ax, 70.0, 80.0, "point features on a $16^{3}$ grid,\n"
                            "3D convolutions $\\rightarrow V$")
    # both descriptors join the connector lane, clear of the captions
    vrule(ax, 42.0, 86.0, 83.0)
    vrule(ax, 86.5, 89.5, 83.0)
    rule(ax, 42.0, 86.5, 83.0)
    arrow(ax, (SPINE, 83.0), (SPINE, 70.0))


def draw_decoder(ax):
    stage_label(ax, 68.5, "(c)", "Implicit decoder")
    # conditioning enters at the decoder input
    vrule(ax, SPINE, 70.0, 65.5)
    rule(ax, 16.0, SPINE, 65.5)
    arrow(ax, (16.0, 65.5), (16.0, 56.8))
    field_box(ax, 16.0, 52.0, 26.0, 9.0,
              "$[\\,z,\\ \\gamma(x),\\ v(x)\\,]$", fs=7.6)
    caption(ax, 16.0, 45.0, "decoder input")
    arrow(ax, (29.5, 52.0), (33.0, 52.0))
    layer_block(ax, 34.0, 78.0, 52.0, [512] * 8, tallest=11.0, gap=0.34)
    skip_arc(ax, 36.0, 58.0, 57.5)
    caption(ax, 60.0, 61.5, "skip connection at layer 4", fs=6.4, ha="left")
    caption(ax, 56.0, 44.0, "8 hidden layers, width 512")
    # trunk splits into the two coupled heads
    vrule(ax, 72.0, 46.5, 40.0)
    rule(ax, 38.0, 72.0, 40.0)
    arrow(ax, (38.0, 40.0), (38.0, 37.5))
    arrow(ax, (62.0, 40.0), (62.0, 37.5))
    field_box(ax, 38.0, 34.0, 22.0, 7.0, "$f_{\\mathrm{endo}}(x)$")
    field_box(ax, 62.0, 34.0, 22.0, 7.0, "$\\delta(x) > 0$")
    vrule(ax, 38.0, 30.5, 28.0)
    vrule(ax, 62.0, 30.5, 28.0)
    rule(ax, 38.0, 62.0, 28.0)


def draw_output(ax, fig, data):
    stage_label(ax, 22.5, "(d)", "Coupled fields and surfaces")
    vrule(ax, SPINE, 28.0, 19.0)
    rule(ax, 30.0, SPINE, 19.0)
    arrow(ax, (30.0, 19.0), (30.0, 17.0))
    ax.text(30.0, 13.5,
            "$f_{\\mathrm{epi}}(x) = f_{\\mathrm{endo}}(x) - \\delta(x)$",
            ha="center", va="center", fontsize=9.2, color=INK, zorder=6)
    caption(ax, 30.0, 7.5,
            "the strictly positive offset keeps the\n"
            "epicardium outside the endocardium")
    caption(ax, 30.0, 2.0, "zero level sets meshed on a $96^{3}$ grid",
            color=INK, fs=7.0)
    draw_surfaces_3d(fig, [52.0 / W, 0.5 / H, 46.0 / W, 24.0 / H],
                     data["model_endo_v"], data["model_endo_f"],
                     data["model_epi_v"], data["model_epi_f"])


# ----------------------------------------------------------------------- main

def main():
    OUT_DIR.mkdir(exist_ok=True)
    with np.load(DEMO) as npz:
        data = {k: npz[k] for k in (
            "contours_xyz_mm", "contours_tissue", "model_endo_v",
            "model_endo_f", "model_epi_v", "model_epi_f")}

    fig = plt.figure(figsize=(5.4, 5.4 * H / W))
    fig.patch.set_facecolor(WHITE)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    for y in (148.5, 110.0, 71.5, 24.5):
        divider(ax, y)

    draw_input(ax, fig, data)
    draw_encoder(ax)
    draw_decoder(ax)
    draw_output(ax, fig, data)

    for suffix in ("png", "pdf"):
        out = OUT_DIR / f"fig_model_architecture.{suffix}"
        fig.savefig(out, facecolor=WHITE)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
