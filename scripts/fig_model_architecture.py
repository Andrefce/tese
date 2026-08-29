"""Schematic of the proposed model architecture.

Monochrome plate read from top to bottom: sparse SAX contour input, two-path
contour encoder, implicit decoder, coupled output heads and marching-cubes
surfaces.

Layer widths, feature dimensions and injection points follow the reference
implementation in test-new-model/cardiosdf2/model.py and
scripts/eval_demo/cardiosdf_model.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch, Polygon,
                                Rectangle)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "images"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 7.0,
    "mathtext.fontset": "dejavuserif",
    "figure.dpi": 160,
    "savefig.dpi": 400,
})

INK = "#000000"
GREY = "#555555"
FILL = "#EDEDED"
WHITE = "#FFFFFF"
DASH = (0, (3, 2))

# per-stage (panel fill, panel edge, title colour)
STAGE = {
    "input": ("#EEF3FA", "#9DB5D6", "#23486F"),
    "encode": ("#EDF5F0", "#9CC4AE", "#215B3F"),
    "query": ("#F2EFF8", "#B4A8D2", "#453270"),
    "decode": ("#FDF4E7", "#DFBF8C", "#8A5A15"),
    "surface": ("#FBEFEF", "#D6A6A6", "#7E2E2E"),
}

W, H = 100.0, 104.0

# centres of the two encoder paths, reused by every vertical connector
CX_L, CX_R = 26.0, 74.0


# ---------------------------------------------------------------- primitives

def panel(ax, y0, y1, title, key):
    face, edge, ink = STAGE[key]
    ax.add_patch(Rectangle((3.0, y0), 94.0, y1 - y0, facecolor=face,
                           edgecolor=edge, linewidth=0.7, zorder=0))
    ax.text(5.5, y1 - 2.6, title, ha="left", va="center", fontsize=7.6,
            fontweight="bold", color=ink, zorder=6)


def slabs(ax, x0, x1, y0, y1, n, highlight=None, ratio=0.5, scale=None):
    """Row of layer slabs, optionally scaled by relative layer width."""
    span = (x1 - x0) / n
    centres = [x0 + span * (i + 0.5) for i in range(n)]
    cy, half = 0.5 * (y0 + y1), 0.5 * (y1 - y0)
    for i, cx in enumerate(centres):
        f = 1.0 if scale is None else scale[i]
        ax.add_patch(Rectangle((cx - 0.5 * span * ratio, cy - f * half),
                               span * ratio, 2.0 * f * half,
                               facecolor=FILL if i == highlight else WHITE,
                               edgecolor=INK, linewidth=0.5, zorder=3))
    return centres


def strip(ax, x0, x1, y0, y1, n=12):
    """Feature-vector glyph."""
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=WHITE,
                           edgecolor=INK, linewidth=0.6, zorder=3))
    for i in range(1, n):
        x = x0 + (x1 - x0) * i / n
        ax.add_line(Line2D([x, x], [y0, y1], color=GREY, lw=0.3, zorder=4))


def grid_glyph(ax, x0, x1, y0, y1, ncols, nrows):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=WHITE,
                           edgecolor=INK, linewidth=0.6, zorder=3))
    for i in range(1, ncols):
        x = x0 + (x1 - x0) * i / ncols
        ax.add_line(Line2D([x, x], [y0, y1], color=GREY, lw=0.35, zorder=4))
    for j in range(1, nrows):
        y = y0 + (y1 - y0) * j / nrows
        ax.add_line(Line2D([x0, x1], [y, y], color=GREY, lw=0.35, zorder=4))


def cube(ax, x0, y0, size, depth=0.34):
    """Oblique feature volume with a gridded front face."""
    dx, dy = depth * size, depth * size * 0.55
    faces = (
        ([(x0, y0 + size), (x0 + dx, y0 + size + dy),
          (x0 + size + dx, y0 + size + dy), (x0 + size, y0 + size)], "#F1F1F1"),
        ([(x0 + size, y0), (x0 + size + dx, y0 + dy),
          (x0 + size + dx, y0 + size + dy), (x0 + size, y0 + size)], "#E1E1E1"),
        ([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size),
          (x0, y0 + size)], WHITE),
    )
    for poly, face in faces:
        ax.add_patch(Polygon(poly, closed=True, facecolor=face, edgecolor=INK,
                             linewidth=0.55, zorder=3))
    for i in range(1, 4):
        t = i / 4.0
        ax.add_line(Line2D([x0 + t * size] * 2, [y0, y0 + size], color=GREY,
                           lw=0.3, zorder=4))
        ax.add_line(Line2D([x0, x0 + size], [y0 + t * size] * 2, color=GREY,
                           lw=0.3, zorder=4))


def node(ax, cx, cy, symbol, r=1.6):
    ax.add_patch(Circle((cx, cy), r, facecolor=WHITE, edgecolor=INK,
                        linewidth=0.6, zorder=4))
    ax.text(cx, cy, symbol, ha="center", va="center", fontsize=7.0,
            color=INK, zorder=5)


def box(ax, x0, y0, x1, y1, text, fs=6.2, fill=WHITE):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fill,
                           edgecolor=INK, linewidth=0.6, zorder=3))
    ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), text, ha="center", va="center",
            fontsize=fs, color=INK, linespacing=1.45, zorder=4)


def arr(ax, p0, p1, rad=0.0, dashed=False, lw=0.7):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=6.5,
        connectionstyle=f"arc3,rad={rad}", linewidth=lw,
        linestyle=DASH if dashed else "-", color=INK,
        shrinkA=0.6, shrinkB=0.6, zorder=5))


def line(ax, pts, dashed=False, lw=0.7):
    xs, ys = zip(*pts)
    ax.add_line(Line2D(xs, ys, color=INK, lw=lw,
                       linestyle=DASH if dashed else "-", zorder=5))


def txt(ax, x, y, s, fs=6.0, ha="center", va="center", color=INK,
        style="normal"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, style=style,
            linespacing=1.4, zorder=6)


# -------------------------------------------------------------- stage: input

def draw_input(ax):
    panel(ax, 85.0, 103.0, "Sparse contour input", "input")

    for k in range(5):
        t = k / 4.0
        cy = 87.8 + 2.4 * k
        we, he = 5.2 + 5.6 * t, 1.0 + 0.9 * t
        ax.add_patch(Ellipse((17.0, cy), we + 2.4, he + 0.7, fill=False,
                             edgecolor=INK, lw=0.5, ls=DASH, zorder=3))
        ax.add_patch(Ellipse((17.0, cy), we, he, fill=False, edgecolor=INK,
                             lw=0.5, zorder=4))
    txt(ax, 17.0, 85.9, "endocardium and epicardium", fs=5.6, color=GREY)

    arr(ax, (26.5, 92.2), (32.5, 92.2))

    grid_glyph(ax, 34.0, 49.0, 88.0, 96.5, 5, 6)
    for i, s in enumerate([r"$x$", r"$y$", r"$z$", "tissue", "phase"]):
        ax.text(34.0 + 3.0 * (i + 0.5), 96.9, s, ha="center", va="bottom",
                fontsize=5.0, rotation=90, color=INK, zorder=6)
    txt(ax, 41.5, 85.9, r"contour tensor $P \in \mathbb{R}^{N \times 5}$",
        fs=6.0)

    for i, s in enumerate([r"$N \leq 1200$ points per case",
                           "one row per contour point",
                           "ED and ES phases"]):
        txt(ax, 55.0, 95.0 - 3.4 * i, s, fs=6.1, ha="left")

    arr(ax, (50.0, 85.0), (50.0, 83.0))


# ------------------------------------------------------------ stage: encoder

def draw_encoder(ax):
    panel(ax, 56.0, 83.0, "Contour encoding", "encode")

    txt(ax, CX_L, 77.8, "global path", fs=6.0, color=GREY, style="italic")
    txt(ax, CX_R, 77.8, "local path", fs=6.0, color=GREY, style="italic")

    centres = slabs(ax, 15.0, 37.0, 70.5, 76.0, 3, ratio=0.34,
                    scale=(0.62, 0.80, 1.0))
    for cx, dim in zip(centres, ("64", "128", "256")):
        txt(ax, cx, 69.5, dim, fs=5.6, color=GREY)
    arr(ax, (CX_L, 68.6), (CX_L, 67.9))
    box(ax, 9.0, 63.4, 43.0, 67.9,
        "tissue-wise max-pool\nconcatenate and project", fs=6.0)
    arr(ax, (CX_L, 63.4), (CX_L, 62.1))
    strip(ax, 14.0, 38.0, 59.1, 62.1)
    txt(ax, CX_L, 56.9, r"global code $z \in \mathbb{R}^{256}$", fs=6.1)

    centres = slabs(ax, 63.0, 85.0, 70.5, 76.0, 3, ratio=0.34,
                    scale=(0.62, 0.62, 0.44))
    for cx, dim in zip(centres, ("64", "64", "32")):
        txt(ax, cx, 69.5, dim, fs=5.6, color=GREY)
    arr(ax, (CX_R, 68.6), (CX_R, 68.0))
    txt(ax, CX_R, 67.2, r"scatter to a $16^{3}$ grid, 3D CNN", fs=6.0)
    arr(ax, (CX_R, 66.4), (CX_R, 65.3))

    cube(ax, 70.5, 57.8, 6.0)
    txt(ax, CX_R, 56.9, r"feature volume $V \in \mathbb{R}^{32 \times 16^{3}}$",
        fs=6.1)

    arr(ax, (50.0, 56.0), (50.0, 54.0))


# -------------------------------------------------------------- stage: query

def draw_query(ax):
    panel(ax, 42.0, 54.0, "Query points and conditioning", "query")

    box(ax, 8.0, 44.5, 26.0, 49.5,
        "query points\n" r"$X \in \mathbb{R}^{Q \times 3}$", fs=6.0)
    arr(ax, (26.0, 47.0), (30.0, 47.0))
    box(ax, 30.0, 44.5, 52.0, 49.5,
        "Fourier encoding\n" r"$L = 3$", fs=6.0)
    txt(ax, 41.0, 43.2, r"$\gamma(x) \in \mathbb{R}^{21}$", fs=6.0,
        color=GREY)
    arr(ax, (52.0, 47.0), (56.0, 47.0))
    box(ax, 56.0, 44.5, 78.0, 49.5,
        "trilinear sampling\n" r"of $V$", fs=6.0)
    txt(ax, 67.0, 43.2, r"$v(x) \in \mathbb{R}^{32}$", fs=6.0, color=GREY)

    arr(ax, (50.0, 42.0), (50.0, 40.0))


# ------------------------------------------------------------ stage: decoder

def draw_decoder(ax):
    panel(ax, 18.0, 40.0, "Implicit decoding", "decode")

    box(ax, 5.0, 31.5, 17.0, 35.0, r"$z \in \mathbb{R}^{256}$", fs=6.0)
    box(ax, 5.0, 26.5, 17.0, 30.0, r"$\gamma(x) \in \mathbb{R}^{21}$",
        fs=6.0)
    box(ax, 5.0, 21.5, 17.0, 25.0, r"$v(x) \in \mathbb{R}^{32}$", fs=6.0)

    box(ax, 21.0, 26.5, 30.0, 35.0, "concat\n277", fs=6.0, fill=FILL)
    arr(ax, (17.0, 33.25), (21.0, 33.25))
    arr(ax, (17.0, 28.25), (21.0, 28.25))

    box(ax, 34.0, 26.0, 62.0, 35.0,
        "MLP trunk\n" r"8 layers $\times$ 512, skip at layer 4", fs=6.0)
    arr(ax, (30.0, 30.75), (34.0, 30.75))

    line(ax, [(17.0, 23.25), (48.0, 23.25)], dashed=True)
    arr(ax, (48.0, 23.25), (48.0, 26.0), dashed=True)
    txt(ax, 30.0, 22.1, "local conditioning", fs=5.7, color=GREY,
        style="italic")

    box(ax, 66.0, 31.0, 93.0, 35.0,
        r"endocardial head $f_{\mathrm{endo}}(x)$", fs=6.0)
    box(ax, 66.0, 26.0, 93.0, 30.0,
        r"positive-offset head $\delta(x) > 0$", fs=6.0)
    arr(ax, (62.0, 30.5), (66.0, 33.0))
    arr(ax, (62.0, 30.5), (66.0, 28.0))

    txt(ax, 79.5, 21.5,
        r"$f_{\mathrm{epi}}(x) = f_{\mathrm{endo}}(x) - \delta(x)$", fs=6.6)

    arr(ax, (50.0, 18.0), (50.0, 16.0))


# ------------------------------------------------------------------ stage (d)

A_EPI, C_EPI = 0.58, 1.00
A_ENDO, C_ENDO = 0.40, 0.85
Z_BASE = 0.30


def _half_width(z, a, c):
    return a * np.sqrt(np.clip(1.0 - (z / c) ** 2, 0.0, None))


def draw_lv_section(ax, cx, y_apex, height):
    """Long-axis cut through the two extracted surfaces, myocardium shaded."""
    s = height / (Z_BASE + C_EPI)

    def curve(z, a, c, side):
        return (cx + side * s * _half_width(z, a, c),
                y_apex + s * (z + C_EPI))

    z_out = np.linspace(Z_BASE, -C_EPI, 200)
    z_in = np.linspace(Z_BASE, -C_ENDO, 200)
    wall = np.vstack([
        np.column_stack(curve(z_out, A_EPI, C_EPI, -1.0)),
        np.column_stack(curve(z_out[::-1], A_EPI, C_EPI, 1.0)),
        np.column_stack(curve(z_in, A_ENDO, C_ENDO, 1.0)),
        np.column_stack(curve(z_in[::-1], A_ENDO, C_ENDO, -1.0)),
    ])
    ax.add_patch(Polygon(wall, closed=True, facecolor=FILL, edgecolor=INK,
                         linewidth=0.6, zorder=3))

    px, py = curve(0.20, A_EPI, C_EPI, 1.0)
    line(ax, [(px, py), (px + 2.6, py + 1.0)], lw=0.5)
    txt(ax, px + 3.0, py + 1.2, "epicardium", fs=5.8, ha="left")

    qx, qy = curve(-0.10, A_ENDO, C_ENDO, 1.0)
    line(ax, [(qx, qy), (qx + 4.2, qy - 1.0)], lw=0.5)
    txt(ax, qx + 4.6, qy - 1.2, "endocardium", fs=5.8, ha="left")


def draw_output(ax):
    panel(ax, 1.0, 16.0, "Surface extraction", "surface")

    box(ax, 8.0, 5.5, 30.0, 10.5,
        r"dense grid $96^{3}$" "\n"
        r"evaluate $f_{\mathrm{endo}}, f_{\mathrm{epi}}$", fs=6.0)
    arr(ax, (30.0, 8.0), (35.0, 8.0))
    box(ax, 35.0, 5.5, 55.0, 10.5,
        "marching cubes\nzero level set", fs=6.0)
    arr(ax, (55.0, 8.0), (60.0, 8.0))

    draw_lv_section(ax, 68.0, 2.4, 9.6)


# ----------------------------------------------------------------------- main

def main():
    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 7.0 * H / W))
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_input(ax)
    draw_encoder(ax)
    draw_query(ax)
    draw_decoder(ax)
    draw_output(ax)

    out_png = OUT_DIR / "fig_model_architecture.png"
    out_pdf = OUT_DIR / "fig_model_architecture.pdf"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02,
                facecolor=WHITE)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02,
                facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
