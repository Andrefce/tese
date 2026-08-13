"""
Generate a high-level schematic of the proposed model architecture:

  fig_model_architecture.png — sparse SAX contours -> contour encoder -> latent
  code -> implicit decoder -> coupled endocardial/epicardial fields -> meshes
  and wall-thickness map.

Output goes to images/. Requires: numpy, matplotlib.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.transforms import Affine2D, Bbox

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "images"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7.5,
    "mathtext.fontset": "dejavuserif",
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})

C_ENDO = "#0072B2"
C_EPI = "#D55E00"
C_DARK = "#243447"
C_PANEL = "#F6F8FA"
C_EDGE = "#C3CDD8"
C_BLOCK = "#DCE6F1"
C_BLOCK2 = "#F3E2D3"
C_LATENT = "#2E7D6F"

# canvas
W, H = 118.0, 46.0


# frame + title artists, excluded when the panel content is re-centred
PANEL_FRAME = []


def panel(ax, x0, x1, y0, y1, label):
    bg = ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=1.4",
        facecolor=C_PANEL, edgecolor=C_EDGE, linewidth=0.9, zorder=0))
    # centred above the box so long titles cannot overflow the panel edges
    title = ax.text(0.5 * (x0 + x1), y1 + 0.8, label, ha="center", va="bottom",
                    fontsize=7.6, fontweight="bold", color=C_DARK, zorder=5)
    PANEL_FRAME.extend([bg, title])


def centre_content(ax, renderer, artists, x0, x1, y0, y1):
    boxes = []
    for a in artists:
        bb = a.get_window_extent(renderer)
        if bb.width > 0 and bb.height > 0:
            boxes.append(bb)
    if not boxes:
        return
    bb = Bbox.union(boxes).transformed(ax.transData.inverted())
    dx = 0.5 * (x0 + x1) - 0.5 * (bb.x0 + bb.x1)
    dy = 0.5 * (y0 + y1) - 0.5 * (bb.y0 + bb.y1)
    for a in artists:
        # every artist is laid out in data coordinates, so replace (not compose)
        # the artist transform to avoid patches with their own unit transform
        a.set_transform(Affine2D().translate(dx, dy) + ax.transData)


def block(ax, xc, yc, w, h, text, fc=C_BLOCK, ec=C_DARK, fs=7.5, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.7",
        facecolor=fc, edgecolor=ec, linewidth=0.8, zorder=3))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs,
            color=C_DARK, zorder=4, linespacing=1.35, fontweight=weight)


def arrow(ax, p0, p1, color=C_DARK, lw=1.0, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=8,
        connectionstyle=f"arc3,rad={rad}", linewidth=lw, linestyle=ls,
        color=color, shrinkA=1.5, shrinkB=1.5, zorder=4))


# ---------------------------------------------------------------------------
# (a) sparse SAX contour observation
# ---------------------------------------------------------------------------

def draw_contours(ax):
    x0, x1, y0, y1 = 1.0, 17.0, 9.0, 44.0
    panel(ax, x0, x1, y0, y1, "(a) Sparse SAX contours")

    levels = 6
    for k in range(levels):
        t = k / (levels - 1)
        cy = 15.0 + 21.0 * t
        cx = 8.2 + 1.6 * t
        w_endo = 3.2 + 4.4 * t
        h_endo = 1.1 + 1.5 * t
        ax.add_patch(Ellipse((cx, cy), (w_endo + 2.6), (h_endo + 0.9),
                             facecolor="none", edgecolor=C_EPI,
                             linewidth=1.1, zorder=2))
        ax.add_patch(Ellipse((cx, cy), w_endo, h_endo,
                             facecolor="none", edgecolor=C_ENDO,
                             linewidth=1.1, zorder=3))
        th = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        ax.plot(cx + 0.5 * w_endo * np.cos(th),
                cy + 0.5 * h_endo * np.sin(th),
                ls="none", marker="o", ms=1.2, mfc=C_ENDO, mec="none",
                zorder=4)
        ax.plot(cx + 0.5 * (w_endo + 2.6) * np.cos(th),
                cy + 0.5 * (h_endo + 0.9) * np.sin(th),
                ls="none", marker="o", ms=1.2, mfc=C_EPI, mec="none",
                zorder=4)

    arrow(ax, (2.9, 16.5), (2.9, 37.0), color="#7A8896", lw=0.8)
    ax.text(2.9, 38.2, "base", fontsize=6.4, color="#54606D", ha="center")
    ax.text(2.9, 15.4, "apex", fontsize=6.4, color="#54606D", ha="center",
            va="top")
    ax.text(9.0, 10.2, "10 levels $\\times$ 60 pts\nendo / epi + phase",
            ha="center", va="bottom", fontsize=6.4, color=C_DARK,
            linespacing=1.3)
    return x0, x1, y0, y1


# ---------------------------------------------------------------------------
# (b) contour encoder
# ---------------------------------------------------------------------------

def draw_encoder(ax):
    x0, x1, y0, y1 = 20.0, 45.0, 9.0, 44.0
    panel(ax, x0, x1, y0, y1, "(b) Contour encoder")
    xc = 0.5 * (x0 + x1)

    block(ax, xc, 37.0, 21.0, 5.0,
          "shared point network $\\phi_{\\mathrm{enc}}(p_i)$")
    arrow(ax, (xc - 5.2, 34.5), (xc - 5.2, 31.0))
    arrow(ax, (xc + 5.2, 34.5), (xc + 5.2, 31.0))

    block(ax, xc - 5.2, 28.0, 9.4, 6.0,
          "max-pool\nendocardial", fc=C_BLOCK)
    block(ax, xc + 5.2, 28.0, 9.4, 6.0,
          "max-pool\nepicardial", fc=C_BLOCK2)
    arrow(ax, (xc - 5.2, 25.0), (xc - 2.0, 22.0), rad=0.15)
    arrow(ax, (xc + 5.2, 25.0), (xc + 2.0, 22.0), rad=-0.15)

    block(ax, xc, 19.5, 15.0, 4.4, "concatenate + linear")
    arrow(ax, (xc, 17.3), (xc, 15.4))

    # latent code strip
    n = 18
    bw = 0.78
    x_start = xc - n * bw / 2
    for i in range(n):
        shade = 0.30 + 0.55 * (0.5 + 0.5 * np.sin(2.1 * i + 0.7))
        ax.add_patch(FancyBboxPatch(
            (x_start + i * bw, 11.6), bw * 0.86, 3.0,
            boxstyle="square,pad=0",
            facecolor=plt.cm.viridis(shade), edgecolor="white",
            linewidth=0.35, zorder=3))
    ax.text(xc, 10.4, "latent code $z \\in \\mathbb{R}^{256}$",
            ha="center", va="bottom", fontsize=7.2, color=C_DARK)

    ax.text(xc, 32.6, "permutation-invariant pooling",
            ha="center", va="center", fontsize=6.4, color="#54606D",
            style="italic")
    return x0, x1, y0, y1


# ---------------------------------------------------------------------------
# (c) implicit decoder
# ---------------------------------------------------------------------------

def draw_decoder(ax):
    x0, x1, y0, y1 = 48.0, 72.0, 9.0, 44.0
    panel(ax, x0, x1, y0, y1, "(c) Implicit SDF decoder")

    nlayer = 8
    bw, gap = 1.7, 1.05
    total = nlayer * bw + (nlayer - 1) * gap
    xs = 0.5 * (x0 + x1) - total / 2 + np.arange(nlayer) * (bw + gap)
    ytop, ybot = 35.5, 20.5
    for i, xl in enumerate(xs):
        ax.add_patch(FancyBboxPatch(
            (xl, ybot), bw, ytop - ybot,
            boxstyle="round,pad=0,rounding_size=0.35",
            facecolor="#CBD9EA" if i != 4 else "#A9C4E0",
            edgecolor=C_DARK, linewidth=0.7, zorder=3))

    # skip connection
    ax.add_patch(FancyArrowPatch(
        (xs[0] + bw / 2, ytop), (xs[4] + bw / 2, ytop),
        arrowstyle="-|>", mutation_scale=8, linewidth=0.9,
        connectionstyle="arc3,rad=-0.30", color="#8A5A9B", zorder=5))
    ax.text(0.5 * (xs[0] + xs[4]) + bw / 2, 39.2, "skip connection",
            ha="center", va="center", fontsize=6.6, color="#8A5A9B")

    ax.text(0.5 * (x0 + x1), 18.4,
            "8 fully connected layers\nwidth 512, softplus",
            ha="center", va="center", fontsize=6.8, color=C_DARK,
            linespacing=1.3)

    # query-point branch
    block(ax, 0.5 * (x0 + x1), 13.0, 21.0, 6.2,
          "query point $x \\in \\mathbb{R}^{3}$\n"
          "Fourier features $\\gamma(x)$, $L=3$", fc="#EDE4F3")
    arrow(ax, (0.5 * (x0 + x1) + 7.5, 16.2), (0.5 * (x0 + x1) + 7.5, 20.2))
    return x0, x1, y0, y1


# ---------------------------------------------------------------------------
# (d) coupled output heads
# ---------------------------------------------------------------------------

def draw_heads(ax):
    x0, x1, y0, y1 = 75.0, 95.0, 9.0, 44.0
    panel(ax, x0, x1, y0, y1, "(d) Positive-wall coupling")
    xc = 0.5 * (x0 + x1)

    block(ax, xc, 36.0, 17.0, 5.2,
          "endocardial field\n$f_{\\mathrm{endo}}(x,z)$", fc=C_BLOCK)
    block(ax, xc, 27.0, 17.0, 6.4,
          "bounded offset\n$\\delta = \\tau_{\\min} + "
          "(\\delta_{\\mathrm{cap}}-\\tau_{\\min})\\,\\sigma(\\cdot)$",
          fc=C_BLOCK2, fs=6.8)
    arrow(ax, (xc, 33.4), (xc, 30.2))
    arrow(ax, (xc, 23.8), (xc, 20.6))

    block(ax, xc, 18.0, 17.0, 5.2,
          "$f_{\\mathrm{epi}} = f_{\\mathrm{endo}} - \\delta$", fc="#D9EAD9")

    ax.add_patch(FancyBboxPatch(
        (x0 + 1.4, 10.2), (x1 - x0) - 2.8, 4.8,
        boxstyle="round,pad=0,rounding_size=0.6",
        facecolor="white", edgecolor=C_LATENT, linewidth=0.9, zorder=3))
    ax.text(xc, 12.6,
            "$\\delta \\geq \\tau_{\\min} > 0$:\n"
            "positive wall thickness\nby construction",
            ha="center", va="center", fontsize=6.4, color=C_LATENT,
            linespacing=1.25, zorder=4)
    return x0, x1, y0, y1


# ---------------------------------------------------------------------------
# (e) reconstructed surfaces
# ---------------------------------------------------------------------------

def draw_outputs(ax):
    x0, x1, y0, y1 = 98.0, 117.0, 9.0, 44.0
    panel(ax, x0, x1, y0, y1, "(e) Reconstructed surfaces")
    xc, yc = 0.5 * (x0 + x1), 28.5

    th = np.linspace(0, 2 * np.pi, 361)
    r_endo = 4.6 + 0.45 * np.cos(3 * th) + 0.25 * np.sin(2 * th)
    r_epi = r_endo + 1.9 + 0.85 * np.cos(th - 0.6) + 0.35 * np.cos(2 * th + 1.1)

    ring = np.concatenate([
        np.column_stack([xc + r_epi * np.cos(th), yc + r_epi * np.sin(th)]),
        np.column_stack([xc + r_endo * np.cos(th[::-1]),
                         yc + r_endo * np.sin(th[::-1])]),
    ])
    ax.add_patch(Polygon(ring, closed=True, facecolor="#E3DCD2",
                         edgecolor="none", zorder=3))

    ax.plot(xc + r_endo * np.cos(th), yc + r_endo * np.sin(th),
            color=C_ENDO, lw=1.2, zorder=4)
    ax.plot(xc + r_epi * np.cos(th), yc + r_epi * np.sin(th),
            color=C_EPI, lw=1.2, zorder=4)

    ax.text(xc, 39.6, "marching cubes on a $96^{3}$ grid",
            ha="center", va="center", fontsize=6.5, color=C_DARK)
    ax.text(xc, 17.0,
            "watertight endocardial and\nepicardial surfaces",
            ha="center", va="center", fontsize=6.8, color=C_DARK,
            linespacing=1.3)
    return x0, x1, y0, y1


def main():
    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    ax.set_xlim(0, W)
    ax.set_ylim(7.8, 45.2)
    ax.set_aspect("equal")
    ax.axis("off")

    draws = [draw_contours, draw_encoder, draw_decoder, draw_heads,
             draw_outputs]
    groups = []
    for fn in draws:
        before = {id(a) for a in ax.get_children()}
        rect = fn(ax)
        content = [a for a in ax.get_children()
                   if id(a) not in before and a not in PANEL_FRAME]
        groups.append((rect, content))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for rect, content in groups:
        centre_content(ax, renderer, content, *rect)

    # inter-panel flow
    arrow(ax, (17.0, 26.5), (20.0, 26.5), lw=1.3)
    arrow(ax, (45.0, 26.5), (48.0, 26.5), lw=1.3)
    arrow(ax, (72.0, 26.5), (75.0, 26.5), lw=1.3)
    arrow(ax, (95.0, 26.5), (98.0, 26.5), lw=1.3)
    ax.text(46.5, 28.0, "$z$", ha="center", va="bottom", fontsize=7.5,
            color=C_DARK)

    out = OUT_DIR / "fig_model_architecture.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
