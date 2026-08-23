"""Full-page portrait schematic of the proposed model architecture.

Monochrome line-art plate: sparse SAX contour observation -> contour encoder
(global shape path and local context path) -> implicit decoder -> coupled
output heads -> marching-cubes surfaces.

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
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle

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
RULE = "#B5B5B5"
FILL = "#EDEDED"
WHITE = "#FFFFFF"
DASH = (0, (3, 2))

W, H = 100.0, 140.0


# ---------------------------------------------------------------- primitives

def rule(ax, y):
    ax.add_line(Line2D([4.0, 96.0], [y, y], color=RULE, lw=0.5, zorder=1))


def header(ax, y, tag, title):
    ax.text(4.0, y, f"({tag})", ha="left", va="center", fontsize=7.2,
            fontweight="bold", color=INK, zorder=6)
    ax.text(9.0, y, title, ha="left", va="center", fontsize=7.2,
            fontweight="bold", color=INK, zorder=6)


def box(ax, x0, y0, x1, y1, text, fs=6.2):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=WHITE,
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


# ------------------------------------------------------------------ stage (a)

def draw_input(ax):
    header(ax, 138.0, "a", "Input: sparse short-axis contour observation")

    for k in range(6):
        t = k / 5.0
        cy = 121.6 + 2.4 * k
        cx = 15.0 + 0.9 * t
        we, he = 4.0 + 3.6 * t, 1.0 + 0.85 * t
        ax.add_patch(Ellipse((cx, cy), we + 2.2, he + 0.85, fill=False,
                             edgecolor=INK, lw=0.65, ls=DASH, zorder=3))
        ax.add_patch(Ellipse((cx, cy), we, he, fill=False, edgecolor=INK,
                             lw=0.65, zorder=4))
        th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ax.plot(cx + 0.5 * we * np.cos(th), cy + 0.5 * he * np.sin(th),
                ls="none", marker="o", ms=0.7, color=INK, zorder=5)

    arr(ax, (5.5, 121.5), (5.5, 133.5), lw=0.55)
    txt(ax, 5.5, 135.9, "base", fs=5.7, color=GREY)
    txt(ax, 5.5, 119.8, "apex", fs=5.7, color=GREY)

    line(ax, [(6.0, 118.3), (9.0, 118.3)])
    txt(ax, 9.8, 118.3, "endocardium", fs=5.7, ha="left", color=GREY)
    line(ax, [(23.0, 118.3), (26.0, 118.3)], dashed=True)
    txt(ax, 26.8, 118.3, "epicardium", fs=5.7, ha="left", color=GREY)

    lines = [
        r"$P \in \mathbb{R}^{N \times 5}$   (contour tensor)",
        r"$p_i = (x_i,\; y_i,\; z_i,\; \mathrm{tissue}_i,\; "
        r"\mathrm{phase}_i)$",
        r"tissue: endocardial or epicardial;   phase: ED or ES",
        r"$\leq 10$ SAX levels $\times$ 2 rings $\times$ 60 points,"
        r"   $N \leq 1200$",
    ]
    for i, s in enumerate(lines):
        txt(ax, 36.0, 134.4 - 3.0 * i, s, fs=6.2, ha="left")
    txt(ax, 36.0, 122.0,
        "at inference the model reads contour points only:\n"
        "no image intensities and no mesh",
        fs=5.9, ha="left", color=GREY, style="italic")


# ------------------------------------------------------------------ stage (b)

def draw_encoder(ax):
    header(ax, 115.0, "b", "Contour encoder")

    txt(ax, 4.0, 112.8, "global shape path", fs=5.9, ha="left", color=GREY,
        style="italic")
    arr(ax, (15.0, 111.6), (15.0, 109.7))
    txt(ax, 16.4, 110.7, r"$P$", fs=5.9, ha="left", color=GREY)
    box(ax, 4.0, 98.0, 26.0, 109.5,
        "point network $\\phi_{\\mathrm{enc}}$\n"
        r"$5 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 256$")
    txt(ax, 27.5, 110.7, r"$[N \times 256]$", fs=5.7, color=GREY)

    arr(ax, (26.0, 107.0), (29.0, 107.0))
    arr(ax, (26.0, 100.5), (29.0, 100.5))
    box(ax, 29.0, 104.5, 47.0, 109.5, "max-pool over\nendocardial points")
    box(ax, 29.0, 98.0, 47.0, 103.0, "max-pool over\nepicardial points")

    arr(ax, (47.0, 107.0), (50.0, 105.6), rad=-0.12)
    arr(ax, (47.0, 100.5), (50.0, 102.4), rad=0.12)
    box(ax, 50.0, 100.0, 68.0, 108.0,
        r"concatenate $\mathbb{R}^{512}$" "\n" r"linear $\rightarrow 256$")
    arr(ax, (68.0, 104.0), (71.0, 104.0))
    box(ax, 71.0, 101.5, 92.0, 106.5,
        r"global code $z \in \mathbb{R}^{256}$")
    txt(ax, 40.0, 95.8, "pooling is invariant to contour-point order",
        fs=5.8, color=GREY, style="italic")

    txt(ax, 4.0, 94.0, "local context path", fs=5.9, ha="left", color=GREY,
        style="italic")
    arr(ax, (15.0, 92.8), (15.0, 90.2))
    txt(ax, 16.4, 91.6, r"$P$", fs=5.9, ha="left", color=GREY)
    box(ax, 4.0, 83.0, 26.0, 90.0,
        "point network $\\phi_{\\mathrm{loc}}$\n"
        r"$5 \rightarrow 64 \rightarrow 64 \rightarrow 32$")
    arr(ax, (26.0, 86.5), (29.0, 86.5))
    box(ax, 29.0, 83.0, 49.0, 90.0,
        "scatter-average into\na $16^3$ voxel grid")
    arr(ax, (49.0, 86.5), (52.0, 86.5))
    box(ax, 52.0, 83.0, 70.0, 90.0,
        "3D CNN\n" r"$3 \times$ conv$_{3^3}$, 32 ch.")
    arr(ax, (70.0, 86.5), (73.0, 86.5))
    box(ax, 73.0, 84.5, 96.0, 88.5, r"$V \in \mathbb{R}^{32 \times 16^3}$")

    arr(ax, (84.5, 84.5), (84.5, 81.0))
    box(ax, 34.0, 76.0, 92.0, 81.0,
        r"trilinear sampling of $V$ at the query point $x$"
        r"$\;\rightarrow\;$"
        r"$v(x) \in \mathbb{R}^{32}$", fs=6.0)


# ------------------------------------------------------------------ stage (c)

def draw_decoder(ax):
    header(ax, 71.5, "c", "Implicit decoder")

    box(ax, 4.0, 54.0, 23.0, 63.0,
        r"query point $x \in \mathbb{R}^{3}$" "\n"
        "Fourier features\n"
        r"$\gamma(x) \in \mathbb{R}^{21}$ ($L=3$)", fs=6.0)
    arr(ax, (23.0, 58.5), (26.0, 58.5))
    box(ax, 26.0, 54.0, 53.0, 63.0,
        r"concatenate $[\,z,\, \gamma(x)\,] \in \mathbb{R}^{277}$" "\n"
        r"linear $\rightarrow 512$" "\n"
        r"$+\; W_{\mathrm{loc}}\, v(x)$", fs=6.0)
    arr(ax, (53.0, 58.5), (56.0, 58.5))

    nb, x_lo, x_hi, bw = 8, 56.0, 82.0, 2.0
    step = (x_hi - x_lo - bw) / (nb - 1)
    centres = [x_lo + i * step + bw / 2 for i in range(nb)]
    for i, cx in enumerate(centres):
        ax.add_patch(Rectangle((cx - bw / 2, 54.0), bw, 9.0,
                               facecolor=FILL if i == 4 else WHITE,
                               edgecolor=INK, linewidth=0.55, zorder=3))
    arr(ax, (82.0, 58.5), (85.0, 58.5))
    box(ax, 85.0, 56.0, 96.0, 61.0, r"$h \in \mathbb{R}^{512}$")

    ax.add_patch(FancyArrowPatch(
        (53.0, 63.0), (centres[4], 63.2), arrowstyle="-|>",
        mutation_scale=6.5, connectionstyle="arc3,rad=-0.18", linewidth=0.65,
        color=INK, zorder=5))
    txt(ax, 62.0, 66.8,
        r"skip: $[\,h,\, z,\, \gamma(x)\,]$ re-entered at layer 4",
        fs=5.8, color=GREY)

    line(ax, [(centres[0], 51.5), (centres[4], 51.5)], dashed=True)
    for cx in (centres[0], centres[4]):
        arr(ax, (cx, 51.5), (cx, 54.0), dashed=True)
    txt(ax, centres[0] - 1.6, 51.5, r"$v(x)$", fs=5.9, ha="right", color=GREY)
    txt(ax, 69.0, 49.0,
        "8 fully connected layers, width 512, softplus activation",
        fs=5.9, color=GREY)


# ------------------------------------------------------------------ stage (d)

def draw_heads(ax):
    header(ax, 43.5, "d", "Output heads and field coupling")

    box(ax, 4.0, 32.0, 28.0, 39.0,
        "endocardial head\n" r"linear $512 \rightarrow 1$" "\n"
        r"$f_{\mathrm{endo}}(x)$")
    box(ax, 31.0, 32.0, 55.0, 39.0,
        "offset head\n" r"linear $512 \rightarrow 1$" "\n"
        r"$+\; W_{v}\, v(x)$")
    arr(ax, (55.0, 35.5), (58.0, 35.5))
    box(ax, 58.0, 32.0, 96.0, 39.0,
        r"$\delta(x) = \tau_{\min} + (\delta_{\mathrm{cap}} - \tau_{\min})"
        r"\, \sigma(\cdot)$" "\n"
        r"$\tau_{\min} = 0.05$,   $\delta_{\mathrm{cap}} = 0.45$"
        "   (normalised units)")

    arr(ax, (16.0, 32.0), (16.0, 30.0))
    arr(ax, (62.0, 32.0), (52.0, 30.2), rad=0.16)
    box(ax, 4.0, 25.0, 60.0, 30.0,
        r"$f_{\mathrm{epi}}(x) = f_{\mathrm{endo}}(x) - \delta(x)$,"
        r"   $\delta \geq \tau_{\min} > 0$")
    txt(ax, 77.0, 27.5,
        "the offset orders the two implicit fields;\n"
        "it is not a measured wall thickness",
        fs=5.8, color=GREY, style="italic")


# --------------------------------------------------------------- LV mesh (e)

A_EPI, C_EPI = 3.30, 5.10
A_ENDO, C_ENDO = 2.25, 4.25
Z_BASE = 1.55
CUT_V0, CUT_V1 = np.radians(-95.0), np.radians(-5.0)

# near side is -Y and the camera sits slightly above the base plane
TOWARD_VIEWER = np.array([0.0, -0.94, 0.34])
LIGHT = np.array([-0.35, -0.75, 0.56])
LIGHT /= np.linalg.norm(LIGHT)


def _radius(z, a, c):
    return a * np.sqrt(max(0.0, 1.0 - (z * z) / (c * c)))


def _project(pts, xc, yc, s=1.45):
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    return np.column_stack([xc + s * x, yc + s * (0.94 * z + 0.34 * y)])


def _shade(pts, darken=1.0):
    normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return 0.85
    normal /= norm
    if normal @ TOWARD_VIEWER < 0:
        normal = -normal
    return float(np.clip((0.74 + 0.26 * max(0.0, normal @ LIGHT)) * darken,
                         0.0, 1.0))


def _shell_quads(a, c, z0, z1, v0, v1, nz, nv, darken):
    zs = np.linspace(z0, z1, nz + 1)
    vs = np.linspace(v0, v1, nv + 1)
    quads = []
    for i in range(nz):
        for j in range(nv):
            corners = [(zs[i], vs[j]), (zs[i], vs[j + 1]),
                       (zs[i + 1], vs[j + 1]), (zs[i + 1], vs[j])]
            pts = np.array([[_radius(z, a, c) * np.cos(v),
                             _radius(z, a, c) * np.sin(v), z]
                            for z, v in corners])
            quads.append((pts, str(_shade(pts, darken)), "#8A8A8A", 0.22))
    return quads


def _cut_face_quads(v, nz):
    zs = np.linspace(-C_EPI, Z_BASE, nz + 1)
    quads = []
    for i in range(nz):
        za, zb = zs[i], zs[i + 1]
        pts = np.array([
            [_radius(za, A_ENDO, C_ENDO) * np.cos(v),
             _radius(za, A_ENDO, C_ENDO) * np.sin(v), za],
            [_radius(za, A_EPI, C_EPI) * np.cos(v),
             _radius(za, A_EPI, C_EPI) * np.sin(v), za],
            [_radius(zb, A_EPI, C_EPI) * np.cos(v),
             _radius(zb, A_EPI, C_EPI) * np.sin(v), zb],
            [_radius(zb, A_ENDO, C_ENDO) * np.cos(v),
             _radius(zb, A_ENDO, C_ENDO) * np.sin(v), zb],
        ])
        quads.append((pts, "0.86", INK, 0.3))
    return quads


def _base_ring_quads(v0, v1, nv):
    vs = np.linspace(v0, v1, nv + 1)
    r_in = _radius(Z_BASE, A_ENDO, C_ENDO)
    r_out = _radius(Z_BASE, A_EPI, C_EPI)
    quads = []
    for j in range(nv):
        va, vb = vs[j], vs[j + 1]
        pts = np.array([
            [r_in * np.cos(va), r_in * np.sin(va), Z_BASE],
            [r_out * np.cos(va), r_out * np.sin(va), Z_BASE],
            [r_out * np.cos(vb), r_out * np.sin(vb), Z_BASE],
            [r_in * np.cos(vb), r_in * np.sin(vb), Z_BASE],
        ])
        quads.append((pts, "0.86", INK, 0.3))
    return quads


def draw_lv_mesh(ax, xc, yc):
    keep0, keep1 = CUT_V1, CUT_V0 + 2.0 * np.pi
    quads = []
    quads += _shell_quads(A_EPI, C_EPI, -C_EPI, Z_BASE, keep0, keep1,
                          16, 30, 1.0)
    quads += _shell_quads(A_ENDO, C_ENDO, -C_ENDO, Z_BASE, keep0, keep1,
                          14, 30, 0.90)
    quads += _cut_face_quads(CUT_V0, 16) + _cut_face_quads(CUT_V1, 16)
    quads += _base_ring_quads(keep0, keep1, 30)

    # painter's algorithm: larger depth is farther from the camera
    quads.sort(key=lambda q: 0.94 * q[0][:, 1].mean()
               - 0.34 * q[0][:, 2].mean(), reverse=True)
    for index, (pts, face, edge, lw) in enumerate(quads):
        ax.add_patch(Polygon(_project(pts, xc, yc), closed=True,
                             facecolor=face, edgecolor=edge, linewidth=lw,
                             zorder=3 + index * 1e-3))

    vs = np.linspace(keep0, keep1, 240)
    for a, c in ((A_EPI, C_EPI), (A_ENDO, C_ENDO)):
        r = _radius(Z_BASE, a, c)
        rim = np.column_stack([r * np.cos(vs), r * np.sin(vs),
                               np.full_like(vs, Z_BASE)])
        pts = _project(rim, xc, yc)
        ax.plot(pts[:, 0], pts[:, 1], color=INK, lw=0.45, zorder=5)


def draw_output(ax):
    header(ax, 19.5, "e", "Surface extraction")

    box(ax, 4.0, 5.0, 26.0, 13.0,
        r"evaluate $f_{\mathrm{endo}}$ and $f_{\mathrm{epi}}$" "\n"
        r"on a $96^{3}$ query grid")
    arr(ax, (26.0, 9.0), (29.0, 9.0))
    box(ax, 29.0, 5.0, 55.0, 13.0,
        "marching cubes at each\nzero level set, then\n"
        "component and hole clean-up")
    arr(ax, (55.0, 9.0), (60.0, 9.0))

    draw_lv_mesh(ax, 66.0, 11.96)
    txt(ax, 66.0, 2.0, "reconstructed LV surfaces (cut-away)", fs=5.8,
        color=GREY, style="italic")

    line(ax, [(60.8, 16.0), (62.2, 13.2)], lw=0.5)
    txt(ax, 60.5, 16.3, "epicardium", fs=5.8, ha="right")
    line(ax, [(75.4, 14.3), (68.8, 11.6)], lw=0.5)
    txt(ax, 75.8, 14.5, "endocardium", fs=5.8, ha="left")

    txt(ax, 74.0, 6.5,
        "wall thickness is measured\n"
        "between the extracted surfaces,\n"
        r"not from the field offset $\delta$",
        fs=5.8, ha="left", color=GREY, style="italic")


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
    draw_decoder(ax)
    draw_heads(ax)
    draw_output(ax)

    for y in (117.0, 74.0, 46.0, 22.0):
        rule(ax, y)

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
