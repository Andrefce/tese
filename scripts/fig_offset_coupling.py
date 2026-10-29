"""Positive-offset coupling figure for the implicit-decoder subsection.

Top row, schematic with analytic fields: what two independent output heads
would permit.
  (a) Long-axis section. Both surfaces pass through every contour point on the
      acquired SAX planes, yet between two planes the epicardium touches the
      endocardium (left wall) or crosses inside it (right wall).
  (b) Fields along the ray marked in (a): an interval where f_endo < 0 and
      f_epi > 0, i.e. points inside the cavity and outside the epicardium.

Bottom row, the proposed model (v2 checkpoint) on ACDC patient002 at ED.
  (c) Long-axis section of the predicted fields: level sets of f_endo, the
      endocardium {f_endo = 0} and the epicardium {f_epi = 0} = {f_endo = delta}.
  (d) Fields along a ray halfway between two acquired planes: f_epi lies below
      f_endo everywhere, so the cavity is always enclosed by the epicardium.

The raw decoder output carries a ripple of about 0.5 mm with a period of about
3 mm near the endocardium; the mesh pipeline removes it before marching cubes.
For display only, f_endo and delta are smoothed in the section plane with a
2 mm Gaussian. Smoothing is linear, so f_epi = f_endo - delta and delta > 0
still hold exactly for the displayed fields. Field values are converted to
millimetres with the case normalisation scale, and the section is cut at the
most basal acquired plane, as the extracted meshes are.

The network evaluation is cached in ``CACHE``; pass ``--refresh`` to recompute.

Outputs: images/fig_offset_coupling.pdf and images/fig_offset_coupling.png

Run:
    C:/Python313/python.exe scripts/fig_offset_coupling.py [--refresh]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches, patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

THESIS = Path(__file__).resolve().parents[1]
EVAL_DEMO = THESIS / "scripts" / "eval_demo"
V2_DIR = THESIS / "test-new-model"
MODEL_PATH = V2_DIR / "runs" / "u1u2_e50" / "cardiosdf_v2_best.pt"
PATIENT = "patient002"
ED_FRAME = 1  # ED frame listed in notebooks/patient002/Info.cfg
SEG_PATH = THESIS / "notebooks" / PATIENT / f"{PATIENT}_frame{ED_FRAME:02d}_gt.nii"
CACHE = EVAL_DEMO / "outputs" / f"offset_coupling_v2_{PATIENT}.npz"
OUT_STEM = THESIS / "images" / "fig_offset_coupling"

SECTION_ANGLE_DEG = 0.0
STEP_MM = 0.25
U_RANGE_MM = (-68.0, 68.0)
Z_RANGE_MM = (-70.0, 52.0)
SMOOTH_MM = 2.0
RAY_TARGET_Z_MM = -10.0
RAY_LENGTH_MM = 56.0
POINTS_PER_RING = 60

FIG_W, FIG_H = 6.30, 6.05

INK = "#1F2933"
GREY = "#5F6B7A"
ENDO = "#0072B2"
EPI = "#D55E00"
CAVITY = "#DCEAF6"
WALL = "#FBE1CF"
CONTRA = "#F3C3C0"
CONTRA_EDGE = "#C0392B"
PLANE = "#9AA5B1"
ISO = "#8C99A6"
DELTA_FILL = "#7B8794"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.0,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.cal": "cmsy10",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
    "hatch.linewidth": 0.6,
    "hatch.color": CONTRA_EDGE,
})

F_ENDO = r"$f_{\mathrm{endo}}$"
F_EPI = r"$f_{\mathrm{epi}}$"
HALO = [patheffects.withStroke(linewidth=2.2, foreground="white")]


# ──────────────────────────────────────────────────────────────────────────
# Data: proposed model, long-axis section and ray
# ──────────────────────────────────────────────────────────────────────────
def ring_plane_crossings(xyz_mm: np.ndarray, tissue: np.ndarray,
                         angle: float) -> np.ndarray:
    """Points where each contour ring crosses the section plane: (u, z, tissue)."""
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-np.sin(angle), np.cos(angle)])
    rings = xyz_mm.reshape(-1, POINTS_PER_RING, 3)
    ring_tissue = tissue.reshape(-1, POINTS_PER_RING)[:, 0]
    rows = []
    for ring, t in zip(rings, ring_tissue):
        d = ring[:, :2] @ normal
        a = ring[:, :2] @ direction
        nxt = np.roll(np.arange(len(ring)), -1)
        for i in np.flatnonzero(np.sign(d) != np.sign(d[nxt])):
            j = nxt[i]
            w = d[i] / (d[i] - d[j])
            rows.append((a[i] + w * (a[j] - a[i]), ring[i, 2], t))
    return np.asarray(rows, np.float64)


def compute_case() -> dict:
    sys.path[:0] = [str(V2_DIR), str(EVAL_DEMO)]
    warnings.filterwarnings("ignore")
    from cardiosdf2.model import load_v2  # noqa: PLC0415
    from cardiosdf_model import DEVICE, encode_contours, query_points  # noqa: PLC0415
    from geometry import extract_contours, load_segmentation  # noqa: PLC0415

    net, cfg, meta = load_v2(MODEL_PATH, DEVICE)
    print(f"v2 checkpoint {MODEL_PATH.relative_to(THESIS)} (epoch {meta['epoch']})")
    contours = extract_contours(load_segmentation(SEG_PATH))
    xyz, tissue = contours["xyz"], contours["tissue"]
    scale = float(contours["scale"])
    cond = encode_contours(net, xyz, tissue, cfg, phase_val=0.0)

    angle = np.deg2rad(SECTION_ANGLE_DEG)
    u = np.arange(U_RANGE_MM[0], U_RANGE_MM[1] + 0.5 * STEP_MM, STEP_MM)
    z = np.arange(Z_RANGE_MM[0], Z_RANGE_MM[1] + 0.5 * STEP_MM, STEP_MM)
    uu, zz = np.meshgrid(u, z)
    pts = np.column_stack([uu.ravel() * np.cos(angle), uu.ravel() * np.sin(angle),
                           zz.ravel()]) / scale
    f_endo, _, delta = query_points(net, cond, pts.astype(np.float32))
    sigma = SMOOTH_MM / STEP_MM
    f_endo = gaussian_filter(f_endo.reshape(uu.shape) * scale, sigma, mode="nearest")
    delta = gaussian_filter(delta.reshape(uu.shape) * scale, sigma, mode="nearest")

    xyz_mm = xyz.astype(np.float64) * scale
    planes = np.unique(np.round(xyz_mm[:, 2], 3))
    mids = 0.5 * (planes[1:] + planes[:-1])
    ray_z = float(mids[np.argmin(np.abs(mids - RAY_TARGET_Z_MM))])

    out = {
        "u": u, "z": z, "f_endo": f_endo.astype(np.float32),
        "delta": delta.astype(np.float32), "planes": planes, "ray_z": ray_z,
        "crossings": ring_plane_crossings(xyz_mm, tissue, angle),
        "scale": scale, "epoch": np.int64(meta["epoch"]),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f"cached fields to {CACHE.relative_to(THESIS)}")
    return out


def load_case(refresh: bool) -> dict:
    if CACHE.exists() and not refresh:
        with np.load(CACHE) as data:
            print(f"using cached fields: {CACHE.relative_to(THESIS)}")
            return {key: data[key] for key in data.files}
    return compute_case()


def first_crossing(r: np.ndarray, f: np.ndarray) -> float:
    i = int(np.flatnonzero((f[:-1] < 0) & (f[1:] >= 0))[0])
    return float(r[i] - f[i] * (r[i + 1] - r[i]) / (f[i + 1] - f[i]))


# ──────────────────────────────────────────────────────────────────────────
# Schematic geometry: independent heads
# ──────────────────────────────────────────────────────────────────────────
Z_BASE = 45.0
S_PLANES = Z_BASE - 17.0 * np.arange(6)
ENDO_SHAPE = (30.0, 92.0)  # radius at the base, base-to-apex length
EPI_SHAPE = (39.0, 104.0)
CROSS_GAP = (-6.0, 11.0)  # right wall: epicardium crosses inside
TOUCH_GAP = (-40.0, -23.0)  # left wall: epicardium touches the endocardium
S_RAY_Z = 2.5
ENDO_BULGE = 4.0


def radius(z, shape):
    a, length = shape
    t = np.clip((Z_BASE - np.asarray(z, float)) / length, 0.0, 1.0)
    return a * np.sqrt(1.0 - t ** 2)


def bump(z, gap):
    lo, hi = gap
    z = np.asarray(z, float)
    s = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    return np.sin(np.pi * s) ** 2


def touch_depth():
    zc = 0.5 * sum(TOUCH_GAP)
    return float(radius(zc, EPI_SHAPE) - radius(zc, ENDO_SHAPE))


def schematic_radii(z):
    """Radii of the independently predicted surfaces on the left and right walls."""
    endo_l = radius(z, ENDO_SHAPE) + ENDO_BULGE * bump(z, TOUCH_GAP)
    endo_r = radius(z, ENDO_SHAPE) + 5.0 * bump(z, CROSS_GAP)
    epi_l = radius(z, EPI_SHAPE) - (touch_depth() - ENDO_BULGE) * bump(z, TOUCH_GAP)
    epi_r = radius(z, EPI_SHAPE) - 10.0 * bump(z, CROSS_GAP)
    return endo_l, endo_r, epi_l, epi_r


def outline(shape, side_radii_index):
    """Closed-bottom outline: left wall down to the apex, right wall back up."""
    phi = np.linspace(0.0, 0.5 * np.pi, 400)
    z = Z_BASE - shape[1] * np.sin(phi)
    radii = schematic_radii(z)
    left, right = radii[side_radii_index], radii[side_radii_index + 1]
    return (np.concatenate([-left, right[::-1]]), np.concatenate([z, z[::-1]]))


# ──────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────
def panel_title(fig, ax, y, tag, title):
    x = ax.get_position().x0
    fig.text(x, y, tag, ha="left", va="bottom", fontsize=7.6,
             fontweight="bold", color=INK)
    fig.text(x + 0.033, y, title, ha="left", va="bottom", fontsize=7.6,
             color=INK)


def region_fill(ax, uu, zz, mask, color, hatch=None):
    ax.contourf(uu, zz, mask.astype(float), levels=[0.5, 1.5], colors=[color],
                hatches=[hatch], zorder=1)


def draw_planes(ax, planes, x0, x1):
    for zp in planes:
        ax.plot([x0, x1], [zp, zp], color=PLANE, lw=0.5, ls=(0, (1.5, 1.8)),
                zorder=2)


def draw_points(ax, u, z, color):
    ax.scatter(u, z, s=11, color=color, edgecolor="white", linewidth=0.6,
               zorder=6)


def ray_arrow(ax, z, x1, label):
    ax.annotate("", xy=(x1, z), xytext=(0.0, z), zorder=7,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9,
                                mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.plot([0.0], [z], "o", ms=2.6, color=INK, zorder=7)
    ax.text(x1 - 1.0, z + 1.6, label, ha="right", va="bottom", fontsize=6.6,
            color=INK, zorder=7, path_effects=HALO)


def gap_marker(ax, x, z_lo, z_hi, text):
    ax.annotate("", xy=(x, z_hi), xytext=(x, z_lo),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.6,
                                mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.text(x + 1.4, 0.5 * (z_lo + z_hi), text, ha="left", va="center",
            fontsize=6.0, color=GREY, linespacing=1.2)


def section_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def profile_axes(ax, xlim, ylim, xlabel, numeric=True):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(0.0, color=INK, lw=0.6, zorder=3)
    ax.set_xlabel(xlabel, fontsize=6.8, color=INK, labelpad=2)
    ax.set_ylabel("field value (mm)" if numeric else "field value",
                  fontsize=6.8, color=INK, labelpad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, width=0.6, labelsize=6.2, colors=INK, pad=1.5)
    if not numeric:
        ax.set_xticks([])
        ax.set_yticks([])


def vertical_gap(ax, x, y0, y1, text, color, side="right"):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), zorder=8,
                arrowprops=dict(arrowstyle="<->", color=color, lw=0.8,
                                mutation_scale=6, shrinkA=0, shrinkB=0))
    dx = 1.0 if side == "right" else -1.0
    ax.text(x + dx, 0.5 * (y0 + y1), text, ha="left" if side == "right" else "right",
            va="center", fontsize=6.6, color=color, zorder=8, path_effects=HALO)


# ──────────────────────────────────────────────────────────────────────────
# Panels
# ──────────────────────────────────────────────────────────────────────────
XLIM_SECTION = (-64.0, 68.0)
YLIM_SECTION = (-66.0, 47.0)


def panel_schematic_section(ax):
    u = np.arange(XLIM_SECTION[0], XLIM_SECTION[1], 0.2)
    z = np.arange(YLIM_SECTION[0], Z_BASE + 0.1, 0.2)
    uu, zz = np.meshgrid(u, z)
    endo_l, endo_r, epi_l, epi_r = schematic_radii(zz)
    endo_l = np.where(zz >= Z_BASE - ENDO_SHAPE[1], endo_l, 0.0)
    endo_r = np.where(zz >= Z_BASE - ENDO_SHAPE[1], endo_r, 0.0)
    epi_l = np.where(zz >= Z_BASE - EPI_SHAPE[1], epi_l, 0.0)
    epi_r = np.where(zz >= Z_BASE - EPI_SHAPE[1], epi_r, 0.0)
    in_endo = np.where(uu >= 0, uu < endo_r, -uu < endo_l)
    in_epi = np.where(uu >= 0, uu < epi_r, -uu < epi_l)
    region_fill(ax, uu, zz, in_endo & in_epi, CAVITY)
    region_fill(ax, uu, zz, in_epi & ~in_endo, WALL)
    region_fill(ax, uu, zz, in_endo & ~in_epi, CONTRA, hatch="//////")

    draw_planes(ax, S_PLANES, -50.0, 50.0)
    for shape, k, color in ((ENDO_SHAPE, 0, ENDO), (EPI_SHAPE, 2, EPI)):
        x, y = outline(shape, k)
        ax.plot(x, y, color=color, lw=1.3, zorder=5, solid_capstyle="round")
    endo_l, endo_r, epi_l, epi_r = schematic_radii(S_PLANES)
    draw_points(ax, np.concatenate([-endo_l, endo_r]), np.tile(S_PLANES, 2), ENDO)
    draw_points(ax, np.concatenate([-epi_l, epi_r]), np.tile(S_PLANES, 2), EPI)

    ray_arrow(ax, S_RAY_Z, 46.0, "(b)")
    gap_marker(ax, 54.0, CROSS_GAP[0], CROSS_GAP[1], "gap")
    zc = 0.5 * sum(TOUCH_GAP)
    ax.annotate("touch", xy=(-radius(zc, ENDO_SHAPE) - ENDO_BULGE, zc),
                xytext=(-46.0, -52.0), fontsize=6.4, color=CONTRA_EDGE,
                ha="center", va="center", zorder=8,
                arrowprops=dict(arrowstyle="-", color=CONTRA_EDGE, lw=0.6,
                                shrinkA=1.5, shrinkB=1.5))
    ax.annotate("cross", xy=(29.5, -0.5), xytext=(48.0, -24.0), fontsize=6.4,
                color=CONTRA_EDGE, ha="center", va="center", zorder=8,
                arrowprops=dict(arrowstyle="-", color=CONTRA_EDGE, lw=0.6,
                                shrinkA=1.5, shrinkB=1.5))
    section_axes(ax, XLIM_SECTION, YLIM_SECTION)


def panel_schematic_profile(ax):
    _, endo_r, _, epi_r = schematic_radii(S_RAY_Z)
    r = np.linspace(0.0, 48.0, 200)
    f_endo, f_epi = r - endo_r, r - epi_r
    ax.axvspan(0.0, epi_r, color=CAVITY, lw=0, zorder=0)
    ax.axvspan(epi_r, endo_r, facecolor=CONTRA, edgecolor=CONTRA_EDGE,
               hatch="//////", lw=0, zorder=0)
    ax.fill_between(r, f_endo, f_epi, color=CONTRA_EDGE, alpha=0.12, lw=0,
                    zorder=1)
    ax.plot(r, f_endo, color=ENDO, lw=1.4, zorder=5)
    ax.plot(r, f_epi, color=EPI, lw=1.4, zorder=5)
    ax.plot([endo_r, epi_r], [0, 0], "o", ms=3.4, color=INK, zorder=6)

    ax.text(44.0, 44.0 - endo_r - 3.0, F_ENDO, color=ENDO, fontsize=7.4,
            ha="center", va="top", zorder=8, path_effects=HALO)
    ax.text(42.0, 42.0 - epi_r + 3.0, F_EPI, color=EPI, fontsize=7.4,
            ha="center", va="bottom", zorder=8, path_effects=HALO)
    x_gap = 6.0
    ax.annotate("", xy=(x_gap, x_gap - epi_r), xytext=(x_gap, x_gap - endo_r),
                zorder=8, arrowprops=dict(arrowstyle="<->", color=CONTRA_EDGE,
                                          lw=0.8, mutation_scale=6,
                                          shrinkA=0, shrinkB=0))
    ax.text(x_gap + 1.5, x_gap - endo_r - 1.0,
            r"$f_{\mathrm{endo}}-f_{\mathrm{epi}}<0$", ha="left", va="center",
            fontsize=6.6, color=CONTRA_EDGE, zorder=8, path_effects=HALO)
    ax.annotate(r"$f_{\mathrm{endo}}<0$ and $f_{\mathrm{epi}}>0$:" "\n"
                "inside the cavity and outside\nthe epicardium at once",
                xy=(0.5 * (epi_r + endo_r), 14.0), xytext=(12.0, 19.0),
                fontsize=6.3, color=CONTRA_EDGE, ha="center", va="center",
                zorder=8, linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=CONTRA_EDGE, lw=0.6,
                                shrinkA=2, shrinkB=0))
    ax.text(21.0, -29.5, "cavity", ha="center", va="bottom",
            fontsize=6.4, color=GREY)
    ax.text(0.5 * (endo_r + 48.0), -26.0, "outside", ha="center", va="bottom",
            fontsize=6.4, color=GREY)
    profile_axes(ax, (0.0, 48.0), (-31.0, 26.0),
                 "distance from the long axis along the ray", numeric=False)


def panel_model_section(ax, case):
    u, z = case["u"], case["z"]
    uu, zz = np.meshgrid(u, z)
    # The decoder closes the cavity just below the most basal plane; the cut
    # leaves the base open, as the trimmed meshes are.
    top = float(case["planes"].max()) - 1.5
    f_endo = np.where(zz <= top, case["f_endo"], np.nan)
    f_epi = f_endo - case["delta"]
    in_endo, in_epi = f_endo < 0, f_epi < 0
    if np.any(in_endo & ~in_epi):
        raise RuntimeError("coupled fields violate nesting; check the cache")
    region_fill(ax, uu, zz, in_endo, CAVITY)
    region_fill(ax, uu, zz, in_epi & ~in_endo, WALL)

    # Outer level sets of f_endo, fading with distance from the endocardium.
    for k, level in enumerate(np.arange(3.0, 12.1, 3.0)):
        ax.contour(uu, zz, f_endo, levels=[level], colors=ISO,
                   linewidths=0.45, alpha=0.85 - 0.17 * k, zorder=2)
    draw_planes(ax, case["planes"], -50.0, 50.0)
    ax.contour(uu, zz, f_endo, levels=[0.0], colors=[ENDO], linewidths=1.3,
               zorder=5)
    ax.contour(uu, zz, f_epi, levels=[0.0], colors=[EPI], linewidths=1.3,
               zorder=5)
    pts = case["crossings"]
    for t, color in ((0.0, ENDO), (1.0, EPI)):
        sel = pts[:, 2] == t
        draw_points(ax, pts[sel, 0], pts[sel, 1], color)

    ray_arrow(ax, float(case["ray_z"]), 52.0, "(d)")
    lo = float(case["ray_z"]) - 5.0
    gap_marker(ax, 58.0, lo, lo + 10.0, "gap")
    ax.text(-12.0, 24.0, r"$f_{\mathrm{endo}}=0$", color=ENDO, fontsize=7.0,
            ha="center", va="center", zorder=8, path_effects=HALO)
    ax.annotate(r"$f_{\mathrm{endo}}=\delta$" "\n" r"($f_{\mathrm{epi}}=0$)",
                xy=(-40.0, -30.0), xytext=(-46.0, -54.0), color=EPI,
                fontsize=6.6, ha="center", va="center", zorder=8,
                linespacing=1.25, path_effects=HALO,
                arrowprops=dict(arrowstyle="-", color=EPI, lw=0.6,
                                shrinkA=2, shrinkB=1))
    ax.annotate(r"level sets of $f_{\mathrm{endo}}$", xy=(41.0, -45.0),
                xytext=(52.0, -58.0), color=GREY, fontsize=6.0,
                ha="center", va="center", zorder=8,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5,
                                shrinkA=2, shrinkB=1))
    ax.plot([15.0, 25.0], [-63.0, -63.0], color=INK, lw=1.2,
            solid_capstyle="butt")
    ax.text(20.0, -64.5, "10 mm", ha="center", va="top", fontsize=6.0,
            color=INK)
    section_axes(ax, XLIM_SECTION, YLIM_SECTION)


def panel_model_profile(ax, case):
    interp_e = RegularGridInterpolator((case["z"], case["u"]), case["f_endo"])
    interp_d = RegularGridInterpolator((case["z"], case["u"]), case["delta"])
    r = np.arange(0.0, RAY_LENGTH_MM, 0.1)
    q = np.column_stack([np.full_like(r, float(case["ray_z"])), r])
    f_endo = interp_e(q)
    delta = interp_d(q)
    f_epi = f_endo - delta
    r_endo, r_epi = first_crossing(r, f_endo), first_crossing(r, f_epi)
    print(f"ray z = {float(case['ray_z']):.1f} mm: endocardium {r_endo:.1f} mm, "
          f"epicardium {r_epi:.1f} mm, delta range "
          f"{delta.min():.1f}-{delta.max():.1f} mm")

    ax.axvspan(0.0, r_endo, color=CAVITY, lw=0, zorder=0)
    ax.axvspan(r_endo, r_epi, color=WALL, lw=0, zorder=0)
    ax.fill_between(r, f_epi, f_endo, color=DELTA_FILL, alpha=0.16, lw=0,
                    zorder=1)
    ax.plot(r, f_endo, color=ENDO, lw=1.4, zorder=5)
    ax.plot(r, f_epi, color=EPI, lw=1.4, zorder=5)
    ax.plot([r_endo, r_epi], [0, 0], "o", ms=3.4, color=INK, zorder=6)

    x_label = r[-1] - 4.0
    ax.text(x_label, np.interp(x_label, r, f_endo) + 3.0, F_ENDO, color=ENDO,
            fontsize=7.4, ha="center", va="bottom", zorder=8, path_effects=HALO)
    ax.text(x_label, np.interp(x_label, r, f_epi) - 3.0, F_EPI, color=EPI,
            fontsize=7.4, ha="center", va="top", zorder=8, path_effects=HALO)
    x_delta = 10.0
    vertical_gap(ax, x_delta, np.interp(x_delta, r, f_epi),
                 np.interp(x_delta, r, f_endo), r"$\delta(x)>0$", INK)
    ax.annotate("", xy=(r_epi, 4.5), xytext=(r_endo, 4.5), zorder=8,
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7,
                                mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.text(0.5 * (r_endo + r_epi), 6.0, "wall", ha="center", va="bottom",
            fontsize=6.4, color=INK, path_effects=HALO, zorder=8)
    ax.text(0.5 * r_endo, -29.0, "cavity", ha="center", va="bottom",
            fontsize=6.4, color=GREY)
    ax.text(0.5 * (r_epi + r[-1]), -29.0, "outside", ha="center", va="bottom",
            fontsize=6.4, color=GREY)
    profile_axes(ax, (0.0, r[-1]), (-31.0, 26.0),
                 "distance from the long axis along the ray (mm)")


# ──────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────
def legend(fig):
    handles = [
        Line2D([0], [0], color=ENDO, lw=1.4, label="endocardium"),
        Line2D([0], [0], color=EPI, lw=1.4, label="epicardium"),
        Line2D([0], [0], ls="none", marker="o", ms=4.0, mfc=GREY, mec="white",
               label="contour point on an acquired plane"),
        Line2D([0], [0], color=PLANE, lw=0.8, ls=(0, (1.5, 1.8)),
               label="acquired SAX plane"),
        patches.Patch(facecolor=CAVITY, label="cavity"),
        patches.Patch(facecolor=WALL, label="myocardial wall"),
        patches.Patch(facecolor=CONTRA, edgecolor=CONTRA_EDGE, hatch="//////",
                      lw=0, label="inside cavity, outside epicardium"),
        patches.Patch(facecolor=DELTA_FILL, alpha=0.3,
                      label=r"offset $\delta=f_{\mathrm{endo}}-f_{\mathrm{epi}}$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=6.4, handlelength=1.8, columnspacing=1.4,
               bbox_to_anchor=(0.5, 0.0))


def row_header(fig, y, text):
    fig.text(0.02, y, text, ha="left", va="bottom", fontsize=7.8,
             fontweight="bold", color=INK)
    fig.add_artist(Line2D([0.02, 0.98], [y - 0.006, y - 0.006],
                          transform=fig.transFigure, color="#C9D1DB", lw=0.6))


def build(case: dict) -> plt.Figure:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    left_w, gap, right_x = 0.36, 0.09, 0.53
    right_w = 0.98 - right_x
    row_h = 0.355
    rows = (0.565, 0.115)
    axes = {}
    for key, y in zip(("top", "bottom"), rows):
        axes[key + "_section"] = fig.add_axes([0.02, y, left_w, row_h])
        axes[key + "_profile"] = fig.add_axes([right_x, y + 0.04, right_w,
                                               row_h - 0.07])

    row_header(fig, rows[0] + row_h + 0.045,
               "Independent output heads (schematic)")
    row_header(fig, rows[1] + row_h + 0.045,
               "Positive-offset coupling (proposed model, one ACDC case at ED)")

    panel_schematic_section(axes["top_section"])
    panel_schematic_profile(axes["top_profile"])
    panel_model_section(axes["bottom_section"], case)
    panel_model_profile(axes["bottom_profile"], case)

    for key, y in zip(("top", "bottom"), rows):
        title_y = y + row_h + 0.004
        first, second = (("(a)", "(b)") if key == "top" else ("(c)", "(d)"))
        panel_title(fig, axes[key + "_section"], title_y, first,
                    "Long-axis section")
        panel_title(fig, axes[key + "_profile"], title_y, second,
                    f"Fields along the ray in {first}")
    legend(fig)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true",
                        help="re-evaluate the network instead of using the cache")
    args = parser.parse_args()
    case = load_case(args.refresh)
    fig = build(case)
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {OUT_STEM.with_suffix('.pdf').relative_to(THESIS)} and .png")


if __name__ == "__main__":
    main()
