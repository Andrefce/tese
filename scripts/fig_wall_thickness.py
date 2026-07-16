"""Generate wall-thickness figures for the methodology chapter.

Produces two figures used in the wall-thickness evaluation section:

  images/fig_aha17_explanation.png   -- AHA-17 bullseye with anatomical
      region labels (clinical convention) next to the objective per-vertex
      aggregation used in this thesis.
  images/fig_lv_wall_thickness_3d.png -- 3D LV endocardial mesh coloured by
      local wall thickness with a shared colour bar.

The heavy geometry helpers are reused from
generate_patient002_methodology_figures so both scripts stay consistent.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, patches

from generate_patient002_methodology_figures import (
    OUT_DIR,
    ROOT,
    aha_segment_values,
    draw_aha17,
    draw_ssm_mesh,
    save_rgb_figure,
    ssm_wall_surfaces,
)

THICK_NORM = colors.Normalize(vmin=4.0, vmax=12.0)

# Discrete colours for the 17 AHA segments (shared by the 3D cut and bullseye).
SEG_CMAP = plt.get_cmap("tab20")
SEG_COLORS = {sid: SEG_CMAP((sid - 1) % 20) for sid in range(1, 18)}
SEG_LISTED = colors.ListedColormap([SEG_COLORS[sid] for sid in range(1, 18)])
SEG_NORM = colors.BoundaryNorm(np.arange(0.5, 18.5, 1.0), 17)


def aha_segment_ids(vertices: np.ndarray) -> np.ndarray:
    """Assign each vertex to an AHA-17 segment (mirrors aha_segment_values)."""
    z_norm = (vertices[:, 2] - vertices[:, 2].min()) / max(np.ptp(vertices[:, 2]), 1e-8)
    angles = (np.arctan2(vertices[:, 1], vertices[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)
    ids = np.full(len(vertices), 17, dtype=float)

    segment_id = 1
    for lower, upper, segment_count in [(0.67, 1.01, 6), (0.34, 0.67, 6), (0.10, 0.34, 4)]:
        ring_mask = (z_norm >= lower) & (z_norm < upper)
        for segment_index in range(segment_count):
            angle_lower = 2.0 * np.pi * segment_index / segment_count
            angle_upper = 2.0 * np.pi * (segment_index + 1) / segment_count
            segment_mask = ring_mask & (angles >= angle_lower) & (angles < angle_upper)
            ids[segment_mask] = segment_id
            segment_id += 1
    ids[z_norm < 0.10] = 17
    return ids


def draw_aha17_segments(ax: plt.Axes) -> None:
    """Bullseye coloured by segment identity (not thickness) to show the cut."""
    rings = [
        (0.72, 1.00, 6, 1, 90.0),
        (0.45, 0.72, 6, 7, 90.0),
        (0.20, 0.45, 4, 13, 45.0),
    ]
    for inner_radius, outer_radius, segment_count, first_id, offset in rings:
        for segment_index in range(segment_count):
            segment_id = first_id + segment_index
            theta1 = offset - 360.0 * (segment_index + 1) / segment_count
            theta2 = offset - 360.0 * segment_index / segment_count
            wedge = patches.Wedge(
                (0.0, 0.0), outer_radius, theta1, theta2,
                width=outer_radius - inner_radius,
                facecolor=SEG_COLORS[segment_id], edgecolor="white", linewidth=1.2,
            )
            ax.add_patch(wedge)
            mid_angle = np.deg2rad((theta1 + theta2) / 2.0)
            radius = (inner_radius + outer_radius) / 2.0
            ax.text(radius * np.cos(mid_angle), radius * np.sin(mid_angle),
                    str(segment_id), ha="center", va="center",
                    fontsize=5.8, color="white", weight="bold")
    centre = patches.Circle((0.0, 0.0), 0.20, facecolor=SEG_COLORS[17],
                            edgecolor="white", linewidth=1.2)
    ax.add_patch(centre)
    ax.text(0.0, 0.0, "17", ha="center", va="center",
            fontsize=5.8, color="white", weight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


def make_aha17_cut(endocardium: np.ndarray, faces: np.ndarray) -> Path:
    """Show how the LV surface is cut into AHA-17 segments and unrolled."""
    seg_ids = aha_segment_ids(endocardium)

    fig = plt.figure(figsize=(7.2, 3.9), facecolor="white")

    # Panel (a): 3D endocardial surface partitioned into the 17 segments.
    ax0 = fig.add_axes([0.00, 0.05, 0.40, 0.90], projection="3d")
    draw_ssm_mesh(ax0, endocardium, faces, values=seg_ids,
                  cmap_name=SEG_LISTED, norm=SEG_NORM, stride=2, alpha=0.98)
    ax0.view_init(elev=18, azim=-62)
    ax0.set_title("(a) LV surface cut into 17 segments",
                  fontsize=8.0, style="italic", color="#333333", pad=2, y=0.98)

    # Long-axis ring-boundary annotation on the left.
    z = endocardium[:, 2]
    z0, z1 = z.min(), z.max()
    for frac, name in [(0.67, "basal / mid"), (0.34, "mid / apical"), (0.10, "apical / apex")]:
        ax0.text2D(0.02, 0.14 + 0.66 * frac, f"— {name}",
                   transform=ax0.transAxes, fontsize=5.6, color="#666666")

    # Panel (b): the same segments flattened into the AHA-17 bullseye.
    ax1 = fig.add_axes([0.46, 0.06, 0.52, 0.86])
    draw_aha17_segments(ax1)
    ax1.set_xlim(-1.95, 1.45)
    ax1.set_ylim(-1.35, 1.35)
    ax1.set_title("(b) Unrolled AHA-17 bullseye",
                  fontsize=8.0, style="italic", color="#333333", pad=6)

    wall_labels = [
        (90.0, "Anterior"), (150.0, "Antero-\nseptal"), (210.0, "Infero-\nseptal"),
        (270.0, "Inferior"), (330.0, "Infero-\nlateral"), (30.0, "Antero-\nlateral"),
    ]
    for angle_deg, name in wall_labels:
        angle = np.deg2rad(angle_deg)
        ax1.text(1.24 * np.cos(angle), 1.24 * np.sin(angle), name,
                 ha="center", va="center", fontsize=6.2, color="#243447")

    ring_labels = [
        (1.02, "Basal ring\n(seg. 1--6)"),
        (0.34, "Mid ring\n(seg. 7--12)"),
        (-0.34, "Apical ring\n(seg. 13--16)"),
        (-1.02, "Apex\n(seg. 17)"),
    ]
    for y_pos, name in ring_labels:
        ax1.text(-1.68, y_pos, name, fontsize=5.8, color="#444444",
                 ha="center", va="center")

    # Connecting arrow between the two panels.
    fig.add_artist(patches.FancyArrowPatch(
        (0.40, 0.50), (0.47, 0.50), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=13.0, linewidth=1.1, color="#555555"))

    output = OUT_DIR / "fig_aha17_cut.png"
    save_rgb_figure(fig, output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output



def make_lv_wall_thickness_3d(
    endocardium: np.ndarray, faces: np.ndarray, thickness: np.ndarray
) -> Path:
    """3D LV endocardial mesh coloured by local wall thickness (two views)."""
    fig = plt.figure(figsize=(6.4, 3.4), facecolor="white")
    views = [(20.0, -60.0, "(a) Antero-lateral view"),
             (20.0, 120.0, "(b) Infero-septal view")]
    for idx, (elev, azim, title) in enumerate(views):
        ax = fig.add_axes([0.02 + 0.46 * idx, 0.12, 0.44, 0.82], projection="3d")
        draw_ssm_mesh(ax, endocardium, faces, values=thickness,
                      norm=THICK_NORM, stride=2, alpha=0.98)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=8.0, style="italic", color="#333333",
                     pad=2, y=0.97)

    colorbar_axis = fig.add_axes([0.30, 0.08, 0.40, 0.026])
    scalar_mappable = plt.cm.ScalarMappable(norm=THICK_NORM, cmap="turbo")
    colorbar = fig.colorbar(scalar_mappable, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Wall thickness (mm)", fontsize=6.6, labelpad=1)
    colorbar.ax.tick_params(labelsize=5.8, length=2)

    output = OUT_DIR / "fig_lv_wall_thickness_3d.png"
    save_rgb_figure(fig, output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output


# ---- Method mechanism diagrams -------------------------------------------

ENDO_COLOR = "#0072B2"
EPI_COLOR = "#D55E00"
ARROW_COLOR = "#B00020"
FIELD_COLOR = "#2A9D8F"


def _wall_curves(n: int = 400):
    """Return endocardial and epicardial 2D contours of a wall segment.

    The endocardium is a gently undulating curve; the epicardium sits outside
    it with a spatially varying gap so the diagrams show non-parallel walls.
    """
    x = np.linspace(0.0, 10.0, n)
    endo_y = 1.6 + 0.55 * np.sin(0.55 * x) + 0.18 * np.sin(1.7 * x + 0.7)
    dy = np.gradient(endo_y, x)
    nx, ny = -dy, np.ones_like(dy)
    nlen = np.hypot(nx, ny)
    nx, ny = nx / nlen, ny / nlen
    gap = 2.1 + 0.6 * np.sin(0.5 * x + 0.4)  # varying wall thickness
    # Give the epicardium a slightly different waviness so the walls are
    # genuinely non-parallel; this makes the transmural field lines curve.
    epi_x = x + nx * gap + 0.35 * np.sin(0.8 * x + 1.1)
    epi_y = endo_y + ny * gap + 0.25 * np.sin(0.7 * x + 0.2)
    return x, endo_y, epi_x, epi_y, nx, ny, gap


def _draw_wall(ax, x, endo_y, epi_x, epi_y):
    ax.fill(
        np.concatenate([x, epi_x[::-1]]),
        np.concatenate([endo_y, epi_y[::-1]]),
        color="#F0D9C8", alpha=0.55, zorder=0, linewidth=0,
    )
    ax.plot(x, endo_y, color=ENDO_COLOR, lw=2.0, zorder=3)
    ax.plot(epi_x, epi_y, color=EPI_COLOR, lw=2.0, zorder=3)
    ax.set_xlim(-0.4, 12.2)
    ax.set_ylim(0.2, 6.6)
    ax.set_aspect("equal")
    ax.axis("off")


def _panel_title(ax, label, name):
    ax.set_title(f"{label} {name}", fontsize=8.2, style="italic",
                 color="#333333", pad=3)


def _arrow(ax, p0, p1, color, lw=1.4, alpha=1.0, zorder=4,
           shrink0=0.0, shrink1=2.0, scale=11.0):
    """Draw a clean, consistently styled arrow from p0 to p1."""
    ax.annotate(
        "", xy=p1, xytext=p0,
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
            mutation_scale=scale, shrinkA=shrink0, shrinkB=shrink1,
            capstyle="round", joinstyle="round",
        ),
        zorder=zorder,
    )


def _ray_epi_hit(ox, oy, dx, dy, epi_x, epi_y, t_max=6.0):
    """Return the first intersection length of a ray with the epi polyline."""
    best = None
    for i in range(len(epi_x) - 1):
        ax0, ay0 = epi_x[i], epi_y[i]
        bx, by = epi_x[i + 1] - ax0, epi_y[i + 1] - ay0
        det = dx * (-by) - dy * (-bx)
        if abs(det) < 1e-9:
            continue
        rx, ry = ax0 - ox, ay0 - oy
        t = (rx * (-by) - ry * (-bx)) / det   # length along the ray
        s = (dx * ry - dy * rx) / det          # position on the segment
        if 0.0 <= s <= 1.0 and 0.05 < t < t_max:
            if best is None or t < best:
                best = t
    return best


def _streamline(idx, x, endo_y, epi_x, epi_y, nx, ny, gap, n=48):
    """Curved transmural streamline from the endo point to the epi surface.

    Modelled as a cubic Bezier that leaves the endocardium along its outward
    normal and arrives at the epicardium along the local epi normal, which is
    how a Laplace streamline crosses a curved, non-parallel wall.
    """
    # Endpoints: endo vertex and the point on epi hit by the endo normal.
    p0 = np.array([x[idx], endo_y[idx]])
    L = _ray_epi_hit(x[idx], endo_y[idx], nx[idx], ny[idx], epi_x, epi_y)
    if L is None:
        L = gap[idx]
    p3 = p0 + np.array([nx[idx], ny[idx]]) * L
    # Local epi normal (from neighbouring epi samples).
    lo, hi = max(idx - 4, 0), min(idx + 4, len(epi_x) - 1)
    et = np.array([epi_x[hi] - epi_x[lo], epi_y[hi] - epi_y[lo]])
    et = et / max(np.hypot(*et), 1e-8)
    en = np.array([-et[1], et[0]])            # epi normal
    if en @ np.array([nx[idx], ny[idx]]) < 0:  # keep it outward-consistent
        en = -en
    d0 = np.array([nx[idx], ny[idx]])          # leave endo along its normal
    # Control points sit one-third of the way out along each normal.
    c1 = p0 + d0 * (L / 3.0)
    c2 = p3 - en * (L / 3.0)
    t = np.linspace(0.0, 1.0, n)[:, None]
    curve = ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1
             + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p3)
    return curve[:, 0], curve[:, 1]



def _curved_arrow(ax, px, py, color, lw=1.6, alpha=1.0, zorder=4, scale=11.0):
    """Draw a curved path following (px, py) with an arrow head at the end."""
    ax.plot(px[:-1], py[:-1], color=color, lw=lw, alpha=alpha,
            solid_capstyle="round", zorder=zorder)
    ax.annotate(
        "", xy=(px[-1], py[-1]), xytext=(px[-2], py[-2]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                        mutation_scale=scale, shrinkA=0, shrinkB=0),
        zorder=zorder,
    )




class Line2DProxy:
    """Small helper to build legend handles with a consistent style."""

    def __init__(self, color, label, dashed=False):
        from matplotlib.lines import Line2D
        self.label = label
        self.artist = Line2D([0], [0], color=color, lw=1.8,
                             ls=(0, (4, 2)) if dashed else "-")


def make_method_diagrams() -> Path:
    """2x2 conceptual diagram of how each method measures wall thickness."""
    x, endo_y, epi_x, epi_y, nx, ny, gap = _wall_curves()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), facecolor="white")

    # --- (a) EDT boundary sum: distances to both boundaries summed. ---
    ax = axes[0, 0]
    _draw_wall(ax, x, endo_y, epi_x, epi_y)
    for xi in [2.0, 5.0, 8.0]:
        idx = int(xi / 10.0 * (len(x) - 1))
        px = x[idx] + nx[idx] * gap[idx] * 0.5
        py = endo_y[idx] + ny[idx] * gap[idx] * 0.5
        ax.plot(px, py, "o", color="#333333", ms=3.0, zorder=5)
        _arrow(ax, (px, py), (x[idx], endo_y[idx]), ENDO_COLOR, lw=1.4)
        _arrow(ax, (px, py), (epi_x[idx], epi_y[idx]), EPI_COLOR, lw=1.4)
    ax.text(0.3, 6.05, r"$t = D_{\mathrm{endo}} + D_{\mathrm{epi}}$",
            fontsize=7.6, color="#333333")
    _panel_title(ax, "(a)", "EDT boundary sum")

    # --- (b) Laplace field: nested iso-potential lines + streamline arrows. ---
    ax = axes[0, 1]
    _draw_wall(ax, x, endo_y, epi_x, epi_y)
    for frac in [0.25, 0.5, 0.75]:
        iso_x = (1 - frac) * x + frac * epi_x
        iso_y = (1 - frac) * endo_y + frac * epi_y
        ax.plot(iso_x, iso_y, color=FIELD_COLOR, lw=0.9, ls=(0, (4, 2)),
                alpha=0.9, zorder=2)
    for xi in [2.0, 5.0, 8.0]:
        idx = int(xi / 10.0 * (len(x) - 1))
        sx, sy = _streamline(idx, x, endo_y, epi_x, epi_y, nx, ny, gap)
        _curved_arrow(ax, sx, sy, ARROW_COLOR, lw=1.6)
    ax.text(0.3, 6.05, r"$\nabla^2\psi=0,\; t = 1/|\nabla\psi|$",
            fontsize=7.6, color="#333333")
    _panel_title(ax, "(b)", "Laplace field")

    # --- (c) Yezzi-Prince: u from endo, v from epi, along the same field. ---
    ax = axes[1, 0]
    _draw_wall(ax, x, endo_y, epi_x, epi_y)
    for frac in [0.25, 0.5, 0.75]:
        iso_x = (1 - frac) * x + frac * epi_x
        iso_y = (1 - frac) * endo_y + frac * epi_y
        ax.plot(iso_x, iso_y, color=FIELD_COLOR, lw=0.7, ls=(0, (4, 2)),
                alpha=0.7, zorder=2)
    for xi in [2.5, 6.5]:
        idx = int(xi / 10.0 * (len(x) - 1))
        sx, sy = _streamline(idx, x, endo_y, epi_x, epi_y, nx, ny, gap)
        half = len(sx) // 2
        # u: endo -> mid-wall along the field; v: epi -> mid-wall along the field.
        _curved_arrow(ax, sx[:half + 1], sy[:half + 1], "#1B7F5B", lw=1.6)
        _curved_arrow(ax, sx[::-1][:len(sx) - half], sy[::-1][:len(sy) - half],
                      "#7B2CBF", lw=1.6)
        ax.text(sx[half] + 0.18, sy[half], r"$u{+}v$", fontsize=6.6,
                color="#333333", va="center")
    ax.text(0.3, 6.05, r"$\nabla\psi\!\cdot\!\nabla u=-1,\; t=u+v$",
            fontsize=7.6, color="#333333")
    _panel_title(ax, "(c)", "Yezzi--Prince")

    # --- (d) SDF cone rays: fan of rays around the normal, median hit. ---
    ax = axes[1, 1]
    _draw_wall(ax, x, endo_y, epi_x, epi_y)
    idx = int(5.0 / 10.0 * (len(x) - 1))
    ox, oy = x[idx], endo_y[idx]
    base_ang = np.arctan2(ny[idx], nx[idx])
    lengths = []
    rays = []
    for dang in np.linspace(-np.pi / 6, np.pi / 6, 7):
        ang = base_ang + dang
        dx, dy = np.cos(ang), np.sin(ang)
        L = _ray_epi_hit(ox, oy, dx, dy, epi_x, epi_y)
        if L is None:
            continue
        lengths.append(L)
        rays.append((dx, dy, L))
    median_L = float(np.median(lengths)) if lengths else None
    for dx, dy, L in rays:
        is_median = median_L is not None and abs(L - median_L) < 1e-9
        _arrow(ax, (ox, oy), (ox + dx * L, oy + dy * L),
               ARROW_COLOR if is_median else "#C9A227",
               lw=1.7 if is_median else 1.0,
               alpha=1.0 if is_median else 0.75,
               scale=12.0 if is_median else 9.0)
    ax.plot(ox, oy, "o", color="#333333", ms=3.2, zorder=6)
    ax.text(0.3, 6.05, r"$t=\mathrm{median}_k\, d_k,\; K=7$",
            fontsize=7.6, color="#333333")
    _panel_title(ax, "(d)", "SDF cone rays")

    handles = [
        Line2DProxy(ENDO_COLOR, "Endocardium"),
        Line2DProxy(EPI_COLOR, "Epicardium"),
        Line2DProxy(FIELD_COLOR, "Iso-potential", dashed=True),
        Line2DProxy(ARROW_COLOR, "Thickness path"),
    ]
    fig.legend(handles=[h.artist for h in handles],
               labels=[h.label for h in handles],
               loc="lower center", ncol=4, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=1.4)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.08,
                        wspace=0.05, hspace=0.16)
    output = OUT_DIR / "fig_wall_thickness_methods.png"
    save_rgb_figure(fig, output, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return output


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    global faces_global
    endocardium, _epicardium, faces, thickness = ssm_wall_surfaces()
    faces_global = faces
    outputs = [
        make_method_diagrams(),
        make_lv_wall_thickness_3d(endocardium, faces, thickness),
        make_aha17_cut(endocardium, faces),
    ]
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
