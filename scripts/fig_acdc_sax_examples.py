"""
Generate a publication-quality figure showing:
  - Left panel: Realistic 3D LV with SAX slice planes (solid shading)
  - Right panels: Basal / Mid / Apical SAX cross-sections
Output: images/acdc_sax_examples.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

# Colours
COL_ENDO = "#e63946"
COL_EPI = "#1d3557"
COL_MYO = "#c9a0a0"
COL_WT = "#f4a261"
COL_SLICE = "#457b9d"


# ──────────────────────────────────────────────────────────────────────
# 3D LV geometry helpers
# ──────────────────────────────────────────────────────────────────────

def lv_surface(rx, ry, rz, nu=80, nv=50, z_offset=0.0):
    """Parametric LV surface (truncated prolate-ish ellipsoid, closed apex)."""
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi * 0.92, nv)  # open at base
    U, V = np.meshgrid(u, v)

    # Taper toward apex (power mapping on v gives rounder apex)
    taper = np.sin(V) ** 0.85
    x = rx * taper * np.cos(U)
    y = ry * taper * np.sin(U)
    z = -rz * np.cos(V) + z_offset
    return x, y, z


def draw_3d_lv(ax):
    """Render a solid-shaded 3D LV with translucent slice planes."""
    # Epicardial surface
    x_epi, y_epi, z_epi = lv_surface(2.3, 2.1, 4.8)
    # Endocardial surface (smaller, offset slightly)
    x_en, y_en, z_en = lv_surface(1.5, 1.35, 4.0, z_offset=0.4)

    # Custom shading via LightSource
    ls = LightSource(azdeg=315, altdeg=45)

    # Plot epicardium (solid, slightly translucent)
    ax.plot_surface(x_epi, y_epi, z_epi, alpha=0.35,
                    color="#a8dadc", linewidth=0, antialiased=True,
                    shade=True, lightsource=ls)

    # Add wireframe for depth cues
    ax.plot_wireframe(x_epi, y_epi, z_epi, alpha=0.12, color=COL_EPI,
                      linewidth=0.25, rstride=5, cstride=5)

    # Endocardium visible through translucent epi
    ax.plot_surface(x_en, y_en, z_en, alpha=0.2,
                    color="#e63946", linewidth=0, antialiased=True,
                    shade=True, lightsource=ls)

    # SAX slice planes
    slice_zs = [-0.8, -2.2, -3.6]
    slice_labels = ["Basal", "Mid", "Apical"]
    slice_colors = ["#264653", "#2a9d8f", "#e76f51"]
    disc_r = 3.2

    for sz, lbl, col in zip(slice_zs, slice_labels, slice_colors):
        theta = np.linspace(0, 2 * np.pi, 80)
        # Filled disc
        r_vals = np.linspace(0, disc_r, 3)
        T, R = np.meshgrid(theta, r_vals)
        X = R * np.cos(T)
        Y = R * np.sin(T)
        Z = np.full_like(X, sz)
        ax.plot_surface(X, Y, Z, alpha=0.22, color=col, linewidth=0)
        # Edge ring
        ax.plot(disc_r * np.cos(theta), disc_r * np.sin(theta),
                np.full(80, sz), color=col, alpha=0.7, linewidth=1.0)
        # Label
        ax.text(disc_r + 0.3, 0.5, sz, lbl, fontsize=7.5, color=col,
                ha="left", va="center", fontweight="bold")

    # Long axis
    ax.plot([0, 0], [0, 0], [1.2, -4.9], color="#6c757d", linewidth=1.0,
            linestyle=":", alpha=0.5)
    ax.text(0, 0, 1.6, "Base", fontsize=7, ha="center", color="#6c757d")
    ax.text(0, 0, -5.2, "Apex", fontsize=7, ha="center", color="#6c757d")

    # Basal ring (open top)
    theta = np.linspace(0, 2 * np.pi, 80)
    ax.plot(2.3 * np.cos(theta), 2.1 * np.sin(theta),
            np.full(80, z_epi[0, 0]), color=COL_EPI, linewidth=1.2, alpha=0.6)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-5.5, 2.5)
    ax.view_init(elev=15, azim=-55)
    ax.set_axis_off()
    ax.set_title("SAX acquisition planes", fontsize=10, pad=-8,
                 color="#2b2d42")


# ──────────────────────────────────────────────────────────────────────
# SAX cross-section panels
# ──────────────────────────────────────────────────────────────────────

def draw_sax_panel(ax, cfg):
    """Draw a SAX cross-section with MRI-like background, smooth contours."""
    n = 256
    extent = 3.5
    rng = np.random.default_rng(cfg["seed"])

    yy, xx = np.mgrid[-extent:extent:complex(n), -extent:extent:complex(n)]
    rr = np.sqrt(xx**2 + yy**2)
    img = rng.normal(0.05, 0.015, (n, n))

    # Chest / body background
    body_mask = (rr > 2.6) & (rr < 3.3)
    img[body_mask] += 0.20

    # Surrounding tissue
    surr_mask = (rr > cfg["epi_r"] * 1.05) & (rr < 2.6)
    img[surr_mask] += 0.10

    # Myocardium
    myo_mask = (rr > cfg["endo_r"] * 0.97) & (rr < cfg["epi_r"] * 1.03)
    img[myo_mask] += 0.32

    # Blood pool (bright)
    cavity_mask = rr < cfg["endo_r"] * 0.93
    img[cavity_mask] += 0.60

    # Slight Gaussian blur via convolution (cheap approximation)
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, sigma=1.2)
    img += rng.normal(0, 0.02, (n, n))
    img = np.clip(img, 0, 1)

    ax.imshow(img, extent=[-extent, extent, -extent, extent],
              cmap="gray", aspect="equal", zorder=0)

    # Smooth contours (no jitter)
    t = np.linspace(0, 2 * np.pi, 400, endpoint=True)
    epi_x = cfg["epi_r"] * np.cos(t)
    epi_y = cfg["epi_ry"] * np.sin(t)
    endo_x = cfg["endo_r"] * np.cos(t)
    endo_y = cfg["endo_ry"] * np.sin(t)

    ax.plot(epi_x, epi_y, color=COL_EPI, linewidth=2.0, zorder=4,
            solid_capstyle="round")
    ax.plot(endo_x, endo_y, color=COL_ENDO, linewidth=2.0, zorder=4,
            solid_capstyle="round")

    # Wall thickness arrow (right side)
    angle = 0  # rightmost
    endo_pt = (cfg["endo_r"], 0)
    epi_pt = (cfg["epi_r"], 0)
    ax.annotate(
        "", xy=epi_pt, xytext=endo_pt,
        arrowprops=dict(arrowstyle="<->", color=COL_WT, lw=1.8,
                        shrinkA=2, shrinkB=2),
        zorder=5
    )
    mid_x = (endo_pt[0] + epi_pt[0]) / 2
    ax.text(mid_x, 0.35, cfg["wt_label"], fontsize=7, color=COL_WT,
            ha="center", fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.8))

    # RV crescent (left side, subtle)
    rv_t = np.linspace(0.55 * np.pi, 1.45 * np.pi, 60)
    rv_r_out = cfg["epi_r"] + 0.6
    rv_r_in = cfg["epi_r"] + 0.2
    ax.plot(rv_r_out * np.cos(rv_t), rv_r_out * np.sin(rv_t) * 0.9,
            color="#adb5bd", linewidth=0.9, alpha=0.6, zorder=3)
    ax.plot(rv_r_in * np.cos(rv_t), rv_r_in * np.sin(rv_t) * 0.9,
            color="#adb5bd", linewidth=0.9, alpha=0.6, zorder=3)

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(cfg["title"], pad=8, color=cfg["title_col"])


def create_figure():
    fig = plt.figure(figsize=(13, 4.0), facecolor="white")
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.3, 1, 1, 1], wspace=0.06)

    # 3D panel
    ax3d = fig.add_subplot(gs[0], projection="3d", facecolor="white")
    draw_3d_lv(ax3d)

    # Basal / Mid / Apical SAX panels
    configs = [
        {
            "title": "Basal",
            "title_col": "#264653",
            "endo_r": 1.7, "endo_ry": 1.65,
            "epi_r": 2.5, "epi_ry": 2.45,
            "wt_label": "~10 mm",
            "seed": 42,
        },
        {
            "title": "Mid-cavity",
            "title_col": "#2a9d8f",
            "endo_r": 1.4, "endo_ry": 1.35,
            "epi_r": 2.2, "epi_ry": 2.15,
            "wt_label": "~11 mm",
            "seed": 77,
        },
        {
            "title": "Apical",
            "title_col": "#e76f51",
            "endo_r": 0.9, "endo_ry": 0.85,
            "epi_r": 1.8, "epi_ry": 1.75,
            "wt_label": "~12 mm",
            "seed": 13,
        },
    ]

    for i, cfg in enumerate(configs):
        ax = fig.add_subplot(gs[i + 1])
        draw_sax_panel(ax, cfg)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COL_ENDO, linewidth=2.2, label="Endocardium"),
        Line2D([0], [0], color=COL_EPI, linewidth=2.2, label="Epicardium"),
        Line2D([0], [0], color=COL_WT, linewidth=1.8, marker="|",
               markersize=8, label="Wall thickness"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.62, -0.01))

    out_path = "images/acdc_sax_examples.pdf"
    fig.savefig(out_path)
    print(f"Saved: {out_path}")
    fig.savefig("images/acdc_sax_examples.png")
    print("Saved: images/acdc_sax_examples.png")
    plt.close(fig)


if __name__ == "__main__":
    create_figure()
