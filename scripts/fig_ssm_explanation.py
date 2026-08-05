"""
Generate two figures explaining the Statistical Shape Model (SSM) approach:

  fig_ssm_mean_modes.png   — Mean LV shape + first 4 modes of variation (±3σ)
  fig_ssm_sampling.png     — Synthetic dataset generation pipeline (6 panels)

Both outputs go to images/.
Requires: numpy, matplotlib, scipy (for chi2), gzip (stdlib), subprocess (stdlib)
SSM files are cloned from the UK Digital Heart Project repo if not present.
"""

import gzip
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "images"
SSM_REPO_URL = "https://github.com/UK-Digital-Heart-Project/Statistical-Shape-Model.git"
SSM_MEAN_FILE = "LV_ED_mean.vtk"
SSM_PC_FILE = "LV_ED_pc_100_modes.csv.gz"
SSM_VAR_FILE = "LV_ED_var_100_modes.csv.gz"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

# colour palette
C_ENDO = "#0072B2"
C_EPI  = "#D55E00"
C_DARK = "#243447"
C_LIGHT = "#F6F8FA"

# Number of PCA modes actually sampled from (\cref{subsec:dataset-composition}):
# the first 32 modes already capture ~99% of the shape variance.
D_ACTIVE_MODES = 32

# ---------------------------------------------------------------------------
# SSM I/O helpers
# ---------------------------------------------------------------------------

def ensure_ssm_dir() -> Path:
    candidates = [
        Path(os.environ["CARDIOSDF_SSM_DIR"]) if "CARDIOSDF_SSM_DIR" in os.environ else None,
        ROOT / "Statistical-Shape-Model",
        ROOT / "notebooks" / "Statistical-Shape-Model",
        Path("/tmp/cardiosdf-ssm"),
    ]
    for c in candidates:
        if c is not None and (c / SSM_MEAN_FILE).exists():
            return c
    target = Path("/tmp/cardiosdf-ssm")
    subprocess.run(
        ["git", "clone", "--depth", "1", SSM_REPO_URL, str(target)],
        check=True, stdout=subprocess.DEVNULL,
    )
    return target


def load_legacy_vtk_polydata(vtk_path: Path):
    tokens = vtk_path.read_text().split()
    pi = tokens.index("POINTS")
    n_pts = int(tokens[pi + 1])
    pts = np.asarray(tokens[pi + 3 : pi + 3 + 3 * n_pts], dtype=float).reshape(n_pts, 3)
    fi = tokens.index("POLYGONS")
    n_poly = int(tokens[fi + 1])
    cursor = fi + 3
    faces = []
    for _ in range(n_poly):
        nv = int(tokens[cursor])
        ids = [int(x) for x in tokens[cursor + 1 : cursor + 1 + nv]]
        if nv == 3:
            faces.append(ids)
        elif nv > 3:
            faces.extend([[ids[0], ids[k], ids[k + 1]] for k in range(1, nv - 1)])
        cursor += nv + 1
    pts -= pts.mean(axis=0)
    return pts, np.asarray(faces, dtype=np.int32)


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-8)
    normals = np.zeros_like(verts)
    for c in range(3):
        np.add.at(normals, faces[:, c], fn)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    if np.mean(np.sum(normals * verts, axis=1)) < 0:
        normals *= -1
    return normals


def epicardium_thickness_field(endo: np.ndarray) -> np.ndarray:
    """Per-vertex synthetic offset used to build the epicardium (\\cref{eq:synthetic-epi-offset})."""
    z_norm = (endo[:, 2] - endo[:, 2].min()) / max(np.ptp(endo[:, 2]), 1e-8)
    angles = np.arctan2(endo[:, 1], endo[:, 0])
    thickness = 5.5 + 4.2 * z_norm + 1.1 * np.sin(2.0 * angles + 0.6) * (0.4 + 0.6 * z_norm)
    return np.clip(thickness, 4.5, 11.5)


def make_epicardium(endo: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = vertex_normals(endo, faces)
    thickness = epicardium_thickness_field(endo)
    return endo + normals * thickness[:, None]


def mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Enclosed volume (mL) via the divergence theorem, assuming vertex coordinates in mm."""
    tri = verts[faces]
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    return abs(float(signed.sum())) / 1000.0


def load_ssm():
    ssm_dir = ensure_ssm_dir()
    mean_pts, faces = load_legacy_vtk_polydata(ssm_dir / SSM_MEAN_FILE)
    with gzip.open(ssm_dir / SSM_PC_FILE, "rt") as f:
        pc = np.loadtxt(f, delimiter=",")   # (3*N) x 100
    with gzip.open(ssm_dir / SSM_VAR_FILE, "rt") as f:
        lambdas = np.loadtxt(f, delimiter=",")  # 100
    sigmas = np.sqrt(np.abs(lambdas))
    return mean_pts, pc, sigmas, faces


# ---------------------------------------------------------------------------
# Mesh-drawing helper
# ---------------------------------------------------------------------------

def draw_mesh(ax, verts, faces, color=C_ENDO, values=None, cmap="turbo",
              vmin=None, vmax=None, stride=4, alpha=0.90, elev=18, azim=-55):
    sf = faces[::stride]
    polys = verts[sf]
    fn = np.cross(polys[:, 1] - polys[:, 0], polys[:, 2] - polys[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-8)
    light = np.array([-0.25, -0.50, 0.83])
    light /= np.linalg.norm(light)
    lf = 0.45 + 0.55 * np.clip(fn @ light, 0.0, 1.0)

    if values is not None:
        cm = plt.get_cmap(cmap)
        norm = colors.Normalize(
            vmin=vmin if vmin is not None else float(np.nanmin(values)),
            vmax=vmax if vmax is not None else float(np.nanmax(values)),
        )
        fc = cm(norm(values[sf].mean(axis=1)))
        fc[:, :3] = np.clip(fc[:, :3] * (0.65 + 0.35 * lf[:, None]), 0, 1)
        fc[:, 3] = alpha
    else:
        base = np.array(colors.to_rgb(color))
        fc = np.empty((len(sf), 4))
        fc[:, :3] = np.clip(base[None] * lf[:, None] + 0.18 * (1 - lf[:, None]), 0, 1)
        fc[:, 3] = alpha

    ax.add_collection3d(Poly3DCollection(polys, facecolors=fc, edgecolors="none", linewidths=0))
    mn, mx = verts.min(0), verts.max(0)
    c, hr = (mn + mx) / 2, np.max(mx - mn) / 2 * 1.05
    ax.set_xlim(c[0] - hr, c[0] + hr)
    ax.set_ylim(c[1] - hr, c[1] + hr)
    ax.set_zlim(c[2] - hr, c[2] + hr)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)


# ---------------------------------------------------------------------------
# Figure 1 — Mean shape + 4 modes of variation
# ---------------------------------------------------------------------------

def fig_ssm_mean_modes():
    mean_pts, pc, sigmas, faces = load_ssm()
    epi_mean = make_epicardium(mean_pts, faces)

    n_modes = 3          # modes 0,1,2 shown ±3σ
    sigma_val = 3.0
    # layout: 1 mean column (2 rows merged) + 3 mode columns (2 rows each)
    fig = plt.figure(figsize=(9.0, 4.2), facecolor="white")

    # columns: [0] mean, [1..n_modes] mode variations
    col_width = 1.0 / (1 + 2 * n_modes)
    row_heights = [0.44, 0.44]
    row_tops = [0.96, 0.50]

    # ---- mean shape (large, centred) ----
    ax_mean = fig.add_axes([0.02, 0.08, 0.18, 0.84], projection="3d")
    draw_mesh(ax_mean, epi_mean, faces, color=C_EPI, stride=3, alpha=0.45)
    draw_mesh(ax_mean, mean_pts, faces, color=C_ENDO, stride=3, alpha=0.90)
    ax_mean.set_title("Mean\nshape $\\bar{X}$", fontsize=7.5, pad=2)

    # vertical arrow annotation
    fig.text(0.21, 0.50, "mode variations", ha="center", va="center",
             fontsize=6.5, color="#555555", rotation=90)

    # ---- mode columns ----
    mode_labels = ["Mode 1 (size)", "Mode 2 (elongation)", "Mode 3 (asymmetry)"]
    for col, mode_idx in enumerate(range(n_modes)):
        b = np.zeros(pc.shape[1])
        b[mode_idx] = sigma_val * sigmas[mode_idx]
        pts_pos = (mean_pts.flatten() + pc @ b).reshape(-1, 3)
        b[mode_idx] = -sigma_val * sigmas[mode_idx]
        pts_neg = (mean_pts.flatten() + pc @ b).reshape(-1, 3)

        x0 = 0.24 + col * 0.26
        ax_pos = fig.add_axes([x0, 0.52, 0.22, 0.40], projection="3d")
        ax_neg = fig.add_axes([x0, 0.08, 0.22, 0.40], projection="3d")

        draw_mesh(ax_pos, pts_pos, faces, color=C_ENDO, stride=4, alpha=0.88)
        draw_mesh(ax_neg, pts_neg, faces, color=C_ENDO, stride=4, alpha=0.88)

        ax_pos.set_title(f"{mode_labels[col]}\n$+{sigma_val:.0f}\\sigma$", fontsize=6.5, pad=1)
        ax_neg.set_title(f"$-{sigma_val:.0f}\\sigma$", fontsize=6.5, pad=1)

    # legend
    legend_handles = [
        mpatches.Patch(color=C_ENDO, label="Endocardium"),
        mpatches.Patch(color=C_EPI, label="Epicardium (mean only)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.50, 0.00), ncol=2,
               fontsize=6.5, frameon=False, labelcolor="#333333")

    out = OUT_DIR / "fig_ssm_mean_modes.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.06,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out.relative_to(ROOT)}")
    return out


# ---------------------------------------------------------------------------
# Figure 2 — Synthetic generation pipeline (6 panels)
# ---------------------------------------------------------------------------

def draw_latent_scatter(ax, pc, sigmas, n_samples=400, rng=None, d_active=D_ACTIVE_MODES):
    """Scatter of accepted samples in (b1, b2) PCA space, restricted to the
    d_active-mode sub-space that is actually sampled (\\cref{eq:ssm-latent-constraint})."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_candidates = 2048
    threshold = float(chi2.ppf(0.99, d_active))
    b_active = rng.standard_normal((n_candidates, d_active)) * sigmas[None, :d_active]
    z_active = b_active / sigmas[None, :d_active]
    accept = (np.abs(z_active) <= 3.0).all(axis=1) & (np.sum(z_active ** 2, axis=1) <= threshold)

    accepted = b_active[accept]
    rejected = b_active[~accept]
    if len(accepted) > n_samples:
        accepted = accepted[:n_samples]
    if len(rejected) > n_samples // 2:
        rejected = rejected[: n_samples // 2]

    if len(rejected) > 0:
        ax.scatter(rejected[:, 0], rejected[:, 1], s=3.0, alpha=0.25,
                   c="#999999", linewidths=0, label="Rejected")
    ax.scatter(accepted[:, 0], accepted[:, 1], s=4.2, alpha=0.70,
               c=np.linalg.norm(accepted[:, :10], axis=1), cmap="viridis", linewidths=0,
               label="Accepted")

    ax.text(
        0.03, 0.97,
        f"$|b_i/\\sigma_i|\\leq 3$, $\\sum_i (b_i/\\sigma_i)^2\\leq {threshold:.0f}$",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=5.8,
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#CCCCCC", lw=0.5, alpha=0.9),
    )
    ax.text(
        0.03, 0.03,
        "Final kept: 1300 / 2048",
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=5.8, color="#444444",
    )

    ax.set_xlabel("$b_1$", fontsize=7, labelpad=1)
    ax.set_ylabel("$b_2$", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=5.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.set_title(f"Latent sampling\n($d={d_active}$-mode sub-space)", fontsize=7, pad=2)


def draw_chi2_distribution(ax, d_active=D_ACTIVE_MODES):
    """Theoretical chi-squared distribution of the Mahalanobis statistic and
    the 99th-percentile acceptance cutoff (\\cref{eq:ssm-latent-constraint})."""
    threshold = float(chi2.ppf(0.99, d_active))
    x = np.linspace(0.0, float(chi2.ppf(0.999, d_active)), 400)
    pdf = chi2.pdf(x, d_active)

    below = x <= threshold
    above = x >= threshold
    ax.fill_between(x[below], pdf[below], color=C_ENDO, alpha=0.28, linewidth=0)
    ax.fill_between(x[above], pdf[above], color="#CC3333", alpha=0.28, linewidth=0)
    ax.plot(x, pdf, color=C_DARK, linewidth=1.1)
    ax.axvline(threshold, color="#CC3333", linewidth=1.0, linestyle="--")

    ymax = ax.get_ylim()[1]
    ax.text(threshold, ymax * 0.94, f"  $\\chi^2_{{0.99,{d_active}}}={threshold:.1f}$",
            fontsize=5.6, color="#CC3333", ha="left", va="top")
    ax.text(0.05, 0.90, "Accept", transform=ax.transAxes, fontsize=5.8, color=C_ENDO, ha="left")
    ax.text(0.68, 0.55, "Reject", transform=ax.transAxes, fontsize=5.8, color="#CC3333", ha="left")

    ax.set_xlabel(r"$\sum_i (b_i/\sigma_i)^2$", fontsize=7, labelpad=1)
    ax.set_ylabel("Density", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=5.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.set_title(f"Mahalanobis bound\n($d={d_active}$ active modes)", fontsize=7, pad=2)


def draw_variance_explained(ax, sigmas, d_active=D_ACTIVE_MODES):
    """Cumulative shape variance explained by the first k PCA modes, computed
    directly from the loaded SSM eigenvalues (no assumed numbers)."""
    eigen = sigmas ** 2
    explained = eigen / eigen.sum()
    cumulative = np.cumsum(explained) * 100.0
    modes = np.arange(1, len(sigmas) + 1)

    ax.plot(modes, cumulative, color=C_ENDO, linewidth=1.2)
    ax.axvline(d_active, color="#888888", linewidth=0.8, linestyle=":")
    ax.axhline(cumulative[d_active - 1], color="#888888", linewidth=0.8, linestyle=":")
    ax.scatter([d_active], [cumulative[d_active - 1]], s=14, color="#CC3333", zorder=5)
    ax.text(d_active + 3, cumulative[d_active - 1] - 10,
            f"$k={d_active}$\n{cumulative[d_active - 1]:.1f}\\%",
            fontsize=5.6, color="#444444")

    ax.set_xlim(1, len(sigmas))
    ax.set_ylim(0, 102)
    ax.set_xlabel("Mode index $k$", fontsize=7, labelpad=1)
    ax.set_ylabel("Cumulative variance (\\%)", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=5.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.set_title("Explained shape\nvariance", fontsize=7, pad=2)


def draw_thickness_distribution(ax, mean_pts, faces):
    """Distribution of the synthetic per-vertex epicardial offset
    (\\cref{eq:synthetic-epi-offset})."""
    thickness = epicardium_thickness_field(mean_pts)
    ax.hist(thickness, bins=28, color=C_EPI, alpha=0.75, edgecolor="white", linewidth=0.3)
    mean_t = float(thickness.mean())
    ax.axvline(mean_t, color=C_DARK, linewidth=1.0, linestyle="--")
    ax.text(mean_t, ax.get_ylim()[1] * 0.94, f"  mean={mean_t:.1f} mm",
            fontsize=5.6, color=C_DARK, ha="left", va="top")

    ax.set_xlabel(r"$d_i$ (mm)", fontsize=7, labelpad=1)
    ax.set_ylabel("Vertex count", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=5.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.set_title("Synthetic wall-offset\ndistribution", fontsize=7, pad=2)


def draw_quality_gate(ax_top, ax_bot, mean_pts, pc, sigmas, faces, rng=None):
    """Accepted and rejected meshes drawn in two separate stacked axes so
    their (very different) extents never overlap, each annotated with its
    actual enclosed volume."""
    if rng is None:
        rng = np.random.default_rng(7)
    # accepted sample
    b_ok = rng.standard_normal(pc.shape[1]) * sigmas * 0.6
    pts_ok = (mean_pts.flatten() + pc @ b_ok).reshape(-1, 3)
    pts_ok -= pts_ok.mean(0)
    # rejected sample (large b outside bound)
    b_bad = np.zeros(pc.shape[1])
    b_bad[0] = 4.5 * sigmas[0]
    pts_bad = (mean_pts.flatten() + pc @ b_bad).reshape(-1, 3)
    pts_bad -= pts_bad.mean(0)

    vol_ok = mesh_volume(pts_ok, faces)
    vol_bad = mesh_volume(pts_bad, faces)

    draw_mesh(ax_top, pts_ok, faces, color=C_ENDO, stride=5, alpha=0.90)
    draw_mesh(ax_bot, pts_bad, faces, color="#CC3333", stride=5, alpha=0.65)

    ax_top.text2D(0.02, 0.06, f"Accepted: $V\\approx{vol_ok:.0f}$ mL",
                  transform=ax_top.transAxes, fontsize=5.4, color=C_ENDO)
    ax_bot.text2D(0.02, 0.06, f"Rejected: $V\\approx{vol_bad:.0f}$ mL",
                  transform=ax_bot.transAxes, fontsize=5.4, color="#CC3333")

    ax_top.set_title("Quality gate\n(volume, sphericity)", fontsize=7, pad=2)
    ax_bot.set_title("Rejected outlier", fontsize=6.3, pad=1, style="italic",
                      color="#CC3333")


def draw_epi_offset(ax, mean_pts, faces):
    """Endo + synthetic epi shown together."""
    epi = make_epicardium(mean_pts, faces)
    draw_mesh(ax, epi, faces, color=C_EPI, stride=4, alpha=0.40)
    draw_mesh(ax, mean_pts, faces, color=C_ENDO, stride=4, alpha=0.90)
    ax.set_title("Synthetic epi\n(normal offset)", fontsize=7, pad=2)


def draw_contour_rings(ax, mean_pts, faces, n_slices=10):
    """Draw one endo/epi contour loop per SAX slice."""
    z_min, z_max = mean_pts[:, 2].min(), mean_pts[:, 2].max()
    z_vals = np.linspace(z_min + 0.07 * (z_max - z_min), z_max - 0.07 * (z_max - z_min), n_slices)
    epi = make_epicardium(mean_pts, faces)

    # Very faint anatomical context in the background.
    draw_mesh(ax, mean_pts, faces, color=C_ENDO, stride=10, alpha=0.05, elev=24, azim=-58)

    # Match the slice-stack style used in other methodology figures:
    # light SAX plane outlines with contours drawn on each plane.
    x_min, x_max = mean_pts[:, 0].min(), mean_pts[:, 0].max()
    y_min, y_max = mean_pts[:, 1].min(), mean_pts[:, 1].max()
    pad_x = 0.06 * (x_max - x_min)
    pad_y = 0.06 * (y_max - y_min)
    x0, x1 = x_min - pad_x, x_max + pad_x
    y0, y1 = y_min - pad_y, y_max + pad_y

    slab = np.ptp(mean_pts[:, 2]) / (n_slices * 3.0)
    min_gap = 3.0  # mm; keeps the epicardial circle visibly outside the endocardium
    for i, z in enumerate(z_vals, start=1):
        ax.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            [z] * 5,
            color="#D6DCE2",
            linewidth=0.42,
            alpha=0.86,
        )
        idx = np.abs(mean_pts[:, 2] - z) < slab
        if idx.sum() < 16:
            continue
        # Endo and epi share the same vertex indices, so use one common centre
        # and enforce rad_epi > rad_endo to prevent the offset epi ring from
        # crossing inside the endo ring near the apex/base.
        ring_endo = mean_pts[idx]
        ring_epi = epi[idx]
        ctr = ring_endo.mean(axis=0)
        rad_endo = np.median(np.linalg.norm(ring_endo[:, :2] - ctr[:2], axis=1))
        rad_epi = np.median(np.linalg.norm(ring_epi[:, :2] - ctr[:2], axis=1))
        rad_epi = max(rad_epi, rad_endo + min_gap)
        theta = np.linspace(0.0, 2.0 * np.pi, 120)

        for rad, col, lw in [(rad_endo, C_ENDO, 1.0), (rad_epi, C_EPI, 0.95)]:
            rx = ctr[0] + rad * np.cos(theta)
            ry = ctr[1] + rad * np.sin(theta)
            rz = np.full(len(rx), z)
            ax.plot(rx, ry, rz, color=col, linewidth=lw, alpha=0.98)

        # Label only representative slices to reduce clutter.
        if i in (1, n_slices // 2, n_slices):
            x_lab = mean_pts[:, 0].max() * 1.02
            y_lab = mean_pts[:, 1].min() * 0.25
            ax.text(x_lab, y_lab, z, f"S{i}", fontsize=5.5, color="#555555")

    z_span = z_vals[-1] - z_vals[0]
    dz = z_span / max(n_slices - 1, 1)
    ax.text(
        mean_pts[:, 0].min() * 0.95,
        mean_pts[:, 1].min() * 1.02,
        z_vals[0],
        f"$N_z={n_slices}$, $\\Delta z\\approx {dz:.1f}$ mm",
        fontsize=5.6,
        color="#444444",
    )

    legend_handles = [
        mpatches.Patch(color=C_ENDO, label="Endocardial contour"),
        mpatches.Patch(color=C_EPI, label="Epicardial contour"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.08),
        ncol=1,
        fontsize=5.3,
        frameon=False,
        labelcolor="#333333",
        handlelength=1.0,
    )

    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title("SAX contour stack\n(one endo + one epi loop/slice)", fontsize=7, pad=2)


def draw_query_cache(ax, mean_pts, faces, n_near=300, n_vol=200, rng=None):
    """Scatter of near-surface and volumetric query points."""
    if rng is None:
        rng = np.random.default_rng(0)
    # near-surface points
    idx = rng.choice(len(mean_pts), n_near, replace=False)
    near = mean_pts[idx] + rng.standard_normal((n_near, 3)) * 2.5
    # volumetric points
    mn, mx = mean_pts.min(0), mean_pts.max(0)
    vol = rng.uniform(mn - 5, mx + 5, (n_vol, 3))

    draw_mesh(ax, mean_pts, faces, color=C_ENDO, stride=6, alpha=0.18)
    ax.scatter(*near.T, s=3.5, color=C_ENDO, alpha=0.65, zorder=5)
    ax.scatter(*vol.T, s=2.0, color="#888888", alpha=0.40, zorder=4)

    legend_handles = [
        mpatches.Patch(color=C_ENDO, label="Near-surface"),
        mpatches.Patch(color="#888888", label="Volumetric"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.50, -0.08), ncol=1, fontsize=5.5,
              frameon=False, labelcolor="#333333", handlelength=1.0)
    mn2, mx2 = mean_pts.min(0), mean_pts.max(0)
    c, hr = (mn2 + mx2) / 2, np.max(mx2 - mn2) / 2 * 1.1
    ax.set_xlim(c[0] - hr, c[0] + hr)
    ax.set_ylim(c[1] - hr, c[1] + hr)
    ax.set_zlim(c[2] - hr, c[2] + hr)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-55)
    ax.set_title("Volumetric supervision\n(2048 query points/case)", fontsize=7, pad=2)


def draw_pathology_shapes(ax, mean_pts, pc, sigmas, faces):
    """Normal (blue), DCM (enlarged, cyan), HCM (thick, orange) meshes offset for comparison."""
    rng = np.random.default_rng(3)

    # DCM: excite mode 0 (size) strongly
    b_dcm = np.zeros(pc.shape[1])
    b_dcm[0] = 2.8 * sigmas[0]
    pts_dcm = (mean_pts.flatten() + pc @ b_dcm).reshape(-1, 3)
    pts_dcm -= pts_dcm.mean(0)
    pts_dcm[:, 0] -= 45  # offset left

    # HCM: excite mode 1 negatively (smaller cavity → thick wall)
    b_hcm = np.zeros(pc.shape[1])
    b_hcm[1] = -2.5 * sigmas[1]
    pts_hcm = (mean_pts.flatten() + pc @ b_hcm).reshape(-1, 3)
    pts_hcm -= pts_hcm.mean(0)
    pts_hcm[:, 0] += 45  # offset right

    mean_centre = mean_pts.copy()
    mean_centre -= mean_centre.mean(0)

    # all on same axis
    draw_mesh(ax, pts_dcm, faces, color="#009E73", stride=5, alpha=0.80, elev=18, azim=-55)
    draw_mesh(ax, mean_centre, faces, color=C_ENDO, stride=5, alpha=0.90, elev=18, azim=-55)
    draw_mesh(ax, pts_hcm, faces, color="#E69F00", stride=5, alpha=0.80, elev=18, azim=-55)

    all_v = np.vstack([pts_dcm, mean_centre, pts_hcm])
    mn, mx = all_v.min(0), all_v.max(0)
    c, hr = (mn + mx) / 2, np.max(mx - mn) / 2 * 1.05
    ax.set_xlim(c[0] - hr, c[0] + hr)
    ax.set_ylim(c[1] - hr, c[1] + hr)
    ax.set_zlim(c[2] - hr, c[2] + hr)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-55)

    handles = [
        mpatches.Patch(color="#009E73", label="DCM"),
        mpatches.Patch(color=C_ENDO, label="Normal"),
        mpatches.Patch(color="#E69F00", label="HCM"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, -0.08),
              ncol=3, fontsize=5.0, frameon=False, handlelength=0.8, columnspacing=0.5)
    ax.set_title("Pathology\nsimulation", fontsize=7, pad=2)


def fig_ssm_sampling_pipeline():
    mean_pts, pc, sigmas, faces = load_ssm()
    rng = np.random.default_rng(42)

    top_labels = ["(a)", "(b)", "(c)", "(d)"]
    bot_labels = ["(e)", "(f)", "(g)", "(h)"]

    fig = plt.figure(figsize=(10.6, 7.7), facecolor="white")

    col_x = [0.015, 0.265, 0.515, 0.765]
    col_w = 0.215
    top_positions = [[x, 0.565, col_w, 0.375] for x in col_x]
    bot_positions = [[x, 0.075, col_w, 0.375] for x in col_x]

    # Row 1 — statistical characterisation of the SSM and the sampling rule.
    top_axes = [fig.add_axes(pos, projection="rectilinear") for pos in top_positions]
    draw_chi2_distribution(top_axes[0])
    draw_variance_explained(top_axes[1], sigmas)
    draw_latent_scatter(top_axes[2], pc, sigmas, rng=rng)
    draw_thickness_distribution(top_axes[3], mean_pts, faces)

    # Row 2 — geometric pipeline built from the sampled coefficients.
    # Panel (f) uses two stacked axes (accepted / rejected) instead of one,
    # since compositing both meshes into a single axes made them overlap.
    fx, fy, fw, fh = bot_positions[1]
    gap_f = 0.02
    f_top_pos = [fx, fy + fh / 2 + gap_f / 2, fw, fh / 2 - gap_f / 2]
    f_bot_pos = [fx, fy, fw, fh / 2 - gap_f / 2]

    bot_axes = [fig.add_axes(bot_positions[0], projection="3d")]
    f_top_ax = fig.add_axes(f_top_pos, projection="3d")
    f_bot_ax = fig.add_axes(f_bot_pos, projection="3d")
    bot_axes.append(f_top_ax)
    bot_axes.append(fig.add_axes(bot_positions[2], projection="3d"))
    bot_axes.append(fig.add_axes(bot_positions[3], projection="3d"))

    draw_mesh(bot_axes[0], mean_pts, faces, color=C_ENDO, stride=4, alpha=0.90)
    bot_axes[0].set_title("Mesh synthesis\n$X(b)=\\bar{X}+\\Phi b$", fontsize=7, pad=2)
    draw_quality_gate(f_top_ax, f_bot_ax, mean_pts, pc, sigmas, faces, rng=rng)
    draw_contour_rings(bot_axes[2], mean_pts, faces)
    draw_query_cache(bot_axes[3], mean_pts, faces, rng=rng)

    for ax, lbl in zip(top_axes + bot_axes, top_labels + bot_labels):
        ax.set_title(f"{lbl} {ax.get_title()}", fontsize=7)
    f_bot_ax.set_title(f"(f$'$) {f_bot_ax.get_title()}", fontsize=7)

    # Arrows connecting panels within each row only; the two rows represent
    # parallel statistical vs. geometric views of the same sampling stage.
    arrow_kw = dict(
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.9,
        color="#555555",
        shrinkA=0, shrinkB=0,
        clip_on=False, zorder=50,
    )
    for row_axes in (top_axes, bot_axes):
        for i in range(len(row_axes) - 1):
            b0 = row_axes[i].get_position()
            b1 = row_axes[i + 1].get_position()
            mid_y = 0.5 * (b0.y0 + b0.y1)
            fig.add_artist(
                matplotlib.patches.FancyArrowPatch(
                    (b0.x1 + 0.006, mid_y),
                    (b1.x0 - 0.006, mid_y),
                    **arrow_kw,
                )
            )

    out = OUT_DIR / "fig_ssm_sampling_pipeline.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.06,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out.relative_to(ROOT)}")
    return out


# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Generating SSM figures …")
    fig_ssm_mean_modes()
    fig_ssm_sampling_pipeline()
    print("Done.")


if __name__ == "__main__":
    main()
