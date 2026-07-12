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


def make_epicardium(endo: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = vertex_normals(endo, faces)
    z_norm = (endo[:, 2] - endo[:, 2].min()) / max(np.ptp(endo[:, 2]), 1e-8)
    angles = np.arctan2(endo[:, 1], endo[:, 0])
    thickness = 5.5 + 4.2 * z_norm + 1.1 * np.sin(2.0 * angles + 0.6) * (0.4 + 0.6 * z_norm)
    thickness = np.clip(thickness, 4.5, 11.5)
    return endo + normals * thickness[:, None]


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

def draw_latent_scatter(ax, pc, sigmas, n_samples=400, rng=None):
    """Scatter of accepted samples in (b1, b2) PCA space."""
    if rng is None:
        rng = np.random.default_rng(42)
    d = pc.shape[1]
    accepted = []
    while len(accepted) < n_samples:
        b = rng.standard_normal(d) * sigmas
        if np.all(np.abs(b / sigmas) <= 3.0) and np.sum((b / sigmas) ** 2) <= 200:
            accepted.append(b)
    accepted = np.array(accepted)
    ax.scatter(accepted[:, 0], accepted[:, 1], s=4, alpha=0.55,
               c=np.linalg.norm(accepted[:, :10], axis=1), cmap="viridis", linewidths=0)
    ax.set_xlabel("$b_1$", fontsize=7, labelpad=1)
    ax.set_ylabel("$b_2$", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=5.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.set_title("Latent sampling\n(Mahalanobis bound)", fontsize=7, pad=2)


def draw_quality_gate(ax, mean_pts, pc, sigmas, faces, rng=None):
    """Two overlaid meshes: accepted (blue) vs exaggerated outlier (red)."""
    if rng is None:
        rng = np.random.default_rng(7)
    # accepted sample
    b_ok = rng.standard_normal(pc.shape[1]) * sigmas * 0.6
    pts_ok = (mean_pts.flatten() + pc @ b_ok).reshape(-1, 3)
    # rejected sample (large b outside bound)
    b_bad = np.zeros(pc.shape[1])
    b_bad[0] = 4.5 * sigmas[0]
    pts_bad = (mean_pts.flatten() + pc @ b_bad).reshape(-1, 3)

    draw_mesh(ax, pts_ok, faces, color=C_ENDO, stride=5, alpha=0.88)
    draw_mesh(ax, pts_bad, faces, color="#CC3333", stride=5, alpha=0.35)

    legend_handles = [
        mpatches.Patch(color=C_ENDO, label="Accepted"),
        mpatches.Patch(color="#CC3333", label="Rejected (outlier)"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.50, -0.08), ncol=1, fontsize=5.5,
              frameon=False, labelcolor="#333333", handlelength=1.0)
    ax.set_title("Quality gate\n(volume, sphericity)", fontsize=7, pad=2)


def draw_epi_offset(ax, mean_pts, faces):
    """Endo + synthetic epi shown together."""
    epi = make_epicardium(mean_pts, faces)
    draw_mesh(ax, epi, faces, color=C_EPI, stride=4, alpha=0.40)
    draw_mesh(ax, mean_pts, faces, color=C_ENDO, stride=4, alpha=0.90)
    ax.set_title("Synthetic epi\n(normal offset)", fontsize=7, pad=2)


def draw_contour_rings(ax, mean_pts, faces, n_slices=10):
    """Draw SAX contour rings extracted from the SSM mesh."""
    z_vals = np.linspace(mean_pts[:, 2].min() * 0.85, mean_pts[:, 2].max() * 0.85, n_slices)
    epi = make_epicardium(mean_pts, faces)

    for z in z_vals:
        for pts, col in [(mean_pts, C_ENDO), (epi, C_EPI)]:
            dz = np.abs(pts[:, 2] - z)
            ring_mask = dz < (np.ptp(pts[:, 2]) / (n_slices * 1.6))
            ring = pts[ring_mask]
            if len(ring) > 5:
                angles = np.arctan2(ring[:, 1], ring[:, 0])
                order = np.argsort(angles)
                rx, ry = ring[order, 0], ring[order, 1]
                ax.plot(np.append(rx, rx[0]), np.append(ry, ry[0]),
                        np.full(len(rx) + 1, z), color=col, linewidth=0.8, alpha=0.85)

    ax.view_init(elev=18, azim=-55)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title("SAX contour\nextraction (10 levels)", fontsize=7, pad=2)


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
    ax.set_title("Occupancy cache\n(2048 query points)", fontsize=7, pad=2)


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

    labels = [
        "(a)", "(b)", "(c)",
        "(d)", "(e)", "(f)",
    ]
    fig = plt.figure(figsize=(10.8, 3.8), facecolor="white")

    # Row of 6 panels: 3 flat, 3 3D
    positions = [
        [0.01, 0.10, 0.155, 0.80],   # (a) scatter - 2D
        [0.175, 0.10, 0.135, 0.80],  # (b) mesh - 3D
        [0.330, 0.10, 0.135, 0.80],  # (c) quality - 3D
        [0.490, 0.10, 0.150, 0.80],  # (d) pathology - 3D
        [0.655, 0.10, 0.155, 0.80],  # (e) contours - 3D
        [0.825, 0.10, 0.155, 0.80],  # (f) cache - 3D
    ]
    projections = ["rectilinear", "3d", "3d", "3d", "3d", "3d"]

    axes = [
        fig.add_axes(pos, projection=proj)
        for pos, proj in zip(positions, projections)
    ]

    draw_latent_scatter(axes[0], pc, sigmas, rng=rng)
    draw_mesh(axes[1], mean_pts, faces, color=C_ENDO, stride=4, alpha=0.90)
    axes[1].set_title("Mesh from SSM\n$X(b)=\\bar{X}+\\Phi b$", fontsize=7, pad=2)
    draw_quality_gate(axes[2], mean_pts, pc, sigmas, faces, rng=rng)
    draw_pathology_shapes(axes[3], mean_pts, pc, sigmas, faces)
    draw_contour_rings(axes[4], mean_pts, faces)
    draw_query_cache(axes[5], mean_pts, faces, rng=rng)

    for ax, lbl in zip(axes, labels):
        ax.set_title(f"{lbl} {ax.get_title()}", fontsize=7)

    # arrows between panels
    arrow_kw = dict(
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.9,
        color="#555555",
        shrinkA=0, shrinkB=0,
        clip_on=False, zorder=50,
    )
    import matplotlib.patches as FancyPatch
    for i in range(len(axes) - 1):
        b0 = axes[i].get_position()
        b1 = axes[i + 1].get_position()
        mid_y = 0.5 * (b0.y0 + b0.y1)
        fig.add_artist(
            matplotlib.patches.FancyArrowPatch(
                (b0.x1 + 0.005, mid_y),
                (b1.x0 - 0.005, mid_y),
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
