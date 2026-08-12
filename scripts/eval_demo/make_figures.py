"""Thesis figures for the single-patient method-comparison demo.

Reads the CSV/NPZ products of ``run_demo.py`` and writes vector PDFs:

    fig_phantom_accuracy.pdf     phantom MAE/bias per estimator
    fig_model_vs_voxel.pdf       Bland-Altman of the AHA-17 regional means
    fig_bullseye.pdf             AHA-17 bullseyes, both geometries, ED and ES
    fig_method_distributions.pdf per-vertex thickness distributions
    fig_surfaces.pdf             endocardial thickness maps on both surfaces

Run:
    cd tese/scripts/eval_demo && ../../../.venv/bin/python make_figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
from matplotlib import colors as mcolors                             # noqa: E402
from matplotlib.patches import Wedge                                 # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection              # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIG = OUT / "figures"

THESIS_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
}

WT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "wall_thickness",
    [(0.00, (0.23, 0.51, 0.96)), (0.25, (0.13, 0.76, 0.76)),
     (0.50, (0.13, 0.77, 0.37)), (0.75, (0.96, 0.62, 0.04)),
     (1.00, (0.91, 0.30, 0.37))],
)

# Reporting order: primary estimator first, demoted baselines last.
METHOD_ORDER = [
    "Laplace streamline (symmetric)",
    "Symmetric surface correspondence",
    "EDT boundary sum",
    "Morphological sphere propagation",
    "Decoder offset delta",
    "SDF cone rays",
    "Laplace local gradient",
]
SHORT = {
    "Laplace streamline (symmetric)": "Laplace\nstreamline",
    "Symmetric surface correspondence": "Surface\ncorrespondence",
    "EDT boundary sum": "EDT\nboundary sum",
    "Morphological sphere propagation": "Sphere\npropagation",
    "Decoder offset delta": "Decoder\noffset $\\delta$",
    "SDF cone rays": "Cone\nrays",
    "Laplace local gradient": "Laplace local\n$1/\\|\\nabla\\phi\\|$",
}
METHOD_COLORS = {
    "Laplace streamline (symmetric)": "#1f4e9c",
    "Symmetric surface correspondence": "#2a9d8f",
    "EDT boundary sum": "#5f7d4f",
    "Morphological sphere propagation": "#e9a13b",
    "Decoder offset delta": "#8e5ea2",
    "SDF cone rays": "#c05746",
    "Laplace local gradient": "#8a8a8a",
}


def _order(values) -> list:
    present = list(values)
    return [m for m in METHOD_ORDER if m in present]


# ──────────────────────────────────────────────────────────────────────────
def fig_phantom_accuracy(df: pd.DataFrame) -> None:
    methods = _order(df["method"].unique())
    phantoms = list(df["phantom"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))

    x = np.arange(len(methods))
    width = 0.8 / len(phantoms)
    for ax, metric, title in (
        (axes[0], "mae_mm", "Mean absolute error"),
        (axes[1], "bias_mm", "Signed bias"),
    ):
        for j, phantom in enumerate(phantoms):
            sub = df[df["phantom"] == phantom].set_index("method")
            vals = [sub.loc[m, metric] if m in sub.index else np.nan for m in methods]
            ax.bar(x + j * width - 0.4 + width / 2, vals, width,
                   label=phantom, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in methods], rotation=35, ha="right")
        ax.set_ylabel("mm")
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.7)
    axes[1].axhspan(-0.5, 0.5, color="grey", alpha=0.15, zorder=0)
    axes[0].legend(loc="upper left", ncols=1)
    fig.suptitle("Estimator accuracy on analytic phantoms (known thickness)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig_phantom_accuracy.pdf")
    plt.close(fig)


def fig_model_vs_voxel(aha: pd.DataFrame, agree: pd.DataFrame) -> None:
    phases = sorted(aha["phase"].unique())
    methods = _order(aha["method"].unique())
    fig, axes = plt.subplots(len(phases), 2, figsize=(7.4, 3.4 * len(phases)),
                             squeeze=False)

    for r, phase in enumerate(phases):
        ax_s, ax_ba = axes[r][0], axes[r][1]
        for method in methods:
            sub = aha[(aha["phase"] == phase) & (aha["method"] == method)]
            m = sub[sub["geometry"] == "model"].set_index("segment_id")["mean_mm"]
            v = sub[sub["geometry"] == "voxel"].set_index("segment_id")["mean_mm"]
            common = m.index.intersection(v.index)
            if len(common) == 0:
                continue
            mv, vv = m.loc[common].to_numpy(), v.loc[common].to_numpy()
            colour = METHOD_COLORS[method]
            ax_s.scatter(vv, mv, s=13, alpha=0.8, color=colour,
                         label=SHORT[method].replace("\n", " "))
            ax_ba.scatter((mv + vv) / 2, mv - vv, s=13, alpha=0.8, color=colour)

        lim = [0, np.nanmax(aha[aha["phase"] == phase]["mean_mm"]) * 1.1]
        ax_s.plot(lim, lim, "k--", linewidth=0.8)
        ax_s.set_xlim(lim)
        ax_s.set_ylim(lim)
        ax_s.set_xlabel("Voxel-based model, AHA-17 mean (mm)")
        ax_s.set_ylabel("CardioSDF model, AHA-17 mean (mm)")
        ax_s.set_title(f"{phase} — regional agreement")

        row = agree[(agree["phase"] == phase) &
                    (agree["method"] == "Laplace streamline (symmetric)")]
        if len(row):
            bias = float(row["bias_mm"].iloc[0])
            lo, hi = float(row["loa_lower_mm"].iloc[0]), float(row["loa_upper_mm"].iloc[0])
            ax_ba.axhline(bias, color="#1f4e9c", linewidth=1.0)
            ax_ba.axhline(lo, color="#1f4e9c", linestyle="--", linewidth=0.8)
            ax_ba.axhline(hi, color="#1f4e9c", linestyle="--", linewidth=0.8)
            ax_ba.text(0.99, 0.04,
                       f"Laplace streamline: bias {bias:+.2f} mm, "
                       f"LoA [{lo:+.2f}, {hi:+.2f}]",
                       transform=ax_ba.transAxes, ha="right", fontsize=7.5)
        ax_ba.axhline(0, color="black", linewidth=0.7)
        ax_ba.set_xlabel("Mean of the two geometries (mm)")
        ax_ba.set_ylabel("CardioSDF - voxel (mm)")
        ax_ba.set_title(f"{phase} — Bland-Altman")

    axes[0][0].legend(loc="upper left", fontsize=7)
    fig.suptitle("CardioSDF reconstruction vs voxel-based model, AHA-17 regional means",
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig_model_vs_voxel.pdf")
    plt.close(fig)


def _bullseye(ax, values: np.ndarray, vmin: float, vmax: float, title: str) -> None:
    """Standard AHA-17 bullseye: 6 basal, 6 mid, 4 apical, 1 apex."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rings = [(range(1, 7), 0.75, 1.0, 60, 60),
             (range(7, 13), 0.50, 0.75, 60, 60),
             (range(13, 17), 0.25, 0.50, 90, 45)]
    for seg_ids, r0, r1, span, start in rings:
        for k, seg in enumerate(seg_ids):
            v = values[seg - 1]
            colour = "#dddddd" if not np.isfinite(v) else WT_CMAP(norm(v))
            theta0 = start + k * span
            ax.add_patch(Wedge((0, 0), r1, theta0, theta0 + span, width=r1 - r0,
                               facecolor=colour, edgecolor="white", linewidth=0.8))
            ang = np.radians(theta0 + span / 2)
            rr = (r0 + r1) / 2
            if np.isfinite(v):
                ax.text(rr * np.cos(ang), rr * np.sin(ang), f"{v:.1f}",
                        ha="center", va="center", fontsize=6.5)
    apex = values[16]
    ax.add_patch(plt.Circle((0, 0), 0.25,
                            facecolor="#dddddd" if not np.isfinite(apex)
                            else WT_CMAP(norm(apex)),
                            edgecolor="white", linewidth=0.8))
    if np.isfinite(apex):
        ax.text(0, 0, f"{apex:.1f}", ha="center", va="center", fontsize=6.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)


def fig_bullseye(aha: pd.DataFrame, method: str) -> None:
    phases = sorted(aha["phase"].unique())
    geoms = ["model", "voxel"]
    sub_all = aha[aha["method"] == method]
    vmin = float(np.nanpercentile(sub_all["mean_mm"], 2))
    vmax = float(np.nanpercentile(sub_all["mean_mm"], 98))

    fig, axes = plt.subplots(len(geoms), len(phases),
                             figsize=(3.1 * len(phases), 3.3 * len(geoms)),
                             squeeze=False)
    for r, geom in enumerate(geoms):
        for c, phase in enumerate(phases):
            sub = sub_all[(sub_all["phase"] == phase) & (sub_all["geometry"] == geom)]
            vals = np.full(17, np.nan)
            for _, row in sub.iterrows():
                vals[int(row["segment_id"]) - 1] = row["mean_mm"]
            label = "CardioSDF model" if geom == "model" else "Voxel-based model"
            _bullseye(axes[r][c], vals, vmin, vmax, f"{label} — {phase}")

    sm = plt.cm.ScalarMappable(cmap=WT_CMAP, norm=mcolors.Normalize(vmin, vmax))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.03)
    cbar.set_label("Wall thickness (mm)")
    fig.suptitle(f"AHA-17 wall thickness — {method}", fontsize=10.5, y=0.99)
    fig.savefig(FIG / "fig_bullseye.pdf")
    plt.close(fig)


def fig_method_distributions(npz_paths: dict[str, Path]) -> None:
    phases = list(npz_paths)
    fig, axes = plt.subplots(1, len(phases), figsize=(4.0 * len(phases), 3.6),
                             squeeze=False)
    for c, phase in enumerate(phases):
        ax = axes[0][c]
        data = np.load(npz_paths[phase])
        keys = [k for k in data.files if k.startswith("model_wt_")]
        names = {k: k.replace("model_wt_", "") for k in keys}
        positions, labels, colours = [], [], []
        series = []
        idx = 0
        for method in METHOD_ORDER:
            key = "model_wt_" + method.lower().replace(" ", "_") \
                .replace("(", "").replace(")", "").replace("/", "_")
            for k in keys:
                if names[k] == key.replace("model_wt_", ""):
                    v = data[k]
                    v = v[np.isfinite(v)]
                    if v.size:
                        series.append(v)
                        positions.append(idx)
                        labels.append(SHORT[method])
                        colours.append(METHOD_COLORS[method])
                        idx += 1
                    break
        parts = ax.violinplot(series, positions=positions, widths=0.8,
                              showmedians=True, showextrema=False)
        for body, colour in zip(parts["bodies"], colours):
            body.set_facecolor(colour)
            body.set_alpha(0.65)
            body.set_edgecolor("white")
        parts["cmedians"].set_color("black")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Wall thickness (mm)")
        ax.set_title(f"CardioSDF model — {phase}")
    fig.suptitle("Per-vertex wall-thickness distributions", fontsize=10.5, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "fig_method_distributions.pdf")
    plt.close(fig)


def _surface_panel(ax, verts, faces, values, vmin, vmax, title):
    ax.set_title(title, fontsize=9, pad=0)
    face_val = np.nanmean(values[faces], axis=1)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colours = WT_CMAP(norm(np.nan_to_num(face_val, nan=vmin)))
    colours[~np.isfinite(face_val)] = (0.85, 0.85, 0.85, 1.0)
    coll = Poly3DCollection(verts[faces], facecolors=colours, linewidths=0)
    coll.set_edgecolor(None)
    ax.add_collection3d(coll)
    mins, maxs = verts.min(0), verts.max(0)
    centre = (mins + maxs) / 2
    radius = float((maxs - mins).max()) / 2
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] + radius, centre[2] - radius)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-60)


def fig_surfaces(npz_paths: dict[str, Path], method: str) -> None:
    key = "wt_" + method.lower().replace(" ", "_").replace("(", "") \
        .replace(")", "").replace("/", "_")
    phases = list(npz_paths)
    fig = plt.figure(figsize=(3.4 * len(phases), 6.4))
    all_vals = []
    for phase in phases:
        d = np.load(npz_paths[phase])
        for g in ("model", "voxel"):
            if f"{g}_{key}" in d.files:
                all_vals.append(d[f"{g}_{key}"])
    stacked = np.concatenate([v[np.isfinite(v)] for v in all_vals])
    vmin, vmax = np.percentile(stacked, [2, 98])

    for r, geom in enumerate(("model", "voxel")):
        for c, phase in enumerate(phases):
            d = np.load(npz_paths[phase])
            ax = fig.add_subplot(2, len(phases), r * len(phases) + c + 1,
                                 projection="3d")
            label = "CardioSDF" if geom == "model" else "Voxel-based"
            _surface_panel(ax, d[f"{geom}_endo_v"], d[f"{geom}_endo_f"],
                           d[f"{geom}_{key}"], vmin, vmax, f"{label} — {phase}")
    sm = plt.cm.ScalarMappable(cmap=WT_CMAP, norm=mcolors.Normalize(vmin, vmax))
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.02, pad=0.02)
    cbar.set_label("Wall thickness (mm)")
    fig.suptitle(f"Endocardial wall-thickness maps — {method}", fontsize=10.5, y=0.97)
    fig.savefig(FIG / "fig_surfaces.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--method", default="Laplace streamline (symmetric)",
                        help="primary estimator shown in the map figures")
    args = parser.parse_args()

    out_dir = args.out
    FIG.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(THESIS_STYLE):
        phantom_csv = out_dir / "phantom_validation.csv"
        if phantom_csv.exists():
            fig_phantom_accuracy(pd.read_csv(phantom_csv))
            print("wrote fig_phantom_accuracy.pdf")

        aha = pd.read_csv(out_dir / "aha17.csv")
        agree = pd.read_csv(out_dir / "model_vs_voxel_agreement.csv")
        fig_model_vs_voxel(aha, agree)
        print("wrote fig_model_vs_voxel.pdf")
        fig_bullseye(aha, args.method)
        print("wrote fig_bullseye.pdf")

        npz_paths = {p.stem.split("_")[-1]: p
                     for p in sorted(out_dir.glob("demo_*_*.npz"))}
        if npz_paths:
            fig_method_distributions(npz_paths)
            print("wrote fig_method_distributions.pdf")
            fig_surfaces(npz_paths, args.method)
            print("wrote fig_surfaces.pdf")

    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
