"""Generate patient-level metric figures for the Results chapter.

Reads only the frozen v2 cohort and its derived patient summaries. The figures
show individual patients, distribution summaries, and bootstrap confidence
intervals without exposing patient identifiers.

Outputs:
  images/results_reconstruction_distributions.pdf
  images/results_group_thickness_distributions.pdf
    images/results_thickness_geometry_agreement.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "test-new-model" / "cohort_full_nor_hcm10"
ANALYSIS = COHORT / "analysis"
OUT = ROOT / "images"

GROUPS = ("NOR", "HCM")
COLORS = {"NOR": "#2A6F97", "HCM": "#C1443B"}
RNG_SEED = 20260823
BOOTSTRAP_SAMPLES = 10_000

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_groups() -> dict[str, str]:
    groups: dict[str, str] = {}
    for path in sorted((COHORT / "cache").glob("*_result.json")):
        payload = json.loads(path.read_text())
        groups[str(payload["patient"])] = str(payload["group"])
    if len(groups) != 30:
        raise ValueError(f"expected 30 cached patients, found {len(groups)}")
    return groups


def bootstrap_mean_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def draw_distribution(
    ax: plt.Axes,
    frame: pd.DataFrame,
    column: str,
    title: str,
    panel: str,
    seed: int,
    reference: float | None = None,
    reference_label: str | None = None,
) -> None:
    values = [frame.loc[frame["group"] == group, column].dropna().to_numpy(float)
              for group in GROUPS]
    boxes = ax.boxplot(
        values,
        positions=np.arange(len(GROUPS)),
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1E1E1E", "linewidth": 1.2},
        whiskerprops={"color": "#555555", "linewidth": 0.9},
        capprops={"color": "#555555", "linewidth": 0.9},
    )
    for box, group in zip(boxes["boxes"], GROUPS):
        box.set_facecolor(COLORS[group])
        box.set_alpha(0.24)
        box.set_edgecolor(COLORS[group])
        box.set_linewidth(1.0)

    generator = np.random.default_rng(seed)
    for position, (group, group_values) in enumerate(zip(GROUPS, values)):
        jitter = generator.uniform(-0.105, 0.105, size=len(group_values))
        ax.scatter(
            position + jitter,
            group_values,
            s=20,
            color=COLORS[group],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.88,
            zorder=3,
        )
        low, high = bootstrap_mean_interval(group_values, seed + position + 100)
        mean = float(group_values.mean())
        ax.errorbar(
            position,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            markersize=4.1,
            color="#111111",
            markerfacecolor="white",
            markeredgewidth=0.9,
            capsize=3,
            linewidth=1.0,
            zorder=4,
        )

    if reference is not None:
        ax.axhline(reference, color="#555555", linestyle="--", linewidth=0.9, zorder=1)
        if reference_label:
            ax.text(
                0.99,
                reference,
                reference_label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=7.2,
                color="#555555",
            )

    counts = [len(group_values) for group_values in values]
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([f"{group}\n$n={count}$" for group, count in zip(GROUPS, counts)])
    ax.set_title(f"({panel}) {title}", loc="left", pad=5)
    ax.margins(x=0.24)


def reconstruction_figure(groups: dict[str, str]) -> None:
    reconstruction = pd.read_csv(COHORT / "recon_quality.csv")
    reconstruction["group"] = reconstruction["patient"].map(groups)
    if reconstruction["group"].isna().any():
        raise ValueError("missing group labels in reconstruction data")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    settings = (
        ("endo_chamfer_mm", "Endocardial Chamfer distance", "A", None, None),
        ("endo_hd95_mm", "Endocardial HD95", "B", None, None),
        ("myo_dice", "Myocardial Dice", "C", None, None),
        ("vol_ratio_endo", "Cavity volume ratio", "D", 1.0, "no volume bias"),
    )
    for index, (ax, setting) in enumerate(zip(axes.flat, settings)):
        column, title, panel, reference, reference_label = setting
        draw_distribution(
            ax,
            reconstruction,
            column,
            title,
            panel,
            seed=RNG_SEED + index * 10,
            reference=reference,
            reference_label=reference_label,
        )
        ax.set_ylabel("Dice" if column == "myo_dice" else
                      "Ratio" if column == "vol_ratio_endo" else "Distance (mm)")

    fig.text(
        0.5,
        0.005,
        "Points represent patients; diamonds show group means with 95% bootstrap confidence intervals.",
        ha="center",
        fontsize=7.5,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path = OUT / "results_reconstruction_distributions.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def thickness_figure() -> None:
    summary = pd.read_csv(ANALYSIS / "patient_summaries.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.95))
    settings = (
        ("model_aha_ed_mm", "Mean ED AHA-17 thickness", "A", None, None),
        ("max_segment_ed_laplace_mm", "Maximum ED segment", "B", 15.0, "15 mm threshold"),
        ("model_systolic_thickening_pct", "Systolic thickening", "C", None, None),
    )
    for index, (ax, setting) in enumerate(zip(axes, settings)):
        column, title, panel, reference, reference_label = setting
        draw_distribution(
            ax,
            summary,
            column,
            title,
            panel,
            seed=RNG_SEED + 100 + index * 10,
            reference=reference,
            reference_label=reference_label,
        )
        ax.set_ylabel("Thickening (%)" if column.endswith("pct") else "Thickness (mm)")

    fig.text(
        0.5,
        0.005,
        "Points represent patients; diamonds show group means with 95% bootstrap confidence intervals.",
        ha="center",
        fontsize=7.5,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.4)
    path = OUT / "results_group_thickness_distributions.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def geometry_agreement_figure() -> None:
    wall = pd.read_csv(COHORT / "wall_methods.csv")
    methods = (
        "Laplace field",
        "Yezzi-Prince",
        "SDF cone rays",
        "EDT boundary sum",
    )
    phase_styles = {
        "ED": {"color": "#2A6F97", "marker": "o"},
        "ES": {"color": "#C1443B", "marker": "^"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2))
    for panel_index, (ax, method) in enumerate(zip(axes.flat, methods)):
        method_rows = wall[wall["method"] == method]
        paired_phases: dict[str, pd.DataFrame] = {}
        all_values: list[np.ndarray] = []
        annotations: list[str] = []

        for phase in ("ED", "ES"):
            phase_rows = method_rows[method_rows["phase"] == phase]
            paired = phase_rows.pivot(index="patient", columns="geometry", values="mean_mm").dropna()
            if not {"model", "voxel"}.issubset(paired.columns):
                raise ValueError(f"missing paired geometry values for {method}, {phase}")
            paired_phases[phase] = paired
            all_values.extend([paired["voxel"].to_numpy(float), paired["model"].to_numpy(float)])

            difference = paired["model"] - paired["voxel"]
            correlation = float(paired[["voxel", "model"]].corr().iloc[0, 1])
            annotations.append(
                f"{phase}: bias {difference.mean():+.2f} mm, $r={correlation:.2f}$"
            )

        pooled = np.concatenate(all_values)
        data_min, data_max = float(pooled.min()), float(pooled.max())
        padding = max(0.8, 0.08 * (data_max - data_min))
        lower, upper = data_min - padding, data_max + padding
        ax.plot([lower, upper], [lower, upper], color="#555555", linestyle="--",
                linewidth=0.9, zorder=1, label="identity")

        for phase, paired in paired_phases.items():
            style = phase_styles[phase]
            ax.scatter(
                paired["voxel"],
                paired["model"],
                s=24,
                marker=style["marker"],
                color=style["color"],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.84,
                label=phase,
                zorder=3,
            )

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Derived geometry (mm)")
        ax.set_ylabel("Model geometry (mm)")
        panel = chr(ord("A") + panel_index)
        ax.set_title(f"({panel}) {method}", loc="left", pad=5)
        ax.text(
            0.03,
            0.97,
            "\n".join(annotations),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.8},
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=1.2, w_pad=1.2)
    path = OUT / "results_thickness_geometry_agreement.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    groups = load_groups()
    reconstruction_figure(groups)
    thickness_figure()
    geometry_agreement_figure()


if __name__ == "__main__":
    main()