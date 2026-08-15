"""
Generate the wall-thickness method-comparison figure for the results chapter:
  images/wall_thickness_methods_comparison.pdf

Values are read from the NOR cohort tables written by
``scripts/eval_demo/run_cohort.py`` (end-diastole, CardioSDF model geometry),
restricted to the four methods retained in the methodology: the Laplace field,
Yezzi--Prince, SDF cone-ray, and EDT baseline estimators. The dashed line marks
the input-segmentation reference mean. No values are invented.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THESIS = Path(__file__).resolve().parents[1]
COHORT = THESIS / "test-new-model" / "cohort_full_nor_hcm10"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

LABELS = {
    "Laplace field": "Laplace field",
    "Yezzi-Prince": "Yezzi--Prince",
    "SDF cone rays": "SDF cone rays",
    "EDT boundary sum": "EDT boundary sum",
}
colors = ["#1d3557", "#2a6f97", "#457b9d", "#c9a0a0"]

wall = pd.read_csv(COHORT / "wall_methods.csv")
wall = wall[(wall["phase"] == "ED") & (wall["geometry"] == "model")]
reference = pd.read_csv(COHORT / "reference_thickness.csv")
REFERENCE_MEAN = float(reference[reference["phase"] == "ED"]["ref_mm"].mean())
N_PATIENTS = int(wall["patient"].nunique())

names, means, p5, p95 = [], [], [], []
for method, label in LABELS.items():
    sub = wall[wall["method"] == method]
    names.append(label)
    means.append(sub["mean_mm"].mean())
    p5.append(sub["p5_mm"].mean())
    p95.append(sub["p95_mm"].mean())
means = np.array(means)
p5 = np.array(p5)
p95 = np.array(p95)

x = np.arange(len(names))
# Asymmetric whiskers from p5 to p95 around the mean.
lower = means - p5
upper = p95 - means

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.bar(x, means, width=0.6, color=colors, zorder=3,
       yerr=[lower, upper], capsize=5,
       error_kw=dict(ecolor="#333333", lw=1.1, capthick=1.1, zorder=4))

for xi, mu in zip(x, means):
    ax.text(xi, mu + 0.12, f"{mu:.1f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold")

ax.axhline(REFERENCE_MEAN, color="#e63946", ls="--", lw=1.2, zorder=2,
           label=f"Segmentation reference mean ({REFERENCE_MEAN:.1f} mm)")

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("Wall thickness (mm)")
ax.set_ylim(0, max(p95.max(), REFERENCE_MEAN) * 1.25)
ax.legend(loc="upper right", frameon=False, fontsize=8)
ax.text(0.0, -0.28, "Bars show the mean; whiskers span the 5th--95th percentile "
        f"(n = {N_PATIENTS} patients).",
        transform=ax.transAxes, fontsize=7.5, color="#555555")
fig.tight_layout()
fig.savefig(THESIS / "images" / "wall_thickness_methods_comparison.pdf")
plt.close(fig)
print("Wrote images/wall_thickness_methods_comparison.pdf")
