"""
Generate the wall-thickness method-comparison figure for the results chapter:
  images/wall_thickness_methods_comparison.pdf

Values are taken directly from the 10-method wall-thickness notebook
(lv_wall_thickness_10_methods), restricted to the four methods retained in the
methodology: the Laplace field, Yezzi--Prince, SDF cone-ray, and EDT baseline
estimators, all evaluated on the CardioSDF-generated LV geometry. The dashed
line marks the input-segmentation reference mean. No values are invented.
"""
import numpy as np
import matplotlib.pyplot as plt

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

# Method, mean, p5, p95 (mm) — from the notebook summary table.
METHODS = [
    ("Laplace field\n(state of the art)", 5.076, 2.916, 8.623, "#1d3557"),
    ("Yezzi--Prince\n(recent)",           4.303, 0.964, 5.804, "#2a6f97"),
    ("SDF cone rays\n(promising)",        5.469, 3.245, 7.156, "#457b9d"),
    ("EDT boundary sum\n(baseline)",      3.584, 1.490, 4.807, "#c9a0a0"),
]
REFERENCE_MEAN = 3.61  # input-segmentation KD reference (mm)

names = [m[0] for m in METHODS]
means = np.array([m[1] for m in METHODS])
p5 = np.array([m[2] for m in METHODS])
p95 = np.array([m[3] for m in METHODS])
colors = [m[4] for m in METHODS]

x = np.arange(len(METHODS))
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
ax.set_ylim(0, 10)
ax.set_title("Wall-thickness estimates on the reconstructed LV geometry")
ax.legend(loc="upper right", frameon=False, fontsize=8)
ax.text(0.0, -0.34, "Bars show the mean; whiskers span the 5th--95th percentile.",
        transform=ax.transAxes, fontsize=7.5, color="#555555")
fig.tight_layout()
fig.savefig("images/wall_thickness_methods_comparison.pdf")
plt.close(fig)
print("Wrote images/wall_thickness_methods_comparison.pdf")
