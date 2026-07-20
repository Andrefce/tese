"""
Generate two publication-quality figures for the dataset chapter:
  - images/dataset_pipeline.pdf : source patients -> valid meshes (attrition)
  - images/dataset_splits.pdf   : train/val/test composition per data group

All counts are taken directly from the dataset-preparation notebooks
(datasetED_real, datasetES_real, datasetED_ssm). No values are invented.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

COL_INDEXED = "#a8c4d9"   # light steel
COL_VALID = "#1d3557"     # deep navy
COL_TRAIN = "#2a6f97"
COL_VAL = "#61a5c2"
COL_TEST = "#f4a261"
COL_SSM = "#457b9d"

# ──────────────────────────────────────────────────────────────────────
# Counts from the notebooks
# ──────────────────────────────────────────────────────────────────────
# Real data: 460 patients indexed (ACDC 100 + M&Ms-2 360).
INDEXED = 460
ED_VALID = 444          # datasetED_real: PyG samples
ES_VALID = 423          # datasetES_real: PyG samples
# Synthetic ED (SSM): 1300 accepted out of 2048 candidates.
SSM_SAMPLED = 2048
SSM_ACCEPTED = 1300

# Splits (train / val / test)
ED_SPLIT = (310, 67, 67)
ES_SPLIT = (338, 66, 19)


# ══════════════════════════════════════════════════════════════════════
# Figure 1 — attrition from indexed patients to valid meshes
# ══════════════════════════════════════════════════════════════════════
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.4, 3.2))

    groups = ["Real ED\n(ACDC + M&Ms-2)", "Real ES\n(ACDC + M&Ms-2)",
              "Synthetic ED\n(UK Biobank SSM)"]
    y = np.arange(len(groups))[::-1]
    h = 0.62

    starting = [INDEXED, INDEXED, SSM_SAMPLED]
    kept = [ED_VALID, ES_VALID, SSM_ACCEPTED]

    ax.barh(y, starting, height=h, color=COL_INDEXED,
            label="Candidates", zorder=2)
    ax.barh(y, kept, height=h, color=COL_VALID,
            label="Passed quality gates", zorder=3)

    for yi, s, k in zip(y, starting, kept):
        pct = 100.0 * k / s
        ax.text(s + 25, yi, f"{k} / {s}  ({pct:.0f}%)",
                va="center", ha="left", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.set_xlabel("Number of samples")
    ax.set_xlim(0, SSM_SAMPLED * 1.28)
    ax.set_title("From raw sources to anatomically valid meshes")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig("images/dataset_pipeline.pdf")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Figure 2 — train / val / test composition per group
# ══════════════════════════════════════════════════════════════════════
def fig_splits():
    fig, ax = plt.subplots(figsize=(6.4, 3.2))

    labels = ["Real ED", "Real ES"]
    x = np.arange(len(labels))
    w = 0.55

    train = np.array([ED_SPLIT[0], ES_SPLIT[0]])
    val = np.array([ED_SPLIT[1], ES_SPLIT[1]])
    test = np.array([ED_SPLIT[2], ES_SPLIT[2]])

    ax.bar(x, train, w, color=COL_TRAIN, label="Train", zorder=3)
    ax.bar(x, val, w, bottom=train, color=COL_VAL, label="Validation", zorder=3)
    ax.bar(x, test, w, bottom=train + val, color=COL_TEST, label="Test", zorder=3)

    totals = train + val + test
    for xi, (t, v, te, tot) in enumerate(zip(train, val, test, totals)):
        ax.text(xi, t / 2, str(t), ha="center", va="center",
                color="white", fontsize=8.5)
        ax.text(xi, t + v / 2, str(v), ha="center", va="center",
                color="white", fontsize=8.5)
        ax.text(xi, t + v + te / 2, str(te), ha="center", va="center",
                color="#333333", fontsize=8.5)
        ax.text(xi, tot + 6, f"n = {tot}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Patients")
    ax.set_ylim(0, max(totals) * 1.18)
    ax.set_title("Patient-level train / validation / test split")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=3)
    fig.tight_layout()
    fig.savefig("images/dataset_splits.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_splits()
    print("Wrote images/dataset_pipeline.pdf and images/dataset_splits.pdf")
