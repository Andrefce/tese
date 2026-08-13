"""Generate the dataset composition and split figure for Chapter 3."""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "xtick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 0,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

C_TRAIN = "#31688e"
C_VAL = "#8fb8d4"
C_TEST = "#d98b3a"
C_SYNTH = "#c4c4c4"
C_EDGE = "#404040"

ED_VALID = 444
ES_VALID = 423
SSM_ACCEPTED = 1300
SSM_SAMPLED = 2048

# Patient-level 70/15/15 allocation, rounded to whole patients.
ED_SPLIT = (310, 67, 67)
ES_SPLIT = (296, 63, 64)

STREAMS = (
    ("Real ED", "ACDC + M&Ms-2", ED_VALID, ED_SPLIT),
    ("Real ES", "ACDC + M&Ms-2", ES_VALID, ES_SPLIT),
    ("Synthetic ED", "UK Biobank SSM", SSM_ACCEPTED, None),
)

BAR_H = 0.34


def main():
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(len(STREAMS) - 0.45, -0.75)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of the stream (%)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(C_EDGE)
    ax.tick_params(axis="x", colors=C_EDGE)

    for row, (name, source, total, split) in enumerate(STREAMS):
        ax.text(0, row - 0.30, f"{name} ({source}) — {total} meshes",
                ha="left", va="bottom", fontsize=8, color=C_EDGE)
        if split is None:
            ax.barh(row, 100, height=BAR_H, color=C_SYNTH, zorder=3)
            ax.text(50, row, "pre-training only", ha="center", va="center",
                    fontsize=7.5, color="#202020", zorder=4)
            continue
        cursor = 0.0
        for count, colour in zip(split, (C_TRAIN, C_VAL, C_TEST)):
            width = 100 * count / total
            ax.barh(row, width, left=cursor, height=BAR_H, color=colour,
                    edgecolor="white", linewidth=0.7, zorder=3)
            ax.text(cursor + width / 2, row, str(count), ha="center",
                    va="center", fontsize=7.5, zorder=4,
                    color="white" if colour == C_TRAIN else "#202020")
            cursor += width

    ax.legend(handles=[Patch(facecolor=C_TRAIN, label="Train (70%)"),
                       Patch(facecolor=C_VAL, label="Validation (15%)"),
                       Patch(facecolor=C_TEST, label="Test (15%)"),
                       Patch(facecolor=C_SYNTH, label="Not partitioned")],
              loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4,
              frameon=False, handlelength=1.0, handleheight=0.9,
              columnspacing=1.4, borderpad=0.0)

    fig.savefig("images/dataset_selection_and_splits.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
    print("Wrote images/dataset_selection_and_splits.pdf")
