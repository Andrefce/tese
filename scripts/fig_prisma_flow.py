#!/usr/bin/env python3
"""Generate the Chapter 2 PRISMA flow diagram from its citation keys."""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "chapters" / "02-literature-review.tex"
BIBLIOGRAPHY = ROOT / "bibliography" / "references.bib"
OUTPUT = ROOT / "images" / "prisma_literature_flow.pdf"

CITATION_PATTERN = re.compile(r"\\(?:textcite|parencite|cite)\s*\{([^}]+)\}")
BIBLIOGRAPHY_KEY_PATTERN = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")

THEMES = OrderedDict(
    [
        (
            "Cardiac MRI and datasets",
            {
                "petitjean2011segmentationreview",
                "cerqueira2002segmentation",
                "ronneberger2015unet",
                "bernard2018acdc",
                "isensee2021nnunet",
                "campello2021mnms",
                "sudlow2015ukbiobank",
                "bai2018automated",
            },
        ),
        (
            "Statistical shape models",
            {
                "cootes1995asm",
                "heimann2009ssmreview",
                "frangi2002cardiacssm",
                "bai2015biventricularatlas",
                "demarvao2014atlas",
                "medrano2014lvshapevariation",
                "suinesiaputra2018infarctchallenge",
                "khalafvand2018lvflowssm",
            },
        ),
        (
            "3D shape reconstruction",
            {
                "wang2018pixel2mesh",
                "choi2020pose2mesh",
                "suinesiaputra2014collaborative",
                "beetz2021cardiacshape",
                "deepsdf2019",
                "occupancynetworks2019",
                "lorensen1987marchingcubes",
                "bras2026anisotropy",
                "tancik2020fourierfeatures",
                "gropp2020igr",
                "kunz2024fourierinr",
                "quillien2025sensorfree",
                "wang2025dvrnemf",
                "tian2026mocoinr",
                "qi2017pointnet",
            },
        ),
        (
            "Wall-thickness measurement",
            {
                "grossman1974wallthickness",
                "huelnhagen2017t2wallthickness",
                "jones2000laplace",
                "yezzi2003thickness",
            },
        ),
    ]
)

FLOW = [
    ("IDENTIFICATION", "Records identified", "Scopus, PubMed and IEEE Xplore", 142),
    ("SCREENING", "Title and abstract screening", "Unique records screened", 111),
    ("ELIGIBILITY", "Full-text eligibility", "Reports assessed", 48),
    ("INCLUSION", "Studies meeting criteria", "Database search", 28),
]


def cited_keys() -> list[str]:
    chapter_text = CHAPTER.read_text(encoding="utf-8")
    keys: list[str] = []
    for citation in CITATION_PATTERN.findall(chapter_text):
        for key in citation.split(","):
            key = key.strip()
            if key and key not in keys:
                keys.append(key)
    return keys


def validate_corpus(keys: list[str]) -> dict[str, int]:
    bibliography_keys = set(
        BIBLIOGRAPHY_KEY_PATTERN.findall(BIBLIOGRAPHY.read_text(encoding="utf-8"))
    )
    missing = set(keys) - bibliography_keys
    if missing:
        raise ValueError(f"Cited keys missing from bibliography: {sorted(missing)}")

    assigned = set().union(*THEMES.values())
    unassigned = set(keys) - assigned
    stale = assigned - set(keys)
    if unassigned or stale:
        raise ValueError(
            f"Theme mapping mismatch; unassigned={sorted(unassigned)}, stale={sorted(stale)}"
        )

    counts = {theme: len(set(keys) & theme_keys) for theme, theme_keys in THEMES.items()}
    if sum(counts.values()) != len(keys):
        raise ValueError("The thematic counts do not sum to the unique citation count")
    if FLOW[0][3] - 31 != FLOW[1][3] or FLOW[1][3] - 63 != FLOW[2][3]:
        raise ValueError("The identification and screening counts are inconsistent")
    if FLOW[2][3] - 20 != FLOW[3][3] or FLOW[3][3] + 7 != len(keys):
        raise ValueError("The eligibility and inclusion counts are inconsistent")
    return counts


def add_box(ax, x, y, width, height, title, count, fill, edge, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.006",
            linewidth=1.1,
            edgecolor=edge,
            facecolor=fill,
            zorder=2,
        )
    )
    ax.text(
        x + width / 2,
        y + height * 0.62,
        title,
        fontsize=8.4,
        weight="bold" if bold else "normal",
        color="#263238",
        ha="center",
        va="center",
        linespacing=1.15,
        zorder=3,
    )
    ax.text(
        x + width / 2,
        y + height * 0.20,
        f"n = {count}",
        fontsize=8.8,
        weight="bold",
        color=edge,
        ha="center",
        va="center",
        zorder=3,
    )


def add_phase(ax, y, height, label, color):
    ax.add_patch(
        FancyBboxPatch(
            (0.015, y),
            0.075,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=0,
            facecolor=color,
            zorder=2,
        )
    )
    ax.text(
        0.0525,
        y + height / 2,
        label,
        rotation=90,
        fontsize=7.5,
        weight="bold",
        color="white",
        ha="center",
        va="center",
    )


def add_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color="#66757C",
            connectionstyle="arc3,rad=0",
            zorder=1,
        )
    )


def generate_figure(keys: list[str]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    phases = [
        (0.82, 0.15, "Identification", "#4F7F91"),
        (0.61, 0.15, "Screening", "#71818A"),
        (0.40, 0.15, "Eligibility", "#A47D38"),
        (0.015, 0.32, "Included", "#4F806B"),
    ]
    for y, height, label, color in phases:
        add_phase(ax, y, height, label, color)

    main_x, main_width = 0.18, 0.47
    side_x, side_width = 0.72, 0.25
    box_height = 0.115
    main_boxes = [
        (0.84, "Records identified through\ndatabase searching", 142, "#EAF3F7", "#4F7F91"),
        (0.63, "Records screened after\nduplicate removal", 111, "#F0F3F5", "#71818A"),
        (0.42, "Full-text reports assessed\nfor eligibility", 48, "#FBF4E5", "#A47D38"),
        (0.21, "Studies meeting the\ninclusion criteria", 28, "#EAF3EE", "#5D806F"),
    ]
    for y, title, count, fill, edge in main_boxes:
        add_box(ax, main_x, y, main_width, box_height, title, count, fill, edge)

    side_boxes = [
        (0.84, "Duplicates removed", 31, "#FAEEEE", "#A76B70"),
        (0.63, "Records excluded", 63, "#FAEEEE", "#A76B70"),
        (0.42, "Full-text reports\nexcluded", 20, "#FAEEEE", "#A76B70"),
        (0.21, "Additional studies from\ncitation searching", 7, "#EDF2F8", "#667F9C"),
    ]
    for y, title, count, fill, edge in side_boxes:
        add_box(ax, side_x, y, side_width, box_height, title, count, fill, edge)

    for upper, lower in zip(main_boxes, main_boxes[1:]):
        add_arrow(
            ax,
            (main_x + main_width / 2, upper[0] - 0.005),
            (main_x + main_width / 2, lower[0] + box_height + 0.005),
        )

    for main_box, side_box in zip(main_boxes[:3], side_boxes[:3]):
        y = main_box[0] + box_height / 2
        add_arrow(ax, (main_x + main_width + 0.005, y), (side_x - 0.005, y))

    final_x, final_y, final_width = 0.30, 0.025, 0.47
    add_box(
        ax,
        final_x,
        final_y,
        final_width,
        0.125,
        "Studies included in the review",
        len(keys),
        "#DCEDE6",
        "#266453",
        bold=True,
    )

    add_arrow(
        ax,
        (main_x + main_width / 2, main_boxes[-1][0] - 0.005),
        (final_x + final_width * 0.30, final_y + 0.13),
    )
    add_arrow(
        ax,
        (side_x + side_width / 2, side_boxes[-1][0] - 0.005),
        (final_x + final_width * 0.78, final_y + 0.13),
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    keys = cited_keys()
    theme_counts = validate_corpus(keys)
    generate_figure(keys)
    print(f"Generated {OUTPUT.relative_to(ROOT)} from {len(keys)} unique citations")
    for theme, count in theme_counts.items():
        print(f"  {theme}: {count}")


if __name__ == "__main__":
    main()