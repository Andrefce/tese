"""Generate the CardioSDF fine-tuning loss-curve figure for the Results chapter.

Reads the training ``history`` stored inside the final fine-tuned checkpoint
(``inr_sdf_combined_fresh_ed_mix_v1_final.pt``) and plots the total loss, the
main SDF loss components, and the optimisation diagnostics across the
fine-tuning epochs. The synthetic pre-training phase has no logged history and
is therefore not shown.

Output: images/training_curves.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "notebooks" / "inr_sdf_combined_fresh_ed_mix_v1_final.pt"
OUT = ROOT / "images" / "training_curves.pdf"


def main() -> None:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    history = ckpt["history"]
    best_epoch = int(ckpt.get("epoch", -1))

    ep = [r["epoch"] for r in history]

    def col(name: str) -> list[float]:
        return [r[name] for r in history]

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))

    # (a) Total loss, train vs. validation.
    ax = axes[0, 0]
    ax.plot(ep, col("tr_total"), label="train", color="C0", lw=1.2)
    ax.plot(ep, col("va_total"), label="validation", color="C3", lw=1.2)
    if best_epoch >= 0:
        ax.axvline(best_epoch, color="0.5", ls="--", lw=0.9)
    ax.set_title("(a) Total loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend(frameon=False)

    # (b) Surface loss.
    ax = axes[0, 1]
    ax.plot(ep, col("tr_L_surf"), color="C0", lw=1.2, label="train")
    ax.plot(ep, col("va_L_surf"), color="C3", lw=1.2, label="validation")
    ax.set_title("(b) Surface term")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend(frameon=False)

    # (c) Eikonal loss.
    ax = axes[0, 2]
    ax.plot(ep, col("tr_L_eik"), color="C0", lw=1.2, label="train")
    ax.plot(ep, col("va_L_eik"), color="C3", lw=1.2, label="validation")
    ax.set_title("(c) Eikonal term")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend(frameon=False)

    # (d) Off-surface loss.
    ax = axes[1, 0]
    ax.plot(ep, col("tr_L_off"), color="C0", lw=1.2, label="train")
    ax.plot(ep, col("va_L_off"), color="C3", lw=1.2, label="validation")
    ax.set_title("(d) Off-surface term")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(frameon=False)

    # (e) Mean gradient norm (Eikonal target = 1).
    ax = axes[1, 1]
    ax.plot(ep, col("tr_grad_norm_avg"), color="C2", lw=1.2)
    ax.axhline(1.0, color="0.5", ls="--", lw=0.9)
    ax.set_title(r"(e) Mean $\|\nabla f\|$")
    ax.set_xlabel("epoch")
    ax.set_ylabel("gradient norm")

    # (f) Learning-rate schedule.
    ax = axes[1, 2]
    ax.plot(ep, col("lr"), color="C4", lw=1.2)
    ax.set_title("(f) Learning rate")
    ax.set_xlabel("epoch")
    ax.set_ylabel("lr")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}  (epochs {min(ep)}-{max(ep)}, best={best_epoch})")


if __name__ == "__main__":
    main()
