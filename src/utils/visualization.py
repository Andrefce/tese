"""
visualization.py
================
Reusable plotting functions for mesh inspection, training monitoring,
and result reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)


# ---------------------------------------------------------------------------
# 3D mesh
# ---------------------------------------------------------------------------

def plot_mesh_3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    title: str = "LV Mesh",
    ax: Optional[plt.Axes] = None,
    color: str = "lightcoral",
    alpha: float = 0.7,
    elev: float = 20.0,
    azim: float = 45.0,
) -> plt.Axes:
    """Plot a triangular surface mesh in 3D."""
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=faces, color=color, alpha=alpha,
        edgecolor="darkred", linewidth=0.05,
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.view_init(elev=elev, azim=azim)
    return ax


# ---------------------------------------------------------------------------
# 2D slice contours
# ---------------------------------------------------------------------------

def plot_slice_contours(
    endo: np.ndarray,
    epi: np.ndarray,
    z: float,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot endo/epi contours for a single Z-slice."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    for contour, color, label in [(epi, "red", "Epicardium"), (endo, "blue", "Endocardium")]:
        if len(contour) < 2:
            continue
        closed = np.vstack([contour, contour[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2, label=label)
        ax.scatter(contour[:, 0], contour[:, 1], c=color, s=15, alpha=0.5)

    if len(epi) > 2 and len(endo) > 2:
        epi_c = np.vstack([epi, epi[0]])
        endo_c = np.vstack([endo, endo[0]])
        ax.fill(epi_c[:, 0], epi_c[:, 1], color="red", alpha=0.08)
        ax.fill(endo_c[:, 0], endo_c[:, 1], color="white", alpha=0.9)

    ax.set_title(f"Slice z = {z:.1f} mm", fontweight="bold")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.axis("equal"); ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
    return ax


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------

def plot_graph_3d(
    nodes: np.ndarray,
    edges: np.ndarray,
    node_types: np.ndarray,
    title: str = "GNN Graph",
    max_edges: int = 200,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Visualise a 3D graph coloured by tissue type."""
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

    endo_m = node_types == 0
    epi_m = node_types == 1
    ax.scatter(*nodes[endo_m, :3].T, c="dodgerblue", s=8, alpha=0.7, label="Endo")
    ax.scatter(*nodes[epi_m, :3].T, c="crimson", s=8, alpha=0.7, label="Epi")

    for e in edges[:max_edges:max(1, len(edges) // max_edges)]:
        n1, n2 = e
        ax.plot(
            [nodes[n1, 0], nodes[n2, 0]],
            [nodes[n1, 1], nodes[n2, 1]],
            [nodes[n1, 2], nodes[n2, 2]],
            "gray", alpha=0.1, linewidth=0.5,
        )

    ax.set_title(f"{title}\n({len(nodes)} nodes, {len(edges)} edges)", fontweight="bold")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(fontsize=9); ax.view_init(elev=20, azim=45)
    return ax


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: list[dict],
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot train/val MSE and total loss curves from trainer history."""
    epochs = [r["epoch"] for r in history]
    train_mse = [r["train_mse"] for r in history]
    val_mse = [r["val_mse"] for r in history]
    val_total = [r["val_total"] for r in history]
    best_ep = history[int(np.argmin(val_total))]["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, yscale, ylabel in zip(axes, ["linear", "log"], ["MSE Loss", "MSE Loss (log)"]):
        ax.plot(epochs, train_mse, "b-", lw=2, label="Train MSE", alpha=0.85)
        ax.plot(epochs, val_mse, "r-", lw=2, label="Val MSE", alpha=0.85)
        ax.axvline(best_ep, color="green", ls="--", lw=2, label=f"Best (ep {best_ep})")
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title("Training Progress", fontweight="bold")
        ax.set_yscale(yscale)
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  [viz] Saved training curves → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Prediction vs ground truth
# ---------------------------------------------------------------------------

def plot_prediction_vs_truth(
    truth_pos: np.ndarray,
    pred_pos: np.ndarray,
    title: str = "Reconstruction: Truth vs Prediction",
    save_path: Optional[str | Path] = None,
) -> None:
    """Side-by-side 3D scatter of ground-truth and predicted node positions."""
    fig = plt.figure(figsize=(12, 5))

    for idx, (pos, color, label) in enumerate([
        (truth_pos, "dodgerblue", "Ground Truth"),
        (pred_pos, "crimson", "GNN Prediction"),
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        ax.scatter(*pos.T, c=color, s=2, alpha=0.4)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
        ax.view_init(elev=20, azim=45)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  [viz] Saved prediction comparison → {save_path}")
    plt.show()
