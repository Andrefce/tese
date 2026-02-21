"""
losses.py
=========
Loss functions used during CardiacGNN training.

  - MSE reconstruction loss (primary)
  - Laplacian smoothness loss (regularizer)
  - Combined loss with configurable weighting
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predicted and target positions."""
    return nn.functional.mse_loss(pred, target)


def laplacian_smoothness_loss(
    positions: torch.Tensor,
    edge_index: torch.LongTensor,
) -> torch.Tensor:
    """
    Laplacian smoothness loss.

    Penalises the difference between each node's position and the
    mean position of its graph neighbours. A lower value means the
    predicted mesh is locally smooth (neighbouring nodes move together).

    Parameters
    ----------
    positions   : (N, 3) predicted node positions
    edge_index  : (2, E) edge list (COO format, bidirectional)

    Returns
    -------
    Scalar loss (mean over all nodes).
    """
    row, col = edge_index  # row = source, col = neighbour

    num_nodes = positions.size(0)
    device = positions.device

    # Accumulate neighbour positions
    sum_pos = torch.zeros_like(positions)
    count = torch.zeros(num_nodes, device=device)

    sum_pos.index_add_(0, row, positions[col])
    count.index_add_(0, row, torch.ones(row.size(0), device=device))

    # Avoid division by zero for isolated nodes
    avg_neighbour = sum_pos / (count.unsqueeze(-1).clamp(min=1e-6))

    return torch.mean(torch.norm(positions - avg_neighbour, dim=1))


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_index: torch.LongTensor,
    smoothness_weight: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the total training loss.

    Returns
    -------
    total_loss, l_mse, l_smooth  (all scalar tensors)
    """
    l_mse = mse_loss(pred, target)
    l_smooth = laplacian_smoothness_loss(pred, edge_index)
    total = l_mse + smoothness_weight * l_smooth
    return total, l_mse, l_smooth
