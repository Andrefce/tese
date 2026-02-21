"""
cardiac_gnn.py
==============
CardiacGNN: a deformation-based Graph Convolutional Network for
left ventricular shape reconstruction.

Architecture summary
--------------------
Input:  N nodes × 5 features  [x, y, z, radial_dist, tissue_type]

1. Local Encoder
   GCNConv(5  → H) → BN → LeakyReLU
   GCNConv(H  → H) → BN → LeakyReLU

2. Global Context Branch
   GlobalMeanPool(H) → Linear(H→H) → LeakyReLU → Linear(H→H)
   (broadcast back to all nodes)

3. Decoder
   Concatenate [local_feat, global_feat] → GCNConv(2H → H) → BN → LeakyReLU

4. Deformation Head
   Linear(H → 64) → LeakyReLU → Linear(64 → 3) → tanh × deformation_cap

Output: template_positions + deformation  (N × 3)

Key design choices
------------------
- Zero-initialized final linear layer → identity at init (stable early training).
- ``tanh × deformation_cap`` output → prevents exploding point clouds.
- Global context branch → adapts local predictions to overall shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool


class CardiacGNN(nn.Module):
    """
    Parameters
    ----------
    node_features : int
        Dimensionality of input node features (default: 5).
    hidden_dim : int
        Width of all hidden layers (default: 128).
    deformation_cap : float
        Maximum absolute displacement in mm predicted by the model
        (output is ``tanh(x) * deformation_cap``). Default: 20.0.
    """

    def __init__(
        self,
        node_features: int = 5,
        hidden_dim: int = 128,
        deformation_cap: float = 20.0,
    ) -> None:
        super().__init__()
        self.deformation_cap = deformation_cap

        # --- Local Encoder ---
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # --- Global Context Branch ---
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Decoder ---
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # --- Deformation Head ---
        self.deformation_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 3),
        )

        # Zero-initialize final layer → identity at epoch 0
        nn.init.zeros_(self.deformation_head[-1].weight)
        nn.init.zeros_(self.deformation_head[-1].bias)

    # ------------------------------------------------------------------
    def forward(self, data: Data | Batch) -> torch.Tensor:
        """
        Parameters
        ----------
        data : PyG Data or Batch
            Must have ``x`` (node features), ``edge_index``, and ``batch``.

        Returns
        -------
        positions : (N, 3) tensor
            Predicted 3D node positions = template + capped deformation.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Local encoder
        h = F.leaky_relu(self.bn1(self.conv1(x, edge_index)), negative_slope=0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h, edge_index)), negative_slope=0.2)

        # Global context
        g = global_mean_pool(h, batch)           # (B, H)
        g = self.global_context(g)                # (B, H)
        g_broadcast = g[batch]                    # (N, H)

        # Decoder
        h_combined = torch.cat([h, g_broadcast], dim=1)  # (N, 2H)
        h = F.leaky_relu(self.bn3(self.conv3(h_combined, edge_index)), negative_slope=0.2)

        # Deformation (capped)
        deformation = torch.tanh(self.deformation_head(h)) * self.deformation_cap

        # Residual: template + predicted displacement
        return x[:, :3] + deformation

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
