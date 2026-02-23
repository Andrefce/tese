"""
models/phase1.py
Phase 1: Denoising GNN — full noisy shape → clean shape.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, AttentionalAggregation

from models.utils import dynamic_knn_edges, laplacian_loss, mse_fn


class CardiacGNN(nn.Module):
    """
    Phase 1: full noisy shape → clean shape.
    Input node features (5): [x, y, z, radial_dist, tissue_type]
    """

    def __init__(self, node_features=5, edge_features=6,
                 hidden_dim=64, deform_cap=20.0, k_dynamic=8):
        super().__init__()
        self.cap    = deform_cap
        self._k_dyn = k_dynamic

        self.gat1 = GATv2Conv(
            node_features, hidden_dim, edge_dim=edge_features,
            heads=2, concat=False, dropout=0.3,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=0.3)
        self.bn3  = nn.BatchNorm1d(hidden_dim)

        gate_nn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.pool = AttentionalAggregation(gate_nn)
        self.gctx = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + node_features, 128),
            nn.LeakyReLU(0.2), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.LeakyReLU(0.2), nn.Linear(64, 3),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def _dyn_forward(self, h, batch):
        dyn_ei = dynamic_knn_edges(h.detach(), batch, self._k_dyn)
        row, col = dyn_ei
        msg   = self.edge_mlp(torch.cat([h[row], h[col] - h[row]], dim=1))
        h_agg = torch.zeros(h.shape[0], msg.shape[1], device=h.device)
        h_agg.scatter_add_(0, row.unsqueeze(1).expand_as(msg), msg)
        cnt = torch.zeros(h.shape[0], 1, device=h.device)
        cnt.scatter_add_(0, row.unsqueeze(1),
                         torch.ones(row.shape[0], 1, device=h.device))
        return h_agg / cnt.clamp(min=1.0)

    def forward(self, data, epoch=0):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h      = F.leaky_relu(self.bn1(self.gat1(x, ei, ea)), 0.2)
        h      = F.leaky_relu(self.bn2(self._dyn_forward(h, batch)), 0.2)
        h      = F.leaky_relu(self.bn3(self.gat2(h, ei)), 0.2)
        g      = self.gctx(self.pool(h, batch))
        h_full = torch.cat([h, g[batch], x], dim=1)
        return x[:, :3] + torch.tanh(self.decoder(h_full)) * self.cap

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def p1_total_loss(pred, target, edge_index, lam=0.01):
    lm = mse_fn(pred, target)
    ls = laplacian_loss(pred, edge_index)
    return lm + lam * ls, lm, ls
