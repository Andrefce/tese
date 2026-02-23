"""
models/phase2.py
Phase 2: SSM-Constrained Completion GNN.
Sparse observed slices → K PCA mode weights → full SSM mesh.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, AttentionalAggregation

from models.utils import dynamic_knn_edges, mse_fn


class SSMCardiacGNN(nn.Module):
    """
    Phase 2: sparse observed slices → K PCA mode weights → full SSM mesh.
    Architecture mirrors Phase 1 encoder so weights transfer cleanly.
    """

    def __init__(self, node_features=6, edge_features=6,
                 hidden_dim=64, num_pca_modes=10, k_dynamic=8):
        super().__init__()
        self.k_dyn     = k_dynamic
        self.num_modes = num_pca_modes

        self.gat1 = GATv2Conv(
            node_features, hidden_dim, edge_dim=edge_features,
            heads=2, concat=False, dropout=0.3,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=0.3)
        self.bn3  = nn.BatchNorm1d(hidden_dim)

        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.pool = AttentionalAggregation(gate_nn)

        self.shape_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
        )

        self.pca_head = nn.Linear(hidden_dim, num_pca_modes)
        nn.init.normal_(self.pca_head.weight, std=0.01)
        nn.init.zeros_(self.pca_head.bias)

    def _dyn_conv(self, h, batch):
        dyn_ei = dynamic_knn_edges(h.detach(), batch, self.k_dyn)
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
        h = F.leaky_relu(self.bn1(self.gat1(x, ei, ea)), 0.2)
        h = F.leaky_relu(self.bn2(self._dyn_conv(h, batch)), 0.2)
        h = F.leaky_relu(self.bn3(self.gat2(h, ei)), 0.2)
        g = self.pool(h, batch)
        g = self.shape_embed(g)
        return self.pca_head(g)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Phase 1 → Phase 2 encoder weight transfer ─────────────────────────────

def transfer_encoder_weights(p1_model, p2_model):
    """Copy all matching encoder layers from Phase 1 into Phase 2."""
    state1 = p1_model.state_dict()
    state2 = p2_model.state_dict()
    loaded, skipped = [], []
    for k, v in state1.items():
        if k in state2 and state2[k].shape == v.shape:
            state2[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    p2_model.load_state_dict(state2)
    print(f'✅ Phase 1 → Phase 2 weight transfer')
    print(f'   Transferred : {len(loaded)} tensors')
    print(f'   Skipped     : {skipped}')


# ── GPU-side SSM reconstruction helpers ───────────────────────────────────

def build_ssm_gpu_tensors(mean_verts, scaled_pcs, faces, num_pca_modes, device):
    """
    Pre-compute and upload SSM reconstruction tensors to `device`.
    Returns (MEAN_T, SCALED_T, LAP_EI, LAP_DEG_INV, N_VERTS).
    """
    mean_t   = torch.from_numpy(mean_verts.flatten()).float().to(device)
    scaled_t = torch.from_numpy(scaled_pcs[:, :num_pca_modes].T).float().to(device)

    # Build Laplacian edge index from face list
    fa = np.array(faces)
    rows = np.concatenate([fa[:, 0], fa[:, 1], fa[:, 2],
                           fa[:, 1], fa[:, 2], fa[:, 0]])
    cols = np.concatenate([fa[:, 1], fa[:, 2], fa[:, 0],
                           fa[:, 0], fa[:, 1], fa[:, 2]])
    ei_t   = torch.from_numpy(np.stack([rows, cols], axis=0)).long()
    lap_ei = torch.unique(ei_t, dim=1).to(device)

    lap_row = lap_ei[0]
    lap_col = lap_ei[1]
    n_verts = len(mean_verts)

    deg = torch.zeros(n_verts, device=device)
    deg.scatter_add_(0, lap_row, torch.ones(lap_row.shape[0], device=device))
    lap_deg_inv = (1.0 / deg.clamp(min=1.0)).unsqueeze(-1)

    return mean_t, scaled_t, lap_ei, lap_deg_inv, n_verts


def reconstruct_batch(weights_batch, mean_t, scaled_t, n_verts, device):
    """(B, K) weights → (B, V, 3) vertex positions."""
    w          = weights_batch.to(device)
    verts_flat = mean_t + w @ scaled_t
    return verts_flat.view(w.shape[0], n_verts, 3)


def laplacian_smooth_loss(verts_batch, lap_row, lap_col, lap_deg_inv, n_verts, device):
    """Fully vectorised Laplacian smoothness over a batch (B, V, 3)."""
    B       = verts_batch.shape[0]
    nbr_sum = torch.zeros(B, n_verts, 3, device=device)
    nbr_sum.scatter_add_(
        1,
        lap_row.view(1, -1, 1).expand(B, -1, 3),
        verts_batch[:, lap_col, :],
    )
    avg  = nbr_sum * lap_deg_inv
    diff = verts_batch - avg
    return diff.norm(dim=-1).mean()


def p2_total_loss(pred_w, target_w,
                  mean_t, scaled_t, n_verts,
                  lap_row, lap_col, lap_deg_inv, device,
                  lam_surf=0.5, lam_smooth=0.005):
    l_pca    = mse_fn(pred_w, target_w)
    pred_v   = reconstruct_batch(pred_w,   mean_t, scaled_t, n_verts, device)
    target_v = reconstruct_batch(target_w, mean_t, scaled_t, n_verts, device)
    l_surf   = mse_fn(pred_v, target_v)
    l_smooth = laplacian_smooth_loss(pred_v, lap_row, lap_col, lap_deg_inv, n_verts, device)
    return l_pca + lam_surf * l_surf + lam_smooth * l_smooth, l_pca, l_surf
