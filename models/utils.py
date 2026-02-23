"""
models/utils.py
Shared GNN utility functions used by both Phase 1 and Phase 2.
"""

import torch
import torch.nn as nn


def dynamic_knn_edges(x, batch, k):
    """
    Fully vectorised batched k-NN in latent space.
    x     : (N_total, D)
    batch : (N_total,)  graph index per node
    Returns edge_index (2, E) on the same device as x.
    """
    dev    = x.device
    B      = int(batch.max().item()) + 1
    sizes  = batch.bincount(minlength=B)
    max_n  = int(sizes.max().item())
    D      = x.shape[1]

    padded   = torch.zeros(B, max_n, D, device=dev)
    pad_mask = torch.zeros(B, max_n, dtype=torch.bool, device=dev)
    start    = 0
    for b in range(B):
        n = int(sizes[b].item())
        padded[b, :n]   = x[start:start + n]
        pad_mask[b, :n] = True
        start += n

    dist     = torch.cdist(padded, padded)
    inf_mask = (~pad_mask).unsqueeze(2) | (~pad_mask).unsqueeze(1)
    dist.masked_fill_(inf_mask, float('inf'))
    dist.diagonal(dim1=1, dim2=2).fill_(float('inf'))

    kk        = min(k, max_n - 1)
    knn_local = dist.topk(kk, dim=2, largest=False).indices

    offsets = torch.cat([
        torch.zeros(1, device=dev, dtype=torch.long),
        sizes.cumsum(0)[:-1],
    ])

    edges = []
    start = 0
    for b in range(B):
        n         = int(sizes[b].item())
        src_local = torch.arange(n, device=dev).unsqueeze(1).expand(n, kk)
        dst_local = knn_local[b, :n, :]
        src_g     = src_local.reshape(-1) + offsets[b]
        dst_g     = dst_local.reshape(-1) + offsets[b]
        edges.append(torch.stack([src_g, dst_g], dim=0))
        start += n
    return torch.cat(edges, dim=1)


def laplacian_loss(pos, edge_index):
    """Per-graph Laplacian smoothness loss."""
    row, col = edge_index
    sp  = torch.zeros_like(pos)
    cnt = torch.zeros(pos.shape[0], device=pos.device)
    sp.index_add_(0, row, pos[col])
    cnt.index_add_(0, row, torch.ones(row.size(0), device=pos.device))
    avg = sp / cnt.unsqueeze(-1).clamp(min=1.0)
    return torch.mean((pos - avg).norm(dim=-1))


mse_fn = nn.MSELoss()
