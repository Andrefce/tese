"""
data/datasets.py
PyTorch Dataset wrappers for Phase 1 (denoising) and Phase 2 (SSM completion).
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class DenoisingCardiacDataset(Dataset):
    """
    Phase 1 dataset.
    Returns clean graph tensors; noise is added on-GPU inside the training loop.
    """

    def __init__(self, graph_dir, indices, cache_size=512):
        self.graph_dir  = graph_dir
        self.indices    = indices
        self._cache     = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.indices)

    def _load(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        raw = np.load(f'{self.graph_dir}/graph_{idx:03d}.npz')
        d   = Data(
            x          = torch.from_numpy(raw['nodes']).float(),
            edge_index = torch.from_numpy(raw['edges'].T).long(),
            edge_attr  = torch.from_numpy(raw['edge_feats']).float(),
        )
        if len(self._cache) < self.cache_size:
            self._cache[idx] = d
        return d

    def __getitem__(self, i):
        d = self._load(self.indices[i])
        return Data(
            x          = d.x.clone(),
            edge_index = d.edge_index.clone(),
            edge_attr  = d.edge_attr.clone(),
            y          = d.x[:, :3].clone(),
        )


class SSMCompletionDataset(Dataset):
    """
    Phase 2 dataset.
    Returns full graph + pca_weights target.
    obs_mask sampling (which slices are 'observed') is done on CPU.
    """

    def __init__(self, graph_dir, indices,
                 min_visible=1, max_visible_frac=0.35, cache_size=512):
        self.graph_dir        = graph_dir
        self.indices          = indices
        self.min_visible      = min_visible
        self.max_visible_frac = max_visible_frac
        self._cache           = {}
        self.cache_size       = cache_size

    def __len__(self):
        return len(self.indices)

    def _load(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        raw = np.load(f'{self.graph_dir}/graph_{idx:03d}.npz')
        d   = Data(
            x           = torch.from_numpy(raw['nodes']).float(),
            edge_index  = torch.from_numpy(raw['edges'].T).long(),
            edge_attr   = torch.from_numpy(raw['edge_feats']).float(),
            slice_ids   = torch.from_numpy(raw['slice_ids'].astype(np.int64)),
            pca_weights = torch.from_numpy(raw['pca_weights']).float(),
        )
        if len(self._cache) < self.cache_size:
            self._cache[idx] = d
        return d

    def __getitem__(self, i):
        d          = self._load(self.indices[i])
        num_slices = int(d.slice_ids.max().item()) + 1
        max_vis    = max(self.min_visible, int(num_slices * self.max_visible_frac))
        n_vis      = random.randint(self.min_visible, max_vis)
        vis_ids    = torch.tensor(
            random.sample(range(num_slices), n_vis), dtype=torch.long
        )
        obs_mask = torch.isin(d.slice_ids, vis_ids)           # (N,) bool
        x_in     = torch.cat([d.x, obs_mask.float().unsqueeze(1)], dim=1)  # (N, 6)
        return Data(
            x          = x_in,
            edge_index = d.edge_index.clone(),
            edge_attr  = d.edge_attr.clone(),
            y          = d.pca_weights.clone().unsqueeze(0),
            n_visible  = torch.tensor(n_vis, dtype=torch.long),
        )
