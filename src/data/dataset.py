"""
dataset.py
==========
PyTorch Geometric Dataset that lazily loads pre-built graph NPZ files
from disk. Each graph's target is the 3D node positions (first 3 features).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class CardiacGraphDataset(Dataset):
    """
    Dataset of cardiac LV graphs stored as .npz files.

    Each file is expected to contain:
      - ``nodes``   : (N, 5) float32 — node feature matrix
      - ``edges``   : (E, 2) int32   — edge list

    The target ``y`` for each graph is ``nodes[:, :3]`` (the 3D positions),
    which the model learns to reconstruct.

    Parameters
    ----------
    graph_dir : str or Path
        Directory containing ``graph_000.npz``, ``graph_001.npz``, …
    indices : list[int]
        Which graph indices to include in this split.
    cache_size : int
        Maximum number of graphs to keep in memory. Set to 0 to disable.
    """

    def __init__(
        self,
        graph_dir: str | Path,
        indices: list[int],
        cache_size: int = 256,
    ) -> None:
        self.graph_dir = Path(graph_dir)
        self.indices = indices
        self.cache_size = cache_size
        self._cache: dict[int, Data] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def _load(self, graph_idx: int) -> Data:
        if graph_idx in self._cache:
            return self._cache[graph_idx]

        path = self.graph_dir / f"graph_{graph_idx:03d}.npz"
        raw = np.load(path)

        x = torch.from_numpy(raw["nodes"]).float()
        edge_index = torch.from_numpy(raw["edges"].T).long()  # (2, E)

        data = Data(x=x, edge_index=edge_index)
        data.y = x[:, :3].clone()  # Target: 3D positions

        if len(self._cache) < self.cache_size:
            self._cache[graph_idx] = data
        return data

    def __getitem__(self, idx: int) -> Data:
        graph_idx = self.indices[idx]
        graph = self._load(graph_idx)
        # Return a fresh copy so batching does not corrupt the cache
        return Data(
            x=graph.x.clone(),
            edge_index=graph.edge_index.clone(),
            y=graph.y.clone(),
        )


def make_dataloaders(
    graph_dir: str | Path,
    num_graphs: int,
    val_split: float = 0.2,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Build train and validation DataLoaders with a deterministic split.

    Returns
    -------
    train_loader, val_loader, (idx_train, idx_val)
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from sklearn.model_selection import train_test_split

    indices = list(range(num_graphs))
    idx_train, idx_val = train_test_split(
        indices, test_size=val_split, random_state=seed
    )

    train_ds = CardiacGraphDataset(graph_dir, idx_train)
    val_ds = CardiacGraphDataset(graph_dir, idx_val)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader = PyGDataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = PyGDataLoader(val_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, (idx_train, idx_val)
