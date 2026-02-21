"""
graph_builder.py
================
Converts a 3D LV mesh into a graph suitable for PyTorch Geometric.

Pipeline per shape:
  1. Slice the mesh along the Z-axis at `num_slices` evenly-spaced planes.
  2. For each slice, extract endocardium (inner) and epicardium (outer) contours
     using angular-bin separation.
  3. Resample each contour to a fixed number of points.
  4. Build node feature matrix: [x, y, z, radial_dist, tissue_type].
  5. Build edge list:
       - k-NN within each Z-slice (intra-slice connectivity)
       - k-NN between adjacent Z-slices (inter-slice continuity)
  6. Return a dict ready for np.savez() and PyG Data construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from data.ssm_loader import SSMShape


# ---------------------------------------------------------------------------
# Contour extraction helpers
# ---------------------------------------------------------------------------

def _separate_endo_epi(points_2d: np.ndarray, num_bins: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate endocardium (inner) and epicardium (outer) from a flat 2D slice
    by angular binning: within each angular bin, the closest point to the
    centroid is labelled endocardium and the farthest is epicardium.
    """
    if len(points_2d) < 6:
        return np.empty((0, 2)), np.empty((0, 2))

    centroid = points_2d.mean(axis=0)
    angles = np.arctan2(points_2d[:, 1] - centroid[1], points_2d[:, 0] - centroid[0])
    distances = np.linalg.norm(points_2d - centroid, axis=1)
    bins = np.linspace(-np.pi, np.pi, num_bins + 1)

    endo, epi = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (angles >= lo) & (angles < hi)
        if not mask.any():
            continue
        bin_pts = points_2d[mask]
        bin_dist = distances[mask]
        if len(bin_dist) > 1:
            endo.append(bin_pts[bin_dist.argmin()])
            epi.append(bin_pts[bin_dist.argmax()])
        else:
            epi.append(bin_pts[0])

    def _order(pts: list) -> np.ndarray:
        if not pts:
            return np.empty((0, 2))
        arr = np.array(pts)
        a = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
        return arr[np.argsort(a)]

    return _order(endo), _order(epi)


def _resample_contour(contour: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a 2D contour to exactly `n_points` evenly spaced points."""
    if len(contour) < 2:
        return np.zeros((n_points, 2), dtype=np.float32)
    # Compute cumulative arc length
    diffs = np.diff(contour, axis=0, prepend=contour[-1:])
    arc = np.cumsum(np.linalg.norm(diffs, axis=1))
    arc /= arc[-1]
    targets = np.linspace(0, 1, n_points)
    xs = np.interp(targets, arc, contour[:, 0])
    ys = np.interp(targets, arc, contour[:, 1])
    return np.column_stack([xs, ys]).astype(np.float32)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

@dataclass
class SliceData:
    z_position: float
    endo_contour: np.ndarray  # (P, 2)
    epi_contour: np.ndarray   # (P, 2)
    centroid: np.ndarray      # (2,)


def extract_slices(
    vertices: np.ndarray,
    num_slices: int = 20,
    points_per_contour: int = 80,
    epsilon: float = 2.0,
) -> list[SliceData]:
    """
    Slice a 3D mesh along the Z-axis and extract endo/epi contours per slice.

    Parameters
    ----------
    vertices : (V, 3) array
    num_slices : int
    points_per_contour : int
        Number of resampled points per contour (endo and epi each).
    epsilon : float
        Half-thickness of each slice in mm.

    Returns
    -------
    List of SliceData (may be shorter than num_slices if some slices are empty).
    """
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    z_positions = np.linspace(z_min, z_max, num_slices)
    slices = []

    for z in z_positions:
        mask = np.abs(vertices[:, 2] - z) < epsilon
        pts_2d = vertices[mask, :2]

        if len(pts_2d) < 10:
            continue

        endo, epi = _separate_endo_epi(pts_2d)

        if len(endo) < 4 or len(epi) < 4:
            continue

        endo_r = _resample_contour(endo, points_per_contour)
        epi_r = _resample_contour(epi, points_per_contour)
        centroid = pts_2d.mean(axis=0)

        slices.append(SliceData(
            z_position=float(z),
            endo_contour=endo_r,
            epi_contour=epi_r,
            centroid=centroid.astype(np.float32),
        ))

    return slices


def build_graph(
    slices: list[SliceData],
    knn_intra: int = 8,
    knn_inter: int = 3,
) -> dict:
    """
    Build a graph from extracted Z-axis slices.

    Node features (N × 5): [x, y, z, radial_distance, tissue_type]
      tissue_type: 0 = endocardium, 1 = epicardium

    Edges:
      - knn_intra nearest neighbours within each slice
      - knn_inter nearest neighbours between adjacent slices

    Returns
    -------
    dict with keys:
      nodes        (N, 5) float32
      edges        (E, 2) int32
      node_types   (N,)   int8
      slice_ids    (N,)   int32
      num_nodes    int
      num_edges    int
      num_slices   int
    """
    all_nodes: list[np.ndarray] = []
    node_types: list[np.ndarray] = []
    slice_ids: list[np.ndarray] = []

    for sid, sl in enumerate(slices):
        z = sl.z_position
        c = sl.centroid

        for contour, ttype in [(sl.endo_contour, 0), (sl.epi_contour, 1)]:
            n = len(contour)
            z_col = np.full((n, 1), z, dtype=np.float32)
            r_col = np.linalg.norm(contour - c, axis=1, keepdims=True).astype(np.float32)
            t_col = np.full((n, 1), ttype, dtype=np.float32)
            feats = np.hstack([contour, z_col, r_col, t_col])  # (n, 5)

            all_nodes.append(feats)
            node_types.append(np.full(n, ttype, dtype=np.int8))
            slice_ids.append(np.full(n, sid, dtype=np.int32))

    if not all_nodes:
        return {}

    nodes = np.vstack(all_nodes).astype(np.float32)  # (N, 5)
    node_types_arr = np.concatenate(node_types)
    slice_ids_arr = np.concatenate(slice_ids)

    edges_list: list[np.ndarray] = []

    # --- Intra-slice k-NN ---
    for sid in range(len(slices)):
        idx = np.where(slice_ids_arr == sid)[0]
        if len(idx) < 2:
            continue
        k = min(knn_intra + 1, len(idx))
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(nodes[idx, :3])
        _, neighbors = nbrs.kneighbors(nodes[idx, :3])
        for i, row in enumerate(neighbors):
            for j in row[1:]:  # skip self
                u, v = idx[i], idx[j]
                edges_list.append([u, v])
                edges_list.append([v, u])

    # --- Inter-slice k-NN (adjacent slices only) ---
    for sid in range(len(slices) - 1):
        idx_a = np.where(slice_ids_arr == sid)[0]
        idx_b = np.where(slice_ids_arr == sid + 1)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            continue
        k = min(knn_inter, len(idx_b))
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(nodes[idx_b, :3])
        _, neighbors = nbrs.kneighbors(nodes[idx_a, :3])
        for i, row in enumerate(neighbors):
            for j in row:
                u, v = idx_a[i], idx_b[j]
                edges_list.append([u, v])
                edges_list.append([v, u])

    if edges_list:
        edges = np.unique(np.array(edges_list, dtype=np.int32), axis=0)
    else:
        edges = np.empty((0, 2), dtype=np.int32)

    return {
        "nodes": nodes,
        "edges": edges,
        "node_types": node_types_arr,
        "slice_ids": slice_ids_arr,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_slices": len(slices),
    }


def slices_to_json(slices: list[SliceData], sample_id: int) -> dict:
    """Serialise slice data to a JSON-compatible dict."""
    return {
        "sample_id": sample_id,
        "num_slices": len(slices),
        "slices": [
            {
                "z_position": sl.z_position,
                "endo_contour": sl.endo_contour.tolist(),
                "epi_contour": sl.epi_contour.tolist(),
                "slice_centroid": sl.centroid.tolist(),
                "num_endo_points": len(sl.endo_contour),
                "num_epi_points": len(sl.epi_contour),
            }
            for sl in slices
        ],
    }
