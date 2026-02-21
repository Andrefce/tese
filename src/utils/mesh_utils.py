"""
mesh_utils.py
=============
VTK and trimesh helper functions shared across the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def save_mesh(vertices: np.ndarray, faces: np.ndarray, path: str | Path) -> None:
    """Save a triangular mesh as STL or OBJ (format inferred from extension)."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(str(path))


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a mesh and return (vertices, faces)."""
    mesh = trimesh.load(str(path), process=False)
    return np.array(mesh.vertices, dtype=np.float32), np.array(mesh.faces, dtype=np.int64)


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute signed mesh volume in mm³."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return float(abs(mesh.volume))


def mesh_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute mesh surface area in mm²."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return float(mesh.area)


def compute_wall_thickness(
    endo_vertices: np.ndarray, epi_vertices: np.ndarray
) -> float:
    """
    Estimate average myocardial wall thickness as mean distance from
    each endocardial vertex to the nearest epicardial vertex.
    """
    from sklearn.neighbors import NearestNeighbors

    if endo_vertices is None or epi_vertices is None:
        return float("nan")
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(epi_vertices)
    dists, _ = nbrs.kneighbors(endo_vertices)
    return float(dists.mean())
