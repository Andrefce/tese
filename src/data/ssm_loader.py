"""
ssm_loader.py
=============
Loads the UK Digital Heart Statistical Shape Model (SSM) and generates
synthetic LV shapes by sampling PCA weight vectors.

Expected SSM files (from https://github.com/UK-Digital-Heart-Project/Statistical-Shape-Model):
  - LV_ED_mean.vtk              : Mean shape as VTK PolyData
  - LV_ED_pc_100_modes.csv.gz   : Principal component matrix  (V × M)
  - LV_ED_var_100_modes.csv.gz  : Explained variance per mode (M,)
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import vtk


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SSMShape:
    """A single shape generated from the SSM."""
    vertices: np.ndarray    # (V, 3) — 3D vertex coordinates in mm
    faces: np.ndarray       # (F, 3) — triangle face indices
    pca_weights: np.ndarray # (M,)   — PCA weight vector used for sampling


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_vtk_polydata(path: str | Path) -> vtk.vtkPolyData:
    """Read a legacy VTK PolyData file (.vtk)."""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"VTK file not found: {path}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def _polydata_to_numpy(mesh: vtk.vtkPolyData) -> tuple[np.ndarray, np.ndarray]:
    """Extract vertices and triangle faces from vtkPolyData."""
    pts = mesh.GetPoints().GetData()
    vertices = np.array(
        [pts.GetTuple3(i) for i in range(pts.GetNumberOfTuples())],
        dtype=np.float32,
    )

    polys = mesh.GetPolys().GetData()
    n_ids = polys.GetNumberOfValues()
    faces = []
    i = 0
    while i < n_ids:
        n = polys.GetValue(i)
        if n == 3:
            faces.append([polys.GetValue(i + 1),
                          polys.GetValue(i + 2),
                          polys.GetValue(i + 3)])
        i += n + 1
    return vertices, np.array(faces, dtype=np.int64)


def _load_csv_gz(path: str | Path) -> np.ndarray:
    """Load a gzip-compressed CSV into a NumPy array."""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV.gz file not found: {path}")
    with gzip.open(path, "rt") as f:
        return np.loadtxt(f, delimiter=",")


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class SSMLoader:
    """
    Loads the UK Digital Heart SSM and exposes shape sampling.

    Parameters
    ----------
    ssm_dir : str or Path
        Root directory of the cloned SSM repository.
    mean_mesh_file : str
        Filename of the mean shape VTK file.
    pc_file : str
        Filename of the principal components CSV.gz (V·3 × M).
    var_file : str
        Filename of the variance CSV.gz (M,).
    """

    def __init__(
        self,
        ssm_dir: str | Path,
        mean_mesh_file: str = "LV_ED_mean.vtk",
        pc_file: str = "LV_ED_pc_100_modes.csv.gz",
        var_file: str = "LV_ED_var_100_modes.csv.gz",
    ) -> None:
        ssm_dir = Path(ssm_dir)
        self.mean_vertices, self.faces = _polydata_to_numpy(
            _read_vtk_polydata(ssm_dir / mean_mesh_file)
        )
        self.pcs = _load_csv_gz(ssm_dir / pc_file)       # (V*3, M)
        self.variances = _load_csv_gz(ssm_dir / var_file) # (M,)
        self.num_vertices = self.mean_vertices.shape[0]

        print(
            f"[SSMLoader] Loaded SSM: {self.num_vertices} vertices, "
            f"{len(self.faces)} faces, {self.pcs.shape[1]} PCA modes."
        )

    # ------------------------------------------------------------------
    def sample_shape(
        self,
        num_modes: int = 10,
        sigma_clip: float = 3.0,
        rng: Optional[np.random.Generator] = None,
    ) -> SSMShape:
        """
        Sample a single plausible LV shape.

        Weights are drawn from N(0, 1) and clipped to ±sigma_clip * std,
        ensuring biomechanical plausibility.

        Parameters
        ----------
        num_modes : int
            How many PCA modes to activate (first `num_modes` modes).
        sigma_clip : float
            Maximum ±std deviation allowed for any weight.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        SSMShape
        """
        if rng is None:
            rng = np.random.default_rng()

        weights = np.zeros(self.pcs.shape[1])
        raw = rng.standard_normal(num_modes)
        weights[:num_modes] = np.clip(raw, -sigma_clip, sigma_clip)

        displacement = self.pcs.dot(weights * np.sqrt(self.variances))
        new_vertices = (self.mean_vertices.flatten() + displacement).reshape(-1, 3)

        return SSMShape(
            vertices=new_vertices.astype(np.float32),
            faces=self.faces.copy(),
            pca_weights=weights,
        )

    def sample_batch(
        self,
        n: int,
        num_modes: int = 10,
        sigma_clip: float = 3.0,
        seed: int = 42,
    ) -> list[SSMShape]:
        """Sample `n` shapes with a seeded RNG."""
        rng = np.random.default_rng(seed)
        return [self.sample_shape(num_modes, sigma_clip, rng) for _ in range(n)]
