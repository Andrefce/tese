"""
acdc_loader.py
==============
Loads ACDC challenge NIfTI files, extracts LV surfaces via marching cubes,
and converts them into the same graph format used during training.

ACDC Label conventions:
  0 - Background
  1 - Right ventricle cavity
  2 - LV myocardium  (label == 2 → endo boundary)
  3 - LV cavity      (labels 2|3 → epi boundary)
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from skimage import measure

from data.graph_builder import SliceData, build_graph, _resample_contour


# ---------------------------------------------------------------------------
# NIfTI loading
# ---------------------------------------------------------------------------

def load_acdc_nifti(image_path: str | Path, label_path: str | Path):
    """
    Load an ACDC image + ground-truth label pair.

    Returns
    -------
    image_data : (X, Y, Z) array — intensity volume
    label_data : (X, Y, Z) array — integer label volume
    affine     : (4, 4) array   — voxel → world (mm) transform
    voxel_spacing : (3,) array  — voxel size in mm
    """
    img_nib = nib.load(str(image_path))
    lbl_nib = nib.load(str(label_path))

    image_data = img_nib.get_fdata()
    label_data = lbl_nib.get_fdata()
    affine = lbl_nib.affine
    voxel_spacing = np.abs(np.diag(affine)[:3])

    return image_data, label_data, affine, voxel_spacing


# ---------------------------------------------------------------------------
# Surface extraction
# ---------------------------------------------------------------------------

def _extract_surface_mm(mask: np.ndarray, affine: np.ndarray):
    """Run marching cubes on a binary mask and transform to mm coordinates."""
    if not np.any(mask):
        return None, None
    verts, faces, _, _ = measure.marching_cubes(mask.astype(float), level=0.5)
    # Homogeneous transform: voxel → world mm
    verts_h = np.c_[verts, np.ones(len(verts))]
    verts_mm = (verts_h @ affine.T)[:, :3]
    return verts_mm.astype(np.float32), faces.astype(np.int64)


def extract_lv_surfaces(label_data: np.ndarray, affine: np.ndarray):
    """
    Extract endo and epi surfaces from an ACDC label volume.

    Returns
    -------
    v_endo, f_endo, v_epi, f_epi  (vertices, faces; None if absent)
    """
    v_endo, f_endo = _extract_surface_mm(label_data == 2, affine)
    v_epi, f_epi = _extract_surface_mm((label_data == 2) | (label_data == 3), affine)
    return v_endo, f_endo, v_epi, f_epi


# ---------------------------------------------------------------------------
# Slice extraction from real data
# ---------------------------------------------------------------------------

def extract_lv_slices_from_label(
    label_data: np.ndarray,
    affine: np.ndarray,
    num_slices: int = 20,
    points_per_contour: int = 80,
    epsilon_mm: float = 3.0,
) -> list[SliceData]:
    """
    Extract Z-axis slices directly from a label volume and return SliceData
    objects using the same representation as the SSM-generated training data.

    Strategy
    --------
    - For each of `num_slices` Z positions, extract LV label points and
      separate into endo/epi contours using angular binning.
    - Resample contours to `points_per_contour` points.
    - Apply affine to convert from voxel to mm space.
    """
    from data.graph_builder import _separate_endo_epi

    # All LV voxels → mm coordinates
    lv_mask = (label_data == 2) | (label_data == 3)
    vox_coords = np.argwhere(lv_mask).astype(np.float32)  # (N, 3)
    if len(vox_coords) == 0:
        return []

    h = np.c_[vox_coords, np.ones(len(vox_coords))]
    mm_coords = (h @ affine.T)[:, :3]  # (N, 3) world mm

    z_min, z_max = mm_coords[:, 2].min(), mm_coords[:, 2].max()
    z_positions = np.linspace(z_min, z_max, num_slices)

    slices = []
    for z in z_positions:
        mask = np.abs(mm_coords[:, 2] - z) < epsilon_mm
        pts_2d = mm_coords[mask, :2]

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


def acdc_patient_to_graph(
    label_path: str | Path,
    image_path: str | Path | None = None,
    num_slices: int = 20,
    points_per_contour: int = 80,
    knn_intra: int = 8,
    knn_inter: int = 3,
) -> dict:
    """
    Full pipeline: ACDC NIfTI → graph dict ready for model inference.

    Returns
    -------
    graph_dict as returned by ``build_graph``, plus ``slices`` list.
    """
    _, label_data, affine, _ = load_acdc_nifti(
        image_path or label_path,  # image not strictly needed here
        label_path,
    )

    slices = extract_lv_slices_from_label(
        label_data, affine,
        num_slices=num_slices,
        points_per_contour=points_per_contour,
    )

    graph = build_graph(slices, knn_intra=knn_intra, knn_inter=knn_inter)
    graph["slices"] = slices
    return graph
