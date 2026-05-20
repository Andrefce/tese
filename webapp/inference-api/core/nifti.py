"""Minimal NIfTI loading utilities for the inference API."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class NiftiLoadError(RuntimeError):
    """Raised when medical image loading cannot continue."""


def _nibabel():
    try:
        import nibabel as nib
    except Exception as exc:
        raise NiftiLoadError(
            "nibabel is required to read NIfTI files."
        ) from exc
    return nib


def resolve_nifti_path(path: Path) -> Path:
    """Resolve ACDC-style folders that contain a single .nii or .nii.gz file."""
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        hits = sorted(path.glob("*.nii.gz")) + sorted(path.glob("*.nii"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No NIfTI file found at {path}")


def load_nifti(path: Path) -> dict[str, Any]:
    nib = _nibabel()
    real = resolve_nifti_path(path)
    img = nib.load(str(real))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim not in (3, 4):
        raise NiftiLoadError(f"Expected a 3D or 4D NIfTI volume, got shape {data.shape}")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return {
        "path": str(real),
        "data": data,
        "affine": np.asarray(img.affine, dtype=np.float32),
        "zooms": zooms,
        "shape": tuple(int(v) for v in data.shape),
    }


def load_nifti_from_bytes(file_bytes: bytes) -> dict[str, Any]:
    """Load a NIfTI volume from in-memory bytes."""
    import io
    nib = _nibabel()
    fh = nib.FileHolder(fileobj=io.BytesIO(file_bytes))
    img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim not in (3, 4):
        raise NiftiLoadError(f"Expected a 3D or 4D NIfTI volume, got shape {data.shape}")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return {
        "data": data,
        "affine": np.asarray(img.affine, dtype=np.float32),
        "zooms": zooms,
        "shape": tuple(int(v) for v in data.shape),
    }


def select_frame(data: np.ndarray, frame: int = 0) -> np.ndarray:
    if data.ndim == 3:
        return data
    frame = int(np.clip(frame, 0, data.shape[3] - 1))
    return data[..., frame]
