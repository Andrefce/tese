from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np


class NiftiLoadError(RuntimeError):
    """Raised when medical image loading cannot continue."""


def _nibabel():
    try:
        import nibabel as nib
    except Exception as exc:  # pragma: no cover - depends on local env
        raise NiftiLoadError(
            "nibabel is required to read NIfTI files. Install webapp/requirements.txt."
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


def frame_count(data: np.ndarray) -> int:
    return int(data.shape[3]) if data.ndim == 4 else 1


def select_frame(data: np.ndarray, frame: int = 0) -> np.ndarray:
    if data.ndim == 3:
        return data
    frame = int(np.clip(frame, 0, data.shape[3] - 1))
    return data[..., frame]


def case_metadata(case: dict[str, Any]) -> dict[str, Any]:
    data = case["mri"]["data"]
    z_count = int(data.shape[2])
    return {
        "id": case["id"],
        "name": case["name"],
        "shape": [int(v) for v in data.shape],
        "spacing": [round(float(v), 3) for v in case["mri"]["zooms"]],
        "slices": z_count,
        "frames": frame_count(data),
        "hasSegmentation": case.get("seg") is not None,
        "centerSlice": z_count // 2,
    }


def _robust_window(volume: np.ndarray) -> tuple[float, float]:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _to_uint8(slice_2d: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    lo, hi = window
    arr = np.nan_to_num(slice_2d.astype(np.float32), nan=lo, posinf=hi, neginf=lo)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.round(arr * 255.0).astype(np.uint8)


def _b64_uint8(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def decode_mask_base64(mask_b64: str, width: int, height: int) -> np.ndarray:
    raw = base64.b64decode(mask_b64.encode("ascii"))
    mask = np.frombuffer(raw, dtype=np.uint8)
    expected = int(width) * int(height)
    if mask.size != expected:
        raise ValueError(f"Mask has {mask.size} bytes, expected {expected}")
    return mask.reshape((int(height), int(width)))


def slice_payload(
    case: dict[str, Any],
    z: int,
    frame: int = 0,
) -> dict[str, Any]:
    mri_volume = select_frame(case["mri"]["data"], frame)
    z = int(np.clip(z, 0, mri_volume.shape[2] - 1))
    window = case.setdefault("display_window", _robust_window(mri_volume))

    # Browser image space is rows x columns, so transpose from volume x/y/z.
    image = _to_uint8(mri_volume[:, :, z].T, window)
    height, width = image.shape

    labels = np.zeros((height, width), dtype=np.uint8)
    if case.get("seg") is not None:
        seg_volume = select_frame(case["seg"]["data"], frame)
        if z < seg_volume.shape[2]:
            seg_slice = np.rint(seg_volume[:, :, z]).astype(np.int16).T
            labels[seg_slice == 1] = 1
            labels[seg_slice == 2] = 2
            labels[seg_slice == 3] = 3

    drawn = case.get("drawn_masks", {}).get(z)
    if drawn is None:
        drawn = np.zeros((height, width), dtype=np.uint8)
    else:
        drawn = drawn.astype(np.uint8)

    return {
        "caseId": case["id"],
        "z": z,
        "frame": int(frame),
        "width": int(width),
        "height": int(height),
        "image": _b64_uint8(image),
        "labels": _b64_uint8(labels),
        "drawn": _b64_uint8(drawn),
    }

