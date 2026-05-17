from __future__ import annotations

import numpy as np
import pytest

from app import CASES, _attach_es_phase, _register_case, app
from core.inference import compare_wall_thickness_results


def _synthetic_volume(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    shape = (36, 36, 12)
    spacing = (1.4, 1.4, 8.0)
    yy, xx, zz = np.indices(shape)
    cx, cy = 18.0, 18.0
    z_center = 5.5
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    z_weight = np.clip(1.0 - np.abs(zz - z_center) / 7.0, 0.35, 1.0)
    lv_radius = 5.0 * scale * z_weight
    epi_radius = 8.0 * scale * z_weight
    seg = np.zeros(shape, dtype=np.float32)
    seg[(r <= epi_radius) & (r > lv_radius)] = 2
    seg[r <= lv_radius] = 3
    mri = np.zeros(shape, dtype=np.float32)
    mri += np.exp(-(r / 10.0) ** 2).astype(np.float32) * 120.0
    return mri, seg, spacing


def _nifti_like(path: str, data: np.ndarray, spacing: tuple[float, float, float]) -> dict:
    return {
        "path": path,
        "data": data,
        "affine": np.eye(4, dtype=np.float32),
        "zooms": spacing,
        "shape": tuple(int(v) for v in data.shape),
    }


@pytest.fixture(autouse=True)
def clear_cases():
    CASES.clear()
    yield
    CASES.clear()


def test_paired_inference_requires_es_data():
    mri, seg, spacing = _synthetic_volume()
    case = _register_case(
        "single-phase",
        _nifti_like("ed-mri", mri, spacing),
        _nifti_like("ed-seg", seg, spacing),
    )

    client = app.test_client()
    response = client.post(f"/api/case/{case['id']}/infer-paired", json={"useSdfModel": False})

    assert response.status_code == 400
    assert "both ED and ES" in response.get_json()["error"]


def test_paired_inference_returns_difference_payload():
    ed_mri, ed_seg, spacing = _synthetic_volume(scale=1.0)
    es_mri, es_seg, _ = _synthetic_volume(scale=0.82)
    case = _register_case(
        "paired",
        _nifti_like("ed-mri", ed_mri, spacing),
        _nifti_like("ed-seg", ed_seg, spacing),
    )
    _attach_es_phase(
        case,
        _nifti_like("es-mri", es_mri, spacing),
        _nifti_like("es-seg", es_seg, spacing),
    )

    client = app.test_client()
    response = client.post(f"/api/case/{case['id']}/infer-paired", json={"useSdfModel": False})

    assert response.status_code == 200
    payload = response.get_json()
    assert set(payload) >= {"ed", "es", "difference"}
    assert payload["difference"]["kind"] == "difference"
    assert len(payload["difference"]["aha17Delta"]) == 17
    assert payload["difference"]["meshes"]["endo"]["vertices"]
    assert "marching_cubes" not in payload["difference"]["meshMethod"]


def test_compare_wall_thickness_uses_es_minus_ed_convention():
    vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0]
    faces = [0, 1, 2]
    ed_result = {
        "metrics": {"meanWallThicknessMm": 5.0},
        "aha17": [{"id": i, "name": f"S{i}", "meanMm": 5.0} for i in range(1, 18)],
        "meshes": {"endo": {"vertices": vertices, "faces": faces, "values": [5.0, 5.0, 5.0]}},
    }
    es_result = {
        "metrics": {"meanWallThicknessMm": 7.0},
        "aha17": [{"id": i, "name": f"S{i}", "meanMm": 7.0} for i in range(1, 18)],
        "meshes": {"endo": {"vertices": vertices, "faces": faces, "values": [7.0, 7.0, 7.0]}},
    }

    difference = compare_wall_thickness_results(ed_result, es_result)

    assert difference["metrics"]["meanDeltaWallThicknessMm"] == 2.0
    assert difference["metrics"]["relativeThickeningPct"] == 40.0
    assert difference["aha17Delta"][0]["deltaMm"] == 2.0
    assert difference["meshes"]["endo"]["values"] == [2.0, 2.0, 2.0]