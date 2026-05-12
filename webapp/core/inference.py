from __future__ import annotations

import math
from typing import Any

import numpy as np


LBL_BG, LBL_RV, LBL_MYO, LBL_LV = 0, 1, 2, 3
MYOCARDIAL_DENSITY_G_PER_ML = 1.05


def _optional_measure():
    try:
        from skimage import measure

        return measure
    except Exception:  # pragma: no cover - optional dependency
        return None


def _optional_ndimage():
    try:
        from scipy import ndimage

        return ndimage
    except Exception:  # pragma: no cover - optional dependency
        return None


def _optional_ckdtree():
    try:
        from scipy.spatial import cKDTree

        return cKDTree
    except Exception:  # pragma: no cover - optional dependency
        return None


def _volume_ml(mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
    return float(mask.sum() * np.prod(spacing) / 1000.0)


def _mesh_area_cm2(vertices: np.ndarray, faces: np.ndarray) -> float | None:
    if vertices.size == 0 or faces.size == 0:
        return None
    tri = vertices[faces]
    area_mm2 = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    ).sum()
    return float(area_mm2 / 100.0)


def _reduce_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray | None = None,
    max_faces: int = 18000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if len(faces) <= max_faces:
        return vertices, faces.astype(np.int32), values
    step = max(1, int(math.ceil(len(faces) / max_faces)))
    kept_faces = faces[::step]
    used = np.unique(kept_faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    reduced_values = values[used] if values is not None and len(values) == len(vertices) else None
    return vertices[used], remap[kept_faces].astype(np.int32), reduced_values


def _ellipsoid_mesh(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    shell: float = 1.0,
    rings: int = 28,
    sectors: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

    lo = pts.min(axis=0).astype(np.float32)
    hi = pts.max(axis=0).astype(np.float32)
    center = (lo + hi) * 0.5 * np.asarray(spacing, dtype=np.float32)
    radii = np.maximum((hi - lo + 1.0) * 0.5 * np.asarray(spacing), 2.0) * shell

    vertices = []
    for i in range(rings + 1):
        v = i / rings
        theta = math.pi * v
        z_taper = 0.82 + 0.18 * math.sin(theta)
        for j in range(sectors):
            u = j / sectors
            phi = 2.0 * math.pi * u
            x = radii[0] * math.sin(theta) * math.cos(phi) * z_taper
            y = radii[1] * math.sin(theta) * math.sin(phi) * (0.92 + 0.08 * math.cos(theta))
            z = radii[2] * math.cos(theta)
            vertices.append([center[0] + x, center[1] + y, center[2] + z])

    faces = []
    for i in range(rings):
        for j in range(sectors):
            a = i * sectors + j
            b = i * sectors + (j + 1) % sectors
            c = (i + 1) * sectors + j
            d = (i + 1) * sectors + (j + 1) % sectors
            faces.append([a, c, b])
            faces.append([b, c, d])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _surface_mesh(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    shell: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) < 8:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int32),
            "empty",
        )

    measure = _optional_measure()
    if measure is None:
        vertices, faces = _ellipsoid_mesh(mask, spacing, shell=shell)
        return vertices, faces, "ellipsoid"

    padded = np.pad(mask.astype(np.float32), 1, mode="constant")
    try:
        vertices, faces, _, _ = measure.marching_cubes(
            padded, level=0.5, spacing=tuple(spacing)
        )
        vertices = vertices - np.asarray(spacing, dtype=np.float32)
        return vertices.astype(np.float32), faces.astype(np.int32), "marching_cubes"
    except Exception:
        vertices, faces = _ellipsoid_mesh(mask, spacing, shell=shell)
        return vertices, faces, "ellipsoid"


def _nearest_wall_thickness(
    endo_vertices: np.ndarray,
    epi_vertices: np.ndarray,
) -> np.ndarray | None:
    if len(endo_vertices) == 0 or len(epi_vertices) == 0:
        return None
    cKDTree = _optional_ckdtree()
    if cKDTree is None:
        return None
    distances, _ = cKDTree(epi_vertices).query(endo_vertices, k=1, workers=-1)
    distances = np.asarray(distances, dtype=np.float32)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return None
    return distances


def _regional_wall_stats(endo_vertices: np.ndarray, thickness: np.ndarray | None) -> list[dict[str, Any]]:
    names = ["Basal", "Mid", "Apical"]
    if thickness is None or len(thickness) != len(endo_vertices) or len(endo_vertices) == 0:
        return [{"name": name, "meanMm": None, "status": "unavailable"} for name in names]

    z = endo_vertices[:, 2]
    q1, q2 = np.quantile(z, (1 / 3, 2 / 3))
    masks = [z >= q2, (z >= q1) & (z < q2), z < q1]
    out = []
    for name, mask in zip(names, masks):
        vals = thickness[mask]
        vals = vals[np.isfinite(vals)]
        out.append(
            {
                "name": name,
                "meanMm": round(float(vals.mean()), 2) if vals.size else None,
                "p95Mm": round(float(np.percentile(vals, 95)), 2) if vals.size else None,
                "status": _thickness_status(float(vals.mean())) if vals.size else "unavailable",
            }
        )
    return out


# AHA 17-segment names
AHA_17_NAMES = [
    "Basal Anterior", "Basal Anteroseptal", "Basal Inferoseptal",
    "Basal Inferior", "Basal Inferolateral", "Basal Anterolateral",
    "Mid Anterior", "Mid Anteroseptal", "Mid Inferoseptal",
    "Mid Inferior", "Mid Inferolateral", "Mid Anterolateral",
    "Apical Anterior", "Apical Septal", "Apical Inferior", "Apical Lateral",
    "Apex",
]


def _aha_17_segment_stats(
    endo_vertices: np.ndarray,
    thickness: np.ndarray | None,
    seg_volume: np.ndarray | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> list[dict[str, Any]]:
    """Compute AHA 17-segment wall thickness from endo mesh vertices.

    Uses the RV centroid (from seg_volume) to establish the angular reference
    so that the septal direction aligns correctly with the AHA convention.
    """
    empty = [{"id": i + 1, "name": n, "meanMm": None, "status": "unavailable"}
             for i, n in enumerate(AHA_17_NAMES)]
    if thickness is None or len(thickness) != len(endo_vertices) or len(endo_vertices) == 0:
        return empty

    z = endo_vertices[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    zrange = zmax - zmin
    if zrange < 1e-6:
        return empty

    # LV centroid in XY plane
    cx = float(endo_vertices[:, 0].mean())
    cy = float(endo_vertices[:, 1].mean())

    # Determine the anterior reference angle using the RV centroid.
    # In AHA convention the septum (LV→RV direction) is at ~90° from anterior.
    anterior_angle = 0.0  # fallback if no RV available
    if seg_volume is not None and spacing is not None:
        labels = np.rint(seg_volume).astype(np.int16)
        rv_mask = labels == LBL_RV
        if rv_mask.any():
            rv_pts = np.argwhere(rv_mask).astype(np.float32) * np.asarray(spacing, dtype=np.float32)
            rv_cx = float(rv_pts[:, 0].mean())
            rv_cy = float(rv_pts[:, 1].mean())
            septal_angle = math.atan2(rv_cy - cy, rv_cx - cx)
            # Anterior is 90° counterclockwise from the septal (LV→RV) direction
            anterior_angle = septal_angle - math.pi / 2.0

    # Compute angular position around LV centroid (in XY plane)
    raw_angles = np.arctan2(endo_vertices[:, 1] - cy, endo_vertices[:, 0] - cx)  # -pi to pi
    # Normalise relative to anterior direction, then to 0-360 clockwise
    angles_deg = np.degrees(raw_angles - anterior_angle) % 360.0

    # Normalize z: 0 = base, 1 = apex
    znorm = (z - zmin) / zrange

    # Vectorised segment assignment
    segment_ids = np.full(len(endo_vertices), 16, dtype=np.int32)  # default apex

    basal = znorm < (1.0 / 3.0)
    mid = (znorm >= (1.0 / 3.0)) & (znorm < (2.0 / 3.0))
    apical = (znorm >= (2.0 / 3.0)) & (znorm < 0.85)

    # Basal & Mid: 6 segments of 60° each
    seg6 = (angles_deg / 60.0).astype(np.int32) % 6
    segment_ids[basal] = seg6[basal]           # 0-5
    segment_ids[mid] = 6 + seg6[mid]           # 6-11

    # Apical: 4 segments of 90° each
    seg4 = (angles_deg / 90.0).astype(np.int32) % 4
    segment_ids[apical] = 12 + seg4[apical]    # 12-15

    out = []
    for seg_id in range(17):
        mask = segment_ids == seg_id
        vals = thickness[mask]
        vals = vals[np.isfinite(vals)] if vals.size else vals
        mean_v = float(vals.mean()) if vals.size else None
        out.append({
            "id": seg_id + 1,
            "name": AHA_17_NAMES[seg_id],
            "meanMm": round(mean_v, 2) if mean_v is not None else None,
            "p95Mm": round(float(np.percentile(vals, 95)), 2) if vals.size else None,
            "status": _thickness_status(mean_v),
        })
    return out


def _thickness_status(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "unavailable"
    if value < 5.0:
        return "thin"
    if value > 13.0:
        return "thick"
    return "typical"


def _mesh_payload(
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray | None = None,
    max_faces: int = 40000,
) -> dict[str, Any]:
    vertices, faces, values = _reduce_mesh(vertices, faces, values=values, max_faces=max_faces)
    payload = {
        "vertices": np.round(vertices.astype(np.float32), 3).reshape(-1).tolist(),
        "faces": faces.astype(np.int32).reshape(-1).tolist(),
    }
    if values is not None and len(values) == len(vertices):
        payload["values"] = np.round(values.astype(np.float32), 3).reshape(-1).tolist()
    return payload


def _frame_volumes(seg_data: np.ndarray, spacing: tuple[float, float, float]) -> list[float]:
    if seg_data.ndim != 4:
        return []
    vols = []
    for frame in range(seg_data.shape[3]):
        frame_seg = np.rint(seg_data[..., frame]).astype(np.int16)
        vols.append(_volume_ml(frame_seg == LBL_LV, spacing))
    return vols


def analyse_segmentation(
    seg_volume: np.ndarray,
    spacing: tuple[float, float, float],
    source: str,
    full_segmentation: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.rint(seg_volume).astype(np.int16)
    lv_mask = labels == LBL_LV
    myo_mask = labels == LBL_MYO
    rv_mask = labels == LBL_RV

    ndimage = _optional_ndimage()
    if not myo_mask.any() and lv_mask.any() and ndimage is not None:
        grow_xy = max(1, int(round(7.0 / max(spacing[0], spacing[1], 1e-3))))
        structure = np.zeros((grow_xy * 2 + 1, grow_xy * 2 + 1, 3), dtype=bool)
        yy, xx, zz = np.indices(structure.shape)
        rr = (xx - grow_xy) ** 2 + (yy - grow_xy) ** 2
        structure[(rr <= grow_xy**2) & (np.abs(zz - 1) <= 1)] = True
        epi_guess = ndimage.binary_dilation(lv_mask, structure=structure, iterations=1)
        myo_mask = epi_guess & ~lv_mask

    epi_mask = lv_mask | myo_mask
    if int(lv_mask.sum()) < 8:
        raise ValueError("No left-ventricle label found. ACDC LV should use label 3.")

    endo_vertices, endo_faces, mesh_method = _surface_mesh(lv_mask, spacing)
    epi_vertices, epi_faces, epi_method = _surface_mesh(epi_mask, spacing, shell=1.18)
    wall_values = _nearest_wall_thickness(endo_vertices, epi_vertices)
    wall_for_mesh = None
    if wall_values is not None and len(wall_values) == len(endo_vertices):
        wall_for_mesh = wall_values

    lv_volume_ml = _volume_ml(lv_mask, spacing)
    myo_volume_ml = _volume_ml(myo_mask, spacing)
    epi_volume_ml = _volume_ml(epi_mask, spacing)
    rv_volume_ml = _volume_ml(rv_mask, spacing) if rv_mask.any() else None
    wall_mean = float(np.mean(wall_values)) if wall_values is not None and wall_values.size else None
    wall_p95 = float(np.percentile(wall_values, 95)) if wall_values is not None and wall_values.size else None
    endo_area = _mesh_area_cm2(endo_vertices, endo_faces)
    epi_area = _mesh_area_cm2(epi_vertices, epi_faces)

    ef = None
    edv = None
    esv = None
    stroke = None
    frame_vols = _frame_volumes(full_segmentation, spacing) if full_segmentation is not None else []
    positive_vols = [v for v in frame_vols if v > 0]
    if len(positive_vols) >= 2:
        edv = max(positive_vols)
        esv = min(positive_vols)
        stroke = edv - esv
        ef = (stroke / edv * 100.0) if edv > 0 else None

    metrics = {
        "lvVolumeMl": round(lv_volume_ml, 2),
        "myocardiumVolumeMl": round(myo_volume_ml, 2) if myo_volume_ml > 0 else None,
        "epicardialVolumeMl": round(epi_volume_ml, 2),
        "rvVolumeMl": round(rv_volume_ml, 2) if rv_volume_ml is not None else None,
        "myocardialMassG": round(myo_volume_ml * MYOCARDIAL_DENSITY_G_PER_ML, 2)
        if myo_volume_ml > 0
        else None,
        "meanWallThicknessMm": round(wall_mean, 2) if wall_mean is not None else None,
        "p95WallThicknessMm": round(wall_p95, 2) if wall_p95 is not None else None,
        "endoSurfaceAreaCm2": round(endo_area, 2) if endo_area is not None else None,
        "epiSurfaceAreaCm2": round(epi_area, 2) if epi_area is not None else None,
        "edvMl": round(edv, 2) if edv is not None else None,
        "esvMl": round(esv, 2) if esv is not None else None,
        "strokeVolumeMl": round(stroke, 2) if stroke is not None else None,
        "ejectionFractionPct": round(ef, 1) if ef is not None else None,
        "spacingMm": [round(float(v), 3) for v in spacing],
        "annotatedSlices": int(np.count_nonzero(epi_mask.any(axis=(0, 1)))),
        "lvVoxels": int(lv_mask.sum()),
    }

    return {
        "source": source,
        "modelStatus": "ACDC segmentation inference" if source != "Drawn LV mask" else "Drawn-mask geometric inference",
        "meshMethod": mesh_method if mesh_method == epi_method else f"{mesh_method}/{epi_method}",
        "metrics": metrics,
        "regionalThickness": _regional_wall_stats(endo_vertices, wall_for_mesh),
        "aha17": _aha_17_segment_stats(endo_vertices, wall_for_mesh, seg_volume=labels, spacing=spacing),
        "meshes": {
            "endo": _mesh_payload(endo_vertices, endo_faces, wall_for_mesh),
            "epi": _mesh_payload(epi_vertices, epi_faces, None, max_faces=30000),
        },
    }


def _scale_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        return np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros_like(mask, dtype=bool)
    cx = float(xs.mean())
    cy = float(ys.mean())
    yy, xx = np.indices(mask.shape)
    src_x = np.rint(cx + (xx - cx) / scale).astype(int)
    src_y = np.rint(cy + (yy - cy) / scale).astype(int)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(mask, dtype=bool)
    out[valid] = mask[src_y[valid], src_x[valid]] > 0
    return out


def segmentation_from_drawings(
    shape: tuple[int, int, int],
    drawn_masks: dict[int, np.ndarray],
    spacing: tuple[float, float, float],
) -> np.ndarray:
    seg = np.zeros(shape, dtype=np.uint8)
    if not drawn_masks:
        return seg

    z_radius = max(2, min(5, int(round(18.0 / max(spacing[2], 1e-3)))))
    for z, display_mask in drawn_masks.items():
        if display_mask is None or not np.any(display_mask):
            continue
        base = np.asarray(display_mask > 0, dtype=bool)
        for dz in range(-z_radius, z_radius + 1):
            zz = int(z) + dz
            if zz < 0 or zz >= shape[2]:
                continue
            t = abs(dz) / (z_radius + 1)
            scale = max(0.28, math.sqrt(max(0.0, 1.0 - t**1.7)))
            tapered = _scale_mask(base, scale)
            if tapered.shape != (shape[1], shape[0]):
                continue
            seg[:, :, zz] = np.maximum(seg[:, :, zz], tapered.T.astype(np.uint8) * LBL_LV)
    return seg


def make_demo_case() -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    rng = np.random.default_rng(7)
    shape = (192, 192, 14)
    spacing = (1.35, 1.35, 8.0)
    yy, xx = np.indices(shape[:2])
    mri = np.full(shape, 18.0, dtype=np.float32)
    seg = np.zeros(shape, dtype=np.uint8)

    for z in range(shape[2]):
        t = (z - (shape[2] - 1) / 2.0) / ((shape[2] - 1) / 2.0)
        taper = max(0.18, 1.0 - 0.62 * abs(t) ** 1.7)
        cx = 94 + 9 * t
        cy = 98 - 5 * math.sin(z * 0.45)
        rx = 23 * taper
        ry = 30 * taper * (1.0 - 0.08 * t)
        theta = -0.22 + 0.18 * t
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x = (xx - cx) * cos_t + (yy - cy) * sin_t
        y = -(xx - cx) * sin_t + (yy - cy) * cos_t
        lv = (x / rx) ** 2 + (y / ry) ** 2 <= 1.0
        epi = (x / (rx + 11)) ** 2 + (y / (ry + 12)) ** 2 <= 1.0
        rv = ((xx - (cx - 42)) / (rx * 0.95)) ** 2 + ((yy - (cy + 7)) / (ry * 0.82)) ** 2 <= 1.0
        seg[:, :, z][rv] = LBL_RV
        seg[:, :, z][epi & ~lv] = LBL_MYO
        seg[:, :, z][lv] = LBL_LV

        mri[:, :, z] += rng.normal(0, 4.0, size=shape[:2])
        mri[:, :, z][epi & ~lv] = 132 + rng.normal(0, 8.0, size=int((epi & ~lv).sum()))
        mri[:, :, z][lv] = 86 + rng.normal(0, 7.0, size=int(lv.sum()))
        mri[:, :, z][rv] = 78 + rng.normal(0, 9.0, size=int(rv.sum()))

    return mri, seg, spacing

