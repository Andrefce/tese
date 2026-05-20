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


def _remove_planar_z_caps(
    vertices: np.ndarray,
    faces: np.ndarray,
    spacing: tuple[float, float, float] | None = None,
) -> np.ndarray:
    if len(vertices) == 0 or len(faces) == 0:
        return faces.astype(np.int32)

    z = vertices[:, 2]
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        return faces.astype(np.int32)

    z_tol = max(float(spacing[2]) * 0.55 if spacing else 0.75, (zmax - zmin) * 0.015)
    tri = vertices[faces]
    edge_a = tri[:, 1] - tri[:, 0]
    edge_b = tri[:, 2] - tri[:, 0]
    normals = np.cross(edge_a, edge_b)
    normal_len = np.linalg.norm(normals, axis=1)
    horizontal = np.zeros(len(faces), dtype=bool)
    valid = normal_len > 1e-8
    horizontal[valid] = np.abs(normals[valid, 2]) / normal_len[valid] > 0.92
    near_bottom = np.all(np.abs(tri[:, :, 2] - zmin) <= z_tol, axis=1)
    near_top = np.all(np.abs(tri[:, :, 2] - zmax) <= z_tol, axis=1)
    keep = ~(horizontal & (near_bottom | near_top))
    kept_faces = faces[keep]
    return kept_faces.astype(np.int32) if len(kept_faces) else faces.astype(np.int32)


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
        faces = _remove_planar_z_caps(vertices, faces, spacing)
        return vertices.astype(np.float32), faces.astype(np.int32), "marching_cubes_trimmed"
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
    finite = np.isfinite(distances)
    if not finite.any():
        return None
    distances[~finite] = np.nan
    return distances


def _finite_mean(values: np.ndarray | None) -> float | None:
    if values is None:
        return None
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else None


def _finite_percentile(values: np.ndarray | None, percentile: float) -> float | None:
    if values is None:
        return None
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, percentile)) if vals.size else None


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


def _mesh_vertices(mesh: dict[str, Any] | None) -> np.ndarray:
    if not mesh or not mesh.get("vertices"):
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(mesh["vertices"], dtype=np.float32).reshape(-1, 3)


def _mesh_values(mesh: dict[str, Any] | None) -> np.ndarray | None:
    if not mesh or not mesh.get("values"):
        return None
    return np.asarray(mesh["values"], dtype=np.float32)


def _sample_values_nearest(
    source_vertices: np.ndarray,
    source_values: np.ndarray | None,
    target_vertices: np.ndarray,
) -> np.ndarray | None:
    if source_values is None or len(source_vertices) == 0 or len(target_vertices) == 0:
        return None
    if len(source_values) != len(source_vertices):
        return None
    cKDTree = _optional_ckdtree()
    if cKDTree is None:
        return None
    _, idx = cKDTree(source_vertices).query(target_vertices, k=1, workers=-1)
    return source_values[np.asarray(idx, dtype=np.int64)].astype(np.float32)


def _radial_surface_motion_values(
    ed_vertices: np.ndarray,
    es_vertices: np.ndarray,
    target_vertices: np.ndarray,
) -> np.ndarray | None:
    """Return signed ED→ES radial motion sampled on target vertices.

    Positive values mean the ES surface is inward from the ED surface along
    the local ED radial direction (expected systolic contraction). Negative
    values indicate outward motion/expansion.
    """
    if len(ed_vertices) == 0 or len(es_vertices) == 0 or len(target_vertices) == 0:
        return None
    cKDTree = _optional_ckdtree()
    if cKDTree is None:
        return None

    ed_center = np.nanmean(ed_vertices, axis=0)
    if not np.all(np.isfinite(ed_center)):
        return None

    _, es_idx = cKDTree(es_vertices).query(target_vertices, k=1, workers=-1)
    nearest_es = es_vertices[np.asarray(es_idx, dtype=np.int64)]

    radial = target_vertices - ed_center
    radial_norm = np.linalg.norm(radial, axis=1)
    valid = radial_norm > 1e-6
    if not valid.any():
        return None

    directions = np.zeros_like(radial, dtype=np.float32)
    directions[valid] = radial[valid] / radial_norm[valid, None]
    motion = np.einsum("ij,ij->i", target_vertices - nearest_es, directions).astype(np.float32)

    # At rare near-centroid vertices, fall back to unsigned nearest distance.
    if (~valid).any():
        motion[~valid] = np.linalg.norm(target_vertices[~valid] - nearest_es[~valid], axis=1)
    motion[~np.isfinite(motion)] = np.nan
    return motion


def _contour_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _resample_contour_by_angle(points: np.ndarray, sectors: int) -> np.ndarray:
    center = points.mean(axis=0)
    rel = points - center
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    targets = np.linspace(-math.pi, math.pi, int(sectors), endpoint=False)
    order = np.argsort(angles)
    angles = angles[order]
    pts = points[order]
    closed_angles = np.concatenate([angles - 2.0 * math.pi, angles, angles + 2.0 * math.pi])
    closed_pts = np.vstack([pts, pts, pts])
    out = []
    for angle in targets:
        idx = int(np.argmin(np.abs(closed_angles - angle)))
        out.append(closed_pts[idx])
    return np.asarray(out, dtype=np.float32)


def _angle_delta(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _mask_boundary_ring_by_angle(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    sectors: int,
) -> np.ndarray | None:
    pts = np.argwhere(mask)
    if len(pts) < 8:
        return None

    xy = np.column_stack([
        pts[:, 0].astype(np.float32) * float(spacing[0]),
        pts[:, 1].astype(np.float32) * float(spacing[1]),
    ])
    center = xy.mean(axis=0)
    rel = xy - center
    radii = np.linalg.norm(rel, axis=1)
    valid = radii > 1e-6
    if int(valid.sum()) < 8:
        return None

    xy = xy[valid]
    rel = rel[valid]
    radii = radii[valid]
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    targets = np.linspace(-math.pi, math.pi, int(sectors), endpoint=False)
    base_window = (2.0 * math.pi / float(sectors)) * 1.5
    outward_pad = 0.45 * max(float(spacing[0]), float(spacing[1]))

    ring = []
    all_indices = np.arange(len(xy))
    for target in targets:
        delta = _angle_delta(angles, float(target))
        chosen = None
        window = base_window
        for _ in range(5):
            candidates = all_indices[delta <= window]
            if len(candidates):
                chosen = candidates[int(np.argmax(radii[candidates]))]
                break
            window *= 1.75
        if chosen is None:
            chosen = int(np.argmin(delta))

        direction = rel[chosen] / max(float(radii[chosen]), 1e-6)
        ring.append(xy[chosen] + direction * outward_pad)

    return np.asarray(ring, dtype=np.float32)


def _loft_rings_to_mesh(
    rings: list[np.ndarray],
    sectors: int,
    cap_ends: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.vstack(rings).astype(np.float32)
    faces: list[list[int]] = []
    for ring_idx in range(len(rings) - 1):
        base = ring_idx * sectors
        nxt = (ring_idx + 1) * sectors
        for i in range(sectors):
            j = (i + 1) % sectors
            faces.append([base + i, nxt + i, base + j])
            faces.append([base + j, nxt + i, nxt + j])

    if cap_ends:
        bottom_center = len(vertices)
        top_center = bottom_center + 1
        cap_vertices = np.asarray(
            [rings[0].mean(axis=0), rings[-1].mean(axis=0)],
            dtype=np.float32,
        )
        vertices = np.vstack([vertices, cap_vertices]).astype(np.float32)
        top_base = (len(rings) - 1) * sectors
        for i in range(sectors):
            j = (i + 1) % sectors
            faces.append([bottom_center, j, i])
            faces.append([top_center, top_base + i, top_base + j])

    return vertices, np.asarray(faces, dtype=np.int32)


def _solid_lv_mesh_from_segmentation(
    seg_volume: np.ndarray,
    spacing: tuple[float, float, float],
    sectors: int = 96,
) -> tuple[np.ndarray, np.ndarray, str]:
    measure = _optional_measure()
    labels = np.rint(seg_volume).astype(np.int16)
    lv_mask = labels == LBL_LV
    rings: list[np.ndarray] = []
    used_pixel_ring = False

    for z_idx in range(lv_mask.shape[2]):
        mask = lv_mask[:, :, z_idx].astype(np.uint8)
        if int(mask.sum()) <= 10:
            continue

        ring_xy = None
        if measure is not None:
            try:
                contours = measure.find_contours(mask, 0.5)
                if contours:
                    contour = max(contours, key=_contour_area).astype(np.float32)
                    if len(contour) >= 8:
                        xy = np.column_stack([contour[:, 0] * spacing[0], contour[:, 1] * spacing[1]])
                        ring_xy = _resample_contour_by_angle(xy, sectors=sectors)
            except Exception:
                ring_xy = None

        if ring_xy is None:
            ring_xy = _mask_boundary_ring_by_angle(mask.astype(bool), spacing, sectors)
            used_pixel_ring = True
        if ring_xy is None:
            continue

        z_col = np.full((len(ring_xy), 1), z_idx * spacing[2], dtype=np.float32)
        rings.append(np.column_stack([ring_xy, z_col]))

    if len(rings) < 2:
        return _surface_mesh(lv_mask, spacing)

    vertices, faces = _loft_rings_to_mesh(rings, sectors, cap_ends=True)
    method = "segmentation_loft_solid" if used_pixel_ring else "contour_loft_solid"
    return vertices, faces, method


def _signed_delta_status(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "unavailable"
    if value > 1.0:
        return "thickening"
    if value < -1.0:
        return "thinning"
    return "minimal"


def compare_wall_thickness_results(
    ed_result: dict[str, Any],
    es_result: dict[str, Any],
    ed_seg_volume: np.ndarray | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    ed_segments = {int(s.get("id", 0)): s for s in ed_result.get("aha17", [])}
    es_segments = {int(s.get("id", 0)): s for s in es_result.get("aha17", [])}
    aha_delta = []
    for seg_id, name in enumerate(AHA_17_NAMES, start=1):
        ed_mean = ed_segments.get(seg_id, {}).get("meanMm")
        es_mean = es_segments.get(seg_id, {}).get("meanMm")
        delta = None
        relative = None
        if ed_mean is not None and es_mean is not None:
            delta = float(es_mean) - float(ed_mean)
            if abs(float(ed_mean)) > 0.5:
                relative = delta / float(ed_mean) * 100.0
        aha_delta.append({
            "id": seg_id,
            "name": name,
            "edMeanMm": round(float(ed_mean), 2) if ed_mean is not None else None,
            "esMeanMm": round(float(es_mean), 2) if es_mean is not None else None,
            "deltaMm": round(delta, 2) if delta is not None else None,
            "relativeThickeningPct": round(relative, 1) if relative is not None else None,
            "status": _signed_delta_status(delta),
        })

    ed_mean = ed_result.get("metrics", {}).get("meanWallThicknessMm")
    es_mean = es_result.get("metrics", {}).get("meanWallThicknessMm")
    mean_delta = float(es_mean) - float(ed_mean) if ed_mean is not None and es_mean is not None else None
    rel_delta = mean_delta / float(ed_mean) * 100.0 if mean_delta is not None and abs(float(ed_mean)) > 0.5 else None

    ed_mesh = ed_result.get("meshes", {}).get("endo")
    es_mesh = es_result.get("meshes", {}).get("endo")
    ed_vertices = _mesh_vertices(ed_mesh)
    es_vertices = _mesh_vertices(es_mesh)
    ed_values = _mesh_values(ed_mesh)
    es_values = _mesh_values(es_mesh)

    difference_vertices = ed_vertices
    difference_faces = np.asarray(ed_mesh.get("faces", []), dtype=np.int32).reshape(-1, 3) if ed_mesh else np.empty((0, 3), dtype=np.int32)
    mesh_method = "ed_endocardial_surface_radial_motion"

    # Prefer the true ED endocardial surface because it preserves LV geometry.
    # Only fall back to the segmentation loft when the ED mesh is missing or is
    # the old ellipsoid fallback, which looks like a generic smooth blob.
    ed_mesh_method = str(ed_result.get("meshMethod", ""))
    needs_segmentation_fallback = (
        len(difference_vertices) == 0
        or len(difference_faces) == 0
        or "ellipsoid" in ed_mesh_method
    )
    if needs_segmentation_fallback and ed_seg_volume is not None and spacing is not None:
        solid_vertices, solid_faces, solid_method = _solid_lv_mesh_from_segmentation(ed_seg_volume, spacing)
        if len(solid_vertices) and len(solid_faces):
            difference_vertices = solid_vertices
            difference_faces = solid_faces
            mesh_method = f"{solid_method}_radial_motion"

    diff_values = _radial_surface_motion_values(ed_vertices, es_vertices, difference_vertices)
    difference_method = "ED→ES inward radial endocardial motion sampled on the ED LV surface"
    value_label = "ED→ES radial contraction (mm)"
    if diff_values is None:
        ed_sample = _sample_values_nearest(ed_vertices, ed_values, difference_vertices)
        es_sample = _sample_values_nearest(es_vertices, es_values, difference_vertices)
        if ed_sample is not None and es_sample is not None:
            diff_values = (es_sample - ed_sample).astype(np.float32)
            mesh_method = mesh_method.replace("radial_motion", "wall_thickness_delta")
            difference_method = "ES wall-thickness nearest-neighbor sampled onto ED LV surface"
            value_label = "ES−ED wall-thickness change (mm)"

    finite_diff = diff_values[np.isfinite(diff_values)] if diff_values is not None else np.empty(0, dtype=np.float32)
    abs_max = float(np.percentile(np.abs(finite_diff), 98)) if finite_diff.size else 5.0
    if not np.isfinite(abs_max) or abs_max < 1.0:
        abs_max = 5.0

    return {
        "kind": "difference",
        "source": "Paired ED/ES wall-thickness comparison",
        "meshMethod": mesh_method,
        "differenceMethod": difference_method,
        "valueLabel": value_label,
        "metrics": {
            "edMeanWallThicknessMm": round(float(ed_mean), 2) if ed_mean is not None else None,
            "esMeanWallThicknessMm": round(float(es_mean), 2) if es_mean is not None else None,
            "meanDeltaWallThicknessMm": round(mean_delta, 2) if mean_delta is not None else None,
            "relativeThickeningPct": round(rel_delta, 1) if rel_delta is not None else None,
            "meanRadialMotionMm": round(float(np.nanmean(finite_diff)), 2) if finite_diff.size else None,
            "p95RadialMotionMm": round(float(np.nanpercentile(finite_diff, 95)), 2) if finite_diff.size else None,
        },
        "aha17Delta": aha_delta,
        "colorScale": {"min": round(-abs_max, 2), "max": round(abs_max, 2), "center": 0.0},
        "meshes": {
            "endo": _mesh_payload(difference_vertices, difference_faces, diff_values, max_faces=50000),
        },
    }


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
    wall_mean = _finite_mean(wall_values)
    wall_p95 = _finite_percentile(wall_values, 95)
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

