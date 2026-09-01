"""Geometry layer: SAX segmentation -> contours, surfaces and watertight meshes.

Two geometry sources are produced from the *same* input segmentation so that the
wall-thickness estimators can be applied identically to both:

``model``  CardioSDF/INR reconstruction (marching cubes on the predicted SDF).
``voxel``  Segmentation-derived surfaces (marching cubes on the label mask),
           i.e. the voxel-based reference model.

Both are pushed through the identical repair pipeline (`make_watertight`) so
that any difference in the thickness results comes from the geometry, not from
the mesh post-processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nibabel as nib
import trimesh
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
    label as ndi_label,
    zoom as ndi_zoom,
)
from scipy.spatial import cKDTree
from skimage.measure import find_contours, marching_cubes

from cardiosdf_model import FLIP_Z

LBL_BG, LBL_RV, LBL_MYO, LBL_LV = 0, 1, 2, 3
N_PTS_PER_RING = 60

_STRUCT3 = generate_binary_structure(3, 2)
_STRUCT2 = generate_binary_structure(2, 2)


# ──────────────────────────────────────────────────────────────────────────
# NIfTI loading
# ──────────────────────────────────────────────────────────────────────────
def resolve_nifti(path: Path) -> Path:
    """The demo dataset stores each volume inside a directory named ``*.nii``."""
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        for pattern in ("*.nii.gz", "*.nii"):
            hits = sorted(path.glob(pattern))
            if hits:
                return hits[0]
    raise FileNotFoundError(path)


@dataclass
class Segmentation:
    labels: np.ndarray          # (H, W, S) int16
    spacing: tuple              # (dx, dy, dz) mm
    path: Path

    @property
    def lv(self) -> np.ndarray:
        return self.labels == LBL_LV

    @property
    def myo(self) -> np.ndarray:
        return self.labels == LBL_MYO

    @property
    def epi(self) -> np.ndarray:
        return self.lv | self.myo


def load_segmentation(path: Path, frame: int | None = None) -> Segmentation:
    real = resolve_nifti(path)
    nii = nib.load(str(real))
    data = np.asarray(nii.dataobj)
    if data.ndim == 4:
        idx = 0 if frame is None else max(0, min(data.shape[3] - 1, int(frame) - 1))
        data = data[..., idx]
    zooms = tuple(float(v) for v in nii.header.get_zooms()[:3])
    return Segmentation(np.rint(data).astype(np.int16), zooms, real)


def read_info_cfg(path: Path) -> dict:
    """Tolerant Info.cfg reader (the demo file contains merge-conflict markers)."""
    info: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("<<<<<<<", "=======", ">>>>>>>")):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if key in ("oid", "size", "version"):     # git-lfs pointer residue
            continue
        info.setdefault(key, value)
    return info


# ──────────────────────────────────────────────────────────────────────────
# Voxel <-> world (millimetre) mapping
# ──────────────────────────────────────────────────────────────────────────
def world_affine(spacing: tuple) -> np.ndarray:
    """Right-handed mm world frame matching the training convention.

    The training/inference code used ``x = -col``, ``y = -row``, ``z = slice*dz``.
    The ground-truth NIfTI of this demo case carries a unit affine, so the
    in-plane pixel size would be silently dropped; the physical spacing is
    therefore applied explicitly. Distances in this frame are millimetres.
    """
    dx, dy, dz = spacing
    aff = np.eye(4, dtype=np.float64)
    aff[0, 1] = -dx      # x <- -col
    aff[1, 0] = -dy      # y <- -row
    aff[2, 2] = dz       # z <- slice
    aff[0, 0] = aff[1, 1] = 0.0
    return aff


def voxel_to_world(indices: np.ndarray, spacing: tuple) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.float64)
    dx, dy, dz = spacing
    out = np.empty((len(idx), 3), dtype=np.float64)
    out[:, 0] = -idx[:, 1] * dx
    out[:, 1] = -idx[:, 0] * dy
    out[:, 2] = idx[:, 2] * dz
    return out


def world_to_voxel(points_mm: np.ndarray, spacing: tuple) -> np.ndarray:
    pts = np.asarray(points_mm, dtype=np.float64)
    dx, dy, dz = spacing
    out = np.empty_like(pts)
    out[:, 0] = -pts[:, 1] / dy      # row
    out[:, 1] = -pts[:, 0] / dx      # col
    out[:, 2] = pts[:, 2] / dz       # slice
    return out


# ──────────────────────────────────────────────────────────────────────────
# SAX contour extraction (model input)
# ──────────────────────────────────────────────────────────────────────────
def _polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _resample_ring(ring: np.ndarray, n: int) -> np.ndarray:
    if len(ring) < 3:
        return ring
    d = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    total = d.sum()
    if total < 1e-6:
        return ring
    cum = np.concatenate([[0.0], np.cumsum(d)])
    t = np.linspace(0.0, total, n, endpoint=False)
    return np.column_stack([np.interp(t, cum, ring[:, 0]), np.interp(t, cum, ring[:, 1])])


def extract_contours(seg: Segmentation, n_pts: int = N_PTS_PER_RING) -> dict:
    """Endo/epi SAX rings in normalised model space, plus the mm de-normalisation."""
    labels, spacing = seg.labels, seg.spacing
    slices = [s for s in range(labels.shape[2])
              if (labels[:, :, s] == LBL_LV).any() or (labels[:, :, s] == LBL_MYO).any()]

    rows: list[np.ndarray] = []
    for s in slices:
        plane = labels[:, :, s]
        for tissue, mask in ((0.0, plane == LBL_LV),
                             (1.0, (plane == LBL_MYO) | (plane == LBL_LV))):
            if mask.sum() <= 10:
                continue
            found = find_contours(mask.astype(np.uint8), 0.5)
            if not found:
                continue
            ring = _resample_ring(max(found, key=_polygon_area), n_pts)
            idx = np.column_stack([ring[:, 0], ring[:, 1], np.full(len(ring), s)])
            xyz = voxel_to_world(idx, spacing)
            rows.append(np.column_stack([xyz, np.full(len(ring), tissue)]))

    if not rows:
        raise ValueError(f"No LV/MYO contours found in {seg.path}")

    raw = np.vstack(rows).astype(np.float32)
    xyz, tissue = raw[:, :3], raw[:, 3]
    centroid = xyz.mean(0)
    centred = xyz - centroid
    scale = float(np.linalg.norm(centred[:, :2], axis=1).mean())
    xyz_n = (centred / scale).astype(np.float32)
    if FLIP_Z:
        xyz_n[:, 2] = -xyz_n[:, 2]
    return {
        "xyz": xyz_n,
        "tissue": tissue.astype(np.float32),
        "centroid": centroid.astype(np.float32),
        "scale": scale,
        "slices": np.asarray(slices, dtype=np.int32),
        "xyz_mm": xyz.astype(np.float32),
    }


def denormalise(points_norm: np.ndarray, centroid: np.ndarray, scale: float) -> np.ndarray:
    flip = np.array([1.0, 1.0, -1.0 if FLIP_Z else 1.0], dtype=np.float32)
    return (np.asarray(points_norm, np.float32) * flip) * float(scale) + centroid


def normalise(points_mm: np.ndarray, centroid: np.ndarray, scale: float) -> np.ndarray:
    flip = np.array([1.0, 1.0, -1.0 if FLIP_Z else 1.0], dtype=np.float32)
    return ((np.asarray(points_mm, np.float32) - centroid) / float(scale)) * flip


# ──────────────────────────────────────────────────────────────────────────
# Surface extraction and repair
# ──────────────────────────────────────────────────────────────────────────
def _largest_component_3d(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi_label(mask, structure=_STRUCT3)
    if n <= 1:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def _clean_inside(mask: np.ndarray, min_pixels: int = 12) -> np.ndarray:
    """Single connected, hole-free inside region, per slice and in 3D."""
    out = _largest_component_3d(mask)
    per_slice = np.zeros_like(out)
    for k in range(out.shape[2]):
        plane = out[:, :, k]
        if plane.sum() < min_pixels:
            continue
        lab, n = ndi_label(plane, structure=_STRUCT2)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        keep = lab == int(np.argmax(sizes))
        per_slice[:, :, k] = binary_fill_holes(keep, structure=_STRUCT2)
    out = binary_fill_holes(per_slice, structure=_STRUCT3)
    out[[0, -1], :, :] = False
    out[:, [0, -1], :] = False
    out[:, :, [0, -1]] = False
    return out.astype(bool)


def signed_distance_from_mask(mask: np.ndarray, voxel: np.ndarray,
                              smooth_sigma: float = 0.0) -> np.ndarray:
    """Re-distance a cleaned binary inside-region so marching cubes sees one shell."""
    outside = distance_transform_edt(~mask, sampling=tuple(voxel))
    inside = distance_transform_edt(mask, sampling=tuple(voxel))
    field = (outside - inside).astype(np.float32)
    if smooth_sigma > 0:
        field = gaussian_filter(field, sigma=smooth_sigma).astype(np.float32)
    return field


def marching_cubes_mesh(field: np.ndarray, origin: np.ndarray, voxel: np.ndarray,
                        level: float = 0.0) -> trimesh.Trimesh:
    field = np.nan_to_num(np.asarray(field, np.float32), nan=1e3,
                          posinf=1e3, neginf=-1e3)
    if field.min() > level or field.max() < level:
        return trimesh.Trimesh(process=False)
    verts, faces, _, _ = marching_cubes(field, level=level, spacing=tuple(voxel))
    return trimesh.Trimesh(vertices=(verts + origin).astype(np.float64),
                           faces=faces.astype(np.int64), process=False)


def _enforce_genus_zero(mesh: trimesh.Trimesh, pitch: float = 0.5) -> trimesh.Trimesh:
    """Re-mesh through a cleaned occupancy mask to remove handles/tunnels.

    A mesh can be watertight yet have genus > 0 (Euler != 2). Rasterising the
    solid, keeping one hole-free component and re-extracting the zero level set
    of its signed distance yields a topological sphere.
    """
    origin, shape = isotropic_grid([mesh], pitch, pad_mm=3.0 * pitch)
    inside = _clean_inside(voxelise_surface(mesh, origin, pitch, shape))
    if not inside.any():
        return mesh
    field = signed_distance_from_mask(inside, np.full(3, pitch), smooth_sigma=0.6)
    out = marching_cubes_mesh(field, origin, np.full(3, pitch))
    if len(out.faces) == 0:
        return mesh
    parts = [p for p in out.split(only_watertight=False) if len(p.faces) > 0]
    if len(parts) > 1:
        out = max(parts, key=lambda p: len(p.faces))
    trimesh.repair.fix_normals(out)
    return out


def make_watertight(mesh: trimesh.Trimesh, name: str,
                    taubin_iters: int = 12,
                    target_faces: int | None = None) -> tuple[trimesh.Trimesh, dict]:
    """Degenerate-face removal -> largest component -> hole filling -> normals.

    ``pymeshfix`` is used when available because it guarantees a closed,
    self-intersection-free manifold; the trimesh fallback keeps the pipeline
    runnable without it. The returned report is written to the results table so
    that the mesh quality of both geometries can be stated in the thesis.
    """
    report: dict = {"surface": name}
    mesh = mesh.copy()
    report["faces_in"] = int(len(mesh.faces))
    report["watertight_in"] = bool(mesh.is_watertight)

    if len(mesh.faces) == 0:
        report.update(faces_out=0, watertight_out=False, repaired_with="none")
        return mesh, report

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    parts = [p for p in mesh.split(only_watertight=False) if len(p.faces) > 0]
    if len(parts) > 1:
        report["components_in"] = len(parts)
        mesh = max(parts, key=lambda p: len(p.faces))
    else:
        report["components_in"] = 1

    repaired_with = "trimesh"
    try:
        import pymeshfix

        fixer = pymeshfix.MeshFix(np.asarray(mesh.vertices, np.float64),
                                  np.asarray(mesh.faces, np.int32))
        fixer.repair(joincomp=True, remove_smallest_components=True)
        if len(fixer.faces) > 0:
            mesh = trimesh.Trimesh(vertices=np.asarray(fixer.points),
                                   faces=np.asarray(fixer.faces), process=False)
            repaired_with = "pymeshfix"
    except Exception as exc:                       # pragma: no cover - optional dep
        report["pymeshfix_error"] = str(exc)[:120]

    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)

    # Watertight is not enough: a handle would make the transmural field
    # multiply connected, so any non-sphere topology is re-meshed.
    report["euler_number_in"] = int(mesh.euler_number)
    if mesh.euler_number != 2:
        try:
            mesh = _enforce_genus_zero(mesh)
            trimesh.repair.fix_normals(mesh)
            report["genus_repair"] = True
        except Exception as exc:
            report["genus_repair_error"] = str(exc)[:120]

    if taubin_iters > 0 and len(mesh.faces) > 0:
        try:
            mesh = trimesh.smoothing.filter_taubin(mesh, lamb=0.53, nu=-0.53,
                                                   iterations=taubin_iters)
        except Exception:
            pass

    if target_faces is not None and len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(target_faces)
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass

    report.update(
        repaired_with=repaired_with,
        faces_out=int(len(mesh.faces)),
        vertices_out=int(len(mesh.vertices)),
        watertight_out=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        euler_number=int(mesh.euler_number),
        volume_ml=float(abs(mesh.volume) / 1000.0),
        area_cm2=float(mesh.area / 100.0),
    )
    return mesh, report


def repair_if_invalid(mesh: trimesh.Trimesh, name: str) -> tuple[trimesh.Trimesh, dict]:
    """Preserve a valid genus-zero shell; repair only when validation fails."""
    candidate = mesh.copy()
    trimesh.repair.fix_normals(candidate)
    valid = (
        len(candidate.faces) > 0
        and candidate.is_watertight
        and candidate.is_winding_consistent
        and candidate.euler_number == 2
    )
    if not valid:
        repaired, report = make_watertight(candidate, name, taubin_iters=0)
        report["repair_required"] = True
        return repaired, report

    report = {
        "surface": name,
        "faces_in": int(len(candidate.faces)),
        "faces_out": int(len(candidate.faces)),
        "vertices_out": int(len(candidate.vertices)),
        "components_in": 1,
        "repaired_with": "none",
        "repair_required": False,
        "watertight_in": True,
        "watertight_out": True,
        "winding_consistent": True,
        "euler_number_in": 2,
        "euler_number": 2,
        "volume_ml": float(abs(candidate.volume) / 1000.0),
        "area_cm2": float(candidate.area / 100.0),
    }
    return candidate, report


def _section_polygons(mesh: trimesh.Trimesh, z: float):
    """Cross-section of a closed mesh at z = const, as world-frame shapely polygons."""
    try:
        section = mesh.section(plane_origin=[0.0, 0.0, z], plane_normal=[0.0, 0.0, 1.0])
    except Exception:
        return []
    if section is None:
        return []
    to_2d = np.eye(4)
    to_2d[2, 3] = -z          # keep planar coordinates identical to world (x, y)
    try:
        planar, _ = section.to_planar(to_2D=to_2d)
    except Exception:
        return []
    return [p for p in getattr(planar, "polygons_full", []) if p.area > 0]


def voxelise_surface(mesh: trimesh.Trimesh, origin: np.ndarray, pitch: float,
                     shape: tuple) -> np.ndarray:
    """Watertight mesh -> filled binary mask on a regular isotropic grid.

    Rasterisation is done slice by slice from the exact mesh cross-sections, so
    interior holes are respected and the result does not depend on surface
    sampling density (unlike vertex splatting followed by hole filling).
    """
    import shapely

    xs = origin[0] + np.arange(shape[0]) * pitch
    ys = origin[1] + np.arange(shape[1]) * pitch
    zs = origin[2] + np.arange(shape[2]) * pitch
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()

    mask = np.zeros(shape, dtype=bool)
    z_lo, z_hi = mesh.bounds[0][2], mesh.bounds[1][2]
    for k, z in enumerate(zs):
        if z < z_lo or z > z_hi:
            continue
        polys = _section_polygons(mesh, float(z))
        if not polys:
            continue
        geom = shapely.union_all(polys)
        mask[:, :, k] = shapely.contains_xy(geom, gx, gy).reshape(shape[0], shape[1])

    if not mask.any():                       # degenerate section: fall back to splatting
        idx = np.rint((np.asarray(mesh.vertices) - origin) / pitch).astype(np.int64)
        idx = np.clip(idx, 0, np.asarray(shape) - 1)
        mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        mask = binary_fill_holes(binary_closing(mask, structure=_STRUCT3))
    return binary_fill_holes(mask).astype(bool)


MAX_GRID_VOXELS = 40_000_000


def _check_grid_budget(shape, what: str) -> None:
    n = int(np.prod(shape))
    if n > MAX_GRID_VOXELS:
        raise MemoryError(
            f"{what} would need {n/1e6:.0f}M voxels (cap {MAX_GRID_VOXELS/1e6:.0f}M). "
            "Increase the pitch or tighten the crop."
        )


def isotropic_grid(meshes: list, pitch: float, pad_mm: float = 3.0):
    verts = np.vstack([np.asarray(m.vertices) for m in meshes if len(m.vertices)])
    lo = verts.min(0) - pad_mm
    hi = verts.max(0) + pad_mm
    shape = tuple(int(np.ceil((hi[d] - lo[d]) / pitch)) + 1 for d in range(3))
    _check_grid_budget(shape, f"isotropic grid at {pitch} mm")
    return lo.astype(np.float64), shape


# ──────────────────────────────────────────────────────────────────────────
# Geometry builders
# ──────────────────────────────────────────────────────────────────────────
def build_loft_geometry(contours: dict, taubin_iters: int = 12) -> dict:
    """Non-neural baseline: join corresponding SAX contour points linearly."""
    xyz = np.asarray(contours["xyz_mm"], dtype=np.float64)
    tissue = np.asarray(contours["tissue"], dtype=np.float64)
    meshes: dict[str, trimesh.Trimesh] = {}
    reports: list[dict] = []

    for name, tissue_id in (("endo", 0.0), ("epi", 1.0)):
        surface_points = xyz[np.isclose(tissue, tissue_id)]
        rings = [surface_points[np.isclose(surface_points[:, 2], z)]
                 for z in np.unique(surface_points[:, 2])]
        rings = [ring for ring in rings if len(ring) >= 3]
        if len(rings) < 2:
            raise ValueError(f"Contour lofting needs at least two {name} rings.")

        aligned = [rings[0]]
        for ring in rings[1:]:
            previous = aligned[-1]
            if len(ring) != len(previous):
                ring = _resample_ring(np.vstack([ring, ring[0]]), len(previous))
            ring = min(
                (np.roll(ring, shift, axis=0) for shift in range(len(ring))),
                key=lambda candidate: np.mean(np.sum((candidate[:, :2] - previous[:, :2]) ** 2,
                                                     axis=1)),
            )
            aligned.append(ring)

        vertices = np.vstack(aligned)
        ring_size = len(aligned[0])
        faces: list[list[int]] = []
        for ring_index in range(len(aligned) - 1):
            lower = ring_index * ring_size
            upper = (ring_index + 1) * ring_size
            for point_index in range(ring_size):
                next_index = (point_index + 1) % ring_size
                faces.append([lower + point_index, upper + point_index, upper + next_index])
                faces.append([lower + point_index, upper + next_index, lower + next_index])

        for ring_index in (0, len(aligned) - 1):
            centre_index = len(vertices)
            vertices = np.vstack([vertices, aligned[ring_index].mean(axis=0)])
            offset = ring_index * ring_size
            for point_index in range(ring_size):
                next_index = (point_index + 1) % ring_size
                faces.append([centre_index, offset + point_index, offset + next_index])

        raw = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
        origin, shape = isotropic_grid([raw], pitch=1.0, pad_mm=3.0)
        inside = _clean_inside(voxelise_surface(raw, origin, 1.0, shape))
        field = signed_distance_from_mask(inside, np.ones(3), smooth_sigma=0.0)
        dense = marching_cubes_mesh(field, origin, np.ones(3))
        mesh, report = make_watertight(dense, f"loft-{name}", taubin_iters)
        meshes[name] = mesh
        reports.append(report)

    return {**meshes, "reports": reports, "source": "linear contour lofting"}


def _crop_bounds(mask: np.ndarray, spacing: np.ndarray, margin_mm: float = 8.0):
    """Tight bounding box around ``mask`` plus a margin, in voxel index space."""
    idx = np.argwhere(mask)
    if len(idx) == 0:
        raise ValueError("Cannot crop an empty mask.")
    pad = np.maximum(np.ceil(margin_mm / spacing), 1).astype(np.int64)
    start = np.maximum(idx.min(0) - pad, 0)
    stop = np.minimum(idx.max(0) + pad + 1, np.asarray(mask.shape))
    return start, stop


def build_voxel_geometry(seg: Segmentation, iso_pitch: float = 1.0,
                         taubin_iters: int = 12, margin_mm: float = 8.0) -> dict:
    """Voxel-based reference: label mask -> isotropic resample -> surfaces.

    The SAX stack is cropped to the epicardial bounding box before isotropic
    resampling; a full-field-of-view resample of a 10 mm-slice stack produces a
    grid two orders of magnitude larger than the heart itself.
    """
    spacing = np.asarray(seg.spacing, dtype=np.float64)
    start, stop = _crop_bounds(seg.epi, spacing, margin_mm)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    factors = spacing / iso_pitch
    _check_grid_budget(np.ceil((stop - start) * factors),
                       f"voxel-geometry resample at {iso_pitch} mm")

    def resample(mask: np.ndarray) -> np.ndarray:
        out = ndi_zoom(mask[sl].astype(np.float32), factors, order=1, prefilter=False)
        return out >= 0.5

    lv_iso = _clean_inside(resample(seg.lv))
    epi_iso = _clean_inside(resample(seg.epi) | lv_iso)
    voxel = np.full(3, iso_pitch)

    # Crop corner in the (row, col, slice) millimetre frame.
    origin = start.astype(np.float64) * spacing
    endo_field = signed_distance_from_mask(lv_iso, voxel, smooth_sigma=0.6)
    epi_field = signed_distance_from_mask(epi_iso, voxel, smooth_sigma=0.6)

    endo_raw = marching_cubes_mesh(endo_field, origin, voxel)
    epi_raw = marching_cubes_mesh(epi_field, origin, voxel)

    # index-space (iso voxels) -> mm world, same convention as voxel_to_world
    def to_world(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        if len(mesh.vertices) == 0:
            return mesh
        v = np.asarray(mesh.vertices)
        out = np.column_stack([-v[:, 1], -v[:, 0], v[:, 2]])
        return trimesh.Trimesh(vertices=out, faces=mesh.faces, process=False)

    endo, endo_rep = make_watertight(to_world(endo_raw), "voxel-endo", taubin_iters)
    epi, epi_rep = make_watertight(to_world(epi_raw), "voxel-epi", taubin_iters)
    return {"endo": endo, "epi": epi, "reports": [endo_rep, epi_rep],
            "source": "segmentation voxels"}


def build_model_geometry(net, cfg, contours: dict, grid_res: int = 96,
                         phase_val: float = 0.0, taubin_iters: int = 12,
                         batch: int = 65536) -> dict:
    """CardioSDF surfaces: encode contours -> dense SDF -> cleaned marching cubes."""
    from cardiosdf_model import dense_sdf_grid, encode_contours

    z = encode_contours(net, contours["xyz"], contours["tissue"], cfg, phase_val)
    sdf_e, sdf_p, delta, lo, hi, voxel = dense_sdf_grid(
        net, z, contours["xyz"], cfg, grid_res=grid_res, batch=batch)

    iso = float(cfg.get("iso_level", 0.0))
    meshes = {}
    reports = []
    for key, field in (("endo", sdf_e), ("epi", sdf_p)):
        inside = _clean_inside(field <= iso)
        clean = signed_distance_from_mask(inside, voxel, smooth_sigma=0.8)
        raw = marching_cubes_mesh(clean, lo, voxel, level=0.0)
        if len(raw.vertices):
            raw = trimesh.Trimesh(
                vertices=denormalise(raw.vertices, contours["centroid"], contours["scale"]),
                faces=raw.faces, process=False)
        mesh, report = make_watertight(raw, f"model-{key}", taubin_iters)
        meshes[key] = mesh
        reports.append(report)

    return {**meshes, "reports": reports, "latent": z, "sdf_endo": sdf_e,
            "sdf_epi": sdf_p, "delta": delta, "grid_lo": lo, "grid_hi": hi,
            "grid_voxel": voxel, "source": "CardioSDF/INR checkpoint"}


def _contains_chunked(mesh: trimesh.Trimesh, points: np.ndarray,
                      pitch: float = 0.5) -> np.ndarray:
    """Inside test via slice-wise rasterisation of a watertight mesh.

    ``trimesh.contains`` falls back to a pure-Python ray engine when embree is
    absent, which allocates per-ray candidate lists and exhausts memory on
    meshes of this size. Rasterising once and looking points up is O(1) each.
    """
    points = np.asarray(points, np.float64)
    origin, shape = isotropic_grid([mesh], pitch, pad_mm=2.0 * pitch)
    inside = voxelise_surface(mesh, origin, pitch, shape)
    idx = np.rint((points - origin) / pitch).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
    out = np.zeros(len(points), dtype=bool)
    sel = idx[ok]
    out[ok] = inside[sel[:, 0], sel[:, 1], sel[:, 2]]
    return out


def enforce_nesting(endo: trimesh.Trimesh, epi: trimesh.Trimesh,
                    min_wall_mm: float = 0.3) -> tuple[trimesh.Trimesh, dict]:
    """Push endocardial vertices that fall outside the epicardium back inside.

    Wall thickness is only defined where the endocardium is strictly enclosed by
    the epicardium. Rather than silently dropping those vertices, they are moved
    along the inward direction by the violation depth plus a safety margin, and
    the affected fraction is reported.
    """
    if len(endo.vertices) == 0 or len(epi.faces) == 0:
        return endo, {"nesting_violations": 0, "nesting_fraction": 0.0}
    verts = np.asarray(endo.vertices, np.float64)
    bad = ~_contains_chunked(epi, verts)
    report = {"nesting_violations": int(bad.sum()),
              "nesting_fraction": float(bad.mean())}
    if not bad.any():
        return endo, report
    closest, dist, _ = trimesh.proximity.closest_point(epi, verts[bad])
    direction = closest - verts[bad]
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = np.divide(direction, np.clip(norm, 1e-9, None))
    verts[bad] = closest + direction * min_wall_mm
    fixed = trimesh.Trimesh(vertices=verts, faces=endo.faces, process=False)
    trimesh.repair.fix_normals(fixed)
    report["nesting_max_depth_mm"] = float(dist.max())
    return fixed, report


def outward_normals(endo: trimesh.Trimesh, epi_vertices: np.ndarray) -> np.ndarray:
    """Endocardial vertex normals oriented towards the epicardium."""
    pts = np.asarray(endo.vertices, np.float64)
    normals = np.asarray(endo.vertex_normals, np.float64).copy()
    _, nn = cKDTree(epi_vertices).query(pts, workers=-1)
    if float(np.nanmedian(np.sum(normals * (epi_vertices[nn] - pts), axis=1))) < 0:
        normals *= -1.0
    return normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)


def long_axis_frame(endo: trimesh.Trimesh, seg: Segmentation) -> dict:
    """Base/apex axis and the septal reference direction (for AHA-17)."""
    verts = np.asarray(endo.vertices, np.float64)
    epi_mask = seg.epi
    support = np.flatnonzero(epi_mask.any(axis=(0, 1)))
    lv_area = seg.lv.sum(axis=(0, 1)).astype(float)
    first, last = int(support[0]), int(support[-1])
    base_slice, apex_slice = (first, last) if lv_area[first] >= lv_area[last] else (last, first)
    base_z = base_slice * seg.spacing[2]
    apex_z = apex_slice * seg.spacing[2]

    axis = np.array([0.0, 0.0, np.sign(apex_z - base_z) or 1.0])
    t = (verts[:, 2] - base_z) / (apex_z - base_z)

    centre = verts[:, :2].mean(0)
    anterior_angle = 0.0
    if seg.labels.any() and (seg.labels == LBL_RV).any():
        rv = voxel_to_world(np.argwhere(seg.labels == LBL_RV), seg.spacing)
        septal = np.arctan2(rv[:, 1].mean() - centre[1], rv[:, 0].mean() - centre[0])
        anterior_angle = septal - np.pi / 2.0
    return {"base_z": float(base_z), "apex_z": float(apex_z), "axis": axis,
            "long_axis_t": t.astype(np.float32), "centre_xy": centre,
            "anterior_angle": float(anterior_angle),
            "base_slice": base_slice, "apex_slice": apex_slice}


AHA_17_NAMES = [
    "Basal Anterior", "Basal Anteroseptal", "Basal Inferoseptal",
    "Basal Inferior", "Basal Inferolateral", "Basal Anterolateral",
    "Mid Anterior", "Mid Anteroseptal", "Mid Inferoseptal",
    "Mid Inferior", "Mid Inferolateral", "Mid Anterolateral",
    "Apical Anterior", "Apical Septal", "Apical Inferior", "Apical Lateral",
    "Apex",
]


def assign_aha17(verts: np.ndarray, frame: dict, apical_end: float = 0.85) -> np.ndarray:
    t = np.clip((np.asarray(verts)[:, 2] - frame["base_z"]) /
                (frame["apex_z"] - frame["base_z"]), 0.0, 1.0)
    cx, cy = frame["centre_xy"]
    ang = np.degrees(np.arctan2(verts[:, 1] - cy, verts[:, 0] - cx)
                     - frame["anterior_angle"]) % 360.0
    ids = np.full(len(verts), 17, dtype=np.int16)
    seg6 = (ang / 60.0).astype(np.int16) % 6
    seg4 = (ang / 90.0).astype(np.int16) % 4
    basal = t < 1 / 3
    mid = (t >= 1 / 3) & (t < 2 / 3)
    apical = (t >= 2 / 3) & (t < apical_end)
    ids[basal] = 1 + seg6[basal]
    ids[mid] = 7 + seg6[mid]
    ids[apical] = 13 + seg4[apical]
    return ids
