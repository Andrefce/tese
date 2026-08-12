"""Wall-thickness estimators evaluated in the demo.

All estimators return one value per endocardial vertex, in millimetres, with
``NaN`` where the method could not produce a defensible value (the invalid
fraction is reported instead of being hidden by a fallback).

Volumetric estimators share a single isotropic rasterisation of the two
watertight surfaces, so differences between them are algorithmic, not
discretisation artefacts.

    laplace_streamline   integrated Laplace streamline length   (primary)
    laplace_gradient     local 1/||grad phi||                   (legacy comparator)
    edt_boundary_sum     d_endo + d_epi                         (volumetric baseline)
    sphere_propagation   morphological sphere propagation       (fast comparator)
    surface_correspondence  regularised symmetric matching      (mesh-native)
    cone_rays            normal-cone ray casting                (sensitivity baseline)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy import sparse
from scipy.ndimage import binary_dilation, distance_transform_edt, generate_binary_structure
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree

_STRUCT3 = generate_binary_structure(3, 2)


@dataclass
class ThicknessResult:
    name: str
    family: str
    values: np.ndarray             # per endocardial vertex, mm
    runtime_s: float
    diagnostics: dict = field(default_factory=dict)

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.values)

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean()) if self.values.size else 0.0

    def summary(self) -> dict:
        v = self.values[self.valid]
        if v.size == 0:
            return {"method": self.name, "family": self.family, "n": 0}
        return {
            "method": self.name,
            "family": self.family,
            "n": int(v.size),
            "valid_fraction": round(self.valid_fraction, 4),
            "mean_mm": round(float(v.mean()), 3),
            "median_mm": round(float(np.median(v)), 3),
            "std_mm": round(float(v.std(ddof=1)), 3) if v.size > 1 else 0.0,
            "p5_mm": round(float(np.percentile(v, 5)), 3),
            "p95_mm": round(float(np.percentile(v, 95)), 3),
            "runtime_s": round(self.runtime_s, 2),
        }


# ──────────────────────────────────────────────────────────────────────────
# Shared volumetric context
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class VolumeContext:
    endo_mask: np.ndarray
    epi_mask: np.ndarray
    myo_mask: np.ndarray
    origin: np.ndarray
    pitch: float

    def world_to_index(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, np.float64) - self.origin) / self.pitch


def build_volume_context(endo_mesh, epi_mesh, pitch: float,
                         pad_mm: float = 3.0) -> VolumeContext:
    from geometry import isotropic_grid, voxelise_surface

    origin, shape = isotropic_grid([endo_mesh, epi_mesh], pitch, pad_mm)
    endo_mask = voxelise_surface(endo_mesh, origin, pitch, shape)
    epi_mask = voxelise_surface(epi_mesh, origin, pitch, shape)
    epi_mask |= endo_mask                      # endocardium is inside the epicardium
    myo_mask = epi_mask & ~endo_mask
    return VolumeContext(endo_mask, epi_mask, myo_mask, origin, pitch)


def _trilinear(field: np.ndarray, idx: np.ndarray, fill=np.nan) -> np.ndarray:
    """Trilinear sample of a scalar field at fractional voxel indices."""
    shape = np.asarray(field.shape)
    base = np.floor(idx).astype(np.int64)
    frac = idx - base
    out = np.full(len(idx), fill, dtype=np.float64)
    ok = np.all((base >= 0) & (base < shape - 1), axis=1)
    if not ok.any():
        return out
    b, f = base[ok], frac[ok]
    acc = np.zeros(int(ok.sum()))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((1 - f[:, 0]) if dx == 0 else f[:, 0]) * \
                    ((1 - f[:, 1]) if dy == 0 else f[:, 1]) * \
                    ((1 - f[:, 2]) if dz == 0 else f[:, 2])
                acc += w * field[b[:, 0] + dx, b[:, 1] + dy, b[:, 2] + dz]
    out[ok] = acc
    return out


def _sample_at_vertices(field_vol: np.ndarray, ctx: VolumeContext,
                        verts: np.ndarray, seed_offset: np.ndarray) -> np.ndarray:
    """Nearest-myocardium sample of a volumetric field at endocardial vertices."""
    myo_idx = np.argwhere(ctx.myo_mask)
    if len(myo_idx) == 0:
        return np.full(len(verts), np.nan)
    myo_world = ctx.origin + myo_idx * ctx.pitch
    tree = cKDTree(myo_world)
    dist, nn = tree.query(verts + seed_offset, workers=-1)
    idx = myo_idx[nn]
    vals = field_vol[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float64)
    vals[dist > 3.0 * ctx.pitch + np.linalg.norm(seed_offset, axis=1).max()] = np.nan
    return vals


# ──────────────────────────────────────────────────────────────────────────
# Laplace transmural field
# ──────────────────────────────────────────────────────────────────────────
def solve_laplace(ctx: VolumeContext, tol: float = 1e-8,
                  maxiter: int = 4000) -> tuple[np.ndarray, dict]:
    """phi = 0 on the endocardium, phi = 1 on the epicardium, harmonic in between."""
    myo = ctx.myo_mask
    idx = np.argwhere(myo)
    n = len(idx)
    if n == 0:
        raise ValueError("empty myocardium")
    local = -np.ones(myo.shape, dtype=np.int64)
    local[myo] = np.arange(n)
    h2 = ctx.pitch ** 2

    rows = [np.arange(n)]
    cols = [np.arange(n)]
    diag = np.zeros(n)
    vals = []
    b = np.zeros(n)

    for axis in range(3):
        for step in (-1, 1):
            nb = idx.copy()
            nb[:, axis] += step
            inside = np.all((nb >= 0) & (nb < np.asarray(myo.shape)), axis=1)
            nb_c = np.clip(nb, 0, np.asarray(myo.shape) - 1)
            in_myo = inside & myo[nb_c[:, 0], nb_c[:, 1], nb_c[:, 2]]
            in_endo = inside & ctx.endo_mask[nb_c[:, 0], nb_c[:, 1], nb_c[:, 2]] & ~in_myo
            outside_epi = ~in_myo & ~in_endo     # includes the grid border

            diag += 1.0 / h2
            b[outside_epi] += 1.0 / h2           # Dirichlet phi = 1 on the epicardium
            # in_endo contributes Dirichlet phi = 0 -> no right-hand-side term
            sel = np.flatnonzero(in_myo)
            if sel.size:
                rows.append(sel)
                cols.append(local[nb[sel, 0], nb[sel, 1], nb[sel, 2]])
                vals.append(np.full(sel.size, -1.0 / h2))

    vals = [diag] + vals
    A = sparse.csr_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    phi_vec, info = cg(A, b, rtol=tol, maxiter=maxiter)
    phi = np.full(myo.shape, np.nan, dtype=np.float64)
    phi[myo] = np.clip(phi_vec, 0.0, 1.0)
    return phi, {"laplace_cg_info": int(info), "laplace_unknowns": int(n)}


def _laplace_direction_field(phi: np.ndarray, ctx: VolumeContext) -> np.ndarray:
    """Unit transmural direction v = grad(phi)/||grad(phi)||, defined on the whole grid."""
    filled = np.where(np.isfinite(phi), phi, 0.0)
    filled[ctx.epi_mask & ~ctx.myo_mask & ~ctx.endo_mask] = 1.0
    filled[~ctx.epi_mask] = 1.0
    grad = np.stack(np.gradient(filled, ctx.pitch, edge_order=2), axis=-1)
    norm = np.linalg.norm(grad, axis=-1, keepdims=True)
    return (grad / np.clip(norm, 1e-9, None)).astype(np.float32)


def _integrate_streamlines(direction: np.ndarray, ctx: VolumeContext,
                           seeds: np.ndarray, stop_mask: np.ndarray,
                           sign: float, step_frac: float = 0.25,
                           max_len_mm: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """RK2 march along +/- v until leaving ``stop_mask``; returns arc length and endpoint.

    All seeds are advanced in lockstep so the integration is fully vectorised.
    """
    h = step_frac * ctx.pitch
    max_steps = int(np.ceil(max_len_mm / h))
    pos = seeds.astype(np.float64).copy()
    length = np.zeros(len(seeds))
    alive = np.ones(len(seeds), dtype=bool)

    comp = [np.ascontiguousarray(direction[..., d].astype(np.float64)) for d in range(3)]

    def velocity(points):
        idx = ctx.world_to_index(points)
        v = np.stack([_trilinear(comp[d], idx, fill=0.0) for d in range(3)], axis=1)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        return sign * v / np.clip(norm, 1e-9, None)

    def inside(points):
        idx = np.rint(ctx.world_to_index(points)).astype(np.int64)
        shp = np.asarray(stop_mask.shape)
        ok = np.all((idx >= 0) & (idx < shp), axis=1)
        res = np.zeros(len(points), dtype=bool)
        if ok.any():
            c = idx[ok]
            res[ok] = stop_mask[c[:, 0], c[:, 1], c[:, 2]]
        return res

    for _ in range(max_steps):
        if not alive.any():
            break
        p = pos[alive]
        k1 = velocity(p)
        k2 = velocity(p + 0.5 * h * k1)
        step = h * k2
        new_p = p + step
        pos[alive] = new_p
        length[alive] += np.linalg.norm(step, axis=1)
        still = inside(new_p)
        idx_alive = np.flatnonzero(alive)
        alive[idx_alive[~still]] = False

    hit = length < (max_len_mm - 2 * h)
    length[~hit] = np.nan
    return length, pos


def method_laplace_streamline(ctx: VolumeContext, verts: np.ndarray,
                              normals: np.ndarray, phi: np.ndarray,
                              symmetric: bool = True) -> ThicknessResult:
    """Thickness = arc length of the full endo->epi transmural streamline.

    This is the recommended primary estimator: unlike ``1/||grad phi||`` it is a
    genuine endocardium-to-epicardium path length and therefore stays correct in
    curved, non-parallel walls. The backward integration from the epicardial
    endpoint gives an explicit round-trip consistency check.
    """
    t0 = time.perf_counter()
    direction = _laplace_direction_field(phi, ctx)
    seeds = verts + normals * (0.75 * ctx.pitch)      # start just inside the wall

    fwd_len, epi_pts = _integrate_streamlines(direction, ctx, seeds, ctx.myo_mask, +1.0)
    diagnostics = {"streamline_step_mm": round(0.25 * ctx.pitch, 4)}

    values = fwd_len + 0.75 * ctx.pitch
    if symmetric:
        back_seeds = epi_pts - direction_at(direction, ctx, epi_pts) * (0.75 * ctx.pitch)
        back_len, endo_pts = _integrate_streamlines(direction, ctx, back_seeds,
                                                    ctx.myo_mask, -1.0)
        round_trip = np.linalg.norm(endo_pts - seeds, axis=1)
        both = np.isfinite(fwd_len) & np.isfinite(back_len)
        sym = np.full(len(verts), np.nan)
        sym[both] = 0.5 * (fwd_len[both] + back_len[both]) + 0.75 * ctx.pitch
        values = np.where(both, sym, values)
        diagnostics.update(
            round_trip_median_mm=float(np.nanmedian(round_trip)),
            round_trip_p95_mm=float(np.nanpercentile(round_trip, 95)),
            symmetric_fraction=float(both.mean()),
        )
    diagnostics["terminated_fraction"] = float(np.isfinite(fwd_len).mean())
    return ThicknessResult("Laplace streamline (symmetric)", "PDE / streamline",
                           values.astype(np.float32), time.perf_counter() - t0,
                           diagnostics)


def direction_at(direction: np.ndarray, ctx: VolumeContext, pts: np.ndarray) -> np.ndarray:
    idx = ctx.world_to_index(pts)
    v = np.stack([_trilinear(np.ascontiguousarray(direction[..., d].astype(np.float64)),
                             idx, fill=0.0) for d in range(3)], axis=1)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


def method_laplace_gradient(ctx: VolumeContext, verts: np.ndarray,
                            normals: np.ndarray, phi: np.ndarray) -> ThicknessResult:
    """Legacy local estimate t = 1/||grad phi||, kept for comparison only."""
    t0 = time.perf_counter()
    filled = np.where(np.isfinite(phi), phi, 0.0)
    grad = np.stack(np.gradient(filled, ctx.pitch, edge_order=2), axis=-1)
    mag = np.linalg.norm(grad, axis=-1)
    field = np.full(phi.shape, np.nan)
    ok = ctx.myo_mask & (mag > 1e-9)
    field[ok] = 1.0 / mag[ok]
    values = _sample_at_vertices(field, ctx, verts, normals * (0.75 * ctx.pitch))
    return ThicknessResult("Laplace local gradient", "PDE / local",
                           values.astype(np.float32), time.perf_counter() - t0, {})


# ──────────────────────────────────────────────────────────────────────────
# Volumetric baselines
# ──────────────────────────────────────────────────────────────────────────
def method_edt_boundary_sum(ctx: VolumeContext, verts: np.ndarray,
                            normals: np.ndarray) -> ThicknessResult:
    t0 = time.perf_counter()
    sampling = (ctx.pitch,) * 3
    d_endo = distance_transform_edt(~ctx.endo_mask, sampling=sampling)
    d_epi = distance_transform_edt(ctx.epi_mask, sampling=sampling)
    field = np.full(ctx.myo_mask.shape, np.nan)
    field[ctx.myo_mask] = (d_endo + d_epi)[ctx.myo_mask]
    values = _sample_at_vertices(field, ctx, verts, normals * (0.75 * ctx.pitch))
    return ThicknessResult("EDT boundary sum", "Volumetric baseline",
                           values.astype(np.float32), time.perf_counter() - t0, {})


def _ball_structure(radius: int) -> np.ndarray:
    r = int(radius)
    ax = np.arange(-r, r + 1)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    return (gx ** 2 + gy ** 2 + gz ** 2) <= r ** 2 + 1e-9


def method_sphere_propagation(ctx: VolumeContext, verts: np.ndarray,
                              normals: np.ndarray, se_radius: int = 2,
                              max_iter: int = 120) -> ThicknessResult:
    """Morphological sphere propagation.

    Endocardial dilation and epicardial contraction with a spherical structuring
    element of radius ``se_radius`` voxels; the iteration index at which a
    myocardial voxel is first reached, times ``se_radius * pitch``, gives its
    distance to that boundary. A radius larger than one voxel is used because a
    6- or 26-neighbour element implements the city-block / Chebyshev metric and
    biases the result by more than 10 %.
    """
    t0 = time.perf_counter()
    myo = ctx.myo_mask
    struct = _ball_structure(se_radius)
    step_mm = se_radius * ctx.pitch
    n_endo = np.full(myo.shape, np.inf)
    n_epi = np.full(myo.shape, np.inf)

    for start_mask, store in ((ctx.endo_mask, n_endo), (~ctx.epi_mask, n_epi)):
        front = start_mask.copy()
        for k in range(1, max_iter + 1):
            grown = binary_dilation(front, structure=struct)
            new = grown & ~front & myo
            if not new.any():
                break
            store[new] = k
            front = grown
            if np.isfinite(store[myo]).all():
                break

    field = np.full(myo.shape, np.nan)
    ok = myo & np.isfinite(n_endo) & np.isfinite(n_epi)
    # A voxel first reached at iteration k lies between (k-1) and k steps away.
    field[ok] = ((n_endo[ok] - 0.5) + (n_epi[ok] - 0.5)) * step_mm
    values = _sample_at_vertices(field, ctx, verts, normals * (0.75 * ctx.pitch))
    return ThicknessResult("Morphological sphere propagation", "Volumetric baseline",
                           values.astype(np.float32), time.perf_counter() - t0,
                           {"propagation_se_radius_vox": se_radius,
                            "propagation_covered_fraction": float(ok[myo].mean())})


# ──────────────────────────────────────────────────────────────────────────
# Mesh-native estimators
# ──────────────────────────────────────────────────────────────────────────
def _directed_correspondence(src: trimesh.Trimesh, dst: trimesh.Trimesh,
                             src_normals: np.ndarray, k_candidates: int,
                             lambda_normal: float, lambda_tangential: float,
                             smooth_iters: int, smooth_lambda: float):
    """Regularised transmural matching of ``src`` vertices onto the ``dst`` surface.

    Cost per candidate pair (p_i, q_j):

        C = ||p_i - q_j||^2
            + lambda_n * (1 - |n_i . n_j|) * s^2
            + lambda_t * ||(I - n_i n_i^T)(q_j - p_i)||^2

    with ``s`` the median candidate distance. The winning displacement field is
    then Laplacian-smoothed over the source mesh graph (this is the
    regularisation that removes the tangential jumps of plain nearest-neighbour
    matching) and the smoothed target is projected back onto ``dst`` so that the
    smoothing cannot shrink the correspondence.
    """
    P = np.asarray(src.vertices, np.float64)
    Q = np.asarray(dst.vertices, np.float64)
    dst_normals = np.array(dst.vertex_normals, dtype=np.float64, copy=True)
    dst_normals /= np.clip(np.linalg.norm(dst_normals, axis=1, keepdims=True), 1e-12, None)

    k = min(k_candidates, len(Q))
    dist, cand = cKDTree(Q).query(P, k=k, workers=-1)
    dist = np.atleast_2d(dist)
    cand = np.atleast_2d(cand)
    s2 = float(np.median(dist) ** 2) + 1e-9

    diff = Q[cand] - P[:, None, :]
    normal_align = np.abs(np.einsum("nkd,nd->nk", dst_normals[cand], src_normals))
    along = np.einsum("nkd,nd->nk", diff, src_normals)
    tangential = np.sum(diff ** 2, axis=2) - along ** 2

    cost = (dist ** 2
            + lambda_normal * (1.0 - normal_align) * s2
            + lambda_tangential * tangential)
    cost[along <= 0] = np.inf                              # must point across the wall
    best = np.argmin(cost, axis=1)
    ok = np.isfinite(np.take_along_axis(cost, best[:, None], 1).ravel())
    displacement = Q[cand[np.arange(len(P)), best]] - P

    neighbours = src.vertex_neighbors
    for _ in range(smooth_iters):
        avg = np.array([displacement[nb].mean(0) if len(nb) else displacement[i]
                        for i, nb in enumerate(neighbours)])
        displacement = (1.0 - smooth_lambda) * displacement + smooth_lambda * avg

    target = P + displacement
    try:
        projected, _, _ = trimesh.proximity.closest_point(dst, target)
    except Exception:
        _, nn = cKDTree(Q).query(target, workers=-1)
        projected = Q[nn]

    values = np.linalg.norm(projected - P, axis=1)
    values[~ok] = np.nan
    return values, projected, ok


def method_surface_correspondence(endo: trimesh.Trimesh, epi: trimesh.Trimesh,
                                  normals: np.ndarray, k_candidates: int = 48,
                                  lambda_normal: float = 1.0,
                                  lambda_tangential: float = 0.6,
                                  smooth_iters: int = 4,
                                  smooth_lambda: float = 0.4) -> ThicknessResult:
    """Symmetric endo<->epi correspondence with normal compatibility.

    The correspondence is computed in both directions; the reported thickness is
    the mean of the two path lengths at each matched pair, and the round-trip
    displacement is reported as an explicit consistency diagnostic.
    """
    t0 = time.perf_counter()
    P = np.asarray(endo.vertices, np.float64)
    epi_normals = np.array(epi.vertex_normals, dtype=np.float64, copy=True)
    epi_normals /= np.clip(np.linalg.norm(epi_normals, axis=1, keepdims=True), 1e-12, None)

    fwd, fwd_target, fwd_ok = _directed_correspondence(
        endo, epi, normals, k_candidates, lambda_normal, lambda_tangential,
        smooth_iters, smooth_lambda)
    bwd, bwd_target, bwd_ok = _directed_correspondence(
        epi, endo, -epi_normals, k_candidates, lambda_normal, lambda_tangential,
        smooth_iters, smooth_lambda)

    # Transport the backward estimate to the endocardial vertices via the
    # forward correspondence endpoint.
    _, nn = cKDTree(np.asarray(epi.vertices, np.float64)).query(fwd_target, workers=-1)
    bwd_at_endo = bwd[nn]
    round_trip = np.linalg.norm(bwd_target[nn] - P, axis=1)

    both = fwd_ok & np.isfinite(fwd) & np.isfinite(bwd_at_endo)
    values = np.where(both, 0.5 * (fwd + bwd_at_endo), fwd)

    diagnostics = {
        "correspondence_round_trip_median_mm": float(np.nanmedian(round_trip)),
        "correspondence_round_trip_p95_mm": float(np.nanpercentile(round_trip, 95)),
        "correspondence_outward_fraction": float(fwd_ok.mean()),
        "correspondence_symmetric_fraction": float(both.mean()),
    }
    return ThicknessResult("Symmetric surface correspondence", "Mesh-native",
                           values.astype(np.float32), time.perf_counter() - t0,
                           diagnostics)


def _cone_directions(normal: np.ndarray, k: int, alpha_deg: float) -> np.ndarray:
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    alpha = np.radians(alpha_deg)
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(normal, u)
    phi = np.linspace(0.0, 2 * np.pi, k, endpoint=False)
    dirs = (normal * np.cos(alpha)
            + np.cos(phi)[:, None] * u * np.sin(alpha)
            + np.sin(phi)[:, None] * v * np.sin(alpha))
    dirs /= np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12, None)
    return np.vstack([normal[None, :], dirs])


def method_cone_rays(endo: trimesh.Trimesh, epi: trimesh.Trimesh, normals: np.ndarray,
                     k: int = 7, alpha_deg: float = 30.0,
                     sample_limit: int = 4000) -> ThicknessResult:
    """Ray casting inside a cone around the endocardial normal (median hit)."""
    t0 = time.perf_counter()
    P = np.asarray(endo.vertices, np.float64)
    n = len(P)
    sample = (np.arange(n) if n <= sample_limit
              else np.unique(np.linspace(0, n - 1, sample_limit).astype(np.int64)))

    origins, dirs, owner = [], [], []
    for local, i in enumerate(sample):
        d = _cone_directions(normals[i], k, alpha_deg)
        origins.append(np.repeat(P[i][None, :], len(d), axis=0))
        dirs.append(d)
        owner.extend([local] * len(d))
    origins = np.vstack(origins)
    dirs = np.vstack(dirs)
    owner = np.asarray(owner)

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(epi)
    hit_dist = np.full(len(origins), np.nan)
    # Small chunks: without embree, trimesh builds a per-ray candidate list, so
    # the peak memory of a batch grows with rays x candidate triangles.
    chunk = 512
    for start in range(0, len(origins), chunk):
        stop = min(start + chunk, len(origins))
        locs, ray_idx, _ = intersector.intersects_location(
            ray_origins=origins[start:stop] + 1e-6 * dirs[start:stop],
            ray_directions=dirs[start:stop], multiple_hits=False)
        if not len(locs):
            continue
        d = np.linalg.norm(locs - origins[start:stop][ray_idx], axis=1)
        for value, local_ray in zip(d, ray_idx):
            g = start + int(local_ray)
            if value > 1e-6 and (not np.isfinite(hit_dist[g]) or value < hit_dist[g]):
                hit_dist[g] = value

    sample_values = np.full(len(sample), np.nan)
    for local in range(len(sample)):
        vals = hit_dist[owner == local]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            sample_values[local] = float(np.median(vals))

    values = np.full(n, np.nan)
    values[sample] = sample_values
    if n > len(sample):                                   # interpolate the skipped ones
        known = sample[np.isfinite(sample_values)]
        if known.size:
            _, nn = cKDTree(P[known]).query(P, workers=-1)
            fill = values[known][nn]
            values = np.where(np.isfinite(values), values, fill)
    return ThicknessResult("SDF cone rays", "Ray casting",
                           values.astype(np.float32), time.perf_counter() - t0,
                           {"cone_sampled_vertices": int(len(sample)),
                            "cone_hit_fraction": float(np.isfinite(sample_values).mean())})


def method_decoder_offset(net, latent, verts_mm: np.ndarray, centroid, scale) -> ThicknessResult:
    """Model-native analytic prediction: the decoder offset delta, in millimetres.

    ``delta = f_endo - f_epi`` is what the network itself predicts for the wall;
    it is reported as a predicted quantity, not as a geometric measurement.
    """
    from cardiosdf_model import query_points
    from geometry import normalise

    t0 = time.perf_counter()
    _, _, delta = query_points(net, latent, normalise(verts_mm, centroid, scale))
    return ThicknessResult("Decoder offset delta", "Model-native",
                           (delta * float(scale)).astype(np.float32),
                           time.perf_counter() - t0, {})
