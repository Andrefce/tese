"""SDF cardiac model — architecture, loading, and inference pipeline.

Standalone copy for the inference API service. Functions that were previously
imported from core.inference (_mesh_payload, _reduce_mesh, _aha_17_segment_stats)
are inlined here so the API has no dependency on the webapp's inference module.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Labels ──
LBL_BG, LBL_RV, LBL_MYO, LBL_LV = 0, 1, 2, 3
N_PTS_PER_RING = 60
FLIP_Z = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════
# Model architecture
# ═══════════════════════════════════════════════════════════════════

class FourierPE(nn.Module):
    def __init__(self, L: int = 6):
        super().__init__()
        self.L = L
        freqs = 2.0 ** torch.arange(L).float() * math.pi
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return 3 + 6 * self.L

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        x = xyz.unsqueeze(-1) * self.freqs
        return torch.cat([xyz, torch.sin(x).flatten(-2), torch.cos(x).flatten(-2)], dim=-1)


class PointNetEncoder(nn.Module):
    def __init__(self, input_dim: int = 5, latent_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )
        self.proj = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        f = self.mlp(x)
        neg_inf = torch.finfo(f.dtype).min
        tissue = x[:, :, 3]
        endo_mask = mask & (tissue < 0.5)
        epi_mask = mask & (tissue >= 0.5)

        f_endo = f.masked_fill(~endo_mask.unsqueeze(-1), neg_inf)
        f_epi = f.masked_fill(~epi_mask.unsqueeze(-1), neg_inf)
        z_endo = f_endo.max(dim=1).values
        z_epi = f_epi.max(dim=1).values

        f_global = f.masked_fill(~mask.unsqueeze(-1), neg_inf)
        z_global = f_global.max(dim=1).values
        has_endo = endo_mask.any(dim=1, keepdim=True).float()
        has_epi = epi_mask.any(dim=1, keepdim=True).float()
        z_endo = z_endo * has_endo + z_global * (1.0 - has_endo)
        z_epi = z_epi * has_epi + z_global * (1.0 - has_epi)

        return self.proj(torch.cat([z_endo, z_epi], dim=-1))


class INRDecoderSDF(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        fourier_L: int = 6,
        hidden: int = 512,
        n_layers: int = 8,
        skip_layer: int = 4,
        r0: float = 0.5,
        delta_init_norm: float = 0.28,
        delta_cap: float | None = None,
        activation: str = "relu",
        softplus_beta: float = 100.0,
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.skip_layer = skip_layer
        self.tau_min = float(delta_init_norm)
        self.delta_cap = None if delta_cap is None else float(delta_cap)
        if activation == "softplus":
            beta = float(softplus_beta)
            self._act = lambda x: F.softplus(x, beta=beta, threshold=20.0)
        else:
            self._act = lambda x: F.relu(x, inplace=False)

        in_dim = latent_dim + 3 + 6 * fourier_L
        self.in_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList()
        for li in range(n_layers):
            d_in = hidden + (in_dim if li == skip_layer else 0)
            self.layers.append(nn.Linear(d_in, hidden))
        self.head_endo = nn.Linear(hidden, 1)
        self.head_delta = nn.Linear(hidden, 1)

    def forward(
        self, z: torch.Tensor, fxyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = fxyz.shape
        z_exp = z.unsqueeze(1).expand(B, N, -1)
        h_in = torch.cat([z_exp, fxyz], dim=-1)
        h = self._act(self.in_proj(h_in))
        for li, lyr in enumerate(self.layers):
            if li == self.skip_layer:
                h = torch.cat([h, h_in], dim=-1)
            h = self._act(lyr(h))
        f_endo = self.head_endo(h).squeeze(-1)
        raw_d = self.head_delta(h).squeeze(-1)
        if self.delta_cap is None:
            delta = F.softplus(raw_d) + 1e-4
        else:
            delta = self.tau_min + (self.delta_cap - self.tau_min) * torch.sigmoid(raw_d)
        return f_endo, f_endo - delta, delta


class SDFNetwork(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder = PointNetEncoder(
            input_dim=cfg["input_dim"], latent_dim=cfg["latent_dim"]
        )
        self.fourier = FourierPE(L=cfg["fourier_L"])
        self.decoder = INRDecoderSDF(
            latent_dim=cfg["latent_dim"],
            fourier_L=cfg["fourier_L"],
            hidden=cfg["decoder_hidden"],
            n_layers=cfg["decoder_layers"],
            skip_layer=cfg["skip_layer"],
            r0=cfg["sphere_r0"],
            delta_init_norm=cfg["tau_min_norm"],
            delta_cap=cfg.get("delta_cap_norm"),
            activation=cfg.get("decoder_activation", "relu"),
            softplus_beta=cfg.get("decoder_softplus_beta", 100.0),
            spectral_norm=cfg.get("decoder_spectral_norm", False),
        )

    def encode(self, contour: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(contour, mask)

    def decode(
        self, z: torch.Tensor, query_xyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.decoder(z, self.fourier(query_xyz))


# ═══════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str | Path) -> tuple[SDFNetwork, dict]:
    """Load checkpoint → (model on DEVICE in eval mode, cfg dict)."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = SDFNetwork(cfg)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE).eval()
    return model, cfg


# ═══════════════════════════════════════════════════════════════════
# Mesh helpers (inlined from inference.py for standalone use)
# ═══════════════════════════════════════════════════════════════════

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


# AHA 17-segment names
AHA_17_NAMES = [
    "Basal Anterior", "Basal Anteroseptal", "Basal Inferoseptal",
    "Basal Inferior", "Basal Inferolateral", "Basal Anterolateral",
    "Mid Anterior", "Mid Anteroseptal", "Mid Inferoseptal",
    "Mid Inferior", "Mid Inferolateral", "Mid Anterolateral",
    "Apical Anterior", "Apical Septal", "Apical Inferior", "Apical Lateral",
    "Apex",
]


def _thickness_status(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "unavailable"
    if value < 5.0:
        return "thin"
    if value > 13.0:
        return "thick"
    return "typical"


def _aha_17_segment_stats(
    endo_vertices: np.ndarray,
    thickness: np.ndarray | None,
    seg_volume: np.ndarray | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> list[dict[str, Any]]:
    empty = [{"id": i + 1, "name": n, "meanMm": None, "status": "unavailable"}
             for i, n in enumerate(AHA_17_NAMES)]
    if thickness is None or len(thickness) != len(endo_vertices) or len(endo_vertices) == 0:
        return empty

    z = endo_vertices[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    zrange = zmax - zmin
    if zrange < 1e-6:
        return empty

    cx = float(endo_vertices[:, 0].mean())
    cy = float(endo_vertices[:, 1].mean())

    anterior_angle = 0.0
    if seg_volume is not None and spacing is not None:
        labels = np.rint(seg_volume).astype(np.int16)
        rv_mask = labels == LBL_RV
        if rv_mask.any():
            rv_pts = np.argwhere(rv_mask).astype(np.float32) * np.asarray(spacing, dtype=np.float32)
            rv_cx = float(rv_pts[:, 0].mean())
            rv_cy = float(rv_pts[:, 1].mean())
            septal_angle = math.atan2(rv_cy - cy, rv_cx - cx)
            anterior_angle = septal_angle - math.pi / 2.0

    raw_angles = np.arctan2(endo_vertices[:, 1] - cy, endo_vertices[:, 0] - cx)
    angles_deg = np.degrees(raw_angles - anterior_angle) % 360.0

    znorm = (z - zmin) / zrange

    segment_ids = np.full(len(endo_vertices), 16, dtype=np.int32)

    basal = znorm < (1.0 / 3.0)
    mid = (znorm >= (1.0 / 3.0)) & (znorm < (2.0 / 3.0))
    apical = (znorm >= (2.0 / 3.0)) & (znorm < 0.85)

    seg6 = (angles_deg / 60.0).astype(np.int32) % 6
    segment_ids[basal] = seg6[basal]
    segment_ids[mid] = 6 + seg6[mid]

    seg4 = (angles_deg / 90.0).astype(np.int32) % 4
    segment_ids[apical] = 12 + seg4[apical]

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


# ═══════════════════════════════════════════════════════════════════
# Contour extraction from ACDC segmentation
# ═══════════════════════════════════════════════════════════════════

def _contour_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _sample_contour(pts2d: np.ndarray, n: int = N_PTS_PER_RING) -> np.ndarray:
    pts2d = np.asarray(pts2d, dtype=np.float32)
    if len(pts2d) < 3:
        return pts2d
    if np.linalg.norm(pts2d[0] - pts2d[-1]) > 1e-6:
        pts2d = np.vstack([pts2d, pts2d[0]])
    d = np.sqrt((np.diff(pts2d, axis=0) ** 2).sum(axis=1))
    total = float(d.sum())
    if total < 1e-6:
        return pts2d[:-1]
    t_old = np.concatenate([[0.0], np.cumsum(d)])
    t_new = np.linspace(0.0, total, int(n), endpoint=False)
    rows = np.interp(t_new, t_old, pts2d[:, 0])
    cols = np.interp(t_new, t_old, pts2d[:, 1])
    return np.column_stack([rows, cols]).astype(np.float32)


def extract_contours(
    seg_data: np.ndarray,
    affine: np.ndarray,
    dz: float,
) -> dict[str, Any]:
    from skimage.measure import find_contours

    if seg_data.ndim == 4:
        seg_data = seg_data[..., 0]
    data = np.rint(seg_data).astype(np.int32)

    support = sorted(
        s for s in range(data.shape[2])
        if (data[:, :, s] == LBL_LV).any() or (data[:, :, s] == LBL_MYO).any()
    )
    if not support:
        raise ValueError("No LV/MYO labels found in segmentation.")

    def vox2world(rows, cols, s):
        vox = np.column_stack([cols, rows, np.zeros(len(rows)), np.ones(len(rows))])
        world = (affine @ vox.T).T
        world[:, 2] = s * dz
        return world[:, :3]

    pts = []
    for s in support:
        seg = data[:, :, s]
        for label_mask, tissue_id in [
            (seg == LBL_LV, 0.0),
            ((seg == LBL_MYO) | (seg == LBL_LV), 1.0),
        ]:
            mask = label_mask.astype(np.uint8)
            if mask.sum() <= 10:
                continue
            contours = find_contours(mask, 0.5)
            if not contours:
                continue
            ring = _sample_contour(max(contours, key=_contour_area))
            xyz = vox2world(ring[:, 0], ring[:, 1], s)
            pts.append(
                np.column_stack(
                    [xyz, np.full(len(ring), tissue_id, dtype=np.float32)]
                )
            )

    if not pts:
        raise ValueError("Could not extract any contours from segmentation.")

    raw = np.vstack(pts).astype(np.float32)
    xyz, tissue = raw[:, :3], raw[:, 3]
    centroid = xyz.mean(axis=0)
    xyz_c = xyz - centroid
    scale = float(np.linalg.norm(xyz_c[:, :2], axis=1).mean())
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(xyz_c) + 1e-6)
    xyz_n = (xyz_c / scale).astype(np.float32)
    if FLIP_Z:
        xyz_n[:, 2] = -xyz_n[:, 2]

    return {
        "xyz": xyz_n,
        "tissue": tissue.astype(np.float32),
        "centroid": centroid.astype(np.float32),
        "scale": float(scale),
    }


# ═══════════════════════════════════════════════════════════════════
# SDF Inference pipeline
# ═══════════════════════════════════════════════════════════════════

def _build_contour_tensor(
    contour_xyz: np.ndarray,
    tissue_labels: np.ndarray,
    cfg: dict,
    phase_val: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cont = np.column_stack([contour_xyz, tissue_labels]).astype(np.float32)
    if phase_val is not None and cfg.get("input_dim", 4) == 5:
        cont = np.column_stack(
            [cont, np.full((len(cont), 1), phase_val, dtype=np.float32)]
        )
    cont_t = torch.from_numpy(cont).unsqueeze(0).to(DEVICE)
    mask_t = torch.ones(1, len(cont), dtype=torch.bool, device=DEVICE)
    return cont_t, mask_t


def _build_grid_and_query(
    z: torch.Tensor,
    model: SDFNetwork,
    contour_xyz: np.ndarray,
    cfg: dict,
    grid_res: int,
    batch_query: int = 131072,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bbox_pad = cfg.get("bbox_pad", 0.3)
    lo = contour_xyz.min(0) - bbox_pad
    hi = contour_xyz.max(0) + bbox_pad
    xs = np.linspace(lo[0], hi[0], grid_res)
    ys = np.linspace(lo[1], hi[1], grid_res)
    zs = np.linspace(lo[2], hi[2], grid_res)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float32)

    sdf_e = np.empty(len(grid_pts), np.float32)
    sdf_p = np.empty(len(grid_pts), np.float32)
    dlt = np.empty(len(grid_pts), np.float32)

    with torch.no_grad():
        for s in range(0, len(grid_pts), batch_query):
            chunk = torch.from_numpy(grid_pts[s : s + batch_query]).unsqueeze(0).to(DEVICE)
            fe, fp, dl = model.decode(z, chunk)
            sdf_e[s : s + batch_query] = fe[0].float().cpu().numpy()
            sdf_p[s : s + batch_query] = fp[0].float().cpu().numpy()
            dlt[s : s + batch_query] = dl[0].float().cpu().numpy()

    shape = (grid_res, grid_res, grid_res)
    voxel = (hi - lo) / (grid_res - 1)
    return sdf_e.reshape(shape), sdf_p.reshape(shape), dlt.reshape(shape), lo, hi, voxel


def _snap_mesh_to_contours(
    vertices: np.ndarray,
    contour_xyz: np.ndarray,
    tissue_labels: np.ndarray,
    surface: str = "endo",
) -> np.ndarray:
    target_tissue = 0.0 if surface == "endo" else 1.0
    cmask = np.abs(tissue_labels - target_tissue) < 0.5
    cont = contour_xyz[cmask]

    if len(cont) < 4 or len(vertices) == 0:
        return vertices

    z_unique = np.unique(np.round(cont[:, 2], 5))
    if len(z_unique) < 2:
        return vertices

    ring_data = []
    for zc in np.sort(z_unique):
        rmask = np.abs(cont[:, 2] - zc) < 1e-4
        ring_xy = cont[rmask, :2]
        if len(ring_xy) < 4:
            continue
        ctr = ring_xy.mean(axis=0)
        dx = ring_xy[:, 0] - ctr[0]
        dy = ring_xy[:, 1] - ctr[1]
        ang = np.arctan2(dy, dx)
        rad = np.sqrt(dx ** 2 + dy ** 2)
        order = np.argsort(ang)
        ring_data.append((float(zc), ang[order], rad[order], ctr))

    if len(ring_data) < 2:
        return vertices

    n_rings = len(ring_data)
    ring_z = np.array([r[0] for r in ring_data])
    ring_centroids = np.array([r[3] for r in ring_data])

    verts = vertices.copy()
    vz = verts[:, 2]

    idx_hi = np.searchsorted(ring_z, vz).clip(1, n_rings - 1)
    idx_lo = idx_hi - 1
    z_lo = ring_z[idx_lo]
    z_hi = ring_z[idx_hi]
    dz = z_hi - z_lo
    dz[dz < 1e-10] = 1.0
    t = np.clip((vz - z_lo) / dz, 0.0, 1.0)

    c_interp = (ring_centroids[idx_lo] * (1.0 - t[:, None])
                + ring_centroids[idx_hi] * t[:, None])

    dxy = verts[:, :2] - c_interp
    v_angles = np.arctan2(dxy[:, 1], dxy[:, 0])
    v_radii = np.sqrt(dxy[:, 0] ** 2 + dxy[:, 1] ** 2)
    valid = v_radii > 1e-8

    target_r = v_radii.copy()
    for pi in range(n_rings - 1):
        group = valid & (idx_lo == pi)
        if not group.any():
            continue
        gi = np.where(group)[0]
        _, sa_lo, sr_lo, _ = ring_data[pi]
        _, sa_hi, sr_hi, _ = ring_data[pi + 1]
        r_lo = np.interp(v_angles[gi], sa_lo, sr_lo, period=2.0 * np.pi)
        r_hi = np.interp(v_angles[gi], sa_hi, sr_hi, period=2.0 * np.pi)
        target_r[gi] = r_lo * (1.0 - t[gi]) + r_hi * t[gi]

    dz_min = float(np.min(np.diff(ring_z))) if n_rings > 1 else 1.0
    fade = dz_min * 1.5
    blend = np.ones(len(verts), dtype=np.float32)
    below = vz < ring_z[0]
    above = vz > ring_z[-1]
    if below.any():
        blend[below] = np.clip(1.0 - (ring_z[0] - vz[below]) / fade, 0.0, 1.0)
    if above.any():
        blend[above] = np.clip(1.0 - (vz[above] - ring_z[-1]) / fade, 0.0, 1.0)

    raw_scale = np.ones(len(verts), dtype=np.float32)
    raw_scale[valid] = target_r[valid] / v_radii[valid]
    scale_f = 1.0 + blend * (raw_scale - 1.0)

    verts[:, 0] = c_interp[:, 0] + dxy[:, 0] * scale_f
    verts[:, 1] = c_interp[:, 1] + dxy[:, 1] * scale_f

    return verts


@torch.no_grad()
def predict_sdf_meshes(
    model: SDFNetwork,
    contour_xyz: np.ndarray,
    tissue_labels: np.ndarray,
    cfg: dict,
    grid_res: int | None = None,
    phase_val: float | None = 0.0,
    scale: float = 1.0,
    centroid: np.ndarray | None = None,
    seg_volume: np.ndarray | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    from skimage.measure import marching_cubes
    from scipy.spatial import cKDTree

    if grid_res is None:
        grid_res = cfg.get("grid_res", 96)

    cont_t, mask_t = _build_contour_tensor(contour_xyz, tissue_labels, cfg, phase_val)
    z = model.encode(cont_t, mask_t)
    sdf_e, sdf_p, dlt, lo, hi, voxel = _build_grid_and_query(
        z, model, contour_xyz, cfg, grid_res
    )

    iso = cfg.get("iso_level", 0.0)

    endo_verts, endo_faces = _mc_field(sdf_e, lo, voxel, iso)
    epi_verts, epi_faces = _mc_field(sdf_p, lo, voxel, iso)

    if len(endo_verts) > 0:
        endo_verts = _snap_mesh_to_contours(
            endo_verts, contour_xyz, tissue_labels, surface="endo"
        )
    if len(epi_verts) > 0:
        epi_verts = _snap_mesh_to_contours(
            epi_verts, contour_xyz, tissue_labels, surface="epi"
        )

    if centroid is None:
        centroid = np.zeros(3, dtype=np.float32)
    flip = np.array([1.0, 1.0, -1.0 if FLIP_Z else 1.0], dtype=np.float32)
    endo_verts_mm = (endo_verts * flip) * scale + centroid if len(endo_verts) > 0 else endo_verts
    epi_verts_mm = (epi_verts * flip) * scale + centroid if len(epi_verts) > 0 else epi_verts

    wall_values = None
    if len(endo_verts_mm) > 0 and len(epi_verts_mm) > 0:
        dists, _ = cKDTree(epi_verts_mm).query(endo_verts_mm, k=1, workers=-1)
        wall_values = np.asarray(dists, dtype=np.float32)
        wall_values = wall_values[np.isfinite(wall_values)] if wall_values is not None else None

    wall_for_mesh = None
    if wall_values is not None and len(wall_values) == len(endo_verts_mm):
        wall_for_mesh = wall_values

    wall_mean = float(np.mean(wall_values)) if wall_values is not None and wall_values.size else None
    wall_p95 = float(np.percentile(wall_values, 95)) if wall_values is not None and wall_values.size else None

    endo_area = _mesh_area_cm2(endo_verts_mm, endo_faces) if len(endo_verts_mm) > 0 else None
    epi_area = _mesh_area_cm2(epi_verts_mm, epi_faces) if len(epi_verts_mm) > 0 else None

    regional = _regional_wall_stats(endo_verts_mm, wall_for_mesh)

    metrics = {
        "meanWallThicknessMm": round(wall_mean, 2) if wall_mean is not None else None,
        "p95WallThicknessMm": round(wall_p95, 2) if wall_p95 is not None else None,
        "endoSurfaceAreaCm2": round(endo_area, 2) if endo_area is not None else None,
        "epiSurfaceAreaCm2": round(epi_area, 2) if epi_area is not None else None,
        "gridResolution": grid_res,
        "endoVertices": len(endo_verts_mm),
        "epiVertices": len(epi_verts_mm),
    }

    return {
        "source": "SDF neural model",
        "modelStatus": "SDF INR inference",
        "meshMethod": "sdf_marching_cubes",
        "metrics": metrics,
        "regionalThickness": regional,
        "aha17": _aha_17_segment_stats(endo_verts_mm, wall_for_mesh, seg_volume=seg_volume, spacing=spacing),
        "meshes": {
            "endo": _mesh_payload(endo_verts_mm, endo_faces, wall_for_mesh),
            "epi": _mesh_payload(epi_verts_mm, epi_faces, None, max_faces=30000),
        },
    }


def _mc_field(
    field: np.ndarray, lo: np.ndarray, voxel: np.ndarray, iso: float
) -> tuple[np.ndarray, np.ndarray]:
    from skimage.measure import marching_cubes

    field = np.asarray(field, dtype=np.float32)
    if float(field.min()) > iso or float(field.max()) < iso:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    try:
        verts, faces, _, _ = marching_cubes(field, level=iso, spacing=tuple(voxel))
        verts = verts + lo
        return verts.astype(np.float32), faces.astype(np.int32)
    except Exception:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)


def _mesh_area_cm2(vertices: np.ndarray, faces: np.ndarray) -> float | None:
    if vertices.size == 0 or faces.size == 0:
        return None
    tri = vertices[faces]
    area_mm2 = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    ).sum()
    return float(area_mm2 / 100.0)


def _regional_wall_stats(
    endo_vertices: np.ndarray, thickness: np.ndarray | None
) -> list[dict[str, Any]]:
    names = ["Basal", "Mid", "Apical"]
    if thickness is None or len(thickness) != len(endo_vertices) or len(endo_vertices) == 0:
        return [{"name": n, "meanMm": None, "status": "unavailable"} for n in names]

    z = endo_vertices[:, 2]
    q1, q2 = np.quantile(z, (1 / 3, 2 / 3))
    masks = [z >= q2, (z >= q1) & (z < q2), z < q1]
    out = []
    for name, mask in zip(names, masks):
        vals = thickness[mask]
        vals = vals[np.isfinite(vals)]
        mean_v = float(vals.mean()) if vals.size else None
        out.append({
            "name": name,
            "meanMm": round(mean_v, 2) if mean_v is not None else None,
            "p95Mm": round(float(np.percentile(vals, 95)), 2) if vals.size else None,
            "status": _thickness_status(mean_v),
        })
    return out
