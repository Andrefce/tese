"""
Simple SDF cardiac model demo.
Loads a patient segmentation, runs neural SDF inference, and plots the 3D mesh.

Usage:
    python run_demo.py
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from skimage.measure import find_contours, marching_cubes
import plotly.graph_objects as go


# ─── Config ───────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model" / "inr_sdf_combined_fresh_ed_mix_v1_final.ptrom"
# Use patient001 ED segmentation
DATA_DIR = Path(__file__).parent / "data" / "patient001"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PTS_PER_RING = 60
LBL_BG, LBL_RV, LBL_MYO, LBL_LV = 0, 1, 2, 3
GRID_RES = 96


# ═══════════════════════════════════════════════════════════════════
# Model Architecture (same as training)
# ═══════════════════════════════════════════════════════════════════

class FourierPE(nn.Module):
    def __init__(self, L=6):
        super().__init__()
        self.L = L
        freqs = 2.0 ** torch.arange(L).float() * math.pi
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self):
        return 3 + 6 * self.L

    def forward(self, xyz):
        x = xyz.unsqueeze(-1) * self.freqs
        return torch.cat([xyz, torch.sin(x).flatten(-2), torch.cos(x).flatten(-2)], dim=-1)


class PointNetEncoder(nn.Module):
    def __init__(self, input_dim=5, latent_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )
        self.proj = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x, mask):
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
    def __init__(self, latent_dim=256, fourier_L=6, hidden=512, n_layers=8,
                 skip_layer=4, delta_cap=None, tau_min=0.28,
                 activation="relu", softplus_beta=100.0):
        super().__init__()
        self.skip_layer = skip_layer
        self.tau_min = float(tau_min)
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

    def forward(self, z, fxyz):
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
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PointNetEncoder(input_dim=cfg["input_dim"], latent_dim=cfg["latent_dim"])
        self.fourier = FourierPE(L=cfg["fourier_L"])
        self.decoder = INRDecoderSDF(
            latent_dim=cfg["latent_dim"],
            fourier_L=cfg["fourier_L"],
            hidden=cfg["decoder_hidden"],
            n_layers=cfg["decoder_layers"],
            skip_layer=cfg["skip_layer"],
            delta_cap=cfg.get("delta_cap_norm"),
            tau_min=cfg["tau_min_norm"],
            activation=cfg.get("decoder_activation", "relu"),
            softplus_beta=cfg.get("decoder_softplus_beta", 100.0),
        )

    def encode(self, contour, mask):
        return self.encoder(contour, mask)

    def decode(self, z, query_xyz):
        return self.decoder(z, self.fourier(query_xyz))


# ═══════════════════════════════════════════════════════════════════
# Load model
# ═══════════════════════════════════════════════════════════════════

def load_model(path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = SDFNetwork(cfg)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE).eval()
    return model, cfg


# ═══════════════════════════════════════════════════════════════════
# Contour extraction from segmentation
# ═══════════════════════════════════════════════════════════════════

def _sample_contour(pts2d, n=N_PTS_PER_RING):
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
    t_new = np.linspace(0.0, total, n, endpoint=False)
    rows = np.interp(t_new, t_old, pts2d[:, 0])
    cols = np.interp(t_new, t_old, pts2d[:, 1])
    return np.column_stack([rows, cols]).astype(np.float32)


def _contour_area(pts):
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def extract_contours(seg_data, affine, dz):
    """Extract endo+epi contour rings from a 3D segmentation."""
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
            pts.append(np.column_stack([xyz, np.full(len(ring), tissue_id, dtype=np.float32)]))

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
    xyz_n[:, 2] = -xyz_n[:, 2]  # FLIP_Z

    return xyz_n, tissue.astype(np.float32), centroid, scale


# ═══════════════════════════════════════════════════════════════════
# SDF Inference
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(model, cfg, contour_xyz, tissue_labels, phase_val=0.0):
    """Run SDF inference → returns endo and epi meshes (vertices, faces)."""
    # Build input tensor
    cont = np.column_stack([contour_xyz, tissue_labels]).astype(np.float32)
    if cfg.get("input_dim", 4) == 5:
        cont = np.column_stack([cont, np.full((len(cont), 1), phase_val, dtype=np.float32)])
    cont_t = torch.from_numpy(cont).unsqueeze(0).to(DEVICE)
    mask_t = torch.ones(1, len(cont), dtype=torch.bool, device=DEVICE)

    # Encode
    z = model.encode(cont_t, mask_t)

    # Build 3D query grid
    bbox_pad = cfg.get("bbox_pad", 0.3)
    lo = contour_xyz.min(0) - bbox_pad
    hi = contour_xyz.max(0) + bbox_pad
    xs = np.linspace(lo[0], hi[0], GRID_RES)
    ys = np.linspace(lo[1], hi[1], GRID_RES)
    zs = np.linspace(lo[2], hi[2], GRID_RES)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float32)

    sdf_endo = np.empty(len(grid_pts), np.float32)
    sdf_epi = np.empty(len(grid_pts), np.float32)

    batch = 131072
    for s in range(0, len(grid_pts), batch):
        chunk = torch.from_numpy(grid_pts[s:s+batch]).unsqueeze(0).to(DEVICE)
        fe, fp, _ = model.decode(z, chunk)
        sdf_endo[s:s+batch] = fe[0].cpu().numpy()
        sdf_epi[s:s+batch] = fp[0].cpu().numpy()

    shape = (GRID_RES, GRID_RES, GRID_RES)
    voxel = (hi - lo) / (GRID_RES - 1)

    # Marching cubes
    iso = cfg.get("iso_level", 0.0)
    meshes = {}
    for name, field in [("endo", sdf_endo.reshape(shape)), ("epi", sdf_epi.reshape(shape))]:
        if field.min() > iso or field.max() < iso:
            print(f"  Warning: no iso-surface for {name}")
            continue
        verts, faces, _, _ = marching_cubes(field, level=iso, spacing=tuple(voxel))
        verts = verts + lo
        meshes[name] = (verts.astype(np.float32), faces.astype(np.int32))

    return meshes


# ═══════════════════════════════════════════════════════════════════
# 3D Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_meshes(meshes, centroid, scale):
    """Plot endo and epi meshes using Plotly (styled like the webapp)."""
    fig = go.Figure()

    flip = np.array([1.0, 1.0, -1.0])  # undo FLIP_Z

    # Convert all meshes to mm and compute shared center
    meshes_mm = {}
    all_verts = []
    for name, (verts, faces) in meshes.items():
        verts_mm = verts * flip * scale + centroid
        meshes_mm[name] = (verts_mm, faces)
        all_verts.append(verts_mm)
    center = np.vstack(all_verts).mean(axis=0) if all_verts else np.zeros(3)

    for name, (verts_mm, faces) in meshes_mm.items():
        # Center at origin; swap Y and Z so Z points "up" (base→apex vertical)
        vc = verts_mm - center
        verts_centered = vc[:, [0, 2, 1]]  # x, z, y → makes apex point up

        if name == "endo":
            # Solid red endocardium with wall-thickness-style lighting
            fig.add_trace(go.Mesh3d(
                x=verts_centered[:, 0], y=verts_centered[:, 1], z=verts_centered[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="#e74c5e",
                opacity=1.0,
                flatshading=False,
                lighting=dict(ambient=0.4, diffuse=0.7, specular=0.3, roughness=0.3),
                lightposition=dict(x=50, y=80, z=60),
                name="Endocardium",
                showlegend=True,
            ))
        else:
            # Semi-transparent blue epicardium shell
            fig.add_trace(go.Mesh3d(
                x=verts_centered[:, 0], y=verts_centered[:, 1], z=verts_centered[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="#3b82f6",
                opacity=0.18,
                flatshading=False,
                lighting=dict(ambient=0.5, diffuse=0.6, specular=0.2, roughness=0.5),
                lightposition=dict(x=-40, y=30, z=-30),
                name="Epicardium",
                showlegend=True,
            ))

    fig.update_layout(
        title=dict(text="Cardiac SDF Model — 3D Mesh", font=dict(color="white", size=16)),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        scene=dict(
            aspectmode="data",
            bgcolor="#0d1117",
            xaxis=dict(showgrid=True, gridcolor="#1a2233", showbackground=False,
                       zerolinecolor="#1a2233", color="#666"),
            yaxis=dict(showgrid=True, gridcolor="#1a2233", showbackground=False,
                       zerolinecolor="#1a2233", color="#666"),
            zaxis=dict(showgrid=True, gridcolor="#1a2233", showbackground=False,
                       zerolinecolor="#1a2233", color="#666"),
            camera=dict(eye=dict(x=1.5, y=1.0, z=1.2)),
        ),
        legend=dict(font=dict(color="white")),
        width=900, height=700,
    )
    fig.write_html("output_mesh.html")
    print("Saved interactive 3D plot → output_mesh.html")
    fig.show()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def resolve_nifti(path):
    """Resolve ACDC-style folder that may contain a single .nii file inside."""
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        hits = sorted(p.glob("*.nii.gz")) + sorted(p.glob("*.nii"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No NIfTI file at {p}")


def main():
    print("=" * 60)
    print("  Cardiac SDF Model — Demo Inference")
    print("=" * 60)

    # 1. Load model
    print(f"\n[1/4] Loading model from {MODEL_PATH.name}...")
    model, cfg = load_model(MODEL_PATH)
    print(f"      Model loaded (device={DEVICE})")

    # 2. Load patient segmentation
    seg_path = DATA_DIR / "patient001_frame01_gt.nii"
    print(f"\n[2/4] Loading segmentation: {seg_path.name}")
    real_path = resolve_nifti(seg_path)
    img = nib.load(str(real_path))
    seg_data = np.asarray(img.get_fdata(dtype=np.float32))
    affine = np.asarray(img.affine, dtype=np.float32)
    dz = float(img.header.get_zooms()[2])
    print(f"      Shape: {seg_data.shape}, spacing-z: {dz:.2f} mm")

    # 3. Extract contours and run inference
    print("\n[3/4] Extracting contours & running SDF inference...")
    contour_xyz, tissue, centroid, scale = extract_contours(seg_data, affine, dz)
    print(f"      Contour points: {len(contour_xyz)}, scale: {scale:.2f}")
    meshes = run_inference(model, cfg, contour_xyz, tissue)
    for name, (v, f) in meshes.items():
        print(f"      {name}: {len(v)} vertices, {len(f)} faces")

    # 4. Plot
    print("\n[4/4] Generating 3D plot...")
    plot_meshes(meshes, centroid, scale)
    print("\nDone!")


if __name__ == "__main__":
    main()
