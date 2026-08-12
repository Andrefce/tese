"""CardioSDF/INR checkpoint: architecture, loading and dense SDF inference.

The webapp package ``core.sdf_model`` is not part of this repository, so the
network is re-declared here exactly as it was trained (``notebooks/training.ipynb``)
and the checkpoint ``cfg`` block drives every hyper-parameter. Nothing is
guessed: the state dict is loaded with ``strict=True``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Apex/base orientation flip used by the training data pipeline
# (training convention: apex at low z, base at high z in normalised space).
FLIP_Z = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FourierPE(nn.Module):
    """Fourier positional encoding for 3D coordinates."""

    def __init__(self, L: int = 6):
        super().__init__()
        self.L = L
        freqs = 2.0 ** torch.arange(L).float() * math.pi
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return 3 + 6 * self.L

    def forward(self, xyz):
        x = xyz.unsqueeze(-1) * self.freqs
        return torch.cat([xyz, torch.sin(x).flatten(-2), torch.cos(x).flatten(-2)], dim=-1)


class PointNetEncoder(nn.Module):
    """Mask-aware PointNet encoder with per-tissue max-pool."""

    def __init__(self, input_dim: int = 4, latent_dim: int = 256):
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
        z_endo = f.masked_fill(~endo_mask.unsqueeze(-1), neg_inf).max(dim=1).values
        z_epi = f.masked_fill(~epi_mask.unsqueeze(-1), neg_inf).max(dim=1).values
        z_global = f.masked_fill(~mask.unsqueeze(-1), neg_inf).max(dim=1).values
        has_endo = endo_mask.any(dim=1, keepdim=True).float()
        has_epi = epi_mask.any(dim=1, keepdim=True).float()
        z_endo = z_endo * has_endo + z_global * (1.0 - has_endo)
        z_epi = z_epi * has_epi + z_global * (1.0 - has_epi)
        return self.proj(torch.cat([z_endo, z_epi], dim=-1))


class INRDecoderSDF(nn.Module):
    """Monotone-epi decoder: outputs (f_endo, f_epi = f_endo - delta, delta)."""

    def __init__(self, latent_dim=256, fourier_L=6, hidden=512, n_layers=8,
                 skip_layer=4, r0=0.5, delta_init_norm=0.28, delta_cap=None,
                 activation="relu", softplus_beta=100.0):
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
        self.layers = nn.ModuleList(
            nn.Linear(hidden + (in_dim if li == skip_layer else 0), hidden)
            for li in range(n_layers)
        )
        self.head_endo = nn.Linear(hidden, 1)
        self.head_delta = nn.Linear(hidden, 1)

    def forward(self, z, fxyz):
        B, N, _ = fxyz.shape
        h_in = torch.cat([z.unsqueeze(1).expand(B, N, -1), fxyz], dim=-1)
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
    """Inference-only wrapper (the training losses live in the notebook)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder = PointNetEncoder(input_dim=cfg["input_dim"],
                                       latent_dim=cfg["latent_dim"])
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
        )

    def encode(self, contour, mask):
        return self.encoder(contour, mask)

    def decode(self, z, query_xyz):
        return self.decoder(z, self.fourier(query_xyz))


def load_model(ckpt_path: Path, device: torch.device = DEVICE):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = dict(ckpt["cfg"])
    net = SDFNetwork(cfg).to(device)
    net.load_state_dict(ckpt["model_state"], strict=True)
    net.eval()
    meta = {"epoch": int(ckpt.get("epoch", -1)),
            "val_loss": float(ckpt.get("val_loss", float("nan")))}
    return net, cfg, meta


def build_contour_tensor(contour_xyz, tissue_labels, cfg, phase_val, device=DEVICE):
    cont = np.column_stack([contour_xyz, tissue_labels]).astype(np.float32)
    if phase_val is not None and cfg["input_dim"] == 5:
        cont = np.column_stack([cont, np.full((len(cont), 1), phase_val, np.float32)])
    cont_t = torch.from_numpy(cont).unsqueeze(0).to(device)
    mask_t = torch.ones(1, len(cont), dtype=torch.bool, device=device)
    return cont_t, mask_t


@torch.no_grad()
def encode_contours(net, contour_xyz, tissue_labels, cfg, phase_val=0.0, device=DEVICE):
    cont_t, mask_t = build_contour_tensor(contour_xyz, tissue_labels, cfg, phase_val, device)
    return net.encode(cont_t, mask_t)


@torch.no_grad()
def query_points(net, z, pts_norm, batch=65536, device=DEVICE):
    """Evaluate (f_endo, f_epi, delta) at arbitrary normalised-space points."""
    pts_norm = np.ascontiguousarray(pts_norm, dtype=np.float32)
    fe = np.empty(len(pts_norm), np.float32)
    fp = np.empty(len(pts_norm), np.float32)
    dl = np.empty(len(pts_norm), np.float32)
    for s in range(0, len(pts_norm), batch):
        chunk = torch.from_numpy(pts_norm[s:s + batch]).unsqueeze(0).to(device)
        a, b, c = net.decode(z, chunk)
        fe[s:s + batch] = a[0].float().cpu().numpy()
        fp[s:s + batch] = b[0].float().cpu().numpy()
        dl[s:s + batch] = c[0].float().cpu().numpy()
    return fe, fp, dl


@torch.no_grad()
def dense_sdf_grid(net, z, contour_xyz, cfg, grid_res=96, batch=65536, device=DEVICE):
    """Dense SDF query on the padded contour bounding box (normalised space)."""
    lo = contour_xyz.min(0) - cfg["bbox_pad"]
    hi = contour_xyz.max(0) + cfg["bbox_pad"]
    axes = [np.linspace(lo[d], hi[d], grid_res, dtype=np.float32) for d in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1).astype(np.float32)
    fe, fp, dl = query_points(net, z, pts, batch=batch, device=device)
    shape = (grid_res,) * 3
    voxel = (hi - lo) / (grid_res - 1)
    return (fe.reshape(shape), fp.reshape(shape), dl.reshape(shape),
            lo.astype(np.float32), hi.astype(np.float32), voxel.astype(np.float32))


def slice_residual_mm(net, z, contour_xyz, tissue_labels, scale_mm, device=DEVICE):
    """Mean |SDF| on the input contour points, in millimetres."""
    fe, fp, _ = query_points(net, z, contour_xyz, device=device)
    endo_m = tissue_labels == 0
    epi_m = tissue_labels == 1
    num = 0.0
    den = 0
    if endo_m.any():
        num += float(np.abs(fe[endo_m]).sum())
        den += int(endo_m.sum())
    if epi_m.any():
        num += float(np.abs(fp[epi_m]).sum())
        den += int(epi_m.sum())
    return float(num / max(den, 1) * scale_mm)
