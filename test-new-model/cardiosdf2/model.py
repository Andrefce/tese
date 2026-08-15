"""CardioSDF v2 — global latent + local contour feature volume.

The measured root cause of the reported failures (upgrade plan §1.3) is the
single 256-d global bottleneck: only 64 % of the NOR->HCM wall contrast
survives the encode/decode round trip, and the deficit scales with the true
thickness. This module keeps the trained backbone bit-for-bit and adds a
convolutional-occupancy-style *local* conditioning path:

* ``LocalContourVolume`` scatters per-contour-point features into a coarse 3D
  grid, smooths it with a small 3D CNN and returns trilinearly interpolated
  features at any query point.
* Those features are injected into the decoder trunk (input projection and
  skip layer) and into the wall (delta) head through **zero-initialised**
  projections, so an untrained v2 reproduces the baseline exactly and the whole
  baseline checkpoint loads without surgery.

Because the injection starts at zero, a short fine-tune only has to learn what
the global code cannot carry, which is the point of the upgrade.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from cardiosdf_model import FourierPE, PointNetEncoder, INRDecoderSDF  # noqa: F401

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULTS = dict(local_dim=32, local_res=16, local_extent=1.8, local_hidden=64)


class LocalContourVolume(nn.Module):
    """Contour points -> coarse feature volume -> per-query local features."""

    def __init__(self, input_dim: int = 5, feat_dim: int = 32, res: int = 16,
                 extent: float = 1.8, hidden: int = 64):
        super().__init__()
        self.res = int(res)
        self.extent = float(extent)
        self.feat_dim = int(feat_dim)
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim),
        )
        groups = 4 if feat_dim % 4 == 0 else 1
        self.conv = nn.Sequential(
            nn.Conv3d(feat_dim, feat_dim, 3, padding=1),
            nn.GroupNorm(groups, feat_dim), nn.ReLU(inplace=True),
            nn.Conv3d(feat_dim, feat_dim, 3, padding=1),
            nn.GroupNorm(groups, feat_dim), nn.ReLU(inplace=True),
            nn.Conv3d(feat_dim, feat_dim, 3, padding=1),
        )

    def _voxel_index(self, xyz: torch.Tensor) -> torch.Tensor:
        u = (xyz / self.extent * 0.5 + 0.5) * self.res
        return u.floor().long().clamp_(0, self.res - 1)

    def build(self, contour: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, N, C_in), (B, N) -> feature volume (B, F, R, R, R)."""
        b, _, _ = contour.shape
        r, f = self.res, self.feat_dim
        feats = self.point_mlp(contour) * mask.unsqueeze(-1).to(contour.dtype)
        idx = self._voxel_index(contour[:, :, :3])
        flat = (idx[:, :, 0] * r + idx[:, :, 1]) * r + idx[:, :, 2]        # (B, N)

        acc = contour.new_zeros(b, r ** 3, f)
        acc.scatter_add_(1, flat.unsqueeze(-1).expand(-1, -1, f), feats)
        cnt = contour.new_zeros(b, r ** 3, 1)
        cnt.scatter_add_(1, flat.unsqueeze(-1), mask.unsqueeze(-1).to(contour.dtype))
        vol = acc / cnt.clamp(min=1.0)
        vol = vol.permute(0, 2, 1).reshape(b, f, r, r, r)
        return self.conv(vol)

    def sample(self, volume: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """Trilinear lookup at (B, N, 3) normalised points -> (B, N, F)."""
        b, n, _ = xyz.shape
        g = (xyz / self.extent).clamp(-1.0, 1.0)
        # grid_sample's last axis is (x, y, z) indexing (W, H, D) = (z, y, x) here;
        # align_corners=False makes -1..1 span voxel edges, matching ``_voxel_index``.
        grid = g.flip(-1).view(b, 1, 1, n, 3).to(volume.dtype)
        out = F.grid_sample(volume, grid, mode="bilinear", padding_mode="border",
                            align_corners=False)
        return out.view(b, volume.shape[1], n).permute(0, 2, 1)


class DecoderV2(INRDecoderSDF):
    """Baseline monotone-epi decoder, zero-initialised local injections, and an
    additive headroom term that removes the δ ceiling.

    The baseline bounds the wall by ``δ = τ + (0.45 − τ)·σ(r)``, i.e. 11.65 mm at
    the cohort mean scale. Measured on this cache, 68.4 % of end-systolic HCM
    wall targets lie above that ceiling and 64.9 % of the baseline's own δ
    outputs sit pinned against it. A saturated sigmoid has a vanishing
    derivative, so the offset head receives no gradient exactly where the wall
    is thickest: no amount of training can lift it.

    Refitting the sigmoid to a wider cap was tested and rejected — it moves the
    unsaturated predictions by up to 3.6 mm. Instead a soft-hinge is *added*
    above the saturation knee ``r0``:

        δ = τ + (cap − τ)·σ(r) + s·softplus(β(r − r0))/β

    Below the knee the term is numerically negligible, so everything the
    baseline learned in its usable range is preserved; above it, δ grows
    linearly and unbounded with a live gradient. ``s`` is initialised to the
    slope of the sigmoid at the knee, so the mapping is C¹ at warm start.
    """

    def __init__(self, *args, local_dim: int = 32, headroom: bool = False,
                 headroom_knee: float = 0.99, headroom_beta: float = 8.0, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.in_proj.out_features
        self.local_in = nn.Linear(local_dim, hidden, bias=False)
        self.local_skip = nn.Linear(local_dim, hidden, bias=False)
        self.local_delta = nn.Sequential(
            nn.Linear(local_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 1, bias=False))
        nn.init.zeros_(self.local_in.weight)
        nn.init.zeros_(self.local_skip.weight)
        nn.init.zeros_(self.local_delta[-1].weight)

        self.headroom = bool(headroom) and self.delta_cap is not None
        if self.headroom:
            k = float(headroom_knee)
            span = self.delta_cap - self.tau_min
            self.hr_beta = float(headroom_beta)
            self.register_buffer("hr_r0", torch.tensor(math.log(k / (1.0 - k))))
            self.hr_log_slope = nn.Parameter(torch.tensor(math.log(span * k * (1.0 - k))))

    def _delta(self, raw_d: torch.Tensor) -> torch.Tensor:
        if self.delta_cap is None:
            return F.softplus(raw_d) + 1e-4
        delta = self.tau_min + (self.delta_cap - self.tau_min) * torch.sigmoid(raw_d)
        if self.headroom:
            hinge = F.softplus(self.hr_beta * (raw_d - self.hr_r0)) / self.hr_beta
            delta = delta + self.hr_log_slope.exp() * hinge
        return delta

    def forward(self, z, fxyz, local=None):                       # type: ignore[override]
        b, n, _ = fxyz.shape
        h_in = torch.cat([z.unsqueeze(1).expand(b, n, -1), fxyz], dim=-1)
        pre = self.in_proj(h_in)
        if local is not None:
            pre = pre + self.local_in(local)
        h = self._act(pre)
        for li, lyr in enumerate(self.layers):
            if li == self.skip_layer:
                h = torch.cat([h, h_in], dim=-1)
                pre = lyr(h)
                if local is not None:
                    pre = pre + self.local_skip(local)
            else:
                pre = lyr(h)
            h = self._act(pre)
        f_endo = self.head_endo(h).squeeze(-1)
        raw_d = self.head_delta(h).squeeze(-1)
        if local is not None:
            raw_d = raw_d + self.local_delta(local).squeeze(-1)
        delta = self._delta(raw_d)
        return f_endo, f_endo - delta, delta


class SDFNetworkV2(nn.Module):
    """Encoder + local volume + decoder. Inference API matches the baseline."""

    def __init__(self, cfg: dict):
        super().__init__()
        cfg = {**DEFAULTS, **cfg}
        self.cfg = cfg
        self.encoder = PointNetEncoder(input_dim=cfg["input_dim"],
                                       latent_dim=cfg["latent_dim"])
        self.fourier = FourierPE(L=cfg["fourier_L"])
        self.local = LocalContourVolume(input_dim=cfg["input_dim"],
                                        feat_dim=cfg["local_dim"],
                                        res=cfg["local_res"],
                                        extent=cfg["local_extent"],
                                        hidden=cfg["local_hidden"])
        self.decoder = DecoderV2(
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
            local_dim=cfg["local_dim"],
            headroom=cfg.get("delta_headroom", False),
            headroom_knee=cfg.get("delta_headroom_knee", 0.99),
            headroom_beta=cfg.get("delta_headroom_beta", 8.0),
        )

    def encode(self, contour, mask):
        """Return the conditioning pair (global code, local feature volume)."""
        return self.encoder(contour, mask), self.local.build(contour, mask)

    def decode(self, cond, query_xyz):
        z, volume = cond if isinstance(cond, tuple) else (cond, None)
        local = None if volume is None else self.local.sample(volume, query_xyz)
        return self.decoder(z, self.fourier(query_xyz), local)


# ──────────────────────────────────────────────────────────────────────────
# Warm start
# ──────────────────────────────────────────────────────────────────────────
def load_baseline_into(net: SDFNetworkV2, ckpt_path: Path,
                       device: torch.device = DEVICE) -> dict:
    """Copy every baseline weight into v2 and verify nothing was dropped."""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = {k.replace("module.", "", 1): v for k, v in ckpt["model_state"].items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"baseline keys not consumed by v2: {unexpected}")
    new_prefixes = ("local.", "decoder.local_", "decoder.hr_")
    stale = [k for k in missing if not k.startswith(new_prefixes)]
    if stale:
        raise RuntimeError(f"v2 parameters missing from the baseline: {stale}")
    return {"epoch": int(ckpt.get("epoch", -1)),
            "val_loss": float(ckpt.get("val_loss", math.nan)),
            "cfg": dict(ckpt["cfg"]), "n_new": len(missing)}


@torch.no_grad()
def headroom_report(net: SDFNetworkV2, delta_obs: torch.Tensor) -> dict:
    """How far the headroom term moves the baseline, and what it unlocks.

    ``delta_obs`` are baseline δ values (headroom disabled). The drift is
    reported over the *unsaturated* points only, because the saturated ones are
    exactly the predictions the term is meant to change.
    """
    dec = net.decoder
    if not getattr(dec, "headroom", False):
        return {"headroom": False}
    tau, cap = dec.tau_min, dec.delta_cap
    d = delta_obs.detach().flatten().double()
    p = ((d - tau) / (cap - tau)).clamp(1e-9, 1 - 1e-9)
    r = torch.log(p / (1 - p))
    hinge = F.softplus(dec.hr_beta * (r - dec.hr_r0.double())) / dec.hr_beta
    new = d + dec.hr_log_slope.exp().double() * hinge
    free = d < 0.99 * cap
    return {
        "headroom": True,
        "knee_r0": float(dec.hr_r0), "slope": float(dec.hr_log_slope.exp()),
        "sat_frac": float((~free).double().mean()),
        "drift_unsaturated_max": float((new - d)[free].abs().max()) if free.any() else 0.0,
        "delta_max_before": float(d.max()), "delta_max_after": float(new.max()),
    }


def build_v2(ckpt_path: Path, overrides: dict | None = None,
             device: torch.device = DEVICE):
    """Instantiate v2 from the baseline checkpoint's own cfg block."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = {**dict(ckpt["cfg"]), **DEFAULTS, **(overrides or {})}
    net = SDFNetworkV2(cfg).to(device)
    meta = load_baseline_into(net, ckpt_path, device)
    return net, cfg, meta


def load_v2(ckpt_path: Path, device: torch.device = DEVICE):
    """Load a checkpoint written by ``train.py``."""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = {**DEFAULTS, **dict(ckpt["cfg"])}
    net = SDFNetworkV2(cfg).to(device)
    net.load_state_dict({k.replace("module.", "", 1): v
                         for k, v in ckpt["model_state"].items()}, strict=True)
    net.eval()
    return net, cfg, {"epoch": int(ckpt.get("epoch", -1)),
                      "val_loss": float(ckpt.get("val_loss", math.nan))}
