"""U0 — test-time latent optimisation (DeepSDF-style auto-decoder inference).

The trained network is frozen; only the conditioning of a single case is
optimised, against that case's own contours. Nothing in the feed-forward path
constrains the predicted zero level set to pass through the input slices, so
this recovers the part of the patient's anatomy that the encoder discarded.

Initialisation is the encoder output, and the objective carries an explicit
pull back towards it, so the result can only depart from the feed-forward
prediction where the contours demand it. Works unchanged on the baseline
(conditioning is a latent vector) and on v2 (conditioning is a latent vector
plus a local feature volume).
"""
from __future__ import annotations

import numpy as np
import torch


def _split(cond):
    return cond if isinstance(cond, tuple) else (cond, None)


def _residual(net, cond, pts, tissue):
    fe, fp, _ = net.decode(cond, pts)
    endo, epi = tissue < 0.5, tissue >= 0.5
    total = pts.new_zeros(())
    n = 0
    if bool(endo.any()):
        total = total + fe[0][endo].abs().sum()
        n += int(endo.sum())
    if bool(epi.any()):
        total = total + fp[0][epi].abs().sum()
        n += int(epi.sum())
    return total / max(n, 1)


def refine_latent(net, contour: torch.Tensor, mask: torch.Tensor, *,
                  steps: int = 150, lr: float = 1e-2, lambda_reg: float = 1e-2,
                  lambda_eik: float = 0.1, refine_local: bool = True,
                  scale_mm: float = 1.0, patience: int = 30):
    """Optimise the conditioning of one case. Returns (cond, history)."""
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        z0, vol0 = _split(net.encode(contour, mask))
    z = z0.clone().requires_grad_(True)
    params = [z]
    vol = None
    if vol0 is not None and refine_local:
        vol = vol0.clone().requires_grad_(True)
        params.append(vol)

    pts = contour[:, :, :3].detach()
    tissue = contour[0, :, 3].detach()
    opt = torch.optim.Adam(params, lr=lr)

    history = []
    best = (float("inf"), None)
    stale = 0
    for step in range(steps + 1):
        cond = (z, vol if vol is not None else vol0) if vol0 is not None else z
        res = _residual(net, cond, pts, tissue)
        res_mm = float(res.detach()) * scale_mm
        history.append(res_mm)
        if res_mm < best[0] - 1e-4:
            best = (res_mm, (z.detach().clone(),
                             None if vol is None else vol.detach().clone()))
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
        if step == steps:
            break

        loss = res + lambda_reg * (z - z0).pow(2).mean()
        if lambda_eik > 0:
            q = pts + torch.randn_like(pts) * 0.05
            q.requires_grad_(True)
            fe, _, _ = net.decode(cond, q)
            g = torch.autograd.grad(fe, q, torch.ones_like(fe), create_graph=True)[0]
            loss = loss + lambda_eik * ((g.norm(dim=-1) - 1.0) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    z_b, vol_b = best[1]
    cond = (z_b, vol_b if vol_b is not None else vol0) if vol0 is not None else z_b
    return cond, {"residual_mm": history, "best_mm": best[0],
                  "start_mm": history[0], "steps_run": len(history) - 1}


def refine_from_cache(net, cache_path, cfg, device, **kwargs):
    """Convenience wrapper: build the contour tensor from a cache entry."""
    with np.load(cache_path, allow_pickle=False) as d:
        xyz, tissue = d["contour_xyz"], d["contour_tissue"]
        scale = float(d["scale"])
        phase_val = 0.0 if str(d["phase"]) == "ED" else 1.0
    cols = [xyz, tissue[:, None]]
    if cfg["input_dim"] == 5:
        cols.append(np.full((len(xyz), 1), phase_val, np.float32))
    contour = torch.from_numpy(np.hstack(cols).astype(np.float32)).unsqueeze(0).to(device)
    mask = torch.ones(1, len(xyz), dtype=torch.bool, device=device)
    return refine_latent(net, contour, mask, scale_mm=scale, **kwargs)
