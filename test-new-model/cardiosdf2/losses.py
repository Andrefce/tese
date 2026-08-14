"""Training objectives for CardioSDF v2.

Same core terms as the baseline (``notebooks/training.ipynb``): surface fit,
dense signed-distance L1, eikonal regularisation and off-surface repulsion.
Two things differ.

**U2 — the wall is finally supervised.** ``lambda_wt`` was present in the
baseline loss table but read 0.0 for all 776 recorded epochs: no target was
ever attached to it. Here the cache carries ``surf_endo_wall``, the distance
from each endocardial surface point to the epicardial surface measured on the
same voxel geometry the Results chapter uses as reference, so the offset head
is trained directly on the quantity the thesis reports.

**Saturation is reported, not hidden.** ``delta_sat`` counts the fraction of
supervised wall targets sitting against the decoder's structural δ ceiling.
On the baseline cap that fraction is 68 % for end-systolic HCM, which is the
measured reason the model cannot reach the 15 mm diagnostic threshold.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_WEIGHTS = dict(
    lambda_surf=10.0,
    epi_surf_weight=1.0,
    lambda_sdf_l1=3.0,
    lambda_eik=10.0,
    lambda_off=0.1,
    alpha_off=5.0,
    lambda_wt=2.0,
    wt_huber_beta=0.02,      # ≈ 0.5 mm at the cohort mean scale
    lambda_normal=0.0,
)


def _point_grad(field: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(field, pts, grad_outputs=torch.ones_like(field),
                               create_graph=True, retain_graph=True)[0]


def compute_losses(net, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """Total loss and a scalar report dictionary."""
    w = {**DEFAULT_WEIGHTS, **cfg}
    se, sp = batch["surf_endo_pts"], batch["surf_epi_pts"]
    n_e, n_p = se.shape[1], sp.shape[1]

    cond = net.encode(batch["contour"], batch["contour_mask"])

    # ── Pass A: surfaces + dense SDF targets (no gradients w.r.t. points) ──
    pts_a = torch.cat([se, sp, batch["query_pts"]], dim=1)
    fe_a, fp_a, delta_a = net.decode(cond, pts_a)
    fe_se, fe_sp, fe_q = fe_a[:, :n_e], fe_a[:, n_e:n_e + n_p], fe_a[:, n_e + n_p:]
    fp_se, fp_sp, fp_q = fp_a[:, :n_e], fp_a[:, n_e:n_e + n_p], fp_a[:, n_e + n_p:]

    l_surf = fe_se.abs().mean() + w["epi_surf_weight"] * fp_sp.abs().mean()
    l_sdf = (F.l1_loss(fe_q, batch["query_e_sdf"])
             + F.l1_loss(fp_q, batch["query_p_sdf"]))

    # U2: direct wall supervision at the endocardial surface.
    wall_gt = batch["surf_endo_wall"]
    delta_se = delta_a[:, :n_e]
    l_wt = F.smooth_l1_loss(delta_se, wall_gt, beta=w["wt_huber_beta"])

    # ── Pass B: eikonal + off-surface repulsion (needs ∂f/∂x) ──
    pts_b = torch.cat([batch["near_pts"], batch["free_pts"]], dim=1)
    pts_b = pts_b.detach().requires_grad_(True)
    with torch.enable_grad():
        fe_b, fp_b, _ = net.decode(cond, pts_b)
        g_e = _point_grad(fe_b, pts_b)
        g_p = _point_grad(fp_b, pts_b)
    l_eik = ((g_e.norm(dim=-1) - 1.0) ** 2).mean() + ((g_p.norm(dim=-1) - 1.0) ** 2).mean()

    n_free = batch["free_pts"].shape[1]
    fe_free, fp_free = fe_b[:, -n_free:], fp_b[:, -n_free:]
    a = w["alpha_off"]
    l_off = (torch.exp(-a * fe_free.abs()).mean() + torch.exp(-a * fp_free.abs()).mean())

    total = (w["lambda_surf"] * l_surf + w["lambda_sdf_l1"] * l_sdf
             + w["lambda_eik"] * l_eik + w["lambda_off"] * l_off
             + w["lambda_wt"] * l_wt)

    if w["lambda_normal"] > 0:
        se_g = se.detach().requires_grad_(True)
        sp_g = sp.detach().requires_grad_(True)
        with torch.enable_grad():
            fe_n, _, _ = net.decode(cond, se_g)
            _, fp_n, _ = net.decode(cond, sp_g)
            gn_e = F.normalize(_point_grad(fe_n, se_g), dim=-1)
            gn_p = F.normalize(_point_grad(fp_n, sp_g), dim=-1)
        l_norm = ((1 - (gn_e * batch["surf_endo_n"]).sum(-1)).mean()
                  + (1 - (gn_p * batch["surf_epi_n"]).sum(-1)).mean())
        total = total + w["lambda_normal"] * l_norm
    else:
        l_norm = torch.zeros((), device=total.device)

    cap = net.decoder.delta_cap
    with torch.no_grad():
        scale = batch["scale"].view(-1, 1)
        wall_err_mm = ((delta_se - wall_gt).abs() * scale).mean()
        # Fraction of targets above the plain sigmoid ceiling, and the fraction
        # of predictions that actually reach past it (0 without headroom).
        if cap is None:
            need = reach = torch.zeros(())
        else:
            need = (wall_gt > cap).float().mean()
            reach = (delta_se > cap).float().mean()
        report = {
            "loss": float(total), "surf": float(l_surf), "sdf": float(l_sdf),
            "eik": float(l_eik), "off": float(l_off), "wt": float(l_wt),
            "normal": float(l_norm), "wall_mae_mm": float(wall_err_mm),
            "target_over_cap": float(need), "pred_over_cap": float(reach),
        }
    return total, report
