"""Baseline vs v2 on the same held-out patients, against the same voxel targets.

Everything is measured in the field domain (δ at endocardial surface points,
|f| at the input contour points), never on a re-meshed surface, so the numbers
isolate what the network predicts from what marching cubes and the watertight
repair do afterwards. Predictions and targets are read at identical points, so
the comparison is paired.

    python -m cardiosdf2.evaluate --ckpt runs/u1u2/cardiosdf_v2_best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import BASELINE_CKPT
from . import data as D
from .model import load_v2
from .refine import refine_latent

SEGMENT_LAYOUT = ((2 / 3, 1.0, 6), (1 / 3, 2 / 3, 6), (0.0, 1 / 3, 4))


@torch.no_grad()
def mesh_metrics(net, arr: dict, cfg: dict, phase_val: float, grid_res: int = 96) -> dict:
    """Watertight rate and volume ratios of the extracted surfaces (§3 gates).

    Extraction goes through ``geometry.build_model_geometry`` — the same
    function that produced the baseline figures in the upgrade plan — so the
    two columns differ only by the checkpoint.
    """
    from geometry import build_model_geometry

    contours = {"xyz": arr["contour_xyz"], "tissue": arr["contour_tissue"],
                "centroid": arr["centroid"].astype(np.float64),
                "scale": float(arr["scale"])}
    geo = build_model_geometry(net, cfg, contours, grid_res=grid_res, phase_val=phase_val)
    out = {}
    for tag, ref_ml in (("endo", float(arr["endo_volume_ml"])),
                        ("epi", float(arr["epi_volume_ml"]))):
        mesh = geo[tag]
        vol_ml = abs(float(mesh.volume)) / 1000.0 if len(mesh.faces) else 0.0
        out[f"{tag}_watertight"] = float(bool(len(mesh.faces)) and bool(mesh.is_watertight))
        out[f"{tag}_volume_ratio"] = vol_ml / ref_ml if ref_ml > 1e-9 else float("nan")
    return out


def segment_medians(pts: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Median value per AHA-style segment (6 basal, 6 mid, 4 apical).

    The true AHA-17 frame needs the RV insertion points, which the cache does
    not carry; the angular origin here is therefore arbitrary. That is harmless
    for the *maximum* segment, which is rotation-invariant, and it is why no
    septal-to-lateral ratio is reported.
    """
    z = pts[:, 2]
    span = max(float(z.max() - z.min()), 1e-6)
    t = (z - z.min()) / span
    ang = np.arctan2(pts[:, 1], pts[:, 0]) % (2 * np.pi)
    out = []
    for lo, hi, n_sec in SEGMENT_LAYOUT:
        level = (t >= lo) & (t <= hi if hi == 1.0 else t < hi)
        for s in range(n_sec):
            sector = (ang >= 2 * np.pi * s / n_sec) & (ang < 2 * np.pi * (s + 1) / n_sec)
            sel = level & sector
            out.append(float(np.median(values[sel])) if sel.sum() >= 5 else np.nan)
    return np.array(out)


@torch.no_grad()
def evaluate_sample(net, spec: D.SampleSpec, cfg: dict, device, refine: dict | None = None,
                   with_mesh: bool = False) -> dict:
    with np.load(spec.path, allow_pickle=False) as d:
        arr = {k: d[k] for k in d.files if d[k].dtype.kind in "fiu"}
    scale = float(arr["scale"])
    phase_val = 0.0 if spec.phase == "ED" else 1.0

    n_ctr = len(arr["contour_xyz"])
    cols = [arr["contour_xyz"], arr["contour_tissue"][:, None]]
    if cfg["input_dim"] == 5:
        cols.append(np.full((n_ctr, 1), phase_val, np.float32))
    contour = torch.from_numpy(np.hstack(cols).astype(np.float32)).unsqueeze(0).to(device)
    mask = torch.ones(1, n_ctr, dtype=torch.bool, device=device)
    refine_info = None
    if refine:
        with torch.enable_grad():
            cond, refine_info = refine_latent(net, contour, mask, scale_mm=scale, **refine)
    else:
        cond = net.encode(contour, mask)

    def decode(points):
        pt = torch.from_numpy(np.ascontiguousarray(points, np.float32)).unsqueeze(0).to(device)
        fe, fp, dl = net.decode(cond, pt)
        return (fe[0].float().cpu().numpy(), fp[0].float().cpu().numpy(),
                dl[0].float().cpu().numpy())

    fe_c, fp_c, _ = decode(arr["contour_xyz"])
    tissue = arr["contour_tissue"]
    res = np.where(tissue < 0.5, np.abs(fe_c), np.abs(fp_c))
    residual_mm = float(res.mean() * scale)

    se = arr["surf_endo_pts"]
    _, _, delta = decode(se)
    wall_pred = delta * scale
    wall_gt = arr["surf_endo_wall"] * scale

    fe_q, fp_q, _ = decode(arr["query_pts"])
    sdf_mae_mm = float((np.abs(fe_q - arr["query_e_sdf"]).mean()
                        + np.abs(fp_q - arr["query_p_sdf"]).mean()) / 2 * scale)

    seg_pred = segment_medians(se, wall_pred)
    seg_gt = segment_medians(se, wall_gt)
    ok = ~np.isnan(seg_pred) & ~np.isnan(seg_gt)

    row = {
        "patient": spec.patient, "group": spec.group, "phase": spec.phase,
        "scale_mm": scale, "residual_mm": residual_mm, "sdf_mae_mm": sdf_mae_mm,
        "wall_pred_mm": float(wall_pred.mean()), "wall_gt_mm": float(wall_gt.mean()),
        "wall_mae_mm": float(np.abs(wall_pred - wall_gt).mean()),
        "wall_bias_mm": float((wall_pred - wall_gt).mean()),
        "max_seg_pred_mm": float(np.nanmax(seg_pred)),
        "max_seg_gt_mm": float(np.nanmax(seg_gt)),
        "seg_mae_mm": float(np.abs(seg_pred[ok] - seg_gt[ok]).mean()),
        "seg_r": float(np.corrcoef(seg_pred[ok], seg_gt[ok])[0, 1]) if ok.sum() > 2 else np.nan,
    }
    if refine_info:
        row["refine_start_mm"] = refine_info["start_mm"]
        row["refine_steps"] = refine_info["steps_run"]
    if with_mesh:
        row.update(mesh_metrics(net, arr, cfg, phase_val))
    return row


def _sel(rows, group=None, phase=None):
    return [r for r in rows
            if (group is None or r["group"] == group) and (phase is None or r["phase"] == phase)]


def _m(rows, key):
    vals = [r[key] for r in rows if np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def summarise(rows: list[dict]) -> dict:
    nor_ed, hcm_ed = _sel(rows, "NOR", "ED"), _sel(rows, "HCM", "ED")
    hcm_all = _sel(rows, "HCM")
    out = {
        "n_samples": len(rows),
        "residual_mm": _m(rows, "residual_mm"),
        "sdf_mae_mm": _m(rows, "sdf_mae_mm"),
        "wall_mae_mm": _m(rows, "wall_mae_mm"),
        "wall_bias_mm": _m(rows, "wall_bias_mm"),
        "seg_mae_mm": _m(rows, "seg_mae_mm"),
        "seg_r": _m(rows, "seg_r"),
    }
    for key in ("endo_watertight", "epi_watertight",
                "endo_volume_ratio", "epi_volume_ratio"):
        if key in rows[0]:
            out[key] = _m(rows, key)
            if key.endswith("volume_ratio") and nor_ed:
                out[f"nor_ed_{key}"] = _m(nor_ed, key)
    for tag, sub in (("nor_ed", nor_ed), ("hcm_ed", hcm_ed), ("hcm_all", hcm_all),
                     ("es_all", _sel(rows, phase="ES"))):
        if sub:
            out[f"{tag}_wall_pred_mm"] = _m(sub, "wall_pred_mm")
            out[f"{tag}_wall_gt_mm"] = _m(sub, "wall_gt_mm")
            out[f"{tag}_max_seg_pred_mm"] = _m(sub, "max_seg_pred_mm")
            out[f"{tag}_max_seg_gt_mm"] = _m(sub, "max_seg_gt_mm")

    if nor_ed and hcm_ed:
        gt_gap = _m(hcm_ed, "wall_gt_mm") - _m(nor_ed, "wall_gt_mm")
        pr_gap = _m(hcm_ed, "wall_pred_mm") - _m(nor_ed, "wall_pred_mm")
        out["contrast_gt_mm"] = gt_gap
        out["contrast_pred_mm"] = pr_gap
        out["contrast_retention"] = pr_gap / gt_gap if abs(gt_gap) > 1e-9 else float("nan")
    if hcm_all:
        out["hcm_detect_pred"] = sum(r["max_seg_pred_mm"] > 15.0 for r in hcm_all)
        out["hcm_detect_gt"] = sum(r["max_seg_gt_mm"] > 15.0 for r in hcm_all)
        out["hcm_n"] = len(hcm_all)
    return out


def load_baseline(device):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "eval_demo"))
    from cardiosdf_model import load_model
    return load_model(BASELINE_CKPT, device)


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=here / "cache")
    ap.add_argument("--ckpt", type=Path, default=here / "runs" / "u1u2" / "cardiosdf_v2_best.pt")
    ap.add_argument("--out", type=Path, default=here / "runs" / "u1u2" / "eval.json")
    ap.add_argument("--held-out-only", action="store_true", default=True)
    ap.add_argument("--all-patients", dest="held_out_only", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--refine-steps", type=int, default=0,
                    help="U0 test-time latent optimisation steps (0 = feed-forward only)")
    ap.add_argument("--refine-lr", type=float, default=1e-2)
    ap.add_argument("--refine-lambda-reg", type=float, default=1e-2)
    ap.add_argument("--refine-lambda-eik", type=float, default=0.1)
    ap.add_argument("--refine-local", type=int, default=1)
    ap.add_argument("--mesh", action="store_true",
                    help="also extract surfaces for the watertight/volume gates")
    args = ap.parse_args()

    refine = None
    if args.refine_steps > 0:
        refine = dict(steps=args.refine_steps, lr=args.refine_lr,
                      lambda_reg=args.refine_lambda_reg,
                      lambda_eik=args.refine_lambda_eik,
                      refine_local=bool(args.refine_local))

    device = torch.device(args.device)
    specs = D.index_cache(args.cache)

    v2, cfg_v2, meta = load_v2(args.ckpt, device)
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    held = set(ckpt.get("val_patients", []))
    if args.held_out_only and held:
        specs = [s for s in specs if s.patient in held]
    print(f"v2 epoch {meta['epoch']} | {len(specs)} samples "
          f"({'held-out' if args.held_out_only and held else 'all'} patients)")

    base, cfg_b, meta_b = load_baseline(device)
    rows_b = [evaluate_sample(base, s, cfg_b, device, with_mesh=args.mesh) for s in specs]
    rows_v = [evaluate_sample(v2, s, cfg_v2, device, refine, with_mesh=args.mesh) for s in specs]
    sum_b, sum_v = summarise(rows_b), summarise(rows_v)

    keys = ["residual_mm", "sdf_mae_mm", "wall_mae_mm", "wall_bias_mm", "seg_mae_mm",
            "seg_r", "contrast_retention", "nor_ed_wall_pred_mm", "hcm_ed_wall_pred_mm",
            "hcm_all_max_seg_pred_mm", "es_all_wall_pred_mm", "hcm_detect_pred",
            "endo_watertight", "epi_watertight", "endo_volume_ratio", "epi_volume_ratio"]
    ideal = {"residual_mm": 0.0, "sdf_mae_mm": 0.0, "wall_mae_mm": 0.0,
             "wall_bias_mm": 0.0, "seg_mae_mm": 0.0, "seg_r": 1.0,
             "contrast_retention": 1.0, "endo_watertight": 1.0, "epi_watertight": 1.0,
             "endo_volume_ratio": 1.0, "epi_volume_ratio": 1.0}
    print(f"\n{'metric':<26}{'baseline':>12}{'v2':>12}{'reference':>12}")
    print("-" * 62)
    for k in keys:
        if k not in sum_b:
            continue
        gt_key = k.replace("_pred_mm", "_gt_mm").replace("_pred", "_gt")
        tgt = ideal.get(k, sum_b.get(gt_key, sum_v.get(gt_key)))
        tgt_s = f"{float(tgt):12.3f}" if tgt is not None and np.isfinite(tgt) else " " * 12
        print(f"{k:<26}{sum_b[k]:12.3f}{sum_v[k]:12.3f}{tgt_s}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"baseline": sum_b, "v2": sum_v,
                                    "rows_baseline": rows_b, "rows_v2": rows_v},
                                   indent=2, default=float))
    print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
