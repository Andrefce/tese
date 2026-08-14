"""Fine-tune CardioSDF v2 (U1 + U2 + local latents) from the baseline weights.

    python -m cardiosdf2.train --epochs 10

The run is a *fine-tune*, not a retrain: every baseline parameter is loaded and
the new conditioning path is zero-initialised, so epoch 0 reproduces the
baseline. What changes during training is (a) the group- and phase-balanced
sampling stream, (b) the direct wall supervision, and (c) the widened δ range.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import BASELINE_CKPT
from . import data as D
from .losses import DEFAULT_WEIGHTS, compute_losses
from .model import DEFAULTS, build_v2, headroom_report

SAMPLING = dict(n_surf_endo=768, n_surf_epi=512, n_query_sdf=768,
                n_near=256, n_free=1024, near_sigma=0.05)


def _loader(dataset, batch_size, sampler=None, shuffle=False, workers=0):
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=shuffle, num_workers=workers, collate_fn=D.collate,
                      drop_last=False, persistent_workers=bool(workers))


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _mean_reports(reports: list[dict]) -> dict:
    return {k: float(np.mean([r[k] for r in reports])) for k in reports[0]}


def run(args) -> dict:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    specs = D.index_cache(args.cache)
    if not specs:
        raise SystemExit(f"empty cache at {args.cache} — run build_cache.py first")
    train_specs, val_specs = D.split_by_patient(specs, args.val_fraction, args.seed)

    overrides = {**SAMPLING, **DEFAULTS,
                 "local_dim": args.local_dim, "local_res": args.local_res,
                 "delta_headroom": bool(args.delta_headroom)}
    net, cfg, meta = build_v2(BASELINE_CKPT, overrides, device)
    cfg.update(SAMPLING)
    cfg.update({k: getattr(args, k, DEFAULT_WEIGHTS[k]) for k in
                ("lambda_wt", "lambda_surf", "lambda_sdf_l1", "lambda_eik")})
    print(f"warm start: baseline epoch {meta['epoch']} val {meta['val_loss']:.4f}, "
          f"{meta['n_new']} new tensors (zero-initialised)")

    cfg.update({k: v for k, v in DEFAULT_WEIGHTS.items() if k not in cfg})

    train_cfg = {**cfg, **SAMPLING}
    ds_train = D.LVSDFDataset(train_specs, train_cfg, augment=False, seed=args.seed)
    ds_val = D.LVSDFDataset(val_specs, train_cfg, augment=False, seed=args.seed + 1)

    # U1: every (group, phase) stratum contributes equally per epoch.
    weights = D.balanced_weights(train_specs)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=len(train_specs), replacement=True)
    dl_train = _loader(ds_train, args.batch_size, sampler=sampler, workers=args.workers)
    dl_val = _loader(ds_val, args.batch_size, workers=0)
    print(f"train {len(train_specs)} samples / {len({s.patient for s in train_specs})} patients; "
          f"val {len(val_specs)} / {len({s.patient for s in val_specs})} (patient-disjoint)")

    if args.delta_headroom:
        probe = _to_device(next(iter(_loader(ds_train, 16, shuffle=True))), device)
        net.decoder.headroom = False
        with torch.no_grad():
            cond = net.encode(probe["contour"], probe["contour_mask"])
            _, _, delta_obs = net.decode(cond, probe["surf_endo_pts"])
        net.decoder.headroom = True
        info = headroom_report(net, delta_obs)
        print(f"δ headroom on: {info['sat_frac'] * 100:.1f}% of baseline δ was pinned "
              f"at the {net.decoder.delta_cap:.2f} ceiling; "
              f"reach {info['delta_max_before']:.2f} -> {info['delta_max_after']:.2f}, "
              f"drift below the knee {info['drift_unsaturated_max']:.4f}")

    # Three provenances, three rates: converged baseline weights must not be
    # kicked off their optimum, the zero-initialised local path has to travel
    # from nothing, and the headroom slope is a single log-scalar that Adam
    # would otherwise barely move within a short run.
    groups: dict[str, list] = {"backbone": [], "local": [], "slope": []}
    for name, p in net.named_parameters():
        if name.endswith("hr_log_slope"):
            groups["slope"].append(p)
        elif name.startswith(("local.", "decoder.local_")):
            groups["local"].append(p)
        else:
            groups["backbone"].append(p)
    opt = torch.optim.AdamW([
        {"params": groups["backbone"], "lr": args.lr},
        {"params": groups["local"], "lr": args.local_lr},
        {"params": groups["slope"], "lr": args.headroom_lr, "weight_decay": 0.0},
    ], lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    slope = groups["slope"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best = float("inf")

    for epoch in range(args.epochs):
        net.train()
        t0, reports = time.time(), []
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(dl_train):
            batch = _to_device(batch, device)
            loss, rep = compute_losses(net, batch, cfg)
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            reports.append(rep)
            if args.max_steps and step + 1 >= args.max_steps:
                break
        tr = _mean_reports(reports)

        net.eval()
        val_reports = []
        for batch in dl_val:
            batch = _to_device(batch, device)
            _, rep = compute_losses(net, batch, cfg)
            val_reports.append(rep)
        va = _mean_reports(val_reports)
        sched.step()

        row = {"epoch": epoch, "sec": round(time.time() - t0, 1),
               "train": tr, "val": va, "lr": opt.param_groups[0]["lr"]}
        if slope:
            row["headroom_slope"] = float(slope[0].detach().exp())
        history.append(row)
        print(f"[{epoch:02d}] {row['sec']:5.0f}s  "
              f"train {tr['loss']:7.3f} (wall MAE {tr['wall_mae_mm']:.2f} mm)  "
              f"| val {va['loss']:7.3f} (wall MAE {va['wall_mae_mm']:.2f} mm, "
              f"targets over cap {va['target_over_cap'] * 100:.0f}%, "
              f"predictions over cap {va['pred_over_cap'] * 100:.0f}%)", flush=True)

        if va["loss"] < best:
            best = va["loss"]
            torch.save({"model_state": net.state_dict(), "cfg": cfg, "epoch": epoch,
                        "val_loss": va["loss"],
                        "val_patients": sorted({s.patient for s in val_specs})},
                       out_dir / "cardiosdf_v2_best.pt")

    torch.save({"model_state": net.state_dict(), "cfg": cfg, "epoch": args.epochs - 1,
                "val_loss": history[-1]["val"]["loss"],
                "val_patients": sorted({s.patient for s in val_specs})},
               out_dir / "cardiosdf_v2_last.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest val {best:.4f} -> {out_dir / 'cardiosdf_v2_best.pt'}")
    return {"history": history, "best": best}


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=here / "cache")
    ap.add_argument("--out", type=Path, default=here / "runs" / "u1u2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--local-lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--grad-clip", type=float, default=1.2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--local-dim", type=int, default=DEFAULTS["local_dim"])
    ap.add_argument("--local-res", type=int, default=DEFAULTS["local_res"])
    ap.add_argument("--delta-headroom", type=int, default=1,
                    help="0 keeps the baseline 0.45 ceiling")
    ap.add_argument("--headroom-lr", type=float, default=0.05)
    ap.add_argument("--lambda-wt", type=float, default=DEFAULT_WEIGHTS["lambda_wt"])
    ap.add_argument("--lambda-surf", type=float, default=DEFAULT_WEIGHTS["lambda_surf"])
    ap.add_argument("--lambda-sdf-l1", type=float, default=DEFAULT_WEIGHTS["lambda_sdf_l1"])
    ap.add_argument("--lambda-eik", type=float, default=DEFAULT_WEIGHTS["lambda_eik"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
