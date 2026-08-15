"""Run the thesis cohort pipeline with a CardioSDF v2 checkpoint.

``scripts/eval_demo/run_cohort.py`` loads checkpoints with ``strict=True``
against the baseline layout, so a v2 checkpoint cannot pass through it. Rather
than edit the script that produced the published Results tables, only the model
loader is swapped here; contour extraction, marching cubes, watertight repair,
reconstruction metrics and all four thickness methods stay byte-identical, so
the output CSVs are directly comparable to ``scripts/cohort_nor``.

    python run_cohort_v2.py --model runs/u1u2_e50/cardiosdf_v2_best.pt \
        --data-root training --patients patient074 --out cohort_nor_v2 --workers 1
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts" / "eval_demo"))

import run_cohort  # noqa: E402
from cardiosdf2.model import load_v2  # noqa: E402


def _load_net_v2(model_path):
    if "net" not in run_cohort._MODEL_CACHE:
        import torch
        from cardiosdf_model import DEVICE

        torch.set_num_threads(1)
        net, cfg, meta = load_v2(Path(model_path), DEVICE)
        run_cohort._MODEL_CACHE.update(net=net, cfg=cfg, meta=meta)
    c = run_cohort._MODEL_CACHE
    return c["net"], c["cfg"], c["meta"]


if __name__ == "__main__":
    import multiprocessing

    # Python 3.14 defaults to "forkserver": workers would re-import run_cohort
    # and lose the patch below, falling back to the strict baseline loader.
    # "fork" copies the patched module. Safe here because CUDA is never
    # initialised in the parent (run with CUDA_VISIBLE_DEVICES="").
    multiprocessing.set_start_method("fork", force=True)
    run_cohort.load_net = _load_net_v2
    run_cohort.main()
