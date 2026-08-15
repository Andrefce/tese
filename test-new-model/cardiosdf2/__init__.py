"""CardioSDF architecture upgrade (U1 + U2 + local latents) — test bed.

The modules here reuse the geometry layer of ``scripts/eval_demo`` verbatim so
that the training targets and the evaluation reference are the same objects.
"""
from __future__ import annotations

import sys
from pathlib import Path

THESIS_ROOT = Path(__file__).resolve().parents[2]
EVAL_DEMO = THESIS_ROOT / "scripts" / "eval_demo"
if str(EVAL_DEMO) not in sys.path:
    sys.path.insert(0, str(EVAL_DEMO))

BASELINE_CKPT = THESIS_ROOT / "notebooks" / "inr_sdf_combined_fresh_ed_mix_v1_final.pt"
