"""Generate the representative-NOR-patient data bundle for the Results figures.

Reuses the validated full-cohort pipeline (``compute_results_full_cohort.process``)
to reconstruct one healthy (NOR) patient at both cardiac phases and to store, for
BOTH geometries -- the CardioSDF model surfaces and the segmentation-derived
"no-model" marching-cubes surfaces -- everything the Results figures need:

  * endo/epi meshes (vertices + faces, physical mm) for ED and ES;
  * calibrated Laplace-field wall thickness per endocardial vertex (ED), for the
    model and derived geometries, with the cavity-wall mask and AHA-17 ids;
  * per-AHA-segment mean thickness at ED and ES and the ED->ES thickening (%),
    again for both geometries, so the model can be compared against the no-model
    baseline segment by segment.

Output -> scripts/webapp/notebooks/outputs/cohort_single/representative_nor.npz

Run:
    RESULTS_NOR_PID=patient076 .venv/bin/python scripts/fig_results_nor_data.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("RESULTS_DATA_ROOT", str(ROOT / "notebooks" / "data"))

import compute_results_full_cohort as F  # noqa: E402

PID = os.environ.get("RESULTS_NOR_PID", "patient076")
OUT = Path(os.environ.get(
    "RESULTS_OUT_DIR",
    str(ROOT / "scripts" / "webapp" / "notebooks" / "outputs" / "cohort_single")))
OUT.mkdir(parents=True, exist_ok=True)
N_SEG = 17


def cal_factor(field: np.ndarray, wall: np.ndarray, ref_mean: float) -> float:
    v = np.asarray(field, float)[wall]
    v = v[np.isfinite(v)]
    if v.size == 0 or v.mean() <= 0.1:
        return 1.0
    return float(np.clip(ref_mean / v.mean(), 0.3, 4.0))


def seg_means(field_cal: np.ndarray, aha: np.ndarray, wall: np.ndarray) -> np.ndarray:
    out = np.full(N_SEG, np.nan)
    for sid in range(1, N_SEG + 1):
        m = field_cal[(aha == sid) & wall]
        m = m[np.isfinite(m)]
        if m.size:
            out[sid - 1] = float(m.mean())
    return out


def geometry_bundle(res: dict, phase: str, prefix: str) -> dict:
    """Extract meshes + (ED only) calibrated Laplace thickness for one phase."""
    seg, spacing, ref = res["seg"], res["spacing"], res["ref_mean"]
    out: dict[str, np.ndarray] = {}
    for geom_key, mesh_key in (("cardio", "cardio"), ("derived", "derived")):
        g = res[geom_key]
        out[f"{prefix}_{mesh_key}_endo_v"] = np.asarray(g["endo_mm"].vertices, np.float32)
        out[f"{prefix}_{mesh_key}_endo_f"] = np.asarray(g["endo_mm"].faces, np.int32)
        out[f"{prefix}_{mesh_key}_epi_v"] = np.asarray(g["epi_mm"].vertices, np.float32)
        out[f"{prefix}_{mesh_key}_epi_f"] = np.asarray(g["epi_mm"].faces, np.int32)

    # AHA ids for both geometries' endocardial vertices.
    P_c = np.asarray(res["cardio"]["endo_mm"].vertices, np.float32)
    P_d = np.asarray(res["derived"]["endo_mm"].vertices, np.float32)
    aha_c = F.assign_aha17(P_c, seg, spacing)
    aha_d = F.assign_aha17(P_d, seg, spacing)
    out[f"{prefix}_cardio_aha"] = aha_c.astype(np.int16)
    out[f"{prefix}_derived_aha"] = aha_d.astype(np.int16)

    # Calibrated Laplace thickness per endo vertex (model + derived).
    lap_c = np.asarray(res["fields_c"]["Laplace field"], np.float32)
    lap_d = np.asarray(res["fields_d"]["Laplace field"], np.float32)
    fc = cal_factor(lap_c, res["wall_c"], ref)
    fd = cal_factor(lap_d, res["wall_d"], ref)
    out[f"{prefix}_cardio_thick"] = lap_c * fc
    out[f"{prefix}_derived_thick"] = lap_d * fd
    out[f"{prefix}_cardio_wall"] = res["wall_c"].astype(bool)
    out[f"{prefix}_derived_wall"] = res["wall_d"].astype(bool)
    out[f"{prefix}_cardio_seg"] = seg_means(lap_c * fc, aha_c, res["wall_c"])
    out[f"{prefix}_derived_seg"] = seg_means(lap_d * fd, aha_d, res["wall_d"])
    out[f"{prefix}_ref_mean"] = np.float32(ref)
    return out


def main() -> None:
    print(f"Representative NOR patient: {PID}")
    model, cfg = F.load_model(F.MODEL_PATH)
    bundle: dict[str, np.ndarray] = {}
    seg_store: dict[str, np.ndarray] = {}
    for phase in ("ED", "ES"):
        res = F.process(PID, phase, model, cfg)
        if res is None:
            raise RuntimeError(f"process() returned None for {PID} {phase}")
        bundle.update(geometry_bundle(res, phase, phase))
        seg_store[phase] = res
        print(f"  {phase}: cardio endo v={len(res['cardio']['endo_mm'].vertices)} "
              f"derived endo v={len(res['derived']['endo_mm'].vertices)} "
              f"ref={res['ref_mean']:.2f} mm")

    # ED->ES thickening (%) per AHA segment, model vs derived.
    for geom in ("cardio", "derived"):
        ed = bundle[f"ED_{geom}_seg"]
        es = bundle[f"ES_{geom}_seg"]
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = 100.0 * (es - ed) / ed
        bundle[f"{geom}_thickening_pct"] = pct.astype(np.float32)

    spacing = np.asarray(seg_store["ED"]["spacing"], np.float32)
    bundle["spacing"] = spacing
    bundle["pid"] = np.array(PID)

    out_path = OUT / "representative_nor.npz"
    np.savez_compressed(out_path, **bundle)
    print(f"\nSaved -> {out_path}")
    # Console summary of the segment thickening the figure will show.
    names = F.AHA_17_NAMES
    print("\nAHA-17 ED->ES thickening (%)  [model  vs  derived]")
    for i in range(N_SEG):
        print(f"  {i+1:2d} {names[i]:20s} "
              f"{bundle['cardio_thickening_pct'][i]:+6.1f}   {bundle['derived_thickening_pct'][i]:+6.1f}")


if __name__ == "__main__":
    main()
