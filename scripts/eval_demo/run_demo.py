"""Single-patient demo: new wall-thickness estimators, CardioSDF vs voxel model.

Pipeline
--------
1. Load the demo SAX segmentation (ED and, if present, ES).
2. Build two watertight geometries from the *same* input:
     * ``model`` - CardioSDF/INR reconstruction (SDF -> cleaned marching cubes
       -> pymeshfix/trimesh repair -> Taubin smoothing),
     * ``voxel`` - segmentation labels resampled to an isotropic grid and
       surfaced with the identical repair pipeline.
3. Validate every estimator on analytic phantoms (known thickness).
4. Apply the estimators to both geometries and compare them on AHA-17
   regional means (Bland-Altman, ICC(2,1), Pearson r).

Run:
    cd tese/scripts/eval_demo && ../../../.venv/bin/python run_demo.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

_THREADS = os.environ.get("DEMO_THREADS", "4")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
import trimesh                                                       # noqa: E402

import agreement as agr                                              # noqa: E402
import phantoms as ph                                                # noqa: E402
import thickness as tk                                               # noqa: E402
from cardiosdf_model import load_model, slice_residual_mm            # noqa: E402
from geometry import (                                               # noqa: E402
    AHA_17_NAMES, assign_aha17, build_model_geometry, build_voxel_geometry,
    enforce_nesting, extract_contours, load_segmentation, long_axis_frame,
    outward_normals, read_info_cfg,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_PATIENT = REPO / "tese" / "notebooks" / "patient002"
DEFAULT_MODEL = REPO / "tese" / "notebooks" / "inr_sdf_combined_fresh_ed_mix_v1_final.pt"
OUT_DIR = HERE / "outputs"

# Endocardial vertices in the artificial basal cap / apical tip are excluded:
# the closed mesh has no myocardium there, so thickness is undefined.
VALID_LONG_AXIS_BAND = (0.04, 0.97)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────
# Phantom validation
# ──────────────────────────────────────────────────────────────────────────
def run_phantom_validation(pitch: float) -> pd.DataFrame:
    rows = []
    for factory in ph.ALL_PHANTOMS:
        phantom = factory()
        log(f"phantom: {phantom.name} ({phantom.description})")
        endo, epi = phantom.endo, phantom.epi
        normals = outward_normals(endo, np.asarray(epi.vertices))
        ctx = tk.build_volume_context(endo, epi, pitch)
        phi, _ = tk.solve_laplace(ctx)

        results = [
            tk.method_laplace_streamline(ctx, np.asarray(endo.vertices), normals, phi),
            tk.method_laplace_gradient(ctx, np.asarray(endo.vertices), normals, phi),
            tk.method_edt_boundary_sum(ctx, np.asarray(endo.vertices), normals),
            tk.method_sphere_propagation(ctx, np.asarray(endo.vertices), normals),
            tk.method_surface_correspondence(endo, epi, normals),
            tk.method_cone_rays(endo, epi, normals),
        ]
        for res in results:
            row = {"phantom": phantom.name, "method": res.name, "family": res.family,
                   "true_mean_mm": round(float(phantom.true_thickness[phantom.valid].mean()), 3)}
            row.update(ph.error_metrics(res.values, phantom.true_thickness,
                                        res.runtime_s, phantom.valid))
            rows.append(row)
            log(f"    {res.name:36s} MAE={row.get('mae_mm')} mm  bias={row.get('bias_mm')} mm")
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Patient geometry + estimators
# ──────────────────────────────────────────────────────────────────────────
def analyse_geometry(label: str, endo, epi, seg, pitch: float,
                     model_ctx: dict | None = None) -> dict:
    log(f"  [{label}] endo={len(endo.vertices)} verts  epi={len(epi.vertices)} verts")
    endo, nesting = enforce_nesting(endo, epi)
    normals = outward_normals(endo, np.asarray(epi.vertices))
    verts = np.asarray(endo.vertices, np.float64)

    frame = long_axis_frame(endo, seg)
    t_axis = np.clip((verts[:, 2] - frame["base_z"]) /
                     (frame["apex_z"] - frame["base_z"]), 0.0, 1.0)
    band = (t_axis >= VALID_LONG_AXIS_BAND[0]) & (t_axis <= VALID_LONG_AXIS_BAND[1])
    aha_ids = assign_aha17(verts, frame)

    log(f"  [{label}] building isotropic volume ({pitch} mm)")
    ctx = tk.build_volume_context(endo, epi, pitch)
    log(f"  [{label}] myocardium voxels: {int(ctx.myo_mask.sum())}")
    phi, laplace_diag = tk.solve_laplace(ctx)

    results = []
    log(f"  [{label}] Laplace streamline")
    results.append(tk.method_laplace_streamline(ctx, verts, normals, phi))
    log(f"  [{label}] Laplace local gradient")
    results.append(tk.method_laplace_gradient(ctx, verts, normals, phi))
    log(f"  [{label}] EDT boundary sum")
    results.append(tk.method_edt_boundary_sum(ctx, verts, normals))
    log(f"  [{label}] sphere propagation")
    results.append(tk.method_sphere_propagation(ctx, verts, normals))
    log(f"  [{label}] surface correspondence")
    results.append(tk.method_surface_correspondence(endo, epi, normals))
    log(f"  [{label}] cone rays")
    results.append(tk.method_cone_rays(endo, epi, normals))
    if model_ctx is not None:
        log(f"  [{label}] decoder offset")
        results.append(tk.method_decoder_offset(
            model_ctx["net"], model_ctx["latent"], verts,
            model_ctx["centroid"], model_ctx["scale"]))

    for res in results:
        res.values = np.where(band, res.values, np.nan)
        res.diagnostics.update(laplace_diag)
        res.diagnostics["analysis_band_fraction"] = float(band.mean())

    return {"label": label, "endo": endo, "epi": epi, "normals": normals,
            "aha_ids": aha_ids, "band": band, "frame": frame,
            "results": {r.name: r for r in results}, "nesting": nesting,
            "myo_voxels": int(ctx.myo_mask.sum())}


def run_patient(patient_dir: Path, model_path: Path, pitch: float,
                grid_res: int, phases: list[str]) -> dict:
    info = read_info_cfg(patient_dir / "Info.cfg")
    patient_id = patient_dir.name
    log(f"patient {patient_id}  Info.cfg={info}")

    net, cfg, meta = load_model(model_path)
    log(f"checkpoint epoch={meta['epoch']} val_loss={meta['val_loss']:.4f} "
        f"(fourier_L={cfg['fourier_L']}, grid_res={grid_res})")

    out: dict = {"patient": patient_id, "info": info, "checkpoint": meta,
                 "phases": {}}

    for phase in phases:
        frame_no = int(info[phase.upper()])
        seg_path = patient_dir / f"{patient_id}_frame{frame_no:02d}_gt.nii"
        seg = load_segmentation(seg_path)
        log(f"── {phase.upper()} frame {frame_no}: {seg.path.name} "
            f"shape={seg.labels.shape} spacing={seg.spacing}")

        contours = extract_contours(seg)
        log(f"  contours: {len(contours['xyz'])} pts, scale={contours['scale']:.2f} mm, "
            f"{len(contours['slices'])} slices")

        phase_val = 0.0 if phase.lower() == "ed" else 1.0
        model_geom = build_model_geometry(net, cfg, contours, grid_res=grid_res,
                                          phase_val=phase_val)
        residual = slice_residual_mm(net, model_geom["latent"], contours["xyz"],
                                     contours["tissue"], contours["scale"])
        log(f"  CardioSDF slice residual: {residual:.2f} mm")
        for rep in model_geom["reports"]:
            log(f"    {rep['surface']}: watertight {rep['watertight_in']} -> "
                f"{rep['watertight_out']} via {rep['repaired_with']}, "
                f"{rep['faces_out']} faces, {rep['volume_ml']:.1f} mL")

        voxel_geom = build_voxel_geometry(seg, iso_pitch=pitch)
        for rep in voxel_geom["reports"]:
            log(f"    {rep['surface']}: watertight {rep['watertight_in']} -> "
                f"{rep['watertight_out']} via {rep['repaired_with']}, "
                f"{rep['faces_out']} faces, {rep['volume_ml']:.1f} mL")

        model_ctx = {"net": net, "latent": model_geom["latent"],
                     "centroid": contours["centroid"], "scale": contours["scale"]}
        analysed = {
            "model": analyse_geometry("model", model_geom["endo"], model_geom["epi"],
                                      seg, pitch, model_ctx),
            "voxel": analyse_geometry("voxel", voxel_geom["endo"], voxel_geom["epi"],
                                      seg, pitch),
        }
        out["phases"][phase] = {
            "frame": frame_no, "seg": seg, "contours": contours,
            "slice_residual_mm": residual,
            "mesh_reports": model_geom["reports"] + voxel_geom["reports"],
            "geometries": analysed,
        }
    return out


# ──────────────────────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────────────────────
def build_tables(payload: dict) -> dict:
    method_rows, aha_rows, agree_rows, mesh_rows, diag_rows = [], [], [], [], []

    for phase, block in payload["phases"].items():
        for rep in block["mesh_reports"]:
            mesh_rows.append({"phase": phase.upper(), **rep})

        geoms = block["geometries"]
        method_names = list(geoms["voxel"]["results"].keys())
        for source, geom in geoms.items():
            for name, res in geom["results"].items():
                method_rows.append({"phase": phase.upper(), "geometry": source,
                                    **res.summary()})
                diag_rows.append({"phase": phase.upper(), "geometry": source,
                                  "method": name,
                                  **{k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in res.diagnostics.items()}})
                seg_means = agr.segment_means(res.values, geom["aha_ids"])
                for i, seg_name in enumerate(AHA_17_NAMES):
                    aha_rows.append({"phase": phase.upper(), "geometry": source,
                                     "method": name, "segment_id": i + 1,
                                     "segment": seg_name,
                                     "mean_mm": None if np.isnan(seg_means[i])
                                     else round(float(seg_means[i]), 3)})

        for name in method_names:
            if name not in geoms["model"]["results"]:
                continue
            m = agr.segment_means(geoms["model"]["results"][name].values,
                                  geoms["model"]["aha_ids"])
            v = agr.segment_means(geoms["voxel"]["results"][name].values,
                                  geoms["voxel"]["aha_ids"])
            agree_rows.append({"phase": phase.upper(), "method": name,
                               "model_mean_mm": round(float(np.nanmean(m)), 3),
                               "voxel_mean_mm": round(float(np.nanmean(v)), 3),
                               **agr.agreement(m, v)})

    tables = {
        "method_summary": pd.DataFrame(method_rows),
        "aha17": pd.DataFrame(aha_rows),
        "model_vs_voxel_agreement": pd.DataFrame(agree_rows),
        "mesh_quality": pd.DataFrame(mesh_rows),
        "diagnostics": pd.DataFrame(diag_rows),
    }

    if {"ED", "ES"}.issubset({p.upper() for p in payload["phases"]}):
        rows = []
        for source in ("model", "voxel"):
            ed = payload["phases"]["ED"]["geometries"][source]
            es = payload["phases"]["ES"]["geometries"][source]
            for name in ed["results"]:
                if name not in es["results"]:
                    continue
                a = agr.segment_means(ed["results"][name].values, ed["aha_ids"])
                b = agr.segment_means(es["results"][name].values, es["aha_ids"])
                with np.errstate(invalid="ignore", divide="ignore"):
                    nor = 100.0 * (b - a) / a
                for i, seg_name in enumerate(AHA_17_NAMES):
                    rows.append({"geometry": source, "method": name,
                                 "segment_id": i + 1, "segment": seg_name,
                                 "ed_mm": None if np.isnan(a[i]) else round(float(a[i]), 3),
                                 "es_mm": None if np.isnan(b[i]) else round(float(b[i]), 3),
                                 "nor_pct": None if not np.isfinite(nor[i])
                                 else round(float(nor[i]), 2)})
        tables["aha17_thickening"] = pd.DataFrame(rows)
    return tables


def save_npz(payload: dict, out_dir: Path) -> None:
    for phase, block in payload["phases"].items():
        arrays = {}
        for source, geom in block["geometries"].items():
            arrays[f"{source}_endo_v"] = np.asarray(geom["endo"].vertices, np.float32)
            arrays[f"{source}_endo_f"] = np.asarray(geom["endo"].faces, np.int32)
            arrays[f"{source}_epi_v"] = np.asarray(geom["epi"].vertices, np.float32)
            arrays[f"{source}_epi_f"] = np.asarray(geom["epi"].faces, np.int32)
            arrays[f"{source}_aha"] = geom["aha_ids"]
            arrays[f"{source}_band"] = geom["band"]
            for name, res in geom["results"].items():
                key = name.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
                arrays[f"{source}_wt_{key}"] = res.values
        arrays["contours_xyz_mm"] = block["contours"]["xyz_mm"]
        arrays["contours_tissue"] = block["contours"]["tissue"]
        np.savez_compressed(out_dir / f"demo_{payload['patient']}_{phase.upper()}.npz",
                            **arrays)


def export_meshes(payload: dict, out_dir: Path) -> None:
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for phase, block in payload["phases"].items():
        for source, geom in block["geometries"].items():
            for surf in ("endo", "epi"):
                geom[surf].export(mesh_dir / f"{payload['patient']}_{phase.upper()}_{source}_{surf}.ply")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", type=Path, default=DEFAULT_PATIENT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--pitch", type=float, default=0.75,
                        help="isotropic voxel pitch (mm) for volumetric methods")
    parser.add_argument("--grid-res", type=int, default=96)
    parser.add_argument("--phases", nargs="+", default=["ED", "ES"])
    parser.add_argument("--skip-phantoms", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if not args.skip_phantoms:
        log("=== phantom validation ===")
        phantom_df = run_phantom_validation(args.pitch)
        phantom_df.to_csv(args.out / "phantom_validation.csv", index=False)
        log(f"wrote {args.out / 'phantom_validation.csv'}")

    log("=== patient demo ===")
    payload = run_patient(args.patient, args.model, args.pitch, args.grid_res, args.phases)

    tables = build_tables(payload)
    for name, df in tables.items():
        df.to_csv(args.out / f"{name}.csv", index=False)
        log(f"wrote {args.out / f'{name}.csv'} ({len(df)} rows)")

    save_npz(payload, args.out)
    export_meshes(payload, args.out)

    (args.out / "run_config.json").write_text(json.dumps({
        "patient": str(args.patient), "model": str(args.model),
        "iso_pitch_mm": args.pitch, "sdf_grid_res": args.grid_res,
        "phases": args.phases, "checkpoint": payload["checkpoint"],
        "analysis_band": VALID_LONG_AXIS_BAND,
        "slice_residual_mm": {p: b["slice_residual_mm"] for p, b in payload["phases"].items()},
        "total_runtime_s": round(time.perf_counter() - t0, 1),
    }, indent=2))

    print()
    print(tables["model_vs_voxel_agreement"].to_string(index=False))
    print()
    print(tables["method_summary"].to_string(index=False))
    log(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
