"""Cohort evaluation for the Results chapter: CardioSDF vs voxel-derived geometry.

Computes exactly the quantities reported in ``chapters/04-results.tex`` and
nothing else: the reconstruction-quality table, the wall-thickness table for the
four selected methods, the point-by-point agreement table and the AHA-17
regional thickening table.

Every watertight mesh is written to ``<out>/cache`` as PLY the first time it is
built and reloaded afterwards, so re-running with different metric options never
repeats the reconstruction. Per-patient metric payloads are cached the same way.

    python run_cohort.py --data-root /path/to/ACDC/training --workers 4
    python run_cohort.py --data-root ... --aggregate-only     # tables from cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
import trimesh                                                       # noqa: E402
from scipy.ndimage import distance_transform_edt, zoom as ndi_zoom   # noqa: E402

import thickness as tk                                               # noqa: E402
from geometry import (                                               # noqa: E402
    AHA_17_NAMES, Segmentation, _clean_inside, _crop_bounds, assign_aha17,
    build_model_geometry, build_voxel_geometry, enforce_nesting, extract_contours,
    load_segmentation, long_axis_frame, outward_normals, read_info_cfg,
)
from recon_metrics import reconstruction_quality                     # noqa: E402

HERE = Path(__file__).resolve().parent
THESIS = HERE.parents[1]
DEFAULT_MODEL = THESIS / "notebooks" / "inr_sdf_combined_fresh_ed_mix_v1_final.pt"
DEFAULT_OUT = THESIS / "scripts" / "cohort_nor"

VALID_LONG_AXIS_BAND = (0.04, 0.97)
METHODS = ["Laplace field", "Yezzi-Prince", "SDF cone rays", "EDT boundary sum"]
REFERENCE_METHOD = "Laplace field"
AGREEMENT_POINTS_PER_PATIENT = 560

_MODEL_CACHE: dict = {}
PROVENANCE_EXTRA: dict[str, object] = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_provenance(args, requested: list[Path], payloads: list[dict]) -> None:
    source_paths = {
        "cohort_runner": Path(__file__),
        "geometry": HERE / "geometry.py",
        "model_loader": HERE / "cardiosdf_model.py",
        "reconstruction_metrics": HERE / "recon_metrics.py",
        "thickness": HERE / "thickness.py",
    }
    source_files = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in source_paths.items()
    }
    completed = sorted(str(payload.get("patient", "")) for payload in payloads)
    requested_names = [patient.name for patient in requested]
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python": {"executable": sys.executable, "version": sys.version},
        "model": {
            "path": str(args.model.resolve()),
            "sha256": _sha256(args.model),
        },
        "dataset": {
            "root": str(args.data_root.resolve()),
            "group": args.group,
            "requested_patients": requested_names,
            "completed_patients": completed,
            "failed_patients": sorted(set(requested_names) - set(completed)),
        },
        "evaluation": {
            "voxel_pitch_mm": args.pitch,
            "grid_resolution": args.grid_res,
            "reconstruction_phase": "ED",
            "wall_thickness_phases": ["ED", "ES"],
            "workers": args.workers,
            "force_mesh": args.force_mesh,
            "force_metrics": args.force_metrics,
        },
        "source_files": source_files,
        "extra": PROVENANCE_EXTRA,
    }
    (args.out / "provenance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


# ──────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ──────────────────────────────────────────────────────────────────────────
def find_frame(patient_dir: Path, patient_id: str, frame_no: int) -> Path:
    stem = f"{patient_id}_frame{frame_no:02d}_gt"
    for candidate in (f"{stem}.nii.gz", f"{stem}.nii", stem):
        path = patient_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"{stem} not found in {patient_dir}")


def discover_patients(data_root: Path, group: str | None) -> list[Path]:
    out = []
    for cfg in sorted(data_root.glob("*/Info.cfg")):
        info = read_info_cfg(cfg)
        if group and info.get("Group", "").upper() != group.upper():
            continue
        out.append(cfg.parent)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Cached geometry
# ──────────────────────────────────────────────────────────────────────────
def load_net(model_path: Path):
    if "net" not in _MODEL_CACHE:
        import torch
        torch.set_num_threads(1)
        from cardiosdf_model import load_model
        net, cfg, meta = load_model(model_path)
        _MODEL_CACHE.update(net=net, cfg=cfg, meta=meta)
    return _MODEL_CACHE["net"], _MODEL_CACHE["cfg"], _MODEL_CACHE["meta"]


def cached_geometry(cache: Path, patient_id: str, phase: str, seg: Segmentation,
                    model_path: Path, pitch: float, grid_res: int,
                    force: bool) -> tuple[dict, dict, dict]:
    """Return (model, voxel, reports); build and persist the meshes on a miss."""
    tags = [(src, surf) for src in ("model", "voxel") for surf in ("endo", "epi")]
    paths = {(s, f): cache / f"{patient_id}_{phase}_{s}_{f}.ply" for s, f in tags}
    meta_path = cache / f"{patient_id}_{phase}_meshes.json"

    if not force and meta_path.exists() and all(p.exists() for p in paths.values()):
        meshes = {src: {surf: trimesh.load(paths[(src, surf)], process=False)
                        for surf in ("endo", "epi")} for src in ("model", "voxel")}
        return meshes["model"], meshes["voxel"], json.loads(meta_path.read_text())

    net, cfg, _ = load_net(model_path)
    contours = extract_contours(seg)
    model_geom = build_model_geometry(net, cfg, contours, grid_res=grid_res,
                                      phase_val=0.0 if phase == "ED" else 1.0)
    voxel_geom = build_voxel_geometry(seg, iso_pitch=pitch)

    model = {"endo": model_geom["endo"], "epi": model_geom["epi"]}
    voxel = {"endo": voxel_geom["endo"], "epi": voxel_geom["epi"]}
    for src, meshes in (("model", model), ("voxel", voxel)):
        for surf, mesh in meshes.items():
            mesh.export(paths[(src, surf)])
    reports = {"reports": model_geom["reports"] + voxel_geom["reports"],
               "n_contour_slices": len(contours["slices"])}
    meta_path.write_text(json.dumps(reports, indent=2, default=str))
    return model, voxel, reports


# ──────────────────────────────────────────────────────────────────────────
# Measurements
# ──────────────────────────────────────────────────────────────────────────
def segmentation_reference_mm(seg: Segmentation, pitch: float) -> float:
    """Mean transmural distance measured directly on the input label mask."""
    spacing = np.asarray(seg.spacing, np.float64)
    start, stop = _crop_bounds(seg.epi, spacing, 8.0)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    factors = spacing / pitch

    def resample(mask: np.ndarray) -> np.ndarray:
        return ndi_zoom(mask[sl].astype(np.float32), factors, order=1,
                        prefilter=False) >= 0.5

    lv = _clean_inside(resample(seg.lv))
    epi = _clean_inside(resample(seg.epi) | lv)
    myo = epi & ~lv
    if not myo.any():
        return float("nan")
    d_endo = distance_transform_edt(~lv, sampling=(pitch,) * 3)
    d_epi = distance_transform_edt(epi, sampling=(pitch,) * 3)
    return float((d_endo + d_epi)[myo].mean())


def measure_geometry(endo, epi, seg: Segmentation, pitch: float) -> dict:
    """Four selected wall-thickness methods on one geometry, plus AHA-17 ids."""
    endo, _ = enforce_nesting(endo, epi)
    normals = outward_normals(endo, np.asarray(epi.vertices))
    verts = np.asarray(endo.vertices, np.float64)

    frame = long_axis_frame(endo, seg)
    t_axis = np.clip((verts[:, 2] - frame["base_z"]) /
                     (frame["apex_z"] - frame["base_z"]), 0.0, 1.0)
    band = (t_axis >= VALID_LONG_AXIS_BAND[0]) & (t_axis <= VALID_LONG_AXIS_BAND[1])

    ctx = tk.build_volume_context(endo, epi, pitch)
    phi, _ = tk.solve_laplace(ctx)

    values = {
        "Laplace field": tk.method_laplace_streamline(ctx, verts, normals, phi).values,
        "Yezzi-Prince": tk.method_yezzi_prince(ctx, verts, normals, phi).values,
        "SDF cone rays": tk.method_cone_rays(endo, epi, normals).values,
        "EDT boundary sum": tk.method_edt_boundary_sum(ctx, verts, normals).values,
    }
    values = {k: np.where(band, v, np.nan).astype(np.float64) for k, v in values.items()}
    return {"values": values, "aha": assign_aha17(verts, frame),
            "band_fraction": float(band.mean())}


def stats(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"mean_mm": np.nan, "std_mm": np.nan, "p5_mm": np.nan, "p95_mm": np.nan}
    return {"mean_mm": float(v.mean()),
            "std_mm": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "p5_mm": float(np.percentile(v, 5)),
            "p95_mm": float(np.percentile(v, 95))}


def segment_means(values: np.ndarray, aha: np.ndarray) -> list:
    out = []
    for sid in range(1, 18):
        sel = values[(aha == sid) & np.isfinite(values)]
        out.append(float(sel.mean()) if sel.size else np.nan)
    return out


# ──────────────────────────────────────────────────────────────────────────
# One patient
# ──────────────────────────────────────────────────────────────────────────
def run_patient(patient_dir: Path, model_path: Path, out_dir: Path, pitch: float,
                grid_res: int, force_mesh: bool, force_metrics: bool) -> dict:
    warnings.filterwarnings("ignore")
    patient_id = patient_dir.name
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    payload_path = cache / f"{patient_id}_result.json"
    if payload_path.exists() and not (force_mesh or force_metrics):
        return json.loads(payload_path.read_text())

    info = read_info_cfg(patient_dir / "Info.cfg")
    payload: dict = {"patient": patient_id, "group": info.get("Group", "?"),
                     "recon": None, "wall": [], "aha": [], "agreement": {},
                     "reference_mm": {}}

    measured: dict = {}
    for phase in ("ED", "ES"):
        seg = load_segmentation(find_frame(patient_dir, patient_id, int(info[phase])))
        model, voxel, reports = cached_geometry(cache, patient_id, phase, seg,
                                                model_path, pitch, grid_res, force_mesh)
        payload["reference_mm"][phase] = segmentation_reference_mm(seg, pitch)

        if phase == "ED":
            recon = reconstruction_quality(model, voxel, pitch)
            recon["n_contour_slices"] = reports.get("n_contour_slices")
            payload["recon"] = recon

        measured[phase] = {src: measure_geometry(g["endo"], g["epi"], seg, pitch)
                           for src, g in (("model", model), ("voxel", voxel))}

        for src, block in measured[phase].items():
            for method, values in block["values"].items():
                payload["wall"].append({"patient": patient_id, "phase": phase,
                                        "geometry": src, "method": method,
                                        "valid_fraction": float(np.isfinite(values).mean()),
                                        **stats(values)})
                for sid, mean_mm in enumerate(segment_means(values, block["aha"]), start=1):
                    payload["aha"].append({"patient": patient_id, "phase": phase,
                                           "geometry": src, "method": method,
                                           "segment_id": sid, "mean_mm": mean_mm})

    model_ed = measured["ED"]["model"]["values"]
    finite = np.all([np.isfinite(model_ed[m]) for m in METHODS], axis=0)
    idx = np.flatnonzero(finite)
    if idx.size:
        take = np.linspace(0, idx.size - 1, min(AGREEMENT_POINTS_PER_PATIENT, idx.size))
        idx = idx[np.unique(take.astype(np.int64))]
        payload["agreement"] = {m: model_ed[m][idx].tolist() for m in METHODS}

    payload_path.write_text(json.dumps(payload))
    return payload


def _worker(args) -> dict:
    return run_patient(*args)


# ──────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────
def icc_two_one(a: np.ndarray, b: np.ndarray) -> float:
    matrix = np.column_stack([a, b])
    n, k = matrix.shape
    grand = matrix.mean()
    ms_rows = k * ((matrix.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((matrix.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0) + grand
    ms_err = (residual ** 2).sum() / ((n - 1) * (k - 1))
    return float((ms_rows - ms_err) /
                 (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n))


def aggregate(payloads: list[dict], out_dir: Path) -> None:
    recon = pd.DataFrame([{"patient": p["patient"], **p["recon"]}
                          for p in payloads if p["recon"]])
    wall = pd.DataFrame([r for p in payloads for r in p["wall"]])
    aha = pd.DataFrame([r for p in payloads for r in p["aha"]])
    reference = pd.DataFrame([{"patient": p["patient"], "phase": phase, "ref_mm": value}
                              for p in payloads for phase, value in p["reference_mm"].items()])

    recon.to_csv(out_dir / "recon_quality.csv", index=False)
    wall.to_csv(out_dir / "wall_methods.csv", index=False)
    aha.to_csv(out_dir / "aha17.csv", index=False)
    reference.to_csv(out_dir / "reference_thickness.csv", index=False)

    pooled = {m: np.concatenate([np.asarray(p["agreement"][m]) for p in payloads
                                 if p.get("agreement")]) for m in METHODS}
    rows = []
    ref = pooled[REFERENCE_METHOD]
    for method in METHODS:
        if method == REFERENCE_METHOD:
            continue
        other = pooled[method]
        ok = np.isfinite(ref) & np.isfinite(other)
        a, b = ref[ok], other[ok]
        diff = b - a
        rows.append({"method": method, "n": int(ok.sum()),
                     "pearson_r": float(np.corrcoef(a, b)[0, 1]),
                     "mae_mm": float(np.abs(diff).mean()),
                     "rmse_mm": float(np.sqrt((diff ** 2).mean())),
                     "bias_mm": float(diff.mean()),
                     "loa_halfwidth_mm": float(1.96 * diff.std(ddof=1)),
                     "icc_2_1": icc_two_one(a, b)})
    agreement = pd.DataFrame(rows)
    agreement.to_csv(out_dir / "agreement.csv", index=False)

    write_summary(recon, wall, aha, reference, agreement, out_dir)


def write_summary(recon, wall, aha, reference, agreement, out_dir: Path) -> None:
    lines = [f"patients: {recon['patient'].nunique()}", ""]

    lines.append("== reconstruction quality (ED, mean +/- sd across patients) ==")
    for column in ("endo_chamfer_mm", "epi_chamfer_mm", "endo_assd_mm", "epi_assd_mm",
                   "endo_hd95_mm", "epi_hd95_mm", "endo_dice", "myo_dice", "endo_iou",
                   "myo_iou", "normal_consistency", "fscore_1mm", "fscore_2mm",
                   "vol_ratio_endo", "vol_ratio_epi", "vol_ratio_myo"):
        if column in recon:
            lines.append(f"  {column:22s} {recon[column].mean():.3f} +/- "
                         f"{recon[column].std(ddof=1):.3f}")
    for column in ("endo_watertight", "epi_watertight"):
        lines.append(f"  {column:22s} {100.0 * recon[column].mean():.0f}%")

    ed = wall[wall["phase"] == "ED"]
    lines += ["", "== wall thickness (ED, cohort mean of per-patient statistics) ==",
              f"  segmentation reference: "
              f"{reference[reference['phase'] == 'ED']['ref_mm'].mean():.2f} mm",
              "  method                geometry   mean   std    p5     p95"]
    for method in METHODS:
        for geometry in ("model", "voxel"):
            sub = ed[(ed["method"] == method) & (ed["geometry"] == geometry)]
            lines.append(f"  {method:20s}  {geometry:8s}  "
                         f"{sub['mean_mm'].mean():5.2f}  {sub['std_mm'].mean():5.2f}  "
                         f"{sub['p5_mm'].mean():5.2f}  {sub['p95_mm'].mean():5.2f}")

    lines += ["", "== model vs voxel, per-patient means (ED) =="]
    for method in METHODS:
        pivot = ed[ed["method"] == method].pivot(index="patient", columns="geometry",
                                                 values="mean_mm").dropna()
        if len(pivot) > 1:
            lines.append(f"  {method:20s} model {pivot['model'].mean():5.2f}  "
                         f"voxel {pivot['voxel'].mean():5.2f}  "
                         f"bias {(pivot['model'] - pivot['voxel']).mean():+5.2f}  "
                         f"r {np.corrcoef(pivot['model'], pivot['voxel'])[0, 1]:.2f}")

    lines += ["", "== agreement with the Laplace field reference (pooled points) =="]
    for _, row in agreement.iterrows():
        lines.append(f"  {row['method']:20s} r {row['pearson_r']:.2f}  "
                     f"MAE {row['mae_mm']:.2f}  RMSE {row['rmse_mm']:.2f}  "
                     f"bias {row['bias_mm']:+.2f} (+/-{row['loa_halfwidth_mm']:.2f})  "
                     f"ICC {row['icc_2_1']:.2f}  n={int(row['n'])}")

    lines += ["", "== AHA-17 (Laplace field) =="]
    sub = aha[aha["method"] == REFERENCE_METHOD]
    for geometry in ("model", "voxel"):
        lines.append(f"  -- {geometry} --")
        for sid in range(1, 18):
            block = sub[(sub["geometry"] == geometry) & (sub["segment_id"] == sid)]
            pivot = block.pivot(index="patient", columns="phase", values="mean_mm").dropna()
            if pivot.empty:
                continue
            ed_mm, es_mm = pivot["ED"].mean(), pivot["ES"].mean()
            lines.append(f"   {sid:2d} {AHA_17_NAMES[sid - 1]:22s} "
                         f"{ed_mm:5.2f}  {es_mm:5.2f}  {100 * (es_mm - ed_mm) / ed_mm:+6.1f}%")
        overall = sub[sub["geometry"] == geometry].pivot_table(
            index=["patient", "segment_id"], columns="phase", values="mean_mm").dropna()
        lines.append(f"   overall {overall['ED'].mean():5.2f}  {overall['ES'].mean():5.2f}  "
                     f"{100 * (overall['ES'].mean() - overall['ED'].mean()) / overall['ED'].mean():+6.1f}%")

    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text)
    print(text)


# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory holding patientXXX/ subfolders")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--group", default="NOR")
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pitch", type=float, default=1.0)
    parser.add_argument("--grid-res", type=int, default=96)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--force-mesh", action="store_true")
    parser.add_argument("--force-metrics", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cache").mkdir(exist_ok=True)

    patients = discover_patients(args.data_root, args.group)
    if args.patients:
        patients = [p for p in patients if p.name in set(args.patients)]
    if args.limit:
        patients = patients[:args.limit]
    if not patients:
        raise SystemExit(f"no {args.group} patients under {args.data_root}")
    log(f"{len(patients)} {args.group} patients: {', '.join(p.name for p in patients)}")

    if args.aggregate_only:
        payloads = [json.loads(p.read_text())
                    for p in sorted((args.out / "cache").glob("*_result.json"))]
        aggregate(payloads, args.out)
        return

    t0 = time.perf_counter()
    jobs = [(p, args.model, args.out, args.pitch, args.grid_res,
             args.force_mesh, args.force_metrics) for p in patients]
    payloads = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, job): job[0].name for job in jobs}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    payloads.append(future.result())
                    log(f"done {name}  ({len(payloads)}/{len(jobs)})")
                except Exception as exc:                     # noqa: BLE001
                    log(f"FAILED {name}: {type(exc).__name__}: {exc}")
    else:
        for job in jobs:
            try:
                payloads.append(_worker(job))
                log(f"done {job[0].name}  ({len(payloads)}/{len(jobs)})")
            except Exception as exc:                         # noqa: BLE001
                log(f"FAILED {job[0].name}: {type(exc).__name__}: {exc}")

    log(f"{len(payloads)} patients in {time.perf_counter() - t0:.0f} s")
    if payloads:
        aggregate(payloads, args.out)
        _write_provenance(args, patients, payloads)


if __name__ == "__main__":
    main()
