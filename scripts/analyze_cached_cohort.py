"""Derive patient-level analyses from a ``run_cohort.py`` output directory.

This script does not reconstruct meshes and does not require the raw NIfTI data.
It ranks reconstruction cases, summarises method validity and phase behaviour,
and estimates NOR-versus-HCM effects from patient-level measurements. Outputs
remain unverified unless the source directory contains a complete
``provenance.json`` written by the cohort evaluator.

Example:
    .venv/bin/python scripts/analyze_cached_cohort.py \
        --cohort test-new-model/cohort_full_nor_hcm10 \
        --out /tmp/cardiosdf-derived-analysis
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_METHOD = "Laplace field"
REQUIRED_FILES = ("recon_quality.csv", "wall_methods.csv", "aha17.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_groups(cache_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for result_path in sorted(cache_dir.glob("*_result.json")):
        payload = json.loads(result_path.read_text())
        rows.append({"patient": str(payload["patient"]), "group": str(payload["group"])})
    groups = pd.DataFrame(rows).drop_duplicates()
    if groups.empty or groups["patient"].duplicated().any():
        raise ValueError("cache does not define one group for each patient")
    return groups


def hedges_g(normal_values: np.ndarray, hcm_values: np.ndarray) -> float:
    normal_values = np.asarray(normal_values, dtype=float)
    hcm_values = np.asarray(hcm_values, dtype=float)
    degrees_freedom = len(normal_values) + len(hcm_values) - 2
    if degrees_freedom <= 0:
        return float("nan")
    pooled_variance = (
        (len(normal_values) - 1) * normal_values.var(ddof=1)
        + (len(hcm_values) - 1) * hcm_values.var(ddof=1)
    ) / degrees_freedom
    if pooled_variance <= 0 or not np.isfinite(pooled_variance):
        return float("nan")
    cohens_d = (hcm_values.mean() - normal_values.mean()) / np.sqrt(pooled_variance)
    correction = 1.0 - 3.0 / (4.0 * degrees_freedom - 1.0)
    return float(correction * cohens_d)


def bootstrap_effect(
    normal_values: np.ndarray,
    hcm_values: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    normal_values = np.asarray(normal_values, dtype=float)
    hcm_values = np.asarray(hcm_values, dtype=float)
    normal_draws = generator.choice(
        normal_values, size=(samples, len(normal_values)), replace=True
    )
    hcm_draws = generator.choice(hcm_values, size=(samples, len(hcm_values)), replace=True)
    differences = hcm_draws.mean(axis=1) - normal_draws.mean(axis=1)
    effect_sizes = np.asarray(
        [hedges_g(normal_draw, hcm_draw) for normal_draw, hcm_draw in zip(normal_draws, hcm_draws)],
        dtype=float,
    )
    effect_sizes = effect_sizes[np.isfinite(effect_sizes)]
    return {
        "difference_ci_low": float(np.quantile(differences, 0.025)),
        "difference_ci_high": float(np.quantile(differences, 0.975)),
        "hedges_g_ci_low": float(np.quantile(effect_sizes, 0.025)),
        "hedges_g_ci_high": float(np.quantile(effect_sizes, 0.975)),
    }


def build_patient_summary(
    groups: pd.DataFrame,
    wall_methods: pd.DataFrame,
    aha17: pd.DataFrame,
) -> pd.DataFrame:
    summary = groups.set_index("patient").copy()

    def wall_series(phase: str, geometry: str) -> pd.Series:
        selected = wall_methods[
            (wall_methods["phase"] == phase)
            & (wall_methods["geometry"] == geometry)
            & (wall_methods["method"] == REFERENCE_METHOD)
        ]
        return selected.set_index("patient")["mean_mm"]

    def aha_series(phase: str, geometry: str, aggregate: str) -> pd.Series:
        selected = aha17[
            (aha17["phase"] == phase)
            & (aha17["geometry"] == geometry)
            & (aha17["method"] == REFERENCE_METHOD)
        ]
        grouped = selected.groupby("patient")["mean_mm"]
        return grouped.mean() if aggregate == "mean" else grouped.max()

    summary["global_ed_laplace_mm"] = wall_series("ED", "model")
    summary["global_es_laplace_mm"] = wall_series("ES", "model")
    summary["max_segment_ed_laplace_mm"] = aha_series("ED", "model", "max")
    summary["model_aha_ed_mm"] = aha_series("ED", "model", "mean")
    summary["model_aha_es_mm"] = aha_series("ES", "model", "mean")
    summary["voxel_aha_ed_mm"] = aha_series("ED", "voxel", "mean")
    summary["voxel_aha_es_mm"] = aha_series("ES", "voxel", "mean")
    summary["model_systolic_thickening_pct"] = 100.0 * (
        summary["model_aha_es_mm"] - summary["model_aha_ed_mm"]
    ) / summary["model_aha_ed_mm"]
    summary["voxel_systolic_thickening_pct"] = 100.0 * (
        summary["voxel_aha_es_mm"] - summary["voxel_aha_ed_mm"]
    ) / summary["voxel_aha_ed_mm"]
    return summary.reset_index()


def group_effects(
    patient_summary: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    metrics = (
        "model_aha_ed_mm",
        "max_segment_ed_laplace_mm",
        "model_systolic_thickening_pct",
    )
    rows: list[dict[str, float | int | str]] = []
    for metric_index, metric in enumerate(metrics):
        normal_values = patient_summary.loc[
            patient_summary["group"] == "NOR", metric
        ].dropna().to_numpy(float)
        hcm_values = patient_summary.loc[
            patient_summary["group"] == "HCM", metric
        ].dropna().to_numpy(float)
        if len(normal_values) < 2 or len(hcm_values) < 2:
            raise ValueError(f"{metric} needs at least two patients in each group")
        intervals = bootstrap_effect(
            normal_values,
            hcm_values,
            samples=bootstrap_samples,
            seed=seed + metric_index,
        )
        rows.append(
            {
                "metric": metric,
                "normal_n": len(normal_values),
                "normal_mean": normal_values.mean(),
                "normal_sd": normal_values.std(ddof=1),
                "hcm_n": len(hcm_values),
                "hcm_mean": hcm_values.mean(),
                "hcm_sd": hcm_values.std(ddof=1),
                "hcm_minus_normal": hcm_values.mean() - normal_values.mean(),
                "hedges_g": hedges_g(normal_values, hcm_values),
                **intervals,
            }
        )
    return pd.DataFrame(rows)


def method_finite_fraction(wall_methods: pd.DataFrame) -> pd.DataFrame:
    return (
        wall_methods.groupby(["phase", "geometry", "method"], as_index=False)
        .agg(
            patient_count=("patient", "nunique"),
            finite_value_fraction_mean=("valid_fraction", "mean"),
            finite_value_fraction_sd=("valid_fraction", "std"),
            finite_value_fraction_min=("valid_fraction", "min"),
        )
        .sort_values(["phase", "geometry", "method"])
    )


def phase_method_summary(wall_methods: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    merged = wall_methods.merge(groups, on="patient", validate="many_to_one")
    return (
        merged.groupby(["group", "phase", "geometry", "method"], as_index=False)
        .agg(
            patient_count=("patient", "nunique"),
            mean_mm=("mean_mm", "mean"),
            between_patient_sd_mm=("mean_mm", "std"),
            mean_valid_fraction=("valid_fraction", "mean"),
        )
        .sort_values(["group", "phase", "geometry", "method"])
    )


def reconstruction_cases(reconstruction: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    frame = reconstruction.merge(groups, on="patient", validate="one_to_one")
    frame["cavity_volume_abs_error"] = (frame["vol_ratio_endo"] - 1.0).abs()
    frame["epicardial_volume_abs_error"] = (frame["vol_ratio_epi"] - 1.0).abs()
    specifications = (
        ("endo_chamfer_mm", True),
        ("epi_chamfer_mm", True),
        ("myo_dice", False),
        ("cavity_volume_abs_error", True),
        ("epicardial_volume_abs_error", True),
    )
    rows: list[dict[str, float | str]] = []
    for metric, lower_is_better in specifications:
        ordered = frame.sort_values(metric, ascending=lower_is_better).reset_index(drop=True)
        positions = {"best": 0, "median": len(ordered) // 2, "difficult": len(ordered) - 1}
        for role, position in positions.items():
            selected = ordered.iloc[position]
            rows.append(
                {
                    "metric": metric,
                    "role": role,
                    "patient": selected["patient"],
                    "group": selected["group"],
                    "value": float(selected[metric]),
                }
            )
    return pd.DataFrame(rows)


def provenance_status(cohort_dir: Path) -> tuple[str, dict | None]:
    provenance_path = cohort_dir / "provenance.json"
    if not provenance_path.exists():
        return "unverified_missing_provenance", None
    provenance = json.loads(provenance_path.read_text())
    failed = provenance.get("dataset", {}).get("failed_patients", [])
    return ("verified" if not failed else "incomplete_failed_patients"), provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    missing = [name for name in REQUIRED_FILES if not (args.cohort / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing cohort files: {', '.join(missing)}")
    args.out.mkdir(parents=True, exist_ok=True)

    reconstruction = pd.read_csv(args.cohort / "recon_quality.csv")
    wall_methods = pd.read_csv(args.cohort / "wall_methods.csv")
    aha17 = pd.read_csv(args.cohort / "aha17.csv")
    groups = load_groups(args.cohort / "cache")

    patients = set(groups["patient"])
    if set(reconstruction["patient"]) != patients:
        raise ValueError("reconstruction and cache patient sets differ")
    if set(wall_methods["patient"]) != patients or set(aha17["patient"]) != patients:
        raise ValueError("wall-thickness/AHA and cache patient sets differ")

    patient_summary = build_patient_summary(groups, wall_methods, aha17)
    effects = group_effects(patient_summary, args.bootstrap_samples, args.seed)
    finite_fraction = method_finite_fraction(wall_methods)
    phase_summary = phase_method_summary(wall_methods, groups)
    cases = reconstruction_cases(reconstruction, groups)

    outputs = {
        "patient_summaries.csv": patient_summary,
        "group_effects.csv": effects,
        "method_finite_fraction.csv": finite_fraction,
        "phase_method_summary.csv": phase_summary,
        "reconstruction_cases.csv": cases,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.out / filename, index=False)

    status, provenance = provenance_status(args.cohort)
    source_hashes = {
        name: sha256(args.cohort / name)
        for name in (*REQUIRED_FILES, "agreement.csv")
        if (args.cohort / name).exists()
    }
    manifest = {
        "analysis_status": status,
        "cohort": str(args.cohort.resolve()),
        "patient_count": len(patients),
        "groups": groups["group"].value_counts().sort_index().to_dict(),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "reference_method": REFERENCE_METHOD,
        "source_hashes": source_hashes,
        "source_provenance": provenance,
        "analysis_script_sha256": sha256(Path(__file__)),
        "outputs": sorted(outputs),
    }
    (args.out / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print(f"status: {status}")
    print(f"patients: {len(patients)} ({groups['group'].value_counts().to_dict()})")
    print(effects.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
