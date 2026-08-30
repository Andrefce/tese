# Results Provenance Record

## Status

**Working evidence freeze complete.** The existing cached cohort is the evidence
base used for the thesis revision. Its table files are fixed by the hashes below,
its patient membership and phase coverage are verified, and its patient-level
outputs reproduce the numbers in Chapter 4. The historical cache does not embed
the exact checkpoint identifier or original command; this remains a
reproducibility limitation rather than a reason to discard the completed
results.

## Current Chapter 4 result source

Directory: `test-new-model/cohort_full_nor_hcm10/`

The values in `summary.txt` match the reconstruction, wall-thickness, agreement,
and AHA-17 tables in `chapters/04-results.tex`.

| Artifact | SHA-256 |
| --- | --- |
| `summary.txt` | `5399357e7bc72606979ba23cac93277a6e212bef5906708d0eddf4d16a8bb401` |
| `recon_quality.csv` | `5d6a13f0f5ca1d0d97233f32b592f89d0b52ab2adadb64679f81f0775913c657` |
| `wall_methods.csv` | `819573de0511a5175f129605dc36ee19550a21fe11ecf8a0a07a8a6f078d78e8` |
| `agreement.csv` | `b11a9500de7d6b66a103c48b70746a052e9231b6ee475b3669b2f38816d56776` |
| `aha17.csv` | `f3d5c3fe3aa06b6216949f087c8a5166bd59a925a26989302a999f682fac719c` |

## Matched linear contour-lofting baseline

The baseline was evaluated on 2026-08-24 with the same 30 ED patients and the
cached segmentation-derived comparator meshes used by the reconstruction
quality analysis. The evaluator extracts the same SAX contour rings supplied to
the model, joins corresponding points on adjacent rings, closes the terminal
rings with planar caps, rasterises the result at 1.0 mm, and applies the shared
watertight repair and metric code.

```bash
/home/C052246/tese/.venv/bin/python scripts/evaluate_contour_lofting.py \
  --data-root notebooks/data/training \
  --cohort test-new-model/cohort_full_nor_hcm10 \
  --workers 4 --bootstrap-samples 10000
```

| Artifact | SHA-256 |
| --- | --- |
| `scripts/evaluate_contour_lofting.py` | `2615396e4fe936ad3b60dc826b93c307d5029babd983ade25c4dc31c9541813e` |
| `contour_lofting.csv` | `76917d89db9feb7d95f41ed7af6b69ac000d92828cba51ab018119c67b79080a` |
| `contour_lofting_summary.csv` | `62bac5f5a6221e62c9d094aa2f5f93d884bbd1c8ab2cc9f7b21aadd097c86963` |

The model and lofting baseline were watertight on both surfaces in all 30
patients after repair. The model had lower endocardial/epicardial Chamfer
distance (1.22/1.05 mm versus 1.60/1.86 mm), lower HD95 (3.30/3.47 mm versus
6.88/8.35 mm), and higher myocardial Dice (0.85 versus 0.81). The patient-level
paired bootstrap intervals reported in Chapter 4 were computed from 10,000
resamples with deterministic seeds.

## RBF implicit and shape-model fitting baselines

Both fitting baselines were evaluated on 2026-08-29 with the same 30 ED
patients, the same cached segmentation-derived comparator meshes, and the same
metric code. Their input contour rings are read from the per-patient sample
cache `test-new-model/cache/{patient}_ED.npz`, denormalised back to world
millimetres. The RBF baseline is a thin-plate-spline implicit surface fitted to
the rings with off-surface constraints; the shape-model baseline fits the public
UK Digital Heart Project left-ventricular model by alternating similarity
registration and regularised mode estimation.

```powershell
C:/Python313/python.exe scripts/evaluate_fitting_baselines.py `
  --cohort test-new-model/cohort_full_nor_hcm10 `
  --samples test-new-model/cache --workers 4 --bootstrap-samples 10000
```

| Artifact | SHA-256 |
| --- | --- |
| `scripts/evaluate_fitting_baselines.py` | `28ea27de0cb695e2a0c1419ea3958d6bcb56a4f5d534fce541711ee849750245` |
| `fitting_baselines.csv` | `ea979d845a383cf28944dfbaa7ba5207899867d9d5d03af2a658c6e2c0f8ff2a` |
| `fitting_baselines_summary.csv` | `cf80e1000a7cc1444e089f2aec7154edc328afc1a1a5a2e37c99355965f38335` |
| `fitting_baseline_wall_methods.csv` | `29868ef375e6625ab54b1a89beb5ba7d0250af0fa4cc5f9fe720eca17ca0ff5a` |
| `fitting_baseline_wall_summary.csv` | `5a4332bf74d5bdaab9de9305ae873d8282e3d636f3ce80021fd4a385f9d2aa87` |

Both baselines were watertight on both surfaces in all 30 patients. Endocardial
Chamfer distance was 1.55 mm for the RBF fit and 2.34 mm for the shape-model
fit, against 1.22 mm for the model; myocardial Dice was 0.81 and 0.68 against
0.85. Every paired baseline-minus-model difference excludes zero except the
cavity volume ratio of the RBF fit.

The same four wall-thickness estimators were applied to both fitted mesh pairs
at 1.0 mm isotropic pitch. The RBF fit remained close to the model, with
per-patient mean-thickness correlations from 0.95 to 0.99, while the shape-model
fit correlations ranged from 0.67 to 0.77. Per-patient JSON checkpoints in
`fit_cache/` allow interrupted runs to resume without repeating completed PDE
measurements.

The shape-model baseline is end-diastolic only: the published model ships
end-diastolic modes alone, so no end-systolic fit exists. This is also why the
end-systolic shape-model panel of `fig:recon-ed-es-meshes` is empty.

A defect found during this run is recorded here because it affected earlier
output: `initial_alignment` in `scripts/fig_baseline_rbf_ssm.py` derived the
ventricular long axis from the principal axis of the whole contour cloud, which
points in-plane on short stacks and left the fitted shape model lying on its
side (patient072 endocardial Dice 0.18). The axis is now taken from the line
through the per-slice ring centroids. All reported shape-model numbers come from
the corrected code.

## Verified cohort facts

- 30 ACDC patients: HCM `patient021`--`patient030` and NOR
  `patient061`--`patient080`.
- Group composition: 10 HCM and 20 NOR.
- Reconstruction-quality CSV: ED only.
- Wall-thickness and AHA-17 CSVs: ED and ES.
- All four selected thickness methods are present on model and voxel geometry.
- All four selected thickness methods are present on the ED RBF and shape-model
  fits.
- Cached patient payloads contain mesh-repair reports and per-patient metrics,
  but no checkpoint hash or command line.

## Current evaluator behaviour

The current evaluator reads physical spacing from NIfTI header zooms and applies
that spacing explicitly in the world coordinate transform. Its defaults are a
1.0 mm isotropic voxel comparison grid and a reconstruction grid resolution of
96. Reconstruction quality is calculated at ED; thickness and AHA-17 are
calculated at ED and ES. Model and voxel surfaces pass through the same
watertight-repair layer.

These are verified properties of the current code. Because the historical
command and source hashes were not stored, they do not prove that every cached
mesh was generated by the current code revision.

Future successful cohort runs now write `provenance.json` from
`scripts/eval_demo/run_cohort.py` with the checkpoint hash, command, source-code
hashes, cohort membership, phase coverage, pitch, grid resolution, and failed
patients.

## Candidate checkpoints

| Candidate | SHA-256 | Status |
| --- | --- | --- |
| `u1u2` v2 | `fafea6ae81c1fdbfc93159cba1cce000f73b59bb9bbd5cc78ead9977c64dda0c` | Candidate, not linked to the current Chapter 4 cache. |
| `u1u2_bal` v2 | `4723edc1163aa34d1ab0ebcae0e8f3a138f386949822f801b51dda5599c6e90a` | Candidate, not linked to the current Chapter 4 cache. |
| `u1u2_e50` v2 | `6804c392d8b22937df063a8c7951481ea9c2bdcd263b62667eed8014f366f5d6` | Most likely source, but not proven. |

Earlier-model checkpoints and the result directories under `scripts/cohort_nor/`
and `scripts/cohort_hcm/` are explicitly excluded from thesis tables, figures,
and comparisons.

Prior session history shows that the full 20-NOR plus 10-HCM run was requested
immediately after the `u1u2_e50` checkpoint was evaluated. The result directory
was created later and is used by the v2 figure scripts. This is supporting
history, not sufficient provenance; the cache itself must identify the model.

## Available raw data

The `notebooks/data/` copy contains Git LFS pointer stubs, but a complete
100-patient ACDC tree is available under `test-new-model/training/`. The
evaluator discovers 20 NOR and 20 HCM cases there; representative NIfTI headers
retain the physical spacing, including the 10~mm slice spacing.

## Optional provenance-hardening rerun

The thesis revision uses the existing hashed cache. A future rerun may be used to
embed the checkpoint and command in `provenance.json`; it should use the exact
same 30 patients. For the leading historical checkpoint candidate:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES='' python test-new-model/run_cohort_v2.py \
  --model test-new-model/runs/u1u2_e50/cardiosdf_v2_best.pt \
  --data-root test-new-model/training \
  --group '' \
  --patients patient021 patient022 patient023 patient024 patient025 \
    patient026 patient027 patient028 patient029 patient030 \
    patient061 patient062 patient063 patient064 patient065 patient066 \
    patient067 patient068 patient069 patient070 patient071 patient072 \
    patient073 patient074 patient075 patient076 patient077 patient078 \
    patient079 patient080 \
  --out test-new-model/cohort_authoritative_v2 \
  --pitch 1.0 --grid-res 96 --workers 4 --force-mesh --force-metrics
```

Such a rerun may replace the frozen cache only when:

1. `provenance.json` identifies the selected checkpoint and all 30 patients;
2. no patient appears in `failed_patients`;
3. the summary is reproducible from the patient-level payloads;
4. slice spacing and physical units are verified on representative NIfTI files;
5. gains and regressions of the selected checkpoint are reported together.
