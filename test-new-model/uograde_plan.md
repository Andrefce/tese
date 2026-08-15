# CardioSDF — Architecture Upgrade Plan and Test Protocol

Working document. Everything in the *Evidence* section was measured in this
repository on 2026-08-13 (20 NOR + 10 HCM ACDC patients, 1.0 mm isotropic pitch,
checkpoint `notebooks/inr_sdf_combined_fresh_ed_mix_v1_final.pt`, epoch 695).
No number below is an estimate unless explicitly marked.

---

## 1. Evidence: what the current model does and does not do

### 1.1 Where it works

| Quantity (20 NOR, end-diastole) | Value |
| --- | --- |
| Watertight rate, endo / epi | 100% / 100% |
| Cavity / epicardial volume ratio | 0.978 ± 0.061 / 1.023 ± 0.051 |
| Endocardium / myocardium Dice | 0.863 ± 0.033 / 0.661 ± 0.069 |
| Laplace thickness, model vs voxel geometry | bias +0.04 mm, r = 0.84 |
| Regional (AHA-17) reproducibility, model vs voxel | ICC 0.57, MAE 0.93 mm |

### 1.2 Where it fails

| Failure | Measurement |
| --- | --- |
| End-systolic shrinkage | epi volume ratio 0.807, myocardial 0.769 (all 20 patients < 1) |
| ES thickness shortfall | 10.16 mm vs 12.04 mm on voxel geometry (−1.6 mm) |
| Systolic thickening understated | +39.0% vs +55.9% on the segmentation |
| HCM peak thickness | mean max segment 11.16 mm vs 16.07 mm on voxel (−4.9 mm) |
| **HCM diagnostic threshold (> 15 mm)** | **0/10 detected; voxel geometry detects 8/10** |
| Septal asymmetry | septal/lateral ratio 1.03 (voxel argmax is septal in 7/10) |
| HCM vs NOR effect size | Cohen d 2.00 vs 3.05 on voxel geometry |

### 1.3 Root cause — measured, not inferred

The input contours *do* carry the pathology; the network discards part of it.

| | NOR | HCM |
| --- | --- | --- |
| Wall implied by the **input contours** | 6.78 mm | 10.90 mm (+4.11) |
| Wall predicted by the **decoder** | 5.43 mm | 8.05 mm (+2.62) |
| Deficit | −1.35 mm | −2.84 mm |
| Slice residual (zero level set → input contour points) | 0.72 mm | 1.20 mm |

**Only 64% of the NOR→HCM contrast survives the encode/decode round trip**, and
the deficit scales with true thickness — the signature of regression toward the
training mean. Three architectural choices produce it:

1. **Global latent bottleneck.** The whole shape is compressed to one 256-d
   vector by a PointNet encoder (`cardiosdf_model.py::PointNetEncoder`).
2. **No test-time optimisation.** Inference is a single feed-forward pass, so
   nothing constrains the output surface to pass through the input contours.
3. **Scalar offset head.** The wall is not reconstructed directly but as a
   per-point scalar offset (monotone-epi), the narrowest path in the network,
   and it is where the deficit concentrates.

Compounded by a training distribution centred on normal end-diastolic anatomy
(`ed_mix` checkpoint; synthetic SSM epicardium in pre-training).

### 1.4 Current architecture and training budget

- 2.625 M parameters; `latent_dim` 256, `fourier_L` 3, `decoder_hidden` 512.
- Phase conditioning: a single scalar (0 = ED, 1 = ES).
- Thesis training setup: ≤ 400 epochs per stage, batch 8 with gradient
  accumulation, automatic mixed precision, 2 GPUs in parallel.
- Existing checkpoint history: 776 epochs, best at 695, val loss 0.694.

---

## 2. Upgrade ladder

Ordered by return on effort. Each rung is independently shippable; stop as soon
as the acceptance gates in §3 are met.

### U0 — Test-time latent optimisation *(no training)*

Freeze the decoder; optimise the latent `z` per case against that patient's own
contours (DeepSDF auto-decoder inference). Initialise from the encoder output so
it can only improve on the current behaviour.

```
z ← encode_contours(...)                      # current feed-forward result
for step in 1..N:
    sdf_e, sdf_p, delta = net.decode(z, contour_pts)
    L = |sdf_e[endo]| + |sdf_p[epi]|          # data consistency
      + λ_eik · (‖∇f‖ − 1)²                   # keep the field a valid SDF
      + λ_reg · ‖z − z₀‖²                     # stay near the learned manifold
    z ← Adam(z, lr 1e-2)
```

- **Pilot already run** (patient021, HCM, worst residual in the cohort):
  residual **1.328 → 0.194 mm in 50 steps**, 588 ms/step on 4 CPU cores.
- Cost: ~30 s/case on this CPU, ~1 s/case on the thesis GPUs. Zero training.
- Risk: over-fitting the latent to noisy contours; controlled by `λ_reg` and by
  early stopping on the residual. Must re-validate NOR to prove no regression.
- **Decides where the GPU budget goes** — see the decision tree in §5.

### U1 — Balanced fine-tuning *(1 stage, ≤ 400 epochs)*

Re-run the fine-tuning stage with the pathological groups and both phases
represented. ACDC's other 80 patients (DCM, HCM, MINF, RV) are already
downloaded at `notebooks/data/training/` and currently unused.

- Reuse the synthetic pre-training weights; only the fine-tuning stage repeats.
- Balance sampling by group *and* by phase (the current stream is ED-dominated).
- Cost ≈ half of what produced the current checkpoint.
- Expected to address §1.2 rows 1–3 (ES shrinkage) more than rows 4–6 (peak
  thickness), because it shifts the prior without widening the bottleneck.

### U2 — Supervise the offset head *(1 stage, folds into U1)*

The `L_WT` term is present in the loss table but reads 0.0 throughout the
recorded history — the wall was never directly supervised.

- Add a direct loss on the predicted wall against the thickness measured on the
  training meshes, so the offset head is trained on the quantity the thesis
  actually reports.
- Optionally replace the scalar offset with a small MLP head conditioned on the
  local Fourier features, keeping the monotone (strictly positive) constraint.
- Cost: none beyond U1 if trained jointly.

### U3 — Local latents *(2 stages, full retrain)*

Replace the single global vector with a spatial feature volume
(convolutional-occupancy / IF-Net style): encode contours into a coarse 3D
feature grid, and condition the decoder on trilinearly interpolated local
features plus the global code.

- Removes the bottleneck properly rather than compensating for it.
- Cost: full two-stage retrain plus a heavier encoder — 2–3× the original.
- Only justified if U0 shows the decoder, not the encoder, is the limit.

---

## 3. Acceptance gates

An upgrade is accepted only if it clears **all** *no-regression* gates and at
least the stated *improvement* gates. Baselines are the measured values of §1.

### No-regression (20 NOR, end-diastole)

| Metric | Baseline | Gate |
| --- | --- | --- |
| Watertight rate | 100% / 100% | = 100% / 100% |
| Cavity volume ratio | 0.978 | within [0.95, 1.05] |
| Epicardial volume ratio | 1.023 | within [0.95, 1.05] |
| Endocardium Dice | 0.863 | ≥ 0.85 |
| Model vs voxel bias (Laplace) | +0.04 mm | \|bias\| ≤ 0.15 mm |
| Model vs voxel r (Laplace) | 0.84 | ≥ 0.80 |

### Improvement

| Metric | Baseline | Target | Rung |
| --- | --- | --- | --- |
| Slice residual, NOR / HCM | 0.72 / 1.20 mm | ≤ 0.30 mm both | U0 |
| Contrast retention | 64% | ≥ 85% | U0–U3 |
| ES epicardial volume ratio | 0.807 | ≥ 0.92 | U1 |
| ES myocardial volume ratio | 0.769 | ≥ 0.90 | U1/U2 |
| Systolic thickening (NOR) | +39.0% | within ±8 pts of +55.9% | U1 |
| HCM mean max segment | 11.16 mm | ≥ 14.5 mm | U0–U3 |
| **HCM detection (> 15 mm)** | **0/10** | **≥ 6/10** | U0–U3 |
| Septal/lateral ratio (HCM) | 1.03 | ≥ 1.15 | U2/U3 |
| HCM vs NOR Cohen d | 2.00 | ≥ 2.7 | U1–U3 |
| Regional ICC (model vs voxel) | 0.57 | ≥ 0.70 | U2/U3 |

The HCM detection gate is the decisive one: it is the criterion ACDC itself uses
to define the group, and the voxel geometry reaches 8/10 on the same input.

---

## 4. Test flow

### 4.1 Harness

The evaluation harness already exists and is deterministic; only the geometry
source changes between runs.

```bash
VENV=/home/C052246/tese/.venv/bin/python
DATA=/home/C052246/tese/tese/notebooks/data/training
cd scripts/eval_demo

# baseline is already cached; a new checkpoint needs --force-mesh
$VENV run_cohort.py --data-root $DATA --group NOR --workers 4 \
    --out ../cohort_nor_<variant> --model <new.pt> --force-mesh
$VENV run_cohort.py --data-root $DATA --group HCM --limit 10 --workers 4 \
    --out ../cohort_hcm_<variant> --model <new.pt> --force-mesh
$VENV reference_aha17.py --data-root $DATA --out ../cohort_nor_<variant>
```

Cost per variant: ~1 h for 20 NOR + ~30 min for 10 HCM on 4 CPU cores. Meshes
are cached per patient, so metric-only changes re-aggregate in seconds via
`--aggregate-only`.

### 4.2 Tasks

- [ ] **T0** Promote the throwaway diagnostics in `/tmp` into
      `scripts/eval_demo/` as a permanent regression harness:
      `check_cohort.py` (recon + wall + ES sanity), `check_fit.py`
      (slice residual, contrast retention), `check_hcm.py` (detection,
      septal ratio, NOR/HCM separation). These encode the gates of §3.
- [ ] **T1** Implement U0 as `scripts/eval_demo/latent_refine.py`, exposing
      `--steps`, `--lr`, `--lambda-reg`, `--lambda-eik`; wire an opt-in
      `--refine-latent` flag into `run_cohort.py::cached_geometry`.
- [ ] **T2** Sweep `steps ∈ {50, 150, 300}` × `λ_reg ∈ {0, 1e-3, 1e-2}` on
      5 NOR + 5 HCM. Select on slice residual subject to the no-regression
      gates; guard against latent drift into implausible shapes.
- [ ] **T3** Full U0 evaluation on 20 NOR + 10 HCM. Record every §3 metric.
- [ ] **T4** Decision point (§5).
- [ ] **T5** If retraining: rebuild the data stream with group- and
      phase-balanced sampling; verify the split has no patient leakage.
- [ ] **T6** Train U1 (+U2), ≤ 400 epochs, checkpoint on validation loss.
- [ ] **T7** Full evaluation of the new checkpoint; compare against both the
      baseline and U0.
- [ ] **T8** Update `chapters/04-results.tex` with whichever variant is
      adopted, and keep the baseline as an ablation row.

### 4.3 Ablation matrix for the thesis

| Variant | Encoder | Test-time opt. | Wall supervision | Training data |
| --- | --- | --- | --- | --- |
| B (baseline) | global | no | none | ED-dominated |
| U0 | global | yes | none | unchanged |
| U1 | global | no | none | balanced |
| U1+U0 | global | yes | none | balanced |
| U1+U2 | global | no | direct | balanced |
| U3 | local | optional | direct | balanced |

Reporting B, U0 and U1+U2 is enough to support the argument; U3 is future work
unless time allows.

---

## 5. Decision tree after U0

```
Run U0 on 20 NOR + 10 HCM
│
├─ HCM detection ≥ 6/10 and NOR gates hold
│     → adopt U0. No training at all. Report as an inference-time contribution.
│
├─ Residual ≤ 0.30 mm but detection still low
│     → the encoder is fine, the DECODER prior is the limit.
│       Go to U2 (supervise the wall) before U1.
│
└─ Residual stays high despite optimisation
      → the latent cannot represent the shape: capacity limit.
        Go to U3 (local latents); U1 alone will not fix it.
```

---

## 6. Risks

| Risk | Mitigation |
| --- | --- |
| U0 over-fits the latent to contour noise | `λ_reg` toward `z₀`, early stop on residual, verify NOR no-regression gates |
| Refined latent leaves the learned manifold → broken topology | watertight rate is already a gate; reject any case failing it |
| Balanced fine-tuning degrades NOR performance | NOR gates in §3 are hard constraints, not targets |
| Interpolation uncertainty masks real gains | ±1.32 mm from slice interpolation exceeds most gate margins; **prefer relative model-vs-voxel comparisons on identical input over absolute millimetres** |
| Cohort too small for the HCM claim | 10 patients; extend to all 20 HCM before publishing the detection rate |
| Genus repair inflates meshes 3–4× | already observed on 3/20 NOR; cap with `make_watertight(target_faces=…)` |

---

## 7. Out of scope

- Retraining the synthetic SSM pre-training stage (its history was never
  retained; the weights are reused as-is).
- Replacing the segmentation input: this plan does not attempt to overcome the
  10 mm slice spacing, only to stop discarding the information that is present.
- Any claim of clinical validation. Without expert manual wall-thickness
  measurements the study remains an inter-method reproducibility exercise.
