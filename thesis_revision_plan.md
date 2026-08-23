# Thesis Revision Implementation Plan

## Review decision

The central change proposed below is correct: the thesis should read as a
research investigation supported by experiments, rather than mainly as a model
description. The original plan is retained at the end of this file as a source
of ideas, but it is not the active checklist. Three corrections are required.

First, evidence integrity must come before page rebalancing. The Results chapter
currently matches cached outputs in
`test-new-model/cohort_full_nor_hcm10/`, while other cached cohorts in
`scripts/cohort_nor/` and `scripts/cohort_hcm/` produce different values. The
authoritative model, cohort, spacing calibration, and post-processing path must
be recorded before any table is treated as final.

Second, the page target is guidance rather than a quota. The current compiled
spans are approximately 24 pages for Methodology and 14 pages for Results. A
Methodology target of 18--20 pages therefore requires a reduction of about
4--6 pages, not 7--9. Results should grow only when supported analyses or
experiments are available; it must not be padded to reach a page count.

Third, all thesis Results use the v2 model only. Earlier model checkpoints and
their outputs are implementation history and must not be used for comparisons
in the thesis. In thesis prose, v2 is still called ``the model'' or ``the
proposed model'' because the model has no proper name.

## Active stages

### Stage 0 — Freeze the evidence base

- [x] Select `test-new-model/cohort_full_nor_hcm10/` as the frozen thesis
  evidence set for the v2 model. Its exact v2 checkpoint was not embedded;
  session history identifies `u1u2_e50` as the leading candidate, and this
  uncertainty is recorded in `results_provenance.md`.
- [x] Create `results_provenance.md` containing checkpoint candidates and
  hashes, evaluation
  patient IDs, phase coverage, voxel pitch, slice-spacing calibration, mesh
  extraction, and repair settings.
- [x] Update the cohort evaluator to write `provenance.json` after a successful
  run, including model and source hashes, command, patients, phases, pitch, grid
  resolution, and failures.
- [x] Reconcile the two existing result families:
  - `test-new-model/cohort_full_nor_hcm10/`, which matches the current Chapter 4
    tables and is retained;
  - `scripts/cohort_nor/` and `scripts/cohort_hcm/`, which report different
    reconstruction and thickness values from the earlier model and are excluded
    from the thesis Results.
- [x] Verify that the retained evaluator uses NIfTI header zooms and preserves
  physical spacing, including the 10~mm through-plane spacing in representative
  ACDC cases.
- [x] Freeze one set of summary and patient-level files by hash. Future analyses
  must derive only from this directory and must not combine checkpoints or
  pipelines.

**Exit condition:** one documented result directory reproduces every number used
in Chapter 4.

### Stage 1 — Methodology review (deferred)

- [ ] Do not shorten, move, or restructure Chapter 3 during the current Results
  revision.
- [ ] Revisit Methodology only after the Results chapter is complete and only
  when explicitly requested.
- [ ] Treat the earlier 18--20 page target as optional guidance, not an active
  requirement.

**Exit condition:** deferred by user request.

### Stage 2 — Expand analyses already supported by cached data

- [x] Leave the existing Training section, epoch values, and training-curves
  figure unchanged and treat them as v2 evidence.
- [x] Add only new Results content from this point onward.
- [x] Use the existing ED/ES mesh render for spatial reconstruction findings and
  Matplotlib for new
  metric graphics.
- [x] Add patient-level Matplotlib figures for reconstruction metrics,
  NOR--HCM thickness outcomes, and model-versus-derived thickness agreement.
- [x] Audit all expanded reconstruction and wall-thickness tables against the
  frozen v2 CSVs and correct the HCM apical row transcription.

- [x] Add `scripts/analyze_cached_cohort.py` to derive patient-level group
  effects, bootstrap intervals, finite-value fractions, phase summaries, and
  ranked reconstruction cases. The script labels outputs unverified when the
  source cohort has no provenance manifest.
- [x] Strengthen reconstruction quality with Chamfer, ASSD, HD95, Dice, volume
  ratio, and watertightness from the same authoritative cohort.
- [x] Add a patient-level error analysis for surface distance, tail error,
  cavity bias, and myocardial overlap. The cache does not localise errors to the
  apex or base, and the text states that point-wise maps would be required.
- [x] Add ED versus ES wall-thickness and AHA-17 comparisons from the existing
  phase rows.
- [ ] Add full ES reconstruction-quality metrics only if they are later
  aggregated from the cached ES meshes; do not infer them from thickness data.
- [x] Expand the four-method thickness comparison using mean, dispersion,
  percentiles, model-versus-voxel bias, correlation, agreement, and valid-value
  fraction. Runtime is not reported because it was not measured consistently
  over the full cohort.
- [x] Compute normal-versus-HCM differences at the patient level. Report an
  effect size and bootstrap confidence interval only for pre-specified global or
  maximum-segment summaries; do not treat 17 segments from one heart as
  independent observations.
- [x] Add a concise failure-analysis subsection based on measured cases rather
  than hypothetical apex, base, or wall failures.

**Exit condition:** every new paragraph points to a table, figure, or
patient-level calculation from the frozen evidence base.

### Stage 3 — Add one conventional reconstruction comparator (deferred)

- [ ] Reuse the existing contour-lofting implementation in
  `scripts/webapp/core/inference.py::_loft_rings_to_mesh` rather than writing a
  second lofting algorithm.
- [ ] Expose it as a geometric comparator for both endocardial and epicardial
  contours.
- [ ] Evaluate lofting and the INR on identical patients, contours, phases,
  physical spacing, mesh repair, and metrics.
- [ ] Compare accuracy, volume bias, watertightness, and visible through-plane
  artefacts. Do not claim superiority before the paired results are available.

**Exit condition:** one reproducible table and a small set of matched visual
examples show what the INR adds beyond direct contour connection.

This stage is outside the current existing-results revision because it requires
a new matched experiment. It must not use the earlier model as a comparator.

### Stage 4 — V2-only reporting policy

- [x] Use only v2 outputs in the thesis Results.
- [x] Exclude the earlier model and its cohort outputs from all tables, figures,
  comparisons, and narrative claims.
- [x] Do not add old-versus-v2 ablations or upgrade narratives.
- [x] Refer to v2 only as ``the model'' or ``the proposed model'' in thesis
  prose.
- [x] Treat the existing training-history and epoch content as v2 and leave it
  unchanged.

**Exit condition:** every model-dependent result comes from the frozen v2
evidence directory; no earlier-model comparison appears in the thesis.

### Stage 5 — Rewrite Discussion and Conclusions

- [x] Organise the Discussion around RQ1--RQ3, using demonstrated findings first,
  plausible explanations second, and unsupported claims last.
- [x] Compare the findings with the method families reviewed in Chapter 2,
  without claiming direct superiority where no matched baseline exists.
- [x] State clearly that segmentation-derived geometry tests computational
  agreement, not anatomical or clinical validity.
- [x] Rewrite Chapter 5 with short sections for answers to the research
  questions, supported contributions, limitations, and three prioritised future
  steps.

**Exit condition:** each research question receives one concise answer in the
Discussion and one shorter synthesis in the Conclusion.

## Current status

- [x] Introduction hierarchy and document map improved.
- [x] Literature Review expanded and reorganised around technical background,
  method families, and the research gap; current span is about ten pages.
- [x] Stage 0 evidence freeze (existing cached results adopted; missing embedded
  checkpoint ID documented as a reproducibility limitation).
- [ ] Stage 1 Methodology review (deferred; no current edits).
- [x] Stage 2 supported Results analyses (except optional full ES reconstruction
  metrics, which are not inferred from the existing thickness summaries).
- [ ] Stage 3 lofting comparator (deferred; requires a new matched experiment).
- [x] Stage 4 v2-only reporting policy.
- [x] Stage 5 Discussion and Conclusions.

---

# Original Proposal — Reference Only

## Main structural change

- [ ] **Shorten the methodology chapter**
  - Current feel: methodology is detailed and technically strong, but it occupies a large share of the thesis.
  - Target: reduce the methodology by roughly **7–9 pages**.
  - Suggested final balance:
    - **Methodology:** ~18–20 pages
    - **Results + Discussion:** ~22–26 pages
  - Goal: make the thesis feel more research-driven by moving space from implementation description into experimental analysis.

## 1. Reduce Chapter 3 — Methodology

### 3.2 Data Preparation

- [ ] Reduce the section from roughly **12 pages to 6–7 pages**.
- [ ] Keep the important methodological decisions:
  - Why independent 3D supervision is unavailable.
  - Why an SSM is used to generate synthetic 3D LV meshes.
  - How real clinical data are converted into the same representation.
  - Patient-level train/validation/test splitting.
  - Contour augmentation.
- [ ] Move implementation/cache details and secondary explanations to the appendix.
- [ ] Avoid explaining the same motivation multiple times.

### 3.3 Model Architecture

- [ ] Reduce the section to roughly **3 pages**.
- [ ] Keep the technically important components:
  - Point-cloud/contour encoder.
  - Separate endocardial and epicardial feature aggregation.
  - Local 3D feature volume.
  - Fourier positional encoding.
  - Implicit SDF decoder.
  - Positive/monotone epicardial coupling.
- [ ] Keep the key equations and architecture table.
- [ ] Remove repeated verbal explanations that simply restate the figure/equations.
- [ ] Keep the explanation of *why* each architectural component exists, but make it concise.

### 3.4 Training and Inference

- [ ] Reduce to roughly **3 pages**.
- [ ] Keep:
  - Full loss equation.
  - Purpose of each loss term.
  - Training stages.
  - Optimiser and main hyperparameters.
  - Inference procedure.
  - Marching Cubes and required post-processing.
- [ ] Move lower-level implementation details to the appendix where they are not essential for understanding or reproducing the method.

### 3.5 Wall-Thickness Evaluation Protocol

- [ ] Reduce to roughly **4 pages**.
- [ ] Keep the four thickness methods:
  - Laplace field.
  - Yezzi–Prince.
  - SDF cone-ray.
  - Euclidean Distance Transform (EDT).
- [ ] Briefly explain the principle behind each method.
- [ ] Keep the reason for selecting the methods.
- [ ] Avoid spending excessive space on mathematical details that are not central to the thesis contribution.
- [ ] Keep the local 3D thickness field and AHA-17 aggregation because they are important outputs.

---

# 2. Expand Chapter 4 — Results and Evaluation

## 4.1 Reconstruction Quality

- [ ] Make this a stronger standalone experimental question:
  - **How accurately does the model reconstruct the LV from sparse SAX contours?**
- [ ] Keep the current metrics:
  - Chamfer distance.
  - Dice similarity coefficient.
  - Volume ratio.
  - Watertightness.
- [ ] Add clearer interpretation of each metric.
- [ ] Include representative successful and difficult reconstructions.
- [ ] Discuss where errors occur:
  - Apex.
  - Base.
  - Thin myocardial regions.
  - Highly curved regions.
- [ ] Distinguish clearly between agreement with segmentation-derived geometry and true anatomical accuracy.

## 4.2 Architectural Ablation Study

- [ ] Add an ablation experiment.
- [ ] Compare the full model against variants such as:
  - Full model.
  - **Without SSM pretraining.**
  - **Without Fourier features.**
  - **Without local conditioning.**
  - **Without positive/monotone wall coupling.**
  - **Without wall-thickness loss.**
- [ ] Report the same core metrics for every variant:
  - Chamfer.
  - Dice.
  - Volume ratio.
  - Potentially wall-thickness error.
- [ ] Add visual examples for important ablations.
- [ ] Explain **why** each component improves or degrades reconstruction.
- [ ] This changes the thesis from:
  - “Here is a model that works.”
  - to:
  - “Here is evidence explaining why this model design works.”

## 4.3 Baseline Comparison

- [ ] Add at least one simple reconstruction baseline.
- [ ] A useful baseline would be:
  - Sparse contours → interpolation/lofting → 3D mesh.
- [ ] Compare it directly with:
  - Sparse contours → proposed INR → 3D mesh.
- [ ] Use the same evaluation cohort and metrics.
- [ ] Compare:
  - Surface error.
  - Volume overlap.
  - Volume bias.
  - Surface smoothness.
  - Watertightness.
  - Failure cases.
- [ ] Explain what the INR provides beyond conventional interpolation.
- [ ] This gives the reader a concrete reference point for the contribution.

## 4.4 ED vs ES Analysis

- [ ] Turn the existing phase comparison into a proper experiment.
- [ ] Compare **end-diastole (ED)** vs **end-systole (ES)** for:
  - Chamfer distance.
  - Dice.
  - Volume ratio.
  - Wall thickness.
  - Regional AHA-17 measurements.
- [ ] Show representative ED/ES reconstructions.
- [ ] Analyse why ES is harder.
- [ ] Discuss:
  - ED-only synthetic pretraining.
  - Binary phase conditioning.
  - More challenging systolic geometry.
- [ ] Separate the observation from the explanation and explicitly state which explanations are supported by the experiments.

## 4.5 Wall-Thickness Method Comparison

- [ ] Expand this into a real experimental subsection rather than treating it as secondary.
- [ ] Compare all four methods on the same reconstructed geometry.
- [ ] Report:
  - Mean.
  - Standard deviation.
  - Percentiles.
  - Correlation with the comparator.
  - Mean offset/bias.
  - Runtime.
  - Failure/invalid-value rate.
- [ ] Compare reconstructed geometry against segmentation-derived geometry.
- [ ] Include:
  - 3D thickness maps.
  - Representative cross-sections.
  - Method comparison plots.
- [ ] Explain why the field-based methods agree more closely while the cone-ray method is noisier.
- [ ] Explicitly identify which method is selected for the final regional analysis and why.

## 4.6 Local 3D Wall-Thickness Analysis

- [ ] Give more emphasis to the fact that the proposed pipeline produces a **continuous 3D thickness field**, rather than only per-slice measurements.
- [ ] Show several representative 3D maps.
- [ ] Discuss:
  - Apex-to-base variation.
  - Spatial heterogeneity.
  - Thin/thick regions.
  - Differences between ED and ES.
- [ ] Explain what information is visible in the continuous surface that would be lost through slice-level averaging.

## 4.7 AHA-17 Regional Analysis

- [ ] Expand the regional analysis into a substantive results section.
- [ ] Show the 17-segment pattern for:
  - Normal cohort.
  - HCM cohort.
  - ED.
  - ES.
- [ ] Include bullseye plots and/or tables.
- [ ] Analyse:
  - Basal vs mid vs apical patterns.
  - Septal differences.
  - Regional systolic thickening.
- [ ] Make clear that this is a **descriptive computational analysis**, not clinical validation.

## 4.8 Normal vs HCM Analysis

- [ ] Expand the existing HCM vs normal result.
- [ ] Compare:
  - Global ED thickness.
  - Segment-wise ED thickness.
  - Systolic thickening.
  - Maximum segment thickness.
  - Distribution of thickness values.
- [ ] Add effect sizes or confidence intervals where appropriate.
- [ ] Highlight the observed septal/anteroseptal differences.
- [ ] Show representative 3D thickness maps for normal and HCM cases.
- [ ] State clearly that the analysis demonstrates **biological plausibility / descriptive separation**, not diagnostic performance.

## 4.9 Failure Cases and Error Analysis

- [ ] Add a dedicated error-analysis subsection.
- [ ] Show cases where:
  - The cavity is underestimated.
  - The wall is locally too thick/thin.
  - The apex is poorly reconstructed.
  - Mesh post-processing is heavily used.
  - The phase transition is difficult.
- [ ] Analyse common sources of error rather than only showing average metrics.
- [ ] Connect observed failures to architectural or data limitations.

## 4.10 Discussion

- [ ] Make the discussion more analytical and less repetitive.
- [ ] For every major result, answer:
  - What happened?
  - Why might it have happened?
  - Does it support the research question?
  - How does it compare with the expected behaviour?
  - What does it imply for the proposed method?
- [ ] Explicitly connect the results back to each research question.
- [ ] Separate:
  - Demonstrated findings.
  - Plausible interpretations.
  - Claims that cannot be supported by the current reference data.

---

# 3. Rebalance the thesis narrative

- [ ] Aim for the following overall progression:

  1. **Introduction**
     - Problem.
     - Research questions.
     - Contribution.

  2. **Literature Review**
     - Existing reconstruction approaches.
     - INRs/SSMs/GNNs.
     - Wall-thickness methods.
     - Clear research gap.

  3. **Methodology**
     - Enough detail to understand and reproduce the approach.
     - Avoid excessive implementation narration.

  4. **Results**
     - Does the model reconstruct accurately?
     - Which components matter?
     - Does it beat a simple baseline?
     - Does it work across ED/ES?
     - Which thickness method is most reliable?
     - Does the reconstructed geometry preserve meaningful regional patterns?
     - What are the failure cases?

  5. **Discussion**
     - What the experiments actually demonstrate.
     - Why the method behaves as it does.
     - Relation to previous work.
     - Limitations.

  6. **Conclusion**
     - Concise answer to each research question.
     - Contribution.
     - Future work.

---

# 4. The central change to make

- [ ] Move the thesis from:

  **“Here is a technically detailed model and here are the resulting metrics.”**

- [ ] Toward:

  **“Here is a technically justified model, and here is experimental evidence showing why its design works, where it works, how it compares with alternatives, and where it fails.”**

- [ ] This is the change most likely to make the thesis feel substantially more mature and research-oriented.

---

# 5. What NOT to do

- [ ] Do **not** remove the important technical details of the INR architecture.
- [ ] Do **not** pad the Results chapter with redundant plots.
- [ ] Do **not** claim anatomical truth when the reference is segmentation-derived.
- [ ] Do **not** call the HCM/normal analysis clinical validation.
- [ ] Do **not** add statistical tests just to create more tables; use them only where the experimental design supports them.
- [ ] Do **not** make the methodology superficial just to increase the page count of Results.
- [ ] Do **not** over-interpret the absolute wall-thickness values.

---

# 6. Target outcome

- [ ] Cut approximately **7–9 pages** from Chapter 3.
- [ ] Add approximately **7–11 pages** of substantive experimental analysis to Chapter 4.
- [ ] Prioritise:
  1. **Ablations**
  2. **Baseline comparison**
  3. **ED vs ES**
  4. **Thickness-method comparison**
  5. **Normal vs HCM**
  6. **Failure/error analysis**
- [ ] Keep the strongest technical parts of the existing methodology.
- [ ] Make Chapter 4 answer more independent scientific questions.
- [ ] End with a thesis that reads as a **research investigation**, not primarily a **description of an ML system**.
