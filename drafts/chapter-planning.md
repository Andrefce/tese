# Thesis Chapter Planning — What to Write

This document maps the actual work done in the notebooks to each thesis chapter and section.
It should be used as a scratch planning file before writing the final LaTeX chapters.

---

## What Has Actually Been Done

### Main thesis idea
- The work is not only about making a 3D model. The complete idea is: reconstruct a 3D left-ventricle (LV) model from 2D short-axis (SAX) MRI contours, then use that reconstructed model to measure LV myocardial wall thickness locally and objectively.
- The final proposed method is CardioSDF: a phase-conditioned signed-distance implicit neural representation that predicts endocardial and epicardial SDFs and also gives analytic wall thickness through the learned delta field.
- The wall-thickness part is central: the model enforces a positive wall thickness by construction and the separate wall-thickness notebook compares four measurement algorithms on the same CardioSDF geometry.

### Notebook 1 — `datasetED_ssm.ipynb`
- A synthetic end-diastolic dataset was created from the UK Biobank / UK Digital Heart Project LV Statistical Shape Model.
- The notebook loaded the mean LV mesh and 100 PCA modes. The printed shapes were: mean shape `(22043, 3)`, PCA matrix `(66129, 100)`, eigenvalues `(100,)`.
- The mesh quality pipeline was checked with VTK and trimesh mass properties, using volume, surface area, and sphericity.
- The full synthetic generation run targeted 1300 accepted meshes. The notebook output shows `Accepted: 1300 / 2048 (63.48%)`, with coefficient matrix `B` of shape `(1300, 100)`.
- The latent sampling used Mahalanobis and sigma clipping constraints. The accepted synthetic shapes were filtered by LV volume, surface area, sphericity, and apex/base plausibility.
- Pathology-like examples were simulated by exciting PCA modes: Normal, DCM, HCM, and Post-MI.
- Synthetic epicardial surfaces were generated from the endocardium using variable normal offsets, with thinner offsets near the apex.
- SAX contours were extracted from the synthetic meshes and transformed into training-compatible cache objects.
- Occupancy/query supervision was generated with 2048 query points per sample and ground-truth inside/outside labels.
- A graph-neural-network direction was explored using PyTorch Geometric and GATv2Conv, but this is not the final model used in the thesis.

### Notebook 2 — `datasetED_real.ipynb`
- A real end-diastolic dataset was prepared directly from real segmentation masks instead of from the SSM.
- The target sources were ACDC, M&Ms, and M&Ms-2. The notebook includes dataset discovery for these sources and harmonises label conventions across them.
- The real-data pipeline canonicalises image orientation to RAS+ with `nib.as_closest_canonical`, detects or reads ED/ES frames, and keeps patient-level identity for splitting.
- The pipeline creates endocardial masks from the LV label and epicardial masks from LV plus myocardium.
- Current notebook configuration for the cache is: 10 SAX slices, 8.0 mm slice thickness, 10.0 mm slice spacing, 60 contour points per ring, and 2048 query points.
- Current mesh-processing configuration in the code is: isotropic target spacing 1.5 mm, SDF smoothing scale 1.5 mm, Taubin smoothing with 25 iterations (`λ=0.53`, `μ=-0.55`), 3 Laplacian fairing iterations, basal fraction 0.08, and target 4000 vertices.
- The mesh pipeline performs per-slice cleaning, 3D largest-component filtering, SDF smoothing, marching cubes, Taubin smoothing, basal-plane capping, Laplacian fairing, and quadric decimation.
- Anatomical prechecks are used to reject bad meshes: apical taper, basal support, centreline drift, contour regularity, endo-inside-epi, wall proxy, and long-axis ratio.
- Real patients are split at patient level and then augmented at the contour-input level during training. The notebook writes the cache to `ed_occupancy_cache_v2/`.

### Notebook 3 — `datasetES_real.ipynb`
- A real end-systolic dataset was prepared with the same direct segmentation-to-mesh strategy as ED-real.
- This notebook replaces an older ES approach that force-fitted the diastolic UK Biobank SSM to systolic contours, which the notebook states produced unrealistic ES geometry.
- The current ES cache uses the same main configuration as ED-real: 10 SAX slices, 8.0 mm thickness, 10.0 mm spacing, 60 contour points per ring, 2048 query points, 1.5 mm isotropic target spacing, 1.5 mm SDF smoothing, 25 Taubin iterations, 3 Laplacian iterations, and target 4000 vertices.
- Real ES meshes are built directly from ACDC, M&Ms, and M&Ms-2 segmentations, with patient-level splitting and contour-input augmentation during training.
- The notebook writes the cache to `es_occupancy_cache_v2/`.

### Notebook 4 — `training.ipynb`
- CardioSDF was implemented as the final reconstruction and wall-thickness model.
- The notebook is in combined ED+ES mode with phase conditioning: `input_dim=5`, `latent_dim=256`.
- The model input is a point cloud of SAX contour points with coordinates, tissue label, and cardiac phase.
- The encoder is PointNet-like: a shared MLP extracts point features, then separate endocardial and epicardial max-pooling forms a latent representation.
- The decoder is an implicit signed-distance decoder with an 8-layer MLP, width 512, and a skip connection.
- Fourier positional encoding is used with `L=6`, producing 39-dimensional encoded point features.
- The decoder has two heads: one predicts `f_endo`, and the other predicts the bounded wall-thickness offset `δ`.
- The monotone-epi rule is `f_epi = f_endo - δ`, with `δ = τ_min + (δ_cap - τ_min)·σ(·)`. In the notebook configuration, `τ_min_norm=0.05` and `δ_cap_norm=0.45`, corresponding roughly to 1.25–11.25 mm at a 25 mm scale.
- This means the model does not only reconstruct geometry; it directly provides analytic wall thickness at queried endocardial locations.
- The loss includes surface fit, eikonal regularisation, off-surface repulsion, normal consistency, wall-thickness floor, cached SDF supervision, contour anchoring, cross-wall sign consistency, exterior sign consistency, and bbox-outside positivity, plus latent regularisation.
- Training configuration includes 400 max epochs, batch size 8, gradient accumulation 2, AdamW, gradient clipping, AMP, cosine scheduling, early stopping, and DataParallel on 2 T4 GPUs when available.
- Inference encodes the contour, queries a 96³ grid, extracts endocardial and epicardial meshes with marching cubes, and does not rely on post-processing repair to make the mesh watertight.

### Notebook 5 — `lv_wall_thickness_10_methods.ipynb`
- This notebook estimates LV wall thickness on the trained CardioSDF 3D model, not directly on raw segmentation meshes.
- The notebook explicitly states that the trained model is first used to generate endocardial and epicardial 3D surfaces, and the wall-thickness methods are then applied to that model geometry.
- The input segmentation is used only to extract contour rings for CardioSDF and to orient AHA-17 anatomy. The analysis meshes and volumetric masks are generated from the CardioSDF/INR output.
- Each method returns per-endocardial-vertex wall thickness in millimetres, allowing local maps rather than only one global mean.
- The notebook implements a broader exploratory survey, but the thesis reports four representative methods: the Laplace field, Yezzi–Prince, and SDF cone-ray estimators, with the EDT boundary sum kept as a volumetric baseline.
- The notebook produces a summary table, 3D wall-thickness colour-map figures, an AHA-17 bullseye figure, and an AHA anatomical QA figure for base/apex and septal orientation.
- This is the evidence for the thesis question about local wall-thickness changes and for the objective comparison of LV wall-thickness algorithms.

### What can already be written safely
- Chapter 1 should stay short: 2--3 pages, enough to define the problem, research questions, objectives, and contributions.
- Chapter 2 should mostly stay as it is. The main remaining work there is replacing placeholder figures and checking that claims are well supported, not expanding it heavily.
- Chapter 3 should become the technical core of the thesis. It can be written in detail from the notebooks because the data-preparation pipelines, CardioSDF architecture, training regime, inference, and wall-thickness methods are implemented.
- Chapter 4 can describe the evaluation protocol and figure/table structure now, but exact final metric values should only be inserted after copying them from the final notebook outputs.
- The thesis should not claim full-heart reconstruction. It should state LV reconstruction and LV myocardial wall-thickness measurement.

### Final Thesis Shape to Aim For
- **Chapter 1:** short, direct, 2--3 pages. No padding.
- **Chapter 2:** keep the current literature review, roughly as already drafted. Replace placeholders later.
- **Chapter 3:** main technical core, around 22--30 pages if fully written with tables, equations, and figures.
- **Chapter 4:** proof/evidence chapter, around 12--18 pages depending on final tables and figures.
- **Chapter 5:** short conclusion, around 4--6 pages.

### Where the Technical Core Belongs
- The technical core belongs in **Chapter 3: Methodology**.
- Chapter 3 should answer: what was built, how each component works, why each design was chosen, and how the pipeline can be reproduced.
- Chapter 4 should not re-explain the method. It should answer: what happened when the method was evaluated?
- The most important technical-core sections are: data preparation, cache/contour representation, CardioSDF architecture, monotone-epi wall-thickness parameterisation, loss function, inference, and the wall-thickness measurement protocol.

### What is still missing before the thesis is strong
- **Final comparison table:** Chapter 4 needs one decisive table that compares CardioSDF analytic thickness and the four wall-thickness methods using the same geometry. Include mean, median, std, p5, p95, min, max, valid/finite fraction, runtime, and warnings/failures.
- **A simple baseline:** Add at least one baseline so the thesis is not only "our method works". Good options are direct marching cubes from segmentation masks, KD-tree thickness on segmentation-derived meshes, or an older SSM/ray-based approach if the results exist. If a baseline cannot be completed, state this honestly and present the four-method comparison as an internal reproducibility benchmark.
- **Ground-truth honesty:** The wall-thickness study compares algorithms on CardioSDF geometry. Unless expert clinical measurements exist, do not claim clinical ground-truth accuracy. Claim objective, reproducible, local thickness estimation and method agreement/disagreement.
- **Local wall-thickness evidence:** The thesis must visually prove local thickness, not only report global means. Include 3D surface colour maps and AHA-17 bullseye plots.
- **Dataset counts:** Add exact counts for synthetic ED, real ED, and real ES after final cache generation: train/val/test, number of patients, number of samples, and augmentation status.
- **Failure cases:** Include one small subsection in Chapter 4 or Chapter 5 showing limitations/failures: apex/base instability, segmentation dependency, grid resolution, or methods that return sparse/invalid values.
- **Monotone-epi justification:** The monotone-epi decoder is one of the strongest ideas. Make it explicit why it matters: no negative wall thickness, no endo/epi crossing by construction, and no need for post-hoc surface repair for thickness validity.
- **Results-first conclusion:** Chapter 5 should not only repeat the method. It must answer each research question with the actual evidence from Chapter 4.

### Minimum Evidence Package for Final Thesis
- **Must-have table 1:** Dataset statistics and split table.
- **Must-have table 2:** CardioSDF reconstruction metrics: watertight rate, endo Chamfer, epi Chamfer, slice residual, and possibly Hausdorff/HD95 if available.
- **Must-have table 3:** Wall-thickness comparison across CardioSDF analytic δ and the four methods.
- **Must-have figure 1:** Full CardioSDF pipeline diagram from 2D SAX contours to 3D mesh and wall-thickness map.
- **Must-have figure 2:** Example 3D reconstructions for ED and ES, preferably with input contours overlaid or shown beside them.
- **Must-have figure 3:** Local wall-thickness colour map on the reconstructed LV.
- **Must-have figure 4:** Four-method wall-thickness comparison panel.
- **Must-have figure 5:** AHA-17 bullseye plot for regional wall thickness.
- **Must-have figure 6:** AHA orientation / anatomical QA figure if used to justify regional mapping.
- **Optional but strong:** A baseline comparison figure/table and one failure-case figure.

---

## Chapter 1: Introduction

### 1.1 Motivation
- **Write about:** Why LV wall thickness from MRI matters, not just 3D shape. Explain that local thickening or thinning can be clinically relevant in cardiomyopathy, infarction, remodelling, and disease monitoring.
- **What was done:** The project reconstructs the LV from 2D SAX contours and then measures local myocardial wall thickness on the reconstructed 3D model.
- **From:** General cardiac imaging context + the local thickness maps and AHA-17 plots in `lv_wall_thickness_10_methods.ipynb`.

### 1.2 Problem Statement
- **Write about:** Current approaches often stop at segmentation or 3D mesh extraction. They may produce surfaces that need repair, and wall thickness is often measured afterward with separate algorithms that can fail locally, especially near the apex or base.
- **What was done:** CardioSDF combines 3D LV reconstruction and wall-thickness modelling in one INR. The decoder predicts endocardium, epicardium, and a positive thickness offset `δ`.
- **From:** `training.ipynb`, especially the SDF watertight design and monotone-epi decoder.

### 1.3 Research Questions
- **Simple idea:** This section should ask whether a 3D left-ventricle model can be reconstructed from 2D short-axis MRI slices, and whether that reconstructed model can also be used to measure local wall thickness objectively.
- **Question 1:** Can a 3D model of the left ventricle be reconstructed from 2D SAX cardiac MRI contours?
  - **Thesis version:** Can a phase-conditioned implicit neural representation reconstruct watertight 3D LV endocardial and epicardial surfaces from sparse 2D short-axis contour observations?
  - **How it is answered:** Use CardioSDF from `training.ipynb`: input 2D SAX contour points + tissue label + cardiac phase, encode them with PointNet, decode endo/epi signed-distance fields, and extract 3D meshes with marching cubes on a 96³ grid.
- **Question 2:** Can the reconstructed model show local wall-thickness changes, not only a global average?
  - **Thesis version:** Can local myocardial wall-thickness variation be estimated over the reconstructed LV surface and represented regionally, for example with 3D colour maps and AHA-17 bullseye plots?
  - **How it is answered:** Use the analytic δ output of CardioSDF at endocardial vertices and the four wall-thickness methods from `lv_wall_thickness_10_methods.ipynb`; report local thickness maps, p5/p95 values, and AHA-17 regional plots.
- **Question 3:** Can the model guarantee physically valid LV wall thickness?
  - **Thesis version:** Can the decoder architecture enforce positive myocardial wall thickness by construction instead of depending only on post-processing or manual correction?
  - **How it is answered:** Use the monotone-epi decoder from `training.ipynb`, where δ = τ_min + (δ_cap − τ_min)·σ(·) and f_epi = f_endo − δ, guaranteeing δ ≥ τ_min everywhere.
- **Question 4:** Can an objective computational model be built for LV wall-thickness measurement?
  - **Thesis version:** Can CardioSDF provide a reproducible and objective framework for LV wall-thickness estimation, and how do established thickness algorithms compare when applied to the same reconstructed geometry?
  - **How it is answered:** Compare the analytic CardioSDF thickness with four representative methods evaluated in `lv_wall_thickness_10_methods.ipynb`: the Laplace field, Yezzi–Prince, and SDF cone-ray estimators, with the EDT boundary sum as a volumetric baseline.
- **From:** `training.ipynb` for the reconstruction and analytic wall-thickness model; `lv_wall_thickness_10_methods.ipynb` for local/regional wall-thickness evaluation.

### 1.4 Objectives
- Reconstruct a 3D LV model from 2D SAX contour data, including both endocardial and epicardial surfaces.
- Build an objective wall-thickness model that estimates thickness from the reconstructed geometry rather than relying on manual point selection.
- Measure local wall-thickness variation over the LV surface, not only one global mean value.
- Represent local changes using 3D wall-thickness colour maps and AHA-17 bullseye regional plots.
- Ensure wall thickness is positive by construction with the monotone-epi parameterisation.
- Compare four wall-thickness methods on the same CardioSDF output geometry.
- **From:** What you implemented in `training.ipynb` and `lv_wall_thickness_10_methods.ipynb`.

### 1.5 Contributions
- CardioSDF: signed-distance-field INR with monotone-epi decoder (guarantees δ ≥ τ_min).
- Mixed training regime: synthetic ED (UK Biobank SSM) + real ED/ES (ACDC, M&Ms, M&Ms-2).
- Local LV wall-thickness estimation from the reconstructed model using analytic δ values and regional AHA-17 visualisation.
- Systematic comparison of four wall-thickness algorithms on model output.
- **From:** `training.ipynb` and `lv_wall_thickness_10_methods.ipynb`.

### 1.6 Document Structure
- **Write:** One sentence per chapter (Ch2: literature, Ch3: methodology, Ch4: results, Ch5: conclusions). Mention that the methodology chapter covers both reconstruction and wall-thickness estimation.

### Chapter 1 deliverables
- **No results table here.** Keep this chapter clean and conceptual.
- **Target length:** 2--3 pages.
- **Optional figure:** probably skip unless the introduction feels too abstract. Leave most figures for Chapter 3 and Chapter 4.
- **Must include:** the four research questions exactly aligned with reconstruction, local wall-thickness variation, positive-thickness guarantee, and objective thickness measurement.
- **Must avoid:** claiming full-heart reconstruction. Say LV reconstruction.

---

## Chapter 2: Literature Review

### 2.1 Review Methodology
- **Already written:** Search strategy (Scopus, PubMed, IEEE Xplore, Google Scholar, 2000–2026), Boolean query, inclusion/exclusion criteria.
- **From:** Your systematic search (already in the file).

### 2.2 Cardiac MRI and LV Anatomy
- **Write about:** Short-axis imaging protocol, endocardial and epicardial contours, ED/ES phases, and why sparse slices make 3D reconstruction difficult.
- **What was done:** The notebooks standardise SAX-like contour inputs as 10 slices, 8.0 mm slice thickness, 10.0 mm spacing, and 60 contour points per ring for real ED/ES caches.
- **From:** `datasetED_real.ipynb`, `datasetES_real.ipynb`, plus general cardiac MRI background.

### 2.3 3D Reconstruction Methods
- **Write about:**
  - Classical: marching cubes from segmentation masks, mesh fitting.
  - Statistical Shape Models (SSM): PCA-based mean shape + modes (UK Biobank SSM is an example).
  - Deep learning: PointNet-like encoders, graph neural networks (GATv2Conv explored in `datasetED_ssm.ipynb` but not used in final model).
  - Implicit representations: occupancy networks, signed distance functions, INRs (SIREN, Fourier features).
- **From:** `datasetED_ssm.ipynb` (SSM), `training.ipynb` (PointNet encoder, INR-SDF).

### 2.4 Wall-Thickness Measurement Algorithms
- **Write about:** Brief survey of the main wall-thickness method families (distance, ray, volumetric/EDT, PDE), then focus on the four evaluated in the thesis:
  1. Laplace field (∇²ψ=0) — transmural reference
  2. Yezzi–Prince (Eulerian PDE: ∇ψ·∇u=−1)
  3. SDF cone rays
  4. EDT boundary sum — volumetric baseline
- **From:** `lv_wall_thickness_10_methods.ipynb` — cite the key papers for Laplace and Yezzi–Prince.

### 2.5 Related Work on LV Reconstruction
- **Write about:** Recent papers that combine deep learning + MRI for 3D LV reconstruction, wall-thickness measurement from MRI. Highlight the gap: no method guarantees positive wall thickness via the decoder architecture.
- **From:** Literature search; this is where you cite related papers.

### Chapter 2 deliverables
- **Keep mostly as-is.** Do not spend the main writing effort expanding this chapter unless a supervisor specifically asks for more literature.
- **Replace placeholders:** PRISMA flow, ACDC/SAX examples, and wall-thickness method overview figure.
- **Optional table:** literature comparison table only if time allows. It is useful, but Chapter 3 and Chapter 4 are higher priority.
- **Optional table:** wall-thickness methods taxonomy can be moved to Chapter 3 if it better supports the implemented methods.
- **Citation check:** this chapter already uses many citations. The priority is to keep them verified and avoid unsupported claims.

---

## Chapter 3: Methodology

### Chapter 3 role
- **This is the technical core of the thesis.** It should be detailed, reproducible, and much longer than Chapter 1.
- **Target length:** around 22--30 pages after figures, equations, and tables.
- **Purpose:** explain exactly what was built and how it works, not just describe the idea at a high level.
- **Main risk:** writing Chapter 3 too briefly would make the thesis look like a collection of notebooks instead of a complete scientific method.

### Suggested Chapter 3 page distribution
- Overview and pipeline: 1--2 pages.
- Synthetic ED dataset from SSM: 3--4 pages.
- Real ED and ES dataset preparation: 4--5 pages.
- Cache format and contour representation: 1--2 pages.
- Spatial normalisation and data augmentation: 2--3 pages.
- CardioSDF architecture: 4--5 pages.
- Loss function and training strategy: 3--4 pages.
- Inference and analytic wall-thickness extraction: 1--2 pages.
- Four wall-thickness methods and AHA-17 protocol: 3--4 pages.

### 3.1 Overview
- **Write about:** High-level pipeline: input (2D SAX contours + phase) → PointNet encoder → latent z → INR decoder → SDF (endo + epi) → marching cubes → watertight mesh → analytic wall thickness.
- **From:** `training.ipynb` architecture summary.
- **Figure:** Main methodology pipeline diagram. Show data sources, contour extraction, PointNet encoder, latent vector, INR decoder, endo/epi SDFs, marching cubes, and local wall-thickness map.
- **Table:** Main method summary. Rows: input, encoder, decoder, output surfaces, wall-thickness guarantee, inference grid, post-processing.

### 3.2 Data Preparation

**Important:** This section should be written as a real engineering pipeline, not as a short dataset paragraph. It should explain how raw meshes/segmentations became training examples.

#### 3.2.0 Data Sources, Phases, and Geometry Standardisation
- **Write about:** What each case contains before processing: segmentation masks or SSM mesh, ED/ES phase, LV cavity/myocardium labels, voxel spacing, and SAX slice geometry.
- **Include:** Real data requires orientation correction, label harmonisation, and phase handling because ED and ES represent different cardiac states.
- **Why it matters:** The model does not learn from raw MRI pixels. It learns from a standardised contour/cache representation derived from the raw segmentations or synthetic meshes.

#### 3.2.1 Synthetic ED Dataset (SSM-based)
- **Write about:**
  - UK Biobank Statistical Shape Model (mean + 100 PCA modes).
  - Generated 1300 synthetic meshes by sampling latent shape coefficients (χ² bound at 99%, σ=3.0).
  - Quality filters: volume, surface area, sphericity, apex/base normal variance.
  - Pathology simulation: DCM, HCM, Post-MI via extreme mode excitation.
  - Synthetic epi: variable offset (base 10 mm, apex 5 mm, noise σ=0.5 mm).
  - SAX contour extraction: 10 slices, 8 mm thick, 10 mm spacing.
  - Occupancy cache: 2048 query points per sample (trimesh.contains for GT labels).
  - Train/Val/Test split: 80/10/10.
- **From:** `datasetED_ssm.ipynb`.
- **Figure:** synthetic SSM examples: normal, DCM, HCM, Post-MI. This proves the synthetic generator was not only random noise.
- **Table:** synthetic dataset configuration: SSM source, PCA modes, target samples, accepted samples, query points, slice settings, split.

#### 3.2.2 Real ED Dataset
- **Write about:**
  - Sources: ACDC, M&Ms, M&Ms-2.
  - Pipeline: NIfTI → RAS+ canonical → isotropic resample (current code uses 1.5 mm target spacing) → per-slice cleaning → 3D largest component → SDF → Gaussian smoothing → Marching Cubes → Taubin smoothing (current code: 25 iters, λ=0.53, μ=-0.55) → cotangent Laplacian fairing → basal cap → quadric decimation to 4000 target vertices.
  - Contour/cache settings: 10 slices, 8.0 mm slice thickness, 10.0 mm spacing, 60 contour points per ring, 2048 query points.
  - Anatomical prechecks: apical taper, basal support, centerline drift, contour regularity, endo-inside-epi, wall proxy, long-axis ratio.
  - Patient-level split (no data leakage), followed by contour-input augmentation during training.
  - Acceleration: CPU-parallel processing via ProcessPoolExecutor, with optional CuPy GPU acceleration for Gaussian filtering when available.
- **From:** `datasetED_real.ipynb`.
- **Figure:** real-data preprocessing pipeline: segmentation mask → cleaned mask → SDF/marching cubes → smoothed/capped mesh → contours/cache.
- **Table:** real ED cache configuration: sources, label conventions, slice settings, mesh settings, quality checks, output cache.

#### 3.2.3 Real ES Dataset
- **Write about:** Same direct real-segmentation pipeline as ED-real, but for end-systolic frames. Replaces the earlier SSM-fitting approach that the notebook says produced unrealistic systolic geometry.
- **What was done:** Output cache is `es_occupancy_cache_v2/`, with patient-level splitting and contour-input augmentation during training.
- **From:** `datasetES_real.ipynb`.
- **Figure optional:** ED vs ES example surfaces from the real caches, to show that the model sees both phases.
- **Table row:** include ES in the same dataset/configuration table as ED to avoid repeated text.

#### 3.2.4 Cache Format and Contour Representation
- **Write about:** How all datasets are converted into the same input format: contour points, tissue labels, phase label, query points, SDF/occupancy targets, and mesh metadata.
- **Why it matters:** This is the bridge between the data notebooks and the training notebook. It explains how synthetic and real data can be mixed.
- **Include:** 10 SAX slices, 60 points per ring, tissue labels for endocardium/epicardium, phase conditioning, and 2048 query points.
- **Table:** cache fields and their meaning. Columns: field, shape/type, source, used by model/loss.

#### 3.2.5 Spatial Normalisation and Data Augmentation
- **Write about:** Real masks are canonicalised to RAS+, resampled to isotropic spacing, and converted to a fixed SAX-like contour representation. Synthetic meshes are sliced into the same representation.
- **Spacing details:** 1.5 mm target spacing for real mesh extraction, 10 SAX slices, 8.0 mm slice thickness, 10.0 mm slice spacing, 60 contour points per ring, 2048 query points.
- **Heart movement:** ED and ES are different phases. ES has smaller cavity volume and different myocardial configuration, so phase is included as a model input.
- **Augmentation:** For the thesis, both synthetic and real cases are augmented at the contour-input level. Augmentation simulates observation variability: XY translations, point jitter, rotations, scale jitter, slice dropout, and contour-point dropout.
- **Important wording:** Target meshes and cached SDF/query supervision remain fixed. Augmentation acts as observation noise, not as a new anatomical ground truth.

### 3.3 CardioSDF Architecture

**Important:** This should be one of the longest and clearest parts of the thesis. The reader should understand why CardioSDF is not simply a generic 3D reconstruction network.

#### 3.3.1 PointNet Encoder
- **Write about:**
  - Input: (x, y, z, tissue_label, phase) — 5D per point.
  - Shared MLP: 64 → 128 → 256 → latent_dim=256.
  - Per-tissue max-pool (separate endo/epi global features).
  - Output: 256-d latent vector z.
- **From:** `training.ipynb` PointNetEncoder class.

#### 3.3.2 Fourier Positional Encoding
- **Write about:** L=6 frequency bands → 39-d encoded point features. Helps the MLP learn high-frequency SDF detail.
- **From:** `training.ipynb` FourierPE.

#### 3.3.3 INR Decoder (Monotone-Epi SDF)
- **Write about:**
  - 8-layer MLP (width 512), skip connection at layer 4.
  - Activation: softplus(β=100) for C^∞ iso-surface (no ReLU kinks).
  - Two heads:
    - head_endo → f_endo (SDF), geometric sphere init (bias = −r₀).
    - head_delta → δ via sigmoid-bounded parameterisation: δ = τ_min + (δ_cap − τ_min)·σ(·).
  - Monotone-epi coupling: f_epi = f_endo − δ. Guarantees f_epi > f_endo everywhere, so wall thickness is positive by construction.
  - Notebook configuration: τ_min_norm = 0.05 and δ_cap_norm = 0.45, approximately 1.25–11.25 mm at the 25 mm scale printed by the notebook.
  - Weight clipping (Frobenius norm cap) for Lipschitz control without explicit gradient penalty.
- **From:** `training.ipynb` INRDecoderSDF.
- **Equation:** include the monotone-epi relation and bounded delta equation:
  - δ(x) = τ_min + (δ_cap − τ_min)σ(g_δ(x,z))
  - f_epi(x) = f_endo(x) − δ(x)
- **Figure:** small decoder diagram showing the shared MLP splitting into `head_endo` and `head_delta`.
- **Text emphasis:** this is the technical reason wall thickness is positive by construction.

### 3.4 Training Regime

**Important:** This section should explain the training decisions as engineering choices: why mixed synthetic/real data, why phase conditioning, why contour-input augmentation is applied to all data streams, and why the loss has multiple terms.

#### 3.4.1 Dataset Mix
- **Write about:**
  - Combined ED + ES with phase conditioning (input_dim=5).
  - Synthetic ED: 800 samples (SSM), augmented.
  - Real ED: ACDC, M&Ms, M&Ms-2, augmented at contour-input level.
  - Real ES: ACDC, M&Ms, M&Ms-2, augmented at contour-input level.
  - Patient-level split for real data.
- **From:** `training.ipynb` data loading.

#### 3.4.2 Augmentation
- **Write about:**
  - Applied to synthetic and real samples at the encoder-input/contour level; GT targets unchanged.
  - Per-slice XY translation, per-point jitter, slice dropout, rotation, scale jitter, contour-point dropout.
  - Augmentation is phase-preserving: ED remains ED and ES remains ES.
- **From:** `training.ipynb` augmentation pipeline.

#### 3.4.3 Loss Function
- **Write about:**
  - 10-term multi-objective loss:
    $$\mathcal{L} = λ_{surf}·L_{surf} + λ_{eik}·L_{eik} + λ_{off}·L_{off} + λ_{normal}·L_{normal} + λ_{WT}·L_{WT} + λ_{L1}·L_{L1} + λ_{anchor}·L_{anchor} + λ_{sign}·L_{sign} + λ_{extsign}·L_{extsign} + λ_{bbox}·L_{bbox\_out}$$
  - L_surf: surface fit (|f|=0 on GT surface points).
  - L_eik: eikonal (‖∇f‖=1), computed via torch.autograd.grad outside AMP for numerical stability.
  - L_off: off-surface repulsion (exponential decay).
  - L_WT: wall-thickness floor (ReLU(τ_min − δ)).
  - L_L1: cached query-point SDF supervision.
  - L_anchor: input-contour Huber anchor (f≈0 on input slice points).
  - L_sign: cross-wall sign consistency with margin.
  - L_extsign: bidirectional query-sign hinge.
  - L_bbox_out: forces f>0 outside contour AABB.
  - Latent regularisation: ‖z‖².
- **From:** `training.ipynb` loss implementation.
- **Table:** loss-term table. Columns: term, purpose, where it is applied, why it matters.
- **Equation:** full objective equation, then short prose for each term. Do not leave only the big equation without explanation.

#### 3.4.4 Optimiser and Training Parameters
- **Write about:**
  - Optimizer: AdamW, lr=2e-5, weight_decay=5e-4.
  - Cosine annealing scheduler (T₀=100).
  - AMP (mixed precision), batch size 8, gradient accumulation 2, and notebook comment indicating effective batch size = 32 on the 2×T4 run.
  - Gradient clipping at 1.0.
  - Early stopping (patience=30).
  - Multi-GPU: DataParallel on 2× T4 GPUs (Kaggle).
  - 400 epochs max.
- **From:** `training.ipynb` training loop.

### 3.5 Inference
- **Write about:**
  - Encode input contour → latent z.
  - Query 96³ grid (uniform in normalised space).
  - Marching cubes at level 0 (endo and epi separately).
  - No post-processing (no PyMeshFix, no cap synthesis). Watertight by construction (Sard's theorem on regular value of C^∞ function).
  - Analytic wall thickness: δ(x) at endo vertices × scale → mm.
- **From:** `training.ipynb` predict_mesh_sdf function.
- **Figure:** inference flow: contour input → latent z → 96³ SDF grid → marching cubes → analytic thickness map.
- **Table optional:** inference settings: grid resolution, scale, marching-cubes level, post-processing, thickness extraction.

#### 3.5.1 Analytic Wall-Thickness Extraction
- **Write about:** The analytic thickness is not a separate post-processing distance query. It comes from the decoder's bounded offset `δ` evaluated at endocardial vertices.
- **Why it matters:** This directly connects reconstruction to wall-thickness measurement and answers the objective-model research question.
- **Include:** conversion from normalised units to millimetres and the distinction between analytic CardioSDF thickness and the four external measurement methods.

### 3.6 Wall-Thickness Measurement Methods
- **Write about:**
  - Four algorithms applied to the model output meshes (not to voxel segmentations directly), spanning the main method families.
  - PDE-based: Laplace field (CG solver, transmural reference), Yezzi–Prince (Eulerian PDE).
  - Ray-based: SDF cone rays.
  - Volumetric: EDT boundary sum (baseline).
  - Key equations:
    - EDT boundary sum: t(x) = D_endo(x) + D_epi(x)
    - Laplace: ∇²ψ = 0 with ψ|_endo=0, ψ|_epi=1; t = 1/|∇ψ|
    - Yezzi–Prince: ∇ψ·∇u = −1 (u|_endo=0); t = u + v
    - Cone rays: median of K=7 hits at α=30° half-angle
- **From:** `lv_wall_thickness_10_methods.ipynb`.
- **Table:** four-method implementation table. Columns: method, category, input representation, output, local/global, expected weakness.
- **Equations:** include the main equations for EDT boundary sum, Laplace, Yezzi-Prince, and SDF cone rays. Put longer derivations in Appendix A.
- **Important wording:** these four methods are applied to CardioSDF-generated geometry, not directly to raw segmentation voxels.

### Chapter 3 deliverables
- **Figure 3.1:** complete CardioSDF pipeline.
- **Figure 3.2:** dataset preparation pipeline for synthetic and real data.
- **Figure 3.3:** CardioSDF architecture with PointNet encoder and monotone-epi decoder.
- **Figure 3.4 optional:** ED versus ES examples or synthetic pathology examples.
- **Table 3.1:** dataset/cache configuration.
- **Table 3.2:** cache field schema.
- **Table 3.3:** model hyperparameters.
- **Table 3.4:** loss terms.
- **Table 3.5:** four wall-thickness methods.
- **Equations:** monotone-epi thickness equations, bounded delta, training objective, Chamfer/SDF supervision if needed, and key wall-thickness equations.

---

## Chapter 4: Results and Evaluation

### Chapter 4 role
- **This is the evidence chapter.** It should prove the claims made in Chapter 3, not re-explain the pipeline.
- **Target length:** around 12--18 pages, depending on how many final figures and tables are available.
- **Main risk:** weak results would make the thesis method-heavy but evidence-light.

### 4.1 Experimental Setup

#### 4.1.1 Datasets
- **Write about:**
  - Synthetic ED: 1300 samples (UK Biobank SSM), 80/10/10 split.
  - Real ED: ACDC, M&Ms, M&Ms-2, patient-level split.
  - Real ES: ACDC, M&Ms, M&Ms-2, patient-level split.
  - Mixed training: synthetic ED (800) + real ED + real ES, with contour-input augmentation for all streams.
- **From:** `datasetED_ssm.ipynb`, `datasetED_real.ipynb`, `datasetES_real.ipynb`.
- **Table:** final dataset statistics. Columns: dataset/source, phase, number of patients, number of samples, train, validation, test, augmented yes/no, cache path.
- **Missing value:** exact real ED/ES counts must be copied from the final cache/notebook outputs before final writing.

#### 4.1.2 Hardware and Software
- **Write about:**
  - Training: 2× NVIDIA T4 GPUs (Kaggle), PyTorch 2.x, DataParallel.
  - Wall-thickness computation: CPU-parallel where applicable, with optional CuPy acceleration for volumetric operations when available.
  - Key libraries: trimesh, PyMeshLab, pyezzi (Yezzi–Prince), scipy (sparse CG), scikit-image (marching cubes).
- **From:** `training.ipynb`, `lv_wall_thickness_10_methods.ipynb`.

#### 4.1.3 Evaluation Metrics
- **Write about:**
  - Reconstruction quality:
    - Watertight rate (target: 100 %).
    - Chamfer distance to GT meshes (mm), separate for endo and epi.
  - Wall thickness:
    - Mean, p5, p95 (analytic δ × scale).
    - Comparison across four methods.
  - Slice residual: mean |f| on input contour points (mm).
- **From:** `training.ipynb` evaluation, `lv_wall_thickness_10_methods.ipynb`.
- **Equations:** define Chamfer distance, slice residual, and the wall-thickness summary statistics. If Hausdorff/HD95 is used, define it too.
- **Important wording:** wall-thickness comparison is not clinical validation unless expert measurement ground truth is added.

#### 4.1.4 Baselines and Reference Comparisons
- **Write about:** At least one comparison point so the results are not only CardioSDF in isolation.
- **Preferred baseline:** direct segmentation-derived marching-cubes meshes, with KD-tree or Laplace thickness on those meshes.
- **Alternative baseline:** older SSM/ray-based reconstruction if outputs exist and are fair to report.
- **Minimum fallback if no baseline is finished:** explicitly state that the evaluation focuses on internal reconstruction quality and algorithmic comparison of four wall-thickness methods on a common CardioSDF geometry.
- **Table:** baseline/reference comparison table. Columns: method, input, output, watertightness, positive-thickness guarantee, local thickness support, main limitation.

### 4.2 Reconstruction Results
- **Write about:**
  - Watertight rate: report the percentage (should be 100 % for marching cubes on C^∞ SDF).
  - Chamfer distance: report mean ± std for endo and epi on test set.
  - Slice residual: report mean ± std (how well the model fits the input contours).
  - Visual: show example reconstructions for ED and ES (figures from training.ipynb if they exist).
- **What was done:** The evaluation code is planned in `training.ipynb`, but exact final numbers must be copied from the notebook outputs before writing the final Results text.
- **From:** `training.ipynb` test-set evaluation.
- **Table:** reconstruction metrics table. Rows: CardioSDF, baseline if available. Columns: watertight rate, endo Chamfer, epi Chamfer, slice residual, runtime/inference time if available.
- **Figure:** qualitative ED and ES reconstructions. Show endocardium and epicardium, preferably with input SAX contours overlaid.
- **Figure optional:** error heat map or point-cloud residual visualization if available.

### 4.3 Wall-Thickness Comparison (Four Methods)
- **Write about:**
  - Summary table: four methods × (mean, p5, p95, std) wall thickness in mm.
  - 3D colour-map figure (one subplot per method).
  - AHA-17 bullseye plot for the reference method (regional thickness).
  - AHA anatomical QA figure (base/apex direction + septal orientation check).
  - Qualitative comparison: which methods agree, which are outliers?
- **What was done:** The notebook computes per-endocardial-vertex thickness on the CardioSDF model geometry and produces the summary table, 3D colour maps, AHA-17 bullseye, and anatomical QA figure.
- **From:** `lv_wall_thickness_10_methods.ipynb` outputs.
- **Table:** final four-method comparison. Include CardioSDF analytic δ as its own row if available, then the four methods. Columns: method, category, n, mean, median, std, p5, p95, min, max, finite fraction, runtime, warnings.
- **Figure:** 3D colour-map figure.
- **Figure:** AHA-17 bullseye comparison.
- **Figure:** AHA anatomical QA figure to verify base/apex and septal orientation.
- **Text:** explicitly discuss local variation: where thickness is higher/lower, and whether methods agree regionally.

### 4.4 Local and Regional Wall-Thickness Findings
- **Write about:** This section should directly answer the "can we see local changes?" research question.
- **Use:** per-vertex colour maps, p5/p95 statistics, AHA-17 segment values, and method agreement/disagreement.
- **Table:** AHA-17 regional thickness table if not too large. If too large, put full values in Appendix A and keep a summary table in Chapter 4.
- **Figure:** one clear local-thickness example with anatomical orientation labels.
- **Important wording:** local changes are measured on the reconstructed LV model; do not imply longitudinal patient change unless the same patient/timepoint comparison was actually performed.

### 4.5 Failure Cases and Limitations in Results
- **Write about:** Where the pipeline or thickness methods are weak.
- **Examples to include if available:** apex/base instability, sparse medial-axis output, ray-casting misses, KD-tree underestimation, segmentation artifacts, low-resolution grid effects.
- **Table optional:** method failure/warning summary. Columns: method, failure mode, observed warning, practical consequence.
- **Figure optional:** one representative failure case. This makes the evaluation more credible.

### 4.6 Discussion
- **Write about:**
  - CardioSDF produces watertight geometry with guaranteed positive wall thickness.
  - Wall-thickness methods show agreement in bulk regions but differ near the apex and base.
  - Laplace and Yezzi–Prince are the most theoretically sound (PDE-based).
  - KD-tree methods are fast but sensitive to surface irregularities.
  - SDF cone rays leverage the implicit representation directly.
- **From:** Analysis of `lv_wall_thickness_10_methods.ipynb` results.
- **Must include:** a blunt distinction between algorithmic validation and clinical validation. This thesis can claim objective computational measurement, but not clinical replacement without expert ground truth.

### Chapter 4 deliverables
- **Table 4.1:** final dataset split/statistics.
- **Table 4.2:** reconstruction metrics.
- **Table 4.3:** baseline/reference comparison.
- **Table 4.4:** CardioSDF analytic thickness + four-method thickness summary.
- **Table 4.5 optional:** method failures/warnings or AHA-17 regional values.
- **Figure 4.1:** ED/ES qualitative reconstructions.
- **Figure 4.2:** local wall-thickness colour map on CardioSDF geometry.
- **Figure 4.3:** four-method 3D colour-map panel.
- **Figure 4.4:** AHA-17 bullseye plot.
- **Figure 4.5:** AHA orientation QA.
- **Figure 4.6 optional:** failure case or baseline visual comparison.

---

## Chapter 5: Conclusions

### 5.1 Summary
- **Write about:** One-paragraph recap: we proposed CardioSDF (INR-SDF with monotone-epi guarantee), trained on mixed synthetic + real data, and compared four wall-thickness methods.
- **From:** High-level summary of Chapters 3 and 4.

### 5.2 Revisiting the Research Questions
- **RQ1 — 3D reconstruction from 2D SAX images:** Yes, CardioSDF reconstructs 3D LV endocardial and epicardial surfaces from sparse 2D SAX contour inputs using a PointNet encoder and SDF decoder. The answer should be supported with watertight rate, Chamfer distance, and slice residual results.
- **RQ2 — Local wall-thickness changes:** Yes, local wall-thickness variation can be represented over the reconstructed surface. The answer should use the 3D colour maps, p5/p95 statistics, and AHA-17 bullseye plots from `lv_wall_thickness_10_methods.ipynb`.
- **RQ3 — Physically valid thickness:** Yes, the monotone-epi parameterisation guarantees δ ≥ τ_min everywhere, so the predicted epicardium remains outside the endocardium by construction.
- **RQ4 — Objective LV wall-thickness model:** Yes, the work builds a reproducible computational framework for LV wall-thickness measurement by applying the same CardioSDF geometry to analytic δ estimation and 10 established wall-thickness algorithms.
- **From:** Results in Chapter 4.

### 5.3 Contributions Summary
- **Write:** Repeat the four contributions from 1.5, now backed by the evidence in Chapter 4.

### 5.4 Limitations
- **Write about:**
  - Model trained on short-axis views only (no long-axis).
  - Input comes from contours/segmentations, not raw MRI pixels.
  - LV-only model, not full-heart reconstruction.
  - Wall-thickness methods benchmark is on model output, not direct comparison to expert manual measurements.
  - No clinical validation (no patient outcomes, no inter-rater agreement study).
  - 96³ grid resolution is a trade-off (higher resolution would be more accurate but slower).
  - Results depend on the quality of upstream segmentations and contour extraction.
- **From:** Honest assessment of what was not done.

### 5.5 Future Work
- **Write about:**
  - Multi-view fusion (SAX + long-axis).
  - Clinical validation study (compare to expert cardiologist measurements).
  - Real-time inference (optimise grid resolution, TensorRT/ONNX export).
  - Extension to other cardiac structures (right ventricle, atria).
  - Pathology-specific shape priors (incorporate disease labels into latent space).
- **From:** Natural extensions of the current work.

### Chapter 5 deliverables
- **No new figures required.** Only include a small summary table if it helps answer the research questions clearly.
- **Table optional:** research question answer summary. Columns: RQ, evidence used, conclusion, limitation.
- **Must include:** one honest paragraph separating what was demonstrated from what still requires clinical validation.

---

## Appendix A

### A.1 Hyperparameters
- **Write:** Full table of all training hyperparameters (lr, weight_decay, batch size, loss weights, τ_min, δ_cap, etc.).
- **From:** `training.ipynb` config.

### A.2 Dataset Statistics
- **Write:** Table of dataset sizes (train/val/test counts for synthetic ED, real ED, real ES).
- **From:** `datasetED_ssm.ipynb`, `datasetED_real.ipynb`, `datasetES_real.ipynb`.

### A.3 Wall-Thickness Method Equations (Full Derivations)
- **Write:** Detailed derivations for Laplace and Yezzi–Prince (too long for main text).
- **From:** Cited papers + `lv_wall_thickness_10_methods.ipynb` implementation.

### A.4 Full Result Tables
- **Write:** Put oversized tables here if they interrupt Chapter 4 flow.
- **Include if available:** full AHA-17 values for all four methods, per-case reconstruction metrics, method warnings, and runtime details.
- **From:** `training.ipynb` and `lv_wall_thickness_10_methods.ipynb`.

### A.5 Reproducibility Details
- **Write:** Environment, major libraries, cache paths, model checkpoint path, inference grid, and exact command/notebook order needed to reproduce the results.
- **From:** all notebooks.

---

## Summary of Notebook → Chapter Mapping

| Notebook | Chapter(s) | What to Write |
|----------|------------|---------------|
| `datasetED_ssm.ipynb` | Ch3 §3.2.1, Ch4 §4.1.1 | Synthetic ED dataset (SSM), 1300 meshes, quality filters, pathology simulation, occupancy cache |
| `datasetED_real.ipynb` | Ch3 §3.2.2, Ch4 §4.1.1 | Real ED dataset (ACDC, M&Ms, M&Ms-2), Marching Cubes pipeline, anatomical prechecks |
| `datasetES_real.ipynb` | Ch3 §3.2.3, Ch4 §4.1.1 | Real ES dataset, same pipeline as ED-real |
| `training.ipynb` | Ch3 §3.3–3.5, Ch4 §4.1.2–4.2 | CardioSDF architecture (PointNet + INR), loss function, training regime, inference, reconstruction metrics |
| `lv_wall_thickness_10_methods.ipynb` | Ch2 §2.4, Ch3 §3.6, Ch4 §4.3–4.5, App. A | Four wall-thickness methods (selected from a broader survey), equations, final comparison table, AHA-17 bullseye, colour maps, method warnings |
| Baseline/reference comparison | Ch4 §4.1.4, §4.2, §4.6 | Direct segmentation marching-cubes or older SSM/ray-based comparison if available; otherwise explicitly state as missing clinical/reference limitation |

---

**Next steps:**
1. Expand Chapter 3 first. It is the technical core and should carry the thesis.
2. Extract exact real ED/ES dataset counts and train/val/test splits from the final caches/notebook outputs.
3. Finish or export the final reconstruction metrics table from `training.ipynb`.
4. Finish or export the final wall-thickness comparison table from `lv_wall_thickness_10_methods.ipynb`, including runtime, finite fraction, and warnings.
5. Decide whether a baseline can be reported. If not, state the limitation clearly in Chapter 4 and Chapter 5.
6. Export the required figures: CardioSDF pipeline, dataset preparation pipeline, model architecture, ED/ES reconstruction examples, local thickness map, four-method colour map panel, AHA-17 bullseye, and AHA orientation QA.
7. Add citations to `references.bib` as you write (delegate to thesis-researcher if needed).
8. Build with `latexmk -pdf main.tex` after each section to catch errors early.
