---
name: cardiosdf-results-metrics
description: "Use when asked to compute, recompute, regenerate, verify, or update the Results-chapter metrics or figures of this thesis — the unnamed model's LV reconstruction meshes (ED/ES figure), reconstruction-quality metrics (Chamfer, ASSD, HD95, Dice/IoU, watertight rate, volume ratio), and the myocardial wall-thickness measurements (Laplace, Yezzi-Prince, SDF cone rays, EDT boundary sum, AHA-17). Explains where the inference pipeline lives, how to run the figure/metric scripts, the mandatory slice-spacing calibration fix, the watertight/orientation fixes, known caveats, and which thesis table/figure each script produces. Triggers: 'do the metrics', 'recompute results', 'regenerate the meshes/figure', 'wall thickness numbers', 'reconstruction quality table'."
argument-hint: "Which metric, table, or figure to (re)compute"
---

# Model Results & Metrics Pipeline

The reconstruction model has no proper name. Thesis prose, captions, tables,
and figure labels must call it ``the model'', ``the proposed model'', or ``the
proposed approach''. The lowercase term retained in this skill's folder name,
image filenames, checkpoint paths, and Python identifiers is legacy
implementation vocabulary only.

Authoritative procedure for (re)generating everything in
[chapters/04-results.tex](../../../chapters/04-results.tex). Follow this before
touching any number or figure in the Results chapter.

## When to Use

- Regenerating the ED/ES reconstruction figure (`images/recon_ed_es_meshes.png`).
- Recomputing reconstruction-quality metrics (Chamfer, ASSD, HD95, Dice/IoU,
  normal consistency, F-score, watertight rate, volume ratio).
- Recomputing wall-thickness statistics (Laplace field, Yezzi–Prince, SDF cone
  rays, EDT boundary sum) and the AHA-17 regional table.
- Verifying whether any Results number still holds after a pipeline change.

## Environment

- Interpreter: **`C:/Python313/python.exe`** (system Python 3.13). Always call it
  explicitly; the VS Code "install package" tool targets a different env.
- Installed for this pipeline: `torch`, `numpy`, `nibabel`, `scikit-image`,
  `scipy`, `trimesh`, `pyvista`, `vtk`, `pyezzi`, `pandas`, `matplotlib`.
- Do **not** use here-strings piped to `python -` for long scripts — PSReadLine
  crashes. Put diagnostics in a temp `scripts/_*.py` file, run it, then delete it.

## Where the Pipeline Lives

- Inference/mesh code: **`scripts/webapp/core/`** (`sdf_model.py`, `inference.py`,
  `nifti.py`). This folder is downloaded from
  `github.com/Andrefce/tese/tree/main/webapp/core`; if missing, re-download it
  there (the figure scripts do `sys.path.insert(0, ROOT/"scripts"/"webapp")` then
  `from core.sdf_model import ...`, so it MUST sit at `scripts/webapp/core`).
- Trained model: [notebooks/inr_sdf_combined_fresh_ed_mix_v1_final.pt](../../../notebooks/inr_sdf_combined_fresh_ed_mix_v1_final.pt).
- Demo case shipped in-repo: `notebooks/patient002/` (ED = frame01, ES = frame12).

## Scripts → Thesis outputs

- [scripts/fig_ed_es_meshes.py](../../../scripts/fig_ed_es_meshes.py) →
  `images/recon_ed_es_meshes.png` → `fig:recon-ed-es-meshes` (the ONLY mesh
  figure used in the thesis). Runs on patient002 with the local `.pt` model.
- [scripts/compute_results_cohort.py](../../../scripts/compute_results_cohort.py)
  → the reconstruction-quality table, wall-thickness table, AHA-17 table.
  Needs a cohort at `scripts/webapp/demo-data/training/` and a model at
  `scripts/webapp/model/*.ptrom` — **both absent in the workspace**, so the
  cohort numbers cannot be produced here yet.
- Methodology figure `cardiosdf_pipeline_visual_flow` is made by
  [scripts/generate_patient002_methodology_figures.py](../../../scripts/generate_patient002_methodology_figures.py)
  via a separate matplotlib/SSM path (reads true zooms) — unaffected by the fixes
  below.

## CRITICAL: slice-spacing calibration

Some NIfTIs (e.g. `patient002`, M&Ms-2) store real spacing only in the header
pixdim `(1.367, 1.367, 10.0)` while the affine is identity, so
`nib.as_closest_canonical` reports `(1,1,1)`. Using `dz = |affine[2,2]| = 1`
collapses the 10 mm slice spacing and flattens the LV long axis ~10×.

Fix (already applied in `fig_ed_es_meshes.py::reconstruct`, apply the same to any
new script): keep in-plane from the affine and express the slice spacing in the
affine's in-plane unit —

```python
zooms = np.abs(np.asarray(raw.header.get_zooms()[:3], float))  # raw = nib.load(path)
aff_inplane = float(np.linalg.norm(affine[:3, 0])) or 1.0
true_inplane = float(min(zooms[0], zooms[1])) or 1.0
dz = float(zooms[2]) * (aff_inplane / true_inplane)   # == |affine[2,2]| for well-formed files
contours = extract_contours(seg, affine, dz)
```

This keeps the anatomically-correct in-plane scale (endo ~50 mm) and only
stretches Z. A full-mm affine (`diag(zooms)`) instead OVER-scales in-plane
(endo ~75 mm) and must not be used for the reconstruction.

## Watertight + orientation

- Watertight: call `mesh.merge_vertices()` first (marching cubes leaves unmerged
  vertices), then fill holes, then the base-cap fallback `_cap_open_boundaries`.
  After this, ED/ES endo+epi are all `watertight=True`.
- Orientation: `FLIP_LONG_AXIS_FOR_DISPLAY = True` in `fig_ed_es_meshes.py`
  renders apex-down/base-up (the correct, not-upside-down view).

## Regenerate the ED/ES figure

```powershell
C:/Python313/python.exe scripts/fig_ed_es_meshes.py
```

Expect `watertight=True` printed for all four surfaces and a rewritten
`images/recon_ed_es_meshes.png` showing a prolate LV, apex down, ED cavity fuller
than ES.

## Wall-thickness status (IMPORTANT)

The wall-thickness numbers currently in [chapters/04-results.tex](../../../chapters/04-results.tex)
are flagged **PROVISIONAL** — they were computed on the pre-calibration
(flattened) geometry. Verified on patient002 ED, correcting calibration roughly
**doubles** them:

| Measure | Old (flattened) | Corrected |
| --- | --- | --- |
| Segmentation reference (in-plane EDT) | 4.24 mm | 9.39 mm |
| KD-nearest endo→epi | 3.21 mm | 6.47 mm |
| EDT boundary sum | 7.31 mm | 10.21 mm |

Caveats on a single anisotropic (10 mm-slice) file: the 3D methods are fragile —
Laplace collapses to a near-constant, SDF cone-rays explode, and **pyezzi
(Yezzi–Prince) native-crashes the process (uncatchable segfault) — skip it**.
Trustworthy final numbers require running `compute_results_cohort.py` on the
proper multi-patient cohort (the `.ptrom` model + `demo-data`, not in the repo).
When that data is available, apply the same calibration fix inside
`compute_results_cohort.py` before trusting its output.

## Reuse of method code

The wall-thickness estimators are defined at module level in
[scripts/compute_results_cohort.py](../../../scripts/compute_results_cohort.py):
`method_laplace_field`, `method_yezzi_prince`, `method_sdf_cone_rays`,
`method_edt_boundary_sum`, plus `voxelize_mesh_to_grid`, `voxel_to_world`,
`orient_normals`, `assign_aha17`, `build_full_meshes`. Import them for one-off
recomputes (it imports `pyezzi` and `pandas` at module load).

## LaTeX build recipe (avoids the bookmark abort)

`latexmk` fails (no Perl). The hyperref bookmark for the methodology subsection
"Real ED and ES…" ([chapters/03-methodology.tex](../../../chapters/03-methodology.tex)
line ~284) can hit pdflatex's 100-error limit and abort with no PDF. Delete
`main.out` **before every** pdflatex pass (cross-refs use `main.aux`, only
bookmarks use `main.out`):

```powershell
Remove-Item main.out -EA SilentlyContinue; pdflatex -interaction=nonstopmode main.tex > $null; `
biber main > $null; `
Remove-Item main.out -EA SilentlyContinue; pdflatex -interaction=nonstopmode main.tex > $null; `
Remove-Item main.out -EA SilentlyContinue; pdflatex -interaction=nonstopmode main.tex > $null
```

Then check `main.log` for `Citation undefined`, `Reference … undefined`, and
`Fatal error` before reporting done.

## Integrity rule

Never present numbers computed on mis-calibrated geometry as final. Keep the
`$^{\dagger}$` / `[PROVISIONAL]` flags until the corrected cohort recomputation
replaces them.
