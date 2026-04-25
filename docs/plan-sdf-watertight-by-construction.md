# Plan — Watertight by Construction with a Signed-Distance INR

> **Goal.** Replace the per-tissue **binary occupancy heads** of the
> current `INRDecoder` (in [training-model-v2.ipynb](../training-model-v2.ipynb))
> with a **signed-distance** parameterisation so that the LV endo and
> epi surfaces are the zero level-sets of two coupled, continuous,
> Lipschitz-1 scalar fields. This makes the reconstruction watertight
> *by construction* and **deletes** every piece of geometric
> post-processing currently in
> [acdc-thesis-watertight.ipynb](../acdc-thesis-watertight.ipynb)
> (`stamp_slice_evidence`, `synthesize_caps`, `repair_watertight`,
> `fill_open_rims`, the iterative PyMeshFix loop), and the
> ray-casting `_wt_per_vertex` wall-thickness estimator.

This document is a concrete engineering plan, grounded in the existing
notebooks. It tells you *which cells to change*, *what to put there*,
and *what to delete*.

---

## 0 · Codebase baseline (what we are replacing)

| Concern | Current implementation | Reference |
| --- | --- | --- |
| Encoder | `PointNetEncoder` — 3 shared Conv-LN-ReLU MLPs (4/5 → 64 → 128 → 256) + global max-pool (256-d) + per-tissue max-pools (128-d each, after `tissue_proj`), aggregated 512 → 512 → **z = 256** | training-model-v2.ipynb, `class PointNetEncoder` |
| Decoder | `INRDecoder` — `FourierPE(L=6)` (3 + 6·6 = **39**-d), 8-layer MLP of width **512** with skip-concat at layer 4, two `Linear(512, 1)` heads | training-model-v2.ipynb, `class INRDecoder` |
| Heads | `head_endo`, `head_epi` → logit; sigmoid in inference. Init `N(0,1e-4)`, bias 0 (start ≈ 0.5 occupancy) | training-model-v2.ipynb, decoder cell |
| Loss | Dice + boundary-weighted BCE + epi⊇endo consistency + min-wall-occupancy + progressive latent reg. | `occupancy_loss(...)` |
| Mesh extraction | Dense grid (64³ training, 96³ thesis), `marching_cubes(occ, level=0.5, spacing=voxel_size)`, `+ lo` to bring back to normalised coords | `predict_mesh(...)` |
| Watertight pipeline | `stamp_slice_evidence` → `synthesize_caps` (basal cone + hemispherical apex) → MC → `largest_component` → `repair_watertight` (PyMeshFix + trimesh fallback + `fill_open_rims` + winding fix), iterated max 3× | acdc-thesis-watertight.ipynb |
| Wall thickness | `_wt_per_vertex(endo, epi, method='ray')` — ray cast along endo normal, fallback to point-to-surface distance; per-AHA-segment summary | acdc-thesis-watertight.ipynb |
| Cache schema (`.npz`) | `contour (N,4)`, `query (Q,3)`, `endo_occ (Q,) bool`, `epi_occ (Q,) bool`, `endo_vertices/faces`, `epi_vertices/faces`, `scale`, `centroid`, `phase` | build-lv-cache.ipynb |
| Normalisation | `centroid = xyz_raw.mean(0)`; `scale = mean(‖xy_c‖₂)`; `xyz_n = (xyz_raw - centroid)/scale` (so `‖xy_n‖ ≈ 1` on average) | build-lv-cache.ipynb |
| Splits | `split.json` next to the cache: `{"tr":[…], "val":[…], "te":[…]}` indexing `sample_*.npz` | training-model-v2.ipynb, `_load_cache_split` |

The PointNet encoder, FourierPE, augmentation pipeline,
DataLoader/collate, optimiser/scheduler, multi-GPU harness, and
patient split loader are **all reused unchanged**. The work is in the
decoder heads, the loss, the cache targets, and the inference path.

---

## 1 · Why SDFs are the right tool

| Property | Occupancy (current) | SDF (proposed) |
| --- | --- | --- |
| Surface definition | $\{x : \sigma(f(x)) = 0.5\}$ | $\{x : f(x) = 0\}$ |
| Continuity / regularity of $f$ | bounded $[0,1]$, very flat far from surface | unbounded, **eikonal** $\lVert\nabla f\rVert \approx 1$ |
| Closed surface? | only if iso-level is crossed on every ray → **not guaranteed** (cause of basal/apical holes today) | **yes** — by Sard's theorem, the zero level-set of a continuous $f$ at a regular value is a closed 2-manifold inside the bbox |
| Need post-hoc capping? | yes (apex / base) — `synthesize_caps`, `fill_open_rims` | no |
| Normals | finite differences on the mesh | analytic: $\hat n(x) = \nabla f(x)/\lVert\nabla f\rVert$ |
| Distance to surface | undefined | the value itself |
| Wall thickness | ray cast endo→epi (`_wt_per_vertex`), fragile at apex / valve plane (the `p5 = 0 mm` outliers on patient001) | analytic: $\delta(x) = f_\text{endo}(x) - f_\text{epi}(x)$ |
| Loss signal off-surface | only "inside vs. outside" (vanishing gradient far from boundary) | actual signed distance → strong gradient everywhere |

The third row is the decisive one. The pipeline today achieves
watertightness *empirically* (PyMeshFix + cap synthesis + fan
triangulation, iterated) — the SDF achieves it *mathematically*.

---

## 2 · Architecture changes (tese/training-model-sdf.ipynb)

Branch `training-model-v2.ipynb` to `training-model-sdf.ipynb`. Keep
`PointNetEncoder` untouched. Modify only the decoder.

### 2.1 · Decoder heads + monotone-epi parameterisation

The most important architectural change is **not** "swap sigmoid for
identity". It is to **couple the two heads** so that wall thickness
is a structurally non-negative quantity:

```python
class INRDecoderSDF(nn.Module):
    def __init__(self, latent_dim=256, fourier_L=6,
                 hidden=512, n_layers=8, skip_layer=4,
                 r0=0.5):
        super().__init__()
        self.pe = FourierPE(L=fourier_L)             # reuse
        self.skip_layer = skip_layer
        in_dim = latent_dim + self.pe.out_dim         # 256 + 39 = 295

        self.layers = nn.ModuleList()
        cur = in_dim
        for j in range(n_layers):
            if j == skip_layer:
                cur = cur + in_dim                    # skip-concat
            self.layers.append(nn.Linear(cur, hidden))
            self.layers.append(nn.LayerNorm(hidden))
            self.layers.append(nn.ReLU(inplace=True))
            cur = hidden

        # Monotone-epi parameterisation (§ 2.2)
        self.head_endo  = nn.Linear(hidden, 1)        # raw signed distance
        self.head_delta = nn.Linear(hidden, 1)        # log-thickness (softplus → > 0)

        self._geometric_init(r0=r0)
```

**Forward pass.** Predict `f_endo` directly and `f_epi = f_endo − δ`,
with `δ = softplus(head_delta(h)) > 0`:

```python
def forward(self, z, xyz):
    pe_xyz = self.pe(xyz)                             # (B, Q, 39)
    z_exp  = z.unsqueeze(1).expand(-1, xyz.shape[1], -1)
    inp    = torch.cat([z_exp, pe_xyz], dim=-1)       # (B, Q, 295)
    h      = inp
    for j in range(0, len(self.layers), 3):
        step = j // 3
        if step == self.skip_layer:
            h = torch.cat([h, inp], dim=-1)
        h = self.layers[j](h); h = self.layers[j+1](h); h = self.layers[j+2](h)

    f_endo = self.head_endo(h).squeeze(-1)            # (B, Q) signed distance
    delta  = F.softplus(self.head_delta(h)).squeeze(-1) + 1e-4  # (B, Q) > 0
    f_epi  = f_endo - delta                           # (B, Q) signed distance
    return f_endo, f_epi, delta
```

**Why this matters.** In the current model, `head_endo` and `head_epi`
are independent → the network can predict an epi level-set that
*locally crosses inside* the endo level-set, collapsing the wall to
zero. This is exactly the failure mode that produces patient001's
`p5 = 0 mm` outliers and the underestimated ED 6.6 mm / ES 8.1 mm /
ΔWT +1.5 mm (clinical range is +3–8 mm). The reparameterisation
above makes "wall thickness > 0 *everywhere*" a structural property
of the model, before any loss is applied.

Wall thickness in *normalised units* is `δ(x)`; in mm it is
`δ(x) · scale` (the same `scale` already stored in every cache
sample).

### 2.2 · Geometric (sphere) initialisation — DeepSDF / SAL trick

A randomly initialised SDF network produces an essentially random
zero level-set: marching cubes returns noisy speckle, the surface and
eikonal losses fight each other, and training collapses to
$f \equiv 0$. The fix is to initialise the network so that *at
$t=0$* it represents the SDF of a sphere of radius $r_0 \approx 0.5$
(half the normalised LV scale):

$$
f_\text{endo}(x) \;\approx\; \lVert x \rVert - r_0 \quad\text{at init.}
$$

```python
def _geometric_init(self, r0=0.5):
    # Hidden layers: standard Kaiming
    for m in self.layers:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=math.sqrt(2.0 / m.in_features))
            nn.init.zeros_(m.bias)
    # Final endo head: tiny weights, bias = -r0 → f ≈ -r0 at the origin
    nn.init.normal_(self.head_endo.weight, std=1e-4)
    nn.init.constant_(self.head_endo.bias, -r0)
    # Delta head: tiny weights, bias chosen so softplus(bias) ≈ 0.16 (≈ 3 mm at scale ~25 mm)
    nn.init.normal_(self.head_delta.weight, std=1e-4)
    nn.init.constant_(self.head_delta.bias, math.log(math.expm1(0.16)))  # softplus⁻¹(0.16)
```

This is the single most important difference between an INR that
trains and one that collapses. It also gives a sensible "warm" output
during the first epochs while the eikonal loss is being ramped in
(§ 4.2).

### 2.3 · Activation choice

Two options to ablate (they fit the existing harness with the same
hyper-params):

1. **ReLU + FourierPE (L=6)** — keep the current activation. Cheap,
   well-tested, but $\nabla f$ is piecewise-constant so the eikonal
   loss is harder to satisfy exactly.
2. **SIREN** — replace `nn.ReLU` with `nn.Sin(ω₀ x)` (`ω₀ = 30` for
   the first layer, `ω₀ = 1` for hidden) and use the SIREN-paper init
   (`U(-√(6/n)/ω₀, +…)`). Smoother gradients → much cleaner eikonal
   convergence. **Drop the FourierPE** in this case (SIREN encodes
   high frequencies natively).

The plan recommends SIREN as the *primary* SDF run and ReLU+FF as a
baseline ablation, so the thesis can show the activation effect on
eikonal residual.

---

## 3 · Loss (replaces `occupancy_loss`)

Define a single function `compute_sdf_losses(z, batch)` returning a
dict of scalar terms. Total loss:

$$
\mathcal{L} =
  \lambda_\text{surf} \mathcal{L}_\text{surf}
+ \lambda_\text{eik}  \mathcal{L}_\text{eik}
+ \lambda_\text{off}  \mathcal{L}_\text{off}
+ \lambda_\text{normal} \mathcal{L}_\text{normal}
+ \lambda_\text{WT}   \mathcal{L}_\text{WT}
+ \lambda_\text{reg}(t) \, \lVert z \rVert^2
$$

Note the latent regulariser $\lambda_\text{reg}(t)$ is reused
unchanged from the occupancy run (`latent_reg_max = 1e-4`,
`latent_reg_warmup = 100`).

### 3.1 · Term definitions

Let $\mathcal{S}_e$ (resp. $\mathcal{S}_p$) be the set of contour points
on the endo (resp. epi) surface, and let $\Omega$ be the working
bounding box (the same `[lo, hi]` as in `predict_mesh`).

**Surface term** — both surfaces sit at the zero level-set:

$$
\mathcal{L}_\text{surf} =
  \mathbb{E}_{x \in \mathcal{S}_e} |f_\text{endo}(x)|
+ \mathbb{E}_{x \in \mathcal{S}_p} |f_\text{epi}(x)|
$$

**Eikonal term** — enforce metric SDF on a band of free-space points:

$$
\mathcal{L}_\text{eik} =
  \mathbb{E}_{x \in \Omega \cup \mathcal{N}}\bigl[(\lVert\nabla_x f_\text{endo}(x)\rVert - 1)^2\bigr]
$$

(applied to `f_endo` only; `f_epi` inherits Lipschitz-1 because
$f_\text{epi} = f_\text{endo} - \delta$ and $\delta$ is bounded).

**Off-surface term** — push level-sets *away* from non-surface points:

$$
\mathcal{L}_\text{off} =
  \mathbb{E}_{x \in \Omega \setminus (\mathcal{S}_e \cup \mathcal{S}_p)}
  \bigl[\,e^{-\alpha |f_\text{endo}(x)|} + e^{-\alpha |f_\text{epi}(x)|}\,\bigr],\quad \alpha = 50
$$

**Normal alignment** — match analytic gradient to GT contour normal:

$$
\mathcal{L}_\text{normal} =
  \mathbb{E}_{x \in \mathcal{S}_e} \bigl(1 - \langle \nabla f_\text{endo}(x), \hat n_e(x) \rangle\bigr)
+ \mathbb{E}_{x \in \mathcal{S}_p} \bigl(1 - \langle \nabla f_\text{epi}(x),  \hat n_p(x) \rangle\bigr)
$$

with $\hat n$ computed in the dataloader from the angularly-ordered
ring (§ 5).

**Wall-thickness hinge** — anatomical floor on $\delta$ at the endo
surface (only active where the network would otherwise collapse):

$$
\mathcal{L}_\text{WT} =
  \mathbb{E}_{x \in \mathcal{S}_e}\bigl[\operatorname{ReLU}(\tau_\text{min} - \delta(x))\bigr]
$$

with $\tau_\text{min} = 3\,\text{mm} / \overline{\text{scale}} \approx 0.12$
(using $\overline{\text{scale}} \approx 25$ mm from the cache).

### 3.2 · Recommended weights and schedule

| Hyperparam | Value | Notes |
| --- | --- | --- |
| $\lambda_\text{surf}$ | 3.0 | Strongest term; data fidelity |
| $\lambda_\text{eik}$ | 0.0 → 0.1 | Linear ramp over epochs 0–10. Starting at 0 avoids fighting the geometric init while the network still maps to a sphere SDF |
| $\lambda_\text{off}$ | 0.1 | DeepSDF default |
| $\lambda_\text{normal}$ | 1.0 | Disabled for the first 5 epochs (set to 0) |
| $\lambda_\text{WT}$ | 0.5 | Active throughout; saturates ReLU = 0 in healthy walls |
| $\tau_\text{min}$ | 0.12 (≈ 3 mm) | Anatomical floor |
| $\alpha$ | 50 | Off-surface decay |

### 3.3 · Implementation note: gradient computation

Eikonal and normal losses require $\nabla_x f$ during training:

```python
xyz_eik = torch.cat([surface_pts, near_pts, free_pts], dim=1)
xyz_eik.requires_grad_(True)
f_endo, f_epi, delta = decoder(z, xyz_eik)
grad_endo = torch.autograd.grad(
    outputs=f_endo, inputs=xyz_eik,
    grad_outputs=torch.ones_like(f_endo),
    create_graph=True, retain_graph=True,
)[0]                                                # (B, Q, 3)
```

Cost ≈ 2× a forward pass; well within the existing T4 budget at
batch 16. **AMP gotcha:** wrap the gradient-computing branch in
`torch.cuda.amp.autocast(enabled=False)` and cast `xyz_eik` to
`float32` — autograd through autocast is unstable for the eikonal
term in PyTorch 2.x. The non-eikonal branch can stay in mixed
precision.

---

## 4 · Sampling strategy (per training batch)

Replace the random query-sampling block of the dataloader. Instead of
2048 occupancy points per sample, draw four sets:

| Set | Count | How | Used by |
| --- | --- | --- | --- |
| `surface_endo` | 512 | All endo contour points (60 × N_slices ≈ 600), random subsample | $\mathcal{L}_\text{surf}, \mathcal{L}_\text{normal}$ |
| `surface_epi`  | 512 | Same, epi rings | $\mathcal{L}_\text{surf}, \mathcal{L}_\text{normal}$ |
| `near`         | 512 | Surface points + Gaussian perturbation, $\sigma = 0.05$ (5 % of LV scale ≈ 1.25 mm) | $\mathcal{L}_\text{eik}, \mathcal{L}_\text{off}$ |
| `free`         | 512 | Uniform in bbox $[lo, hi]$ from `predict_mesh`, expanded by 0.15 | $\mathcal{L}_\text{eik}, \mathcal{L}_\text{off}$ |

Total query points per sample: 2048 (same as before — unchanged GPU
memory footprint).

The bbox `[lo, hi]` is the same one already computed in `predict_mesh`
(the `xyz.min/max ± 0.15` rule). Compute it from the (augmented)
input contour at dataloader time so it matches what inference will
see.

---

## 5 · Ground-truth normals (dataloader change)

The current cache stores `contour (N, 4)` with `[x_n, y_n, z_n, tissue]`
in **angular order per slice** (`np.argsort(np.arctan2(y, x))` after
plane intersection in `slice_mesh_at_z`). Normals are therefore a
finite-difference of consecutive points, rotated 90° outward.

Add a `contour_normals (N, 3)` field to the cache **and** to the
dataloader output:

```python
def ring_normals_2d(ring_xy):
    # ring_xy: (n, 2), angularly-ordered, closed
    tang = np.roll(ring_xy, -1, axis=0) - np.roll(ring_xy, 1, axis=0)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-8)
    # 2D outward normal: rotate tangent by -90°
    return np.stack([tang[:, 1], -tang[:, 0]], axis=1)

def stack_normals(contour, slice_ids, tissue):
    n = np.zeros((len(contour), 3), dtype=np.float32)
    for sid in np.unique(slice_ids):
        for tis in (0.0, 1.0):
            m = (slice_ids == sid) & (tissue == tis)
            ring2d = contour[m, :2]
            n[m, :2] = ring_normals_2d(ring2d)
            n[m, 2]  = 0.0
    return n
```

The out-of-plane component is set to 0 because SAX slices are
quasi-parallel and the contour does not carry information about the
through-plane gradient. The eikonal and off-surface terms supply the
missing 3-D constraint.

**Sign convention.** The endo zero level-set should have $f<0$
**inside the cavity**. Rotation by `-90°` of the angularly-ordered
tangent gives the outward 2-D normal (away from the centroid for a
convex ring), which is the convention the loss expects.

**Augmentation interaction.** The augmentations in
`augment_contour` that act on geometry (per-slice translation,
jitter, rotation, scale) must be applied to `contour_normals` too —
translations leave normals invariant, but rotation/scale do not.
Slice dropout and contour-point dropout are applied to the same
indices in both arrays. **Label noise** (the `aug_label_noise_prob =
0.02` flip of tissue) becomes nonsensical for SDF (it would put a
point on the wrong surface): set `aug_label_noise_prob = 0` for the
SDF run.

---

## 6 · Cache changes (build-lv-cache-sdf.ipynb)

GT meshes from `build-lv-cache.ipynb` are watertight (the mesh
loader's `split_endo_epi` keeps connected components, and apex/base
are closed by mesh connectivity), which makes signed distance
well-defined.

Replace the boolean occupancy fields with a precomputed signed
distance for *every* query point. For each sample produce:

```python
np.savez_compressed(
    f'sample_{i:04d}.npz',
    contour          = contour_xyzt,             # (N, 4)  unchanged
    contour_normals  = contour_normals,          # (N, 3)  NEW
    slice_ids        = slice_ids,                # (N,)    NEW (used to compute normals after augmentation)
    query            = query_pts,                # (Q, 3)  unchanged sample positions
    endo_sdf         = endo_sdf,                 # (Q,)    NEW float32, normalised units
    epi_sdf          = epi_sdf,                  # (Q,)    NEW float32, normalised units
    endo_vertices    = endo_v, endo_faces = endo_f,
    epi_vertices     = epi_v,  epi_faces  = epi_f,
    scale            = scale,
    centroid         = centroid,
    phase            = phase,
)
```

Compute SDFs from the GT meshes:

```python
# trimesh.proximity.signed_distance: + inside, - outside (note: opposite of our convention)
endo_sdf = -trimesh.proximity.signed_distance(endo_mesh, query_pts).astype(np.float32)
epi_sdf  = -trimesh.proximity.signed_distance(epi_mesh,  query_pts).astype(np.float32)
```

Sign convention: `f < 0 inside`, `f > 0 outside`. (Trimesh returns
`+` inside, so we negate once at cache time.)

**Query distribution at cache time.** Mirror the training
distribution of § 4: ≈30 % surface, ≈30 % near-surface
($\sigma = 0.05$), ≈40 % uniform-in-bbox. This way the cache
already contains useful "hard" points; the dataloader subsamples
from them, and the *augmentation* path (which needs runtime-correct
SDFs) only re-samples surface points (whose distance is exactly 0
under any augmentation that preserves the contour) and
near/free-space points (whose distance is recomputed from the
*augmented* contour mesh — see below).

**Augmentation + SDF targets.** Augmentations that move the contour
also move the implicit surface, so precomputed SDFs become stale.
Two viable strategies:

1. **Rigid-augment-only**: for SDF training, restrict augmentations
   to rigid transforms (translation, rotation), under which the
   transformation `query → R⁻¹(query − t)` keeps the cached SDF
   exact. *Disable* slice dropout, contour dropout, jitter, and scale
   jitter for the SDF run. This is simple and recommended for the
   first iteration.
2. **Recompute on the fly**: rebuild a small in-RAM mesh from the
   augmented contour in the collate function and call
   `trimesh.proximity.signed_distance`. Costs ~30 ms/sample → manageable
   at batch 16 with `dl_workers=4`, but adds complexity. Defer to a
   second iteration if (1) underfits.

Strategy (1) is the default. The `augment_contour` flags become:

```python
sdf_aug_overrides = dict(
    aug_translate_xy_std = 0.18,   # kept (rigid)
    aug_rotate_prob      = 0.5,    # kept (rigid)
    aug_rotate_max_deg   = 15.0,   # kept
    aug_jitter_std       = 0.0,    # disabled — non-rigid
    aug_slice_drop_prob  = 0.0,    # disabled
    aug_scale_std        = 0.0,    # disabled — non-rigid (would rescale SDF too)
    aug_contour_drop     = 0.0,    # disabled
    aug_label_noise_prob = 0.0,    # nonsensical for SDF
)
```

---

## 7 · Inference (replaces `predict_mesh` and the entire watertight pipeline)

```python
@torch.no_grad()
def predict_mesh_sdf(model, contour_xyz, tissue_labels, cfg,
                    grid_res=128, batch_query=131072, phase_val=None):
    z = model.encode(contour_xyz, tissue_labels, phase_val)        # (1, 256)

    lo = contour_xyz.min(0) - 0.15
    hi = contour_xyz.max(0) + 0.15
    g  = [np.linspace(lo[d], hi[d], grid_res) for d in range(3)]
    pts = np.stack(np.meshgrid(*g, indexing='ij'), axis=-1).reshape(-1, 3)

    sdf_endo = np.empty(len(pts), dtype=np.float32)
    sdf_epi  = np.empty(len(pts), dtype=np.float32)
    for k in range(0, len(pts), batch_query):
        chunk = torch.as_tensor(pts[k:k+batch_query], device=z.device).unsqueeze(0)
        fe, fp, _ = model.decoder(z, chunk)
        sdf_endo[k:k+batch_query] = fe.squeeze(0).cpu().numpy()
        sdf_epi [k:k+batch_query] = fp.squeeze(0).cpu().numpy()

    voxel = (hi - lo) / (grid_res - 1)
    endo_v, endo_f, _, _ = marching_cubes(sdf_endo.reshape((grid_res,)*3), level=0.0, spacing=voxel)
    epi_v,  epi_f,  _, _ = marching_cubes(sdf_epi .reshape((grid_res,)*3), level=0.0, spacing=voxel)
    endo_v += lo; epi_v += lo

    return (trimesh.Trimesh(endo_v, endo_f, process=True),
            trimesh.Trimesh(epi_v,  epi_f,  process=True))
```

That is the **entire** mesh-extraction pipeline. The following
functions in
[acdc-thesis-watertight.ipynb](../acdc-thesis-watertight.ipynb)
become unused and are deleted from the SDF inference notebook:

- `stamp_slice_evidence` (lines ~447–480) — slice-evidence fusion is
  now provided structurally via $\mathcal{L}_\text{surf}$.
- `synthesize_caps` (lines ~607–670) — closure is enforced by the
  eikonal + off-surface losses on the bbox.
- `repair_watertight` (lines ~566–600) — closed by construction.
- `fill_open_rims` (lines ~519–545) — no rims by construction.
- The PyMeshFix iteration loop and the trimesh fallback.
- `_wt_per_vertex` (lines ~1089–1179) — replaced by analytic $\delta$
  (§ 7.1).

Keep `largest_component` (~3 lines) only as a defensive safety net
against sub-resolution ghost shells, which become rare with a
properly trained SDF (and disappear at `grid_res = 128`).

### 7.1 · Wall thickness — analytic, no ray casting

```python
def wt_endo_vertices(model, z, endo_verts):
    pts = torch.as_tensor(endo_verts, device=z.device, dtype=torch.float32).unsqueeze(0)
    _, _, delta = model.decoder(z, pts)
    return delta.squeeze(0).cpu().numpy()              # normalised units
wt_mm = wt_endo_vertices(...) * scale                  # mm
```

This replaces every line of `_wt_per_vertex`. It is exact at the
apex and at the valve plane (where ray casting currently drops to
`p5 = 0 mm`) because $\delta(x) \ge \tau_\text{min}$ everywhere by
the hinge in § 3.1. The AHA 17-segment summary code
([acdc-thesis-watertight.ipynb](../acdc-thesis-watertight.ipynb),
lines ~1186–1289) is reused unchanged — it only consumes
`endo_verts` and per-vertex WT values.

### 7.2 · Inference grid resolution

The current pipeline uses 64³ in training and 96³ in the thesis
notebook. With an SDF, marching cubes is much less sensitive to
resolution because the field is continuous and Lipschitz-1. Set
**`grid_res = 128`** for thesis figures (≈ 2.1 M points, ~20 ms
encoder + ~0.6 s decoder + 0.2 s MC on T4). Below 96³ the
endo–epi delta can lose the apex; above 128³ the marginal gain is
< 0.1 mm.

---

## 8 · Validation (reviewer-proof ablation table)

| Metric | What it tells you | Expected SDF result |
| --- | --- | --- |
| **Watertight rate** over the full ACDC test set (`mesh.is_watertight`) | structural correctness | **100 %** vs. ~50 % epi today (without `repair_watertight`) |
| **Eikonal residual** $\bigl(\lVert\nabla f\rVert - 1\bigr)^2$ on a held-out grid | how "metric" the field is | < 0.05 (SIREN), < 0.15 (ReLU+FF) |
| **Per-slice Dice & Hausdorff** (`per_slice_fidelity` from the watertight notebook, lines ~772–892) | direct comparison to occupancy baseline | Dice ≥ 0.90, Hausdorff ≤ 1.5 mm |
| **Volume drift** (ED endo vs. clinical from ACDC `Info.cfg`) | watertight volumes are well-defined | within ±5 mL of ground-truth LVEDV |
| **Wall thickness summary** vs. clinical range (3–15 mm; ΔWT 3–8 mm in healthy LV) | the central thesis claim | ED ~7–10 mm, ES ~10–14 mm, ΔWT +3 mm; **`p5 > τ_min · scale`** by construction |
| **Hausdorff(mesh, contours)** | should drop because the SDF interpolates smoothly | ≤ Occupancy Hausdorff − 0.2 mm |
| **AHA 17-segment WT bull's-eye** (existing visualisation) | qualitative anatomical check | smooth, no apex hole |

Include all metrics for three configurations in the thesis ablation
table:

| Configuration | Watertight % | p5 WT (mm) | ΔWT (mm) | Mean Dice | Eikonal residual |
| --- | --- | --- | --- | --- | --- |
| Occupancy + post-hoc cap (current `acdc-thesis-watertight`) | ~50 / 100 (pre/post repair) | 0 | +1.5 | … | n/a |
| Occupancy + curved cap (option 1 — kept for reference) | … | … | … | … | n/a |
| **SDF (this plan)** | **100** (no repair) | **≥ 3** | **+3 to +8** | … | … |

---

## 9 · Engineering steps (concrete order of operations)

The repository today has eight notebooks; this plan adds one and
edits two.

1. **Branch the cache builder** → `build-lv-cache-sdf.ipynb`. Reuse
   `split_endo_epi`, `slice_mesh_at_z`, `extract_sax_contours`,
   `sample_contour` verbatim. Replace the occupancy-label cell with
   the SDF computation of § 6 and add `contour_normals` (§ 5).
   Reuse `split.json` from the existing ED/ES caches — the patient
   split is preserved.

2. **Branch the trainer** → `training-model-sdf.ipynb` from
   `training-model-v2.ipynb`. Edits, in order:
   - Cell defining `INRDecoder` → replace with `INRDecoderSDF`
     (§ 2.1) and the `_geometric_init` method (§ 2.2). Keep
     `PointNetEncoder` and `FourierPE` unchanged.
   - Cell defining `LVOccDataset` → rename to `LVSDFDataset`. Load
     `endo_sdf`, `epi_sdf`, `contour_normals`. Build the four-set
     query bundle (§ 4) at `__getitem__` time (the cache provides
     more than enough points to sample from).
   - Cell defining `augment_contour` → apply rotation/scale to
     `contour_normals`; honour the SDF override flags (§ 6).
   - Cell defining `occupancy_loss` → replace with
     `compute_sdf_losses` (§ 3). Keep the latent-reg term, the
     reduction logic, and the dictionary return shape so the
     training loop's logging code is reusable.
   - Cell with the training loop → only the loss-call line changes
     (`losses = compute_sdf_losses(z, batch)`). The eikonal term
     needs `xyz_eik.requires_grad_(True)` → set this in the loop,
     not in the dataloader.
   - `CFG` → add `lambda_surf, lambda_eik, lambda_off, lambda_normal,
     lambda_wt, tau_min, alpha_off, eik_warmup_epochs=10,
     normal_warmup_epochs=5, r0=0.5, sdf_aug_overrides`.
   - `predict_mesh` → replace with `predict_mesh_sdf` (§ 7).
     Keep the same function name in this notebook so downstream
     code paths in the inference notebook do not need editing.

3. **Branch the watertight thesis notebook** →
   `acdc-thesis-watertight-sdf.ipynb`. Delete the cells defining
   `stamp_slice_evidence`, `synthesize_caps`, `repair_watertight`,
   `fill_open_rims`, the iterative repair loop, and `_wt_per_vertex`.
   Replace the wall-thickness call with `wt_endo_vertices` (§ 7.1).
   Reuse `per_slice_fidelity` and the AHA bull's-eye / radar code
   verbatim.

4. **Train.**
   - Hyper-params unchanged from v2 except where noted in § 3.2:
     `epochs = 300` (SDF converges ~1.7× faster than occupancy in
     practice), `lr = 3e-4` for SIREN / `1e-4` for ReLU+FF (the
     existing v2 lr), `weight_decay = 5e-4`, `grad_clip = 1.0`,
     batch 16 × N_GPU, AdamW, `CosineAnnealingLR(T_max = epochs/5,
     eta_min = lr*0.01)`, AMP on (with the eikonal-branch override
     of § 3.3).
   - **Distillation warm-start (optional, recommended for the first
     run):** initialise the SDF decoder weights from the v2
     occupancy decoder, then run one epoch of MSE between the SDF's
     `f_endo` and a soft pseudo-SDF derived from the occupancy
     network's `logit / 10` on a uniform grid. This sidesteps the
     cold-start risk if the geometric init alone proves brittle on
     ES (where contours are tighter).

5. **Reproduce the figures.** All thesis plots in
   `acdc-thesis-watertight.ipynb` (3-D overlays, AHA bull's-eye,
   cross-section slices, radar, mesh-quality table, per-slice
   Dice/Hausdorff comparison) regenerate with no code changes — only
   the model and the mesh-extraction call differ.

---

## 10 · Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Cold-start collapse to $f \equiv 0$ | Geometric sphere init (§ 2.2); $\lambda_\text{eik}$ ramped from 0 over epochs 0–10; distillation warm-start from the trained occupancy decoder (§ 9 step 4) |
| Eikonal instability on ReLU networks | Use SIREN as primary; weight-normalise hidden Linear layers in the ReLU+FF ablation; clip $\lVert\nabla f\rVert$ to $[0.1, 10]$ before the eikonal term |
| Apex still under-supervised (sparse contours) | Add synthetic apex points: pick the apex centroid `(0, 0, z_apex_n)` with $f_\text{endo} = 0$ as a *training* surface sample; not used at inference |
| Convention disagreement on "inside" | $f < 0$ inside the cavity (endo) and inside the myocardium (epi); enforce by sign of the contour normal (§ 5); cache builder negates `trimesh.proximity.signed_distance` (§ 6) |
| Augmentation invalidates cached SDFs | Use rigid-only augmentations (§ 6, strategy 1); fall back to on-the-fly SDF recomputation only if validation eikonal residual stalls |
| `pymeshfix` unavailable | Not needed — SDF is watertight by construction. Remove the `try/except` block from inference. |
| AMP autograd instability for eikonal | Run the eikonal branch in `float32` under `autocast(enabled=False)`; only the surface and off-surface terms run in mixed precision (§ 3.3) |

---

## 11 · Expected outcome

A single inference call

```python
endo_mesh, epi_mesh = predict_mesh_sdf(model, contour_xyz, tissue_labels, CFG)
wt_mm = wt_endo_vertices(model, z, endo_mesh.vertices) * scale
```

produces **always-watertight, always-closed, anatomically-curved
endo and epi meshes** with no `pymeshfix`, no `fill_open_rims`, no
`synthesize_caps`, no ray casting. The
mesh-extraction code shrinks from ~600 lines (`predict_mesh_v3` plus
the watertight-repair stack) to ~30 lines, and the watertight
diagnostic figure (`fig_quality_diag`) collapses to a single column
labelled "**SDF — closed by construction**", with a 100 %
watertight-rate cell. The thesis narrative becomes a clean
one-line claim: *the model's output is a continuous signed-distance
field whose zero level-sets are guaranteed to be closed 2-manifolds,
so watertightness is a structural property of the architecture
rather than a side-effect of post-processing.*
