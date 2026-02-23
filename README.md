# Cardiac Shape Modelling — Phase 1 & Phase 2

## Overview

**Phase 1 — Denoising GNN:** full noisy shape → clean shape. Encoder learns LV geometry.

**Phase 2 — SSM-Constrained Completion:** sparse observed slices → GNN encoder → K PCA mode weights → full SSM mesh.

- Phase 1 checkpoint warm-starts Phase 2 encoder
- Phase 2 output always lives on the SSM manifold (anatomically plausible by construction)
- Works from as few as 1 observed slice

---

## Project Structure

```
tese/
├── config.py              # Shared CFG dict (paths, hyperparameters)
├── train_phase1.py        # Train denoising GNN (Phase 1)
├── train_phase2.py        # Train SSM completion GNN (Phase 2)
├── infer.py               # Evaluation & visualization CLI
├── requirements.txt
├── data/
│   ├── helpers.py         # SSM loading, contour extraction, graph construction
│   └── datasets.py        # DenoisingCardiacDataset, SSMCompletionDataset
└── models/
    ├── utils.py           # dynamic_knn_edges, laplacian_loss
    ├── phase1.py          # CardiacGNN + p1_total_loss
    └── phase2.py          # SSMCardiacGNN + GPU SSM reconstruction helpers
```

---

## Prerequisites

```bash
pip install -r requirements.txt
```

Clone the SSM repository:

```bash
git clone --quiet https://github.com/UK-Digital-Heart-Project/Statistical-Shape-Model.git
```

Required SSM files inside `Statistical-Shape-Model/`:
- `LV_ED_mean.vtk`
- `LV_ED_pc_100_modes.csv.gz`
- `LV_ED_var_100_modes.csv.gz`

---

## Dataset

Expects pre-generated graph `.npz` files at `CFG['graph_dir']` (default: `heart_dataset_gnn/graphs/`).

Each file `graph_NNN.npz` must contain:
- `nodes` — `(N, 5)` float32: `[x, y, z, radial_dist, tissue_type]`
- `edges` — `(E, 2)` int32
- `edge_feats` — `(E, 6)` float32
- `node_types` — `(N,)` int8
- `slice_ids` — `(N,)` int32

`pca_weights` will be added automatically on first run if missing.

---

## Training

```bash
# Phase 1 — train denoising GNN
python train_phase1.py

# Phase 2 — train SSM completion GNN (requires Phase 1 checkpoint)
python train_phase2.py
```

Checkpoints are saved to `CFG['output_dir']/checkpoints/`.

---

## Inference & Evaluation

```bash
# Evaluate Phase 1 on validation set
python infer.py --phase 1

# Evaluate Phase 2 on validation set
python infer.py --phase 2

# Run ACDC inference with Phase 2
python infer.py --phase 2 --acdc
```

Figures are saved to `CFG['output_dir']/viz/`.

---

## Configuration

Edit `config.py` to change paths and hyperparameters. Key settings:

| Key | Default | Description |
|---|---|---|
| `graph_dir` | `heart_dataset_gnn/graphs` | Location of `.npz` graph files |
| `phase1_ckpt` | `.../checkpoints/phase1_denoising.pth` | Phase 1 checkpoint |
| `phase2_ckpt` | `.../checkpoints/phase2_ssm_completion.pth` | Phase 2 checkpoint |
| `acdc_label_path` | *(set manually)* | Path to ACDC `.nii`/`.nii.gz` label file |
| `num_pca_modes` | 10 | Number of SSM modes predicted by Phase 2 |
| `noise_sigma` | 1.5 | Gaussian noise std (mm) for Phase 1 training |
| `epochs` | 150 | Max training epochs |
| `patience` | 30 | Early stopping patience |

On Kaggle, paths are auto-detected from `/kaggle/input`.
