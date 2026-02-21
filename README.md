# 🫀 Cardiac Shape GNN

> **Thesis Project** — Learning Left Ventricular Shape Representations via Graph Neural Networks trained on a Statistical Shape Model (SSM) and evaluated on the ACDC Clinical Dataset.

---

## 📌 Overview

This repository implements a full pipeline for **cardiac left ventricle (LV) shape analysis** using Graph Neural Networks (GNNs). The approach:

1. Loads the **UK Digital Heart Statistical Shape Model** (SSM) — a VTK mesh + 100 PCA modes describing population-level LV shape variation.
2. **Generates a synthetic dataset** of biomechanically plausible LV shapes by sampling PCA weight vectors from a clipped Normal distribution.
3. **Slices each shape** along the Z-axis (simulating MRI cross-sections) and extracts endocardium/epicardium contours.
4. **Constructs graph representations** where nodes are contour sample points and edges encode local (k-NN within slices) and inter-slice connectivity.
5. Trains a **CardiacGNN** model — a deformation-based GCN with a global context branch — to reconstruct 3D node positions from learned embeddings.
6. **Evaluates on real clinical data** from the [ACDC Challenge dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/) using NIfTI segmentation masks.

---

## 🏗️ Architecture

```
cardiac-shape-gnn/
│
├── configs/
│   └── default.yaml          # All hyperparameters & paths
│
├── src/
│   ├── data/
│   │   ├── ssm_loader.py     # Load VTK mesh + PCA modes, sample shapes
│   │   ├── graph_builder.py  # Slice → contours → PyG graph
│   │   ├── dataset.py        # PyTorch Geometric Dataset & DataLoader
│   │   └── acdc_loader.py    # Load ACDC NIfTI, extract LV surfaces
│   │
│   ├── models/
│   │   └── cardiac_gnn.py    # CardiacGNN: GCN encoder + global context + deformation decoder
│   │
│   ├── training/
│   │   ├── trainer.py        # Training loop with checkpointing & early stopping
│   │   └── losses.py         # MSE reconstruction + Laplacian smoothness loss
│   │
│   └── utils/
│       ├── mesh_utils.py     # VTK/trimesh helpers
│       └── visualization.py  # 2D slices, 3D mesh, graph, prediction plots
│
├── scripts/
│   ├── generate_dataset.py   # End-to-end: SSM → graphs → dataset on disk
│   ├── train.py              # Train CardiacGNN from config
│   └── evaluate_acdc.py      # Inference + metrics on ACDC patients
│
├── notebooks/
│   └── demo.ipynb            # Interactive walkthrough (Colab-ready)
│
├── requirements.txt
└── setup.py
```

---

## ⚙️ Model: CardiacGNN

The model is a **deformation network** — it starts from a template mesh and predicts per-node 3D displacement vectors:

```
Input Graph (N nodes × 5 features)
  [x, y, z, radial_dist, tissue_type]
         │
    ┌────▼────┐
    │ GCNConv │  ← Local neighbourhood message passing
    │  + BN   │
    └────┬────┘
         │
    ┌────▼──────┐
    │ GCNConv   │
    │  + BN     │
    └────┬──────┘
         │          ┌──────────────────┐
         ├──────────► Global Mean Pool  │
         │          │ + Linear + LReLU │
         │          └────────┬─────────┘
         │                   │  (broadcast back to all nodes)
    ┌────▼──────────────────▼──┐
    │   Concatenate [local, global]    │
    └───────────┬──────────────────────┘
                │
    ┌───────────▼──────────┐
    │ GCNConv + BN         │
    └───────────┬──────────┘
                │
    ┌───────────▼──────────┐
    │  Linear → tanh × 20  │  ← Displacement capped at ±20 mm
    └───────────┬──────────┘
                │
  Output: template_pos + deformation  (N × 3)
```

**Key design choices:**
- Zero-initialized final layer → model starts as identity (predicts zero deformation), stabilizing early training.
- `tanh × 20` output activation → prevents exploding point clouds.
- Global context branch → enables the model to adapt local predictions based on overall shape context.

---

## 📊 Loss Function

```
L_total = L_MSE + λ · L_smooth

L_MSE    = MSE(predicted_positions, target_positions)
L_smooth = mean‖node_pos − mean_neighbour_pos‖  (Laplacian)
λ = 0.05
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/cardiac-shape-gnn.git
cd cardiac-shape-gnn
pip install -e .
```

### 2. Download the SSM

```bash
git clone https://github.com/UK-Digital-Heart-Project/Statistical-Shape-Model.git
```

Point `ssm_dir` in `configs/default.yaml` to the cloned folder.

### 3. Generate the training dataset

```bash
python scripts/generate_dataset.py --config configs/default.yaml
```

This creates `data/heart_dataset_gnn/` with:
- `meshes/`   — 500 STL heart meshes
- `slices/`   — JSON contour files (simulated MRI slices)
- `graphs/`   — NPZ graph files ready for PyG
- `dataset_metadata.csv`

### 4. Train the GNN

```bash
python scripts/train.py --config configs/default.yaml
```

Checkpoints and logs are saved to `outputs/checkpoints/` and `outputs/logs/`.

### 5. Evaluate on ACDC

Place ACDC patient files (`.nii.gz`) in `data/acdc/` and run:

```bash
python scripts/evaluate_acdc.py \
    --config configs/default.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --patient_dir data/acdc/
```

---

## 📦 Data Format

### Graph Node Features (N × 5)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `x` | X coordinate (mm) |
| 1 | `y` | Y coordinate (mm) |
| 2 | `z` | Z coordinate (mm) |
| 3 | `r` | Radial distance from slice centroid |
| 4 | `t` | Tissue type: `0` = endocardium, `1` = epicardium |

### Graph Edges

Bidirectional. Built from:
- **8-NN within each Z-slice** (local XY connectivity)
- **3-NN between adjacent Z-slices** (inter-level continuity)

---

## 🩻 ACDC Integration

The ACDC evaluation pipeline:
1. Loads NIfTI image + ground-truth segmentation (`.nii.gz`)
2. Applies affine transform to convert voxel → world coordinates (mm)
3. Extracts LV endocardium (label 2) and epicardium (labels 2+3) via marching cubes
4. Resamples contours to a fixed number of points per slice for consistent graph size
5. Builds a PyG graph and runs inference with the trained model
6. Reports volume, surface area, wall thickness predictions vs ground truth

---

## 📈 Metrics

| Metric | Description |
|--------|-------------|
| MSE | Mean squared error on node positions (mm²) |
| Smoothness | Laplacian loss on predicted mesh |
| Dice (ACDC) | Overlap with ground truth segmentation |
| HD95 (ACDC) | 95th percentile Hausdorff distance |

---

## 🔧 Configuration

All hyperparameters live in `configs/default.yaml`:

```yaml
# Data
ssm_dir: "Statistical-Shape-Model"
data_dir: "data/heart_dataset_gnn"
num_samples: 500
num_pca_modes: 10
num_slices: 20
points_per_contour: 80
knn_intra_slice: 8
knn_inter_slice: 3

# Model
node_features: 5
hidden_dim: 128

# Training
batch_size: 16
learning_rate: 3e-4
epochs: 100
smoothness_weight: 0.05
val_split: 0.2
seed: 42

# Outputs
checkpoint_dir: "outputs/checkpoints"
log_dir: "outputs/logs"
```

---

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | Deep learning |
| `torch-geometric` | ≥2.3 | GNN layers |
| `vtk` | ≥9.2 | VTK mesh I/O |
| `nibabel` | ≥5.0 | NIfTI I/O (ACDC) |
| `trimesh` | ≥4.0 | Mesh processing |
| `scikit-image` | ≥0.21 | Marching cubes |
| `scikit-learn` | ≥1.3 | Nearest neighbours |
| `numpy` | ≥1.24 | Numerical ops |
| `matplotlib` | ≥3.7 | Visualization |

