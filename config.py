import os
from pathlib import Path

# ── Kaggle path detection ──────────────────────────────────────────────────
_on_kaggle = os.path.exists('/kaggle/input')

GRAPH_INPUT_DIR = (
    '/kaggle/input/datasets/andrefce/heart-dataset-gnn/heart_dataset_gnn'
    if _on_kaggle else 'heart_dataset_gnn'
)
OUTPUT_DIR = '/kaggle/working' if _on_kaggle else 'heart_dataset_gnn'

CFG = dict(
    # Paths
    ssm_dir             = 'Statistical-Shape-Model',
    graph_dir           = f'{GRAPH_INPUT_DIR}/graphs',
    output_dir          = OUTPUT_DIR,
    phase1_ckpt         = f'{OUTPUT_DIR}/checkpoints/phase1_denoising.pth',
    phase2_ckpt         = f'{OUTPUT_DIR}/checkpoints/phase2_ssm_completion.pth',
    acdc_label_path     = '/kaggle/input/datasets/andrefce/ed-1-p/DCM08-OH-AL_V2_1.nii',
    acdc_visible_slices = 3,

    # Dataset
    num_samples         = 500,
    num_pca_modes       = 10,
    sigma_clip          = 3.0,
    num_slices          = 20,
    points_per_cont     = 50,
    slice_epsilon       = 2.0,
    knn_intra           = 8,
    knn_inter           = 3,

    # Phase 1 model
    p1_node_features    = 5,   # [x, y, z, radial_dist, tissue_type]
    edge_features       = 6,
    hidden_dim          = 64,
    deform_cap          = 20.0,
    k_dynamic           = 8,
    noise_sigma         = 1.5,
    noise_warmup_frac   = 0.01,

    # Phase 2 model
    p2_node_features    = 6,   # adds is_observed flag
    min_visible         = 1,
    max_visible_frac    = 0.35,

    # Training (shared)
    epochs              = 150,
    batch_size          = 4,
    val_split           = 0.2,
    patience            = 30,
    seed                = 42,

    # Phase 1 training
    p1_lr               = 1e-3,
    p1_weight_decay     = 1e-5,
    smoothness_w        = 0.01,

    # Phase 2 training
    p2_lr               = 5e-4,
    p2_weight_decay     = 1e-5,
    w_pca               = 1.0,
    w_surface           = 0.5,
    w_smooth            = 0.005,
)

# Create writable output subdirectories
for sub in ['checkpoints', 'viz']:
    Path(CFG['output_dir'], sub).mkdir(parents=True, exist_ok=True)
