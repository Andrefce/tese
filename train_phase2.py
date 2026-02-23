"""
train_phase2.py
Train the Phase 2 SSM-Constrained Completion GNN.
Phase 1 checkpoint must exist at CFG['phase1_ckpt'].
Run: python train_phase2.py
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as PyGDataLoader

from config import CFG
from data.datasets import SSMCompletionDataset
from data.helpers import load_ssm, sample_shape
from models.phase1 import CardiacGNN
from models.phase2 import (
    SSMCardiacGNN,
    build_ssm_gpu_tensors,
    reconstruct_batch,
    laplacian_smooth_loss,
    p2_total_loss,
    transfer_encoder_weights,
)

# ── Setup ──────────────────────────────────────────────────────────────────

torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count() if DEVICE.type == 'cuda' else 0
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
print(f'Device: {DEVICE}  |  GPUs: {NUM_GPUS}')

# ── SSM ────────────────────────────────────────────────────────────────────

MEAN_VERTS, FACES, PCS, VARIANCES, SCALED_PCS = load_ssm(CFG['ssm_dir'])

MEAN_T, SCALED_T, LAP_EI, LAP_DEG_INV, N_VERTS = build_ssm_gpu_tensors(
    MEAN_VERTS, SCALED_PCS, FACES, CFG['num_pca_modes'], DEVICE
)
LAP_ROW = LAP_EI[0]
LAP_COL = LAP_EI[1]

# ── Dataset ────────────────────────────────────────────────────────────────

graph_dir   = CFG['graph_dir']
graph_files = sorted(Path(graph_dir).glob('graph_*.npz'))
assert len(graph_files) > 0, f'No graph files in {graph_dir}'

rng = np.random.default_rng(CFG['seed'])
for gf in graph_files:
    raw = np.load(gf)
    if 'pca_weights' not in raw.files:
        _, weights = sample_shape(rng, PCS, SCALED_PCS, MEAN_VERTS,
                                  CFG['num_pca_modes'], CFG['sigma_clip'])
        existing = dict(raw)
        existing['pca_weights'] = weights
        np.savez_compressed(str(gf), **existing)

all_idx = list(range(len(graph_files)))
idx_train, idx_val = train_test_split(all_idx, test_size=CFG['val_split'],
                                       random_state=CFG['seed'])
print(f'Train: {len(idx_train)}  Val: {len(idx_val)}')

pin = (DEVICE.type == 'cuda')
train_loader = PyGDataLoader(
    SSMCompletionDataset(graph_dir, idx_train,
                         CFG['min_visible'], CFG['max_visible_frac']),
    batch_size=CFG['batch_size'], shuffle=True,
    num_workers=2, pin_memory=pin, persistent_workers=pin,
)
val_loader = PyGDataLoader(
    SSMCompletionDataset(graph_dir, idx_val,
                         CFG['min_visible'], CFG['max_visible_frac']),
    batch_size=CFG['batch_size'], shuffle=False,
    num_workers=2, pin_memory=pin, persistent_workers=pin,
)

# ── Models ─────────────────────────────────────────────────────────────────

# Build Phase 1 model and load checkpoint for encoder transfer
p1_model = CardiacGNN(
    node_features = CFG['p1_node_features'],
    edge_features = CFG['edge_features'],
    hidden_dim    = CFG['hidden_dim'],
    deform_cap    = CFG['deform_cap'],
    k_dynamic     = CFG['k_dynamic'],
).to(DEVICE)
ckpt1 = torch.load(CFG['phase1_ckpt'], map_location=DEVICE)
p1_model.load_state_dict(ckpt1['model_state_dict'])
print(f'Phase 1 checkpoint loaded (epoch {ckpt1["epoch"]}, val_mse={ckpt1["val_mse"]:.4f})')

p2_model = SSMCardiacGNN(
    node_features = CFG['p2_node_features'],
    edge_features = CFG['edge_features'],
    hidden_dim    = CFG['hidden_dim'],
    num_pca_modes = CFG['num_pca_modes'],
    k_dynamic     = CFG['k_dynamic'],
).to(DEVICE)
print(f'Phase 2 SSMCardiacGNN  |  params: {p2_model.num_params():,}')

transfer_encoder_weights(p1_model, p2_model)

# ── Training ───────────────────────────────────────────────────────────────

P2_CKPT = CFG['phase2_ckpt']
os.makedirs(os.path.dirname(P2_CKPT), exist_ok=True)

optimizer = torch.optim.Adam(p2_model.parameters(), lr=CFG['p2_lr'],
                              weight_decay=CFG['p2_weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG['epochs'], eta_min=1e-6
)

best_val, no_improve, best_epoch = float('inf'), 0, 0

print(f'Phase 2 — GNN output: {CFG["num_pca_modes"]} PCA weights  |  SSM mesh: {N_VERTS:,} vertices')
print('─' * 70)
print(f'{"Epoch":>5}  {"Train PCA":>10}  {"Val PCA":>10}  {"Surface":>10}  {"LR":>8}')
print('─' * 70)

t0 = time.time()
for epoch in range(CFG['epochs']):
    p2_model.train()
    tr_pca_acc  = torch.tensor(0.0, device=DEVICE)
    tr_surf_acc = torch.tensor(0.0, device=DEVICE)

    for batch in train_loader:
        batch  = batch.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred_w = p2_model(batch, epoch=epoch)
        loss, l_pca, l_surf = p2_total_loss(
            pred_w, batch.y,
            MEAN_T, SCALED_T, N_VERTS,
            LAP_ROW, LAP_COL, LAP_DEG_INV, DEVICE,
            CFG['w_surface'], CFG['w_smooth'],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(p2_model.parameters(), 1.0)
        optimizer.step()
        tr_pca_acc  += l_pca.detach()
        tr_surf_acc += l_surf.detach()

    n_tr    = len(train_loader)
    tr_pca  = (tr_pca_acc  / n_tr).item()
    tr_surf = (tr_surf_acc / n_tr).item()

    p2_model.eval()
    vl_pca_acc  = torch.tensor(0.0, device=DEVICE)
    vl_surf_acc = torch.tensor(0.0, device=DEVICE)
    with torch.no_grad():
        for batch in val_loader:
            batch  = batch.to(DEVICE, non_blocking=True)
            pred_w = p2_model(batch, epoch=epoch)
            _, l_pca, l_surf = p2_total_loss(
                pred_w, batch.y,
                MEAN_T, SCALED_T, N_VERTS,
                LAP_ROW, LAP_COL, LAP_DEG_INV, DEVICE,
                CFG['w_surface'], CFG['w_smooth'],
            )
            vl_pca_acc  += l_pca
            vl_surf_acc += l_surf

    n_vl    = len(val_loader)
    vl_pca  = (vl_pca_acc  / n_vl).item()
    vl_surf = (vl_surf_acc / n_vl).item()

    scheduler.step()
    lr = optimizer.param_groups[0]['lr']

    if epoch % 5 == 0 or epoch == 1:
        print(f'{epoch:5d}  {tr_pca:10.4f}  {vl_pca:10.4f}  {vl_surf:10.4f}  {lr:8.2e}')

    if vl_pca < best_val - 1e-5:
        best_val, best_epoch, no_improve = vl_pca, epoch, 0
        torch.save({'model_state_dict': p2_model.state_dict(),
                    'epoch': epoch, 'val_pca': best_val}, P2_CKPT)
        print(f'        ★ New best val_pca={best_val:.4f}  (saved)')
    else:
        no_improve += 1
        if no_improve >= CFG['patience']:
            print(f'\nEarly stopping at epoch {epoch}')
            break

print(f'\n✅ Phase 2 done in {time.time()-t0:.0f}s  '
      f'|  best epoch={best_epoch}  val_pca={best_val:.4f}')
