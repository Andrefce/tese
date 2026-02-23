"""
train_phase1.py
Train the Phase 1 Denoising GNN.
Run: python train_phase1.py
"""

import math
import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as PyGDataLoader

from config import CFG
from data.datasets import DenoisingCardiacDataset
from data.helpers import load_ssm, sample_shape
from models.phase1 import CardiacGNN, p1_total_loss

# ── Setup ──────────────────────────────────────────────────────────────────

torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count() if DEVICE.type == 'cuda' else 0
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
print(f'Device: {DEVICE}  |  GPUs: {NUM_GPUS}')

# ── SSM ────────────────────────────────────────────────────────────────────

MEAN_VERTS, FACES, PCS, VARIANCES, SCALED_PCS = load_ssm(CFG['ssm_dir'])

# ── Dataset ────────────────────────────────────────────────────────────────

from pathlib import Path

graph_dir   = CFG['graph_dir']
graph_files = sorted(Path(graph_dir).glob('graph_*.npz'))
assert len(graph_files) > 0, f'No graph files in {graph_dir}'
print(f'Found {len(graph_files)} graphs')

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
    DenoisingCardiacDataset(graph_dir, idx_train),
    batch_size=CFG['batch_size'], shuffle=True,
    num_workers=2, pin_memory=pin, persistent_workers=pin,
)
val_loader = PyGDataLoader(
    DenoisingCardiacDataset(graph_dir, idx_val),
    batch_size=CFG['batch_size'], shuffle=False,
    num_workers=2, pin_memory=pin, persistent_workers=pin,
)

# ── Model ──────────────────────────────────────────────────────────────────

model = CardiacGNN(
    node_features = CFG['p1_node_features'],
    edge_features = CFG['edge_features'],
    hidden_dim    = CFG['hidden_dim'],
    deform_cap    = CFG['deform_cap'],
    k_dynamic     = CFG['k_dynamic'],
).to(DEVICE)
print(f'Phase 1 CardiacGNN  |  params: {model.num_params():,}')

# ── Training ───────────────────────────────────────────────────────────────

def _lr_lambda(epoch):
    progress = epoch / max(CFG['epochs'] - 1, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))

optimizer  = torch.optim.Adam(model.parameters(), lr=CFG['p1_lr'],
                               weight_decay=CFG['p1_weight_decay'])
scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
noise_sigma_t = torch.tensor(CFG['noise_sigma'], device=DEVICE)

P1_CKPT = CFG['phase1_ckpt']
os.makedirs(os.path.dirname(P1_CKPT), exist_ok=True)

best_val, no_improve, best_epoch = float('inf'), 0, 0

print(f'Phase 1 — σ={CFG["noise_sigma"]}mm  |  epochs={CFG["epochs"]}')
print('─' * 65)
print(f'{"Epoch":>5}  {"Train MSE":>10}  {"Val MSE":>10}  {"Smooth":>8}  {"LR":>8}')
print('─' * 65)

t0 = time.time()
for epoch in range(CFG['epochs']):
    model.train()
    tr_mse_acc = torch.tensor(0.0, device=DEVICE)
    tr_sm_acc  = torch.tensor(0.0, device=DEVICE)

    for batch in train_loader:
        batch          = batch.to(DEVICE, non_blocking=True)
        noisy_x        = batch.x.clone()
        noisy_x[:, :3] += torch.randn_like(noisy_x[:, :3]) * noise_sigma_t
        batch.x        = noisy_x

        optimizer.zero_grad(set_to_none=True)
        pred             = model(batch, epoch=epoch)
        loss, lm, ls     = p1_total_loss(pred, batch.y, batch.edge_index, CFG['smoothness_w'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_mse_acc += lm.detach()
        tr_sm_acc  += ls.detach()

    n_tr   = len(train_loader)
    tr_mse = (tr_mse_acc / n_tr).item()
    tr_sm  = (tr_sm_acc  / n_tr).item()

    model.eval()
    vl_mse_acc = torch.tensor(0.0, device=DEVICE)
    vl_sm_acc  = torch.tensor(0.0, device=DEVICE)
    with torch.no_grad():
        for batch in val_loader:
            batch          = batch.to(DEVICE, non_blocking=True)
            noisy_x        = batch.x.clone()
            noisy_x[:, :3] += torch.randn_like(noisy_x[:, :3]) * noise_sigma_t
            batch.x        = noisy_x
            pred             = model(batch, epoch=epoch)
            _, lm, ls        = p1_total_loss(pred, batch.y, batch.edge_index, CFG['smoothness_w'])
            vl_mse_acc += lm
            vl_sm_acc  += ls

    n_vl   = len(val_loader)
    vl_mse = (vl_mse_acc / n_vl).item()
    vl_sm  = (vl_sm_acc  / n_vl).item()

    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print(f'{epoch:5d}  {tr_mse:10.4f}  {vl_mse:10.4f}  {vl_sm:8.4f}  {lr:8.2e}')

    if vl_mse < best_val - 1e-5:
        best_val, best_epoch, no_improve = vl_mse, epoch, 0
        torch.save({'model_state_dict': model.state_dict(),
                    'epoch': epoch, 'val_mse': best_val}, P1_CKPT)
        print(f'        ★ New best val_mse={best_val:.4f}  (saved)')
    else:
        no_improve += 1
        if no_improve >= CFG['patience']:
            print(f'\nEarly stopping at epoch {epoch}')
            break

print(f'\n✅ Phase 1 done in {time.time()-t0:.0f}s  '
      f'|  best epoch={best_epoch}  val_mse={best_val:.4f}')
