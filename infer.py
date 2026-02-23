"""
infer.py
Inference and evaluation for Phase 1 and Phase 2.

Usage:
  # Evaluate Phase 1 on the validation set
  python infer.py --phase 1

  # Evaluate Phase 2 on the validation set
  python infer.py --phase 2

  # Run full ACDC inference with Phase 2
  python infer.py --phase 2 --acdc
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from skimage import measure
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from config import CFG
from data.datasets import DenoisingCardiacDataset, SSMCompletionDataset
from data.helpers import (
    build_graph,
    extract_slices,
    load_ssm,
    nodes_to_surface,
    sample_shape,
)
from models.phase1 import CardiacGNN, p1_total_loss
from models.phase2 import (
    SSMCardiacGNN,
    build_ssm_gpu_tensors,
    reconstruct_batch,
)

# ── Setup ──────────────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN_VERTS, FACES, PCS, VARIANCES, SCALED_PCS = load_ssm(CFG['ssm_dir'])

graph_dir   = CFG['graph_dir']
graph_files = sorted(Path(graph_dir).glob('graph_*.npz'))
all_idx     = list(range(len(graph_files)))
_, idx_val  = train_test_split(all_idx, test_size=CFG['val_split'],
                                random_state=CFG['seed'])

viz_dir = Path(CFG['output_dir']) / 'viz'
viz_dir.mkdir(parents=True, exist_ok=True)


# ── Phase 1 evaluation ────────────────────────────────────────────────────

def eval_phase1():
    model = CardiacGNN(
        node_features = CFG['p1_node_features'],
        edge_features = CFG['edge_features'],
        hidden_dim    = CFG['hidden_dim'],
        deform_cap    = CFG['deform_cap'],
        k_dynamic     = CFG['k_dynamic'],
    ).to(DEVICE)
    ckpt = torch.load(CFG['phase1_ckpt'], map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Phase 1 checkpoint: epoch={ckpt["epoch"]}  val_mse={ckpt["val_mse"]:.4f}')

    # Single sample qualitative view
    raw0    = np.load(f'{graph_dir}/graph_{idx_val[0]:03d}.npz')
    nt0, sid0 = raw0['node_types'], raw0['slice_ids']
    clean_t = torch.from_numpy(raw0['nodes']).float().to(DEVICE)
    ei0     = torch.from_numpy(raw0['edges'].T).long().to(DEVICE)
    ea0     = torch.from_numpy(raw0['edge_feats']).float().to(DEVICE)

    torch.manual_seed(0)
    noisy_t        = clean_t.clone()
    noisy_t[:, :3] += torch.randn_like(noisy_t[:, :3]) * CFG['noise_sigma']

    with torch.no_grad():
        d0     = Batch.from_data_list([Data(x=noisy_t, edge_index=ei0, edge_attr=ea0)])
        pred_t = model(d0)

    rmse       = (pred_t - clean_t[:, :3]).pow(2).mean().sqrt().item()
    noise_rmse = (noisy_t[:, :3] - clean_t[:, :3]).pow(2).mean().sqrt().item()
    print(f'Val RMSE: {rmse:.3f} mm  |  Noise baseline: {noise_rmse:.3f} mm')

    pred0  = pred_t.cpu().numpy()
    clean0 = clean_t[:, :3].cpu().numpy()
    noisy0 = noisy_t[:, :3].cpu().numpy()

    fig = plt.figure(figsize=(16, 5))
    for i, (pts, title, col) in enumerate([
        (noisy0, f'Noisy Input (σ={CFG["noise_sigma"]}mm)', 'tomato'),
        (clean0, 'Ground Truth',                             'steelblue'),
        (pred0,  f'GNN Reconstruction\nRMSE={rmse:.3f}mm',  'mediumseagreen'),
    ]):
        sv, sf = nodes_to_surface(pts, nt0, sid0)
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        if sv is not None:
            ax.plot_trisurf(sv[:, 0], sv[:, 1], sv[:, 2],
                            triangles=sf, color=col, alpha=0.7, edgecolor='none')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.view_init(20, 45)
    plt.suptitle('Phase 1 — Shape Denoising', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = viz_dir / 'phase1_result.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f'Saved → {out}')


# ── Phase 2 evaluation ────────────────────────────────────────────────────

def eval_phase2(run_acdc=False):
    MEAN_T, SCALED_T, LAP_EI, LAP_DEG_INV, N_VERTS = build_ssm_gpu_tensors(
        MEAN_VERTS, SCALED_PCS, FACES, CFG['num_pca_modes'], DEVICE
    )

    model = SSMCardiacGNN(
        node_features = CFG['p2_node_features'],
        edge_features = CFG['edge_features'],
        hidden_dim    = CFG['hidden_dim'],
        num_pca_modes = CFG['num_pca_modes'],
        k_dynamic     = CFG['k_dynamic'],
    ).to(DEVICE)
    ckpt = torch.load(CFG['phase2_ckpt'], map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Phase 2 checkpoint: epoch={ckpt["epoch"]}  val_pca={ckpt["val_pca"]:.4f}')

    # RMSE by visibility
    pin        = DEVICE.type == 'cuda'
    val_loader = PyGDataLoader(
        SSMCompletionDataset(graph_dir, idx_val,
                             CFG['min_visible'], CFG['max_visible_frac']),
        batch_size=CFG['batch_size'], shuffle=False,
        num_workers=2, pin_memory=pin, persistent_workers=pin,
    )

    thresholds = torch.tensor([1, 2, 3, 5, 10], device=DEVICE)
    rmse_sum   = torch.zeros(len(thresholds), device=DEVICE)
    rmse_cnt   = torch.zeros(len(thresholds), device=DEVICE)

    with torch.no_grad():
        for batch in val_loader:
            batch    = batch.to(DEVICE, non_blocking=True)
            pred_w   = model(batch, epoch=0)
            pred_v   = reconstruct_batch(pred_w,   MEAN_T, SCALED_T, N_VERTS, DEVICE)
            target_v = reconstruct_batch(batch.y,  MEAN_T, SCALED_T, N_VERTS, DEVICE)
            rmse_b   = (pred_v - target_v).pow(2).mean(dim=[1, 2]).sqrt()
            n_vis_b  = batch.n_visible
            for ti, thr in enumerate(thresholds):
                mask          = n_vis_b <= thr
                rmse_sum[ti] += rmse_b[mask].sum()
                rmse_cnt[ti] += mask.sum()

    print('\nSurface RMSE by number of visible slices:')
    print(f'  {"N slices":>10}  {"Mean RMSE (mm)":>15}  {"N samples":>10}')
    for ti, thr in enumerate(thresholds.tolist()):
        cnt = int(rmse_cnt[ti].item())
        if cnt > 0:
            mean_rmse = (rmse_sum[ti] / rmse_cnt[ti]).item()
            print(f'  {int(thr):>10}  {mean_rmse:>15.3f}  {cnt:>10}')

    # Reconstruction from N slices
    graph_path   = f'{graph_dir}/graph_{idx_val[0]:03d}.npz'
    slice_counts = [1, 2, 3, 5]
    fig          = plt.figure(figsize=(20, 5))
    for col_i, n_vis in enumerate(slice_counts):
        pred_v, gt_v, obs, nt, sid, rmse = _infer_graph(
            graph_path, n_vis, model, MEAN_T, SCALED_T, N_VERTS
        )
        ax = fig.add_subplot(1, len(slice_counts) + 1, col_i + 1, projection='3d')
        ax.plot_trisurf(pred_v[:, 0], pred_v[:, 1], pred_v[:, 2],
                        triangles=FACES[::4], color='dodgerblue', alpha=0.25,
                        edgecolor='steelblue', linewidth=0.15)
        obs_pts = np.load(graph_path)['nodes'][:, :3][obs]
        ax.scatter(obs_pts[:, 0], obs_pts[:, 1], obs_pts[:, 2],
                   c='gold', s=15, zorder=5)
        ax.set_title(f'{n_vis} slice{"s" if n_vis > 1 else ""}\nRMSE={rmse:.2f}mm',
                     fontweight='bold', fontsize=11)
        ax.view_init(20, 45)

    ax_gt = fig.add_subplot(1, len(slice_counts) + 1, len(slice_counts) + 1,
                             projection='3d')
    ax_gt.plot_trisurf(gt_v[:, 0], gt_v[:, 1], gt_v[:, 2], triangles=FACES[::4],
                       color='lightcoral', alpha=0.4, edgecolor='darkred', linewidth=0.15)
    ax_gt.set_title('Ground Truth\n(SSM mesh)', fontweight='bold', fontsize=11)
    ax_gt.view_init(20, 45)
    plt.suptitle('SSM Reconstruction from N Visible Slices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = viz_dir / 'phase2_reconstruction.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f'Saved → {out}')

    if run_acdc:
        _run_acdc(model, MEAN_T, SCALED_T, N_VERTS)


def _infer_graph(graph_npz, n_visible, model, mean_t, scaled_t, n_verts):
    raw     = np.load(graph_npz)
    sid_arr = raw['slice_ids']
    num_sl  = int(sid_arr.max()) + 1
    n_vis   = min(n_visible, num_sl)
    vis_ids = np.round(np.linspace(0, num_sl - 1, n_vis)).astype(int)
    obs_np  = np.isin(sid_arr, vis_ids)

    nodes    = torch.from_numpy(raw['nodes']).float().to(DEVICE)
    ei       = torch.from_numpy(raw['edges'].T).long().to(DEVICE)
    ea       = torch.from_numpy(raw['edge_feats']).float().to(DEVICE)
    obs_mask = torch.from_numpy(obs_np).to(DEVICE)
    gt_w     = torch.from_numpy(raw['pca_weights']).float().to(DEVICE)

    x_in = torch.cat([nodes, obs_mask.float().unsqueeze(1)], dim=1)
    d    = Batch.from_data_list([Data(x=x_in, edge_index=ei, edge_attr=ea)]).to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred_w = model(d, epoch=0)
        pred_v = reconstruct_batch(pred_w, mean_t, scaled_t, n_verts, DEVICE)[0]
        gt_v   = reconstruct_batch(gt_w.unsqueeze(0), mean_t, scaled_t, n_verts, DEVICE)[0]
        rmse   = (pred_v - gt_v).pow(2).mean().sqrt().item()

    return pred_v.cpu().numpy(), gt_v.cpu().numpy(), obs_np, raw['node_types'], sid_arr, rmse


# ── ACDC inference ────────────────────────────────────────────────────────

def _load_acdc_surfaces(label_path):
    nii    = nib.load(str(label_path))
    data   = nii.get_fdata()
    affine = nii.affine

    def surface_mm(mask):
        if not np.any(mask):
            return None, None
        v, f, _, _ = measure.marching_cubes(mask.astype(float), level=0.5)
        v_mm = (np.c_[v, np.ones(len(v))] @ affine.T)[:, :3].astype(np.float32)
        return v_mm, f.astype(np.int64)

    return (*surface_mm((data == 3)),
            *surface_mm((data == 2) | (data == 3)),
            data, affine)


def _acdc_to_graph(label_path):
    v_endo, f_endo, v_epi, f_epi, data, affine = _load_acdc_surfaces(label_path)
    if v_endo is None or v_epi is None:
        raise ValueError('Missing LV labels')
    slices_epi  = extract_slices(v_epi,  CFG['num_slices'], CFG['points_per_cont'],
                                 CFG['slice_epsilon'])
    slices_endo = extract_slices(v_endo, CFG['num_slices'], CFG['points_per_cont'],
                                 CFG['slice_epsilon'])
    slices = []
    for se in slices_epi:
        if not slices_endo:
            continue
        si = slices_endo[np.argmin([abs(s['z'] - se['z']) for s in slices_endo])]
        slices.append(dict(z=se['z'], endo=si['endo'], epi=se['epi'],
                           centroid=se['centroid']))
    if not slices:
        raise ValueError('No valid slices')
    graph = build_graph(slices, CFG['knn_intra'], CFG['knn_inter'])
    graph.update(slices=slices, v_endo=v_endo, f_endo=f_endo,
                 v_epi=v_epi, f_epi=f_epi, affine=affine)
    return graph


def _align_acdc_to_ssm(nodes_np):
    acdc_xyz      = nodes_np[:, :3]
    ssm_center    = MEAN_VERTS.mean(axis=0)
    acdc_center   = acdc_xyz.mean(axis=0)
    ssm_xy_range  = np.ptp(MEAN_VERTS[:, :2], axis=0).max()
    acdc_xy_range = np.ptp(acdc_xyz[:, :2], axis=0).max()
    scale_xy      = ssm_xy_range / max(acdc_xy_range, 1e-3)
    ssm_z_range   = np.ptp(MEAN_VERTS[:, 2])
    acdc_z_range  = np.ptp(acdc_xyz[:, 2])
    scale_z       = ssm_z_range / max(acdc_z_range, 1e-3)
    acdc_z_mid    = float((acdc_xyz[:, 2].min() + acdc_xyz[:, 2].max()) / 2.0)

    def transform(xyz):
        centered = xyz - acdc_center
        return centered * np.array([scale_xy, scale_xy, -scale_z]) + ssm_center

    def inverse_transform(xyz):
        out = xyz.copy().astype(np.float64)
        out[:, 0] = (xyz[:, 0] - ssm_center[0]) / scale_xy + acdc_center[0]
        out[:, 1] = (xyz[:, 1] - ssm_center[1]) / scale_xy + acdc_center[1]
        ssm_z_center = float(MEAN_VERTS[:, 2].mean())
        out[:, 2]    = (xyz[:, 2] - ssm_z_center) / scale_xy + acdc_z_mid
        return out.astype(np.float32)

    nodes_aligned            = nodes_np.copy()
    nodes_aligned[:, :3]     = transform(acdc_xyz).astype(np.float32)
    xy_center                = nodes_aligned[:, :2].mean(axis=0)
    nodes_aligned[:, 3]      = np.linalg.norm(nodes_aligned[:, :2] - xy_center, axis=1)
    return nodes_aligned, transform, inverse_transform


def _run_acdc(model, mean_t, scaled_t, n_verts):
    acdc_path = CFG.get('acdc_label_path')
    assert acdc_path and os.path.exists(str(acdc_path)), \
        f'ACDC file not found: {acdc_path}'

    print(f'\nLoading ACDC: {acdc_path}')
    acdc_graph = _acdc_to_graph(acdc_path)

    sid_arr   = acdc_graph['slice_ids']
    n_slices  = acdc_graph['num_slices']
    first_sid = 0
    mid_sid   = n_slices // 2
    last_sid  = n_slices - 1
    vis_sids  = np.array([first_sid, mid_sid, last_sid])
    obs_mask_np = np.isin(sid_arr, vis_sids)
    print(f'Observing slices {vis_sids}  →  {obs_mask_np.sum()} nodes')

    nodes_raw = acdc_graph['nodes'].copy()
    nodes_aligned, _, acdc_inverse = _align_acdc_to_ssm(nodes_raw)

    edges_np    = acdc_graph['edges']
    aligned_xyz = nodes_aligned[:, :3]
    if len(edges_np) > 0:
        diff       = aligned_xyz[edges_np[:, 1]] - aligned_xyz[edges_np[:, 0]]
        dist       = np.linalg.norm(diff, axis=1, keepdims=True).clip(min=1e-8)
        ttype_s    = nodes_aligned[edges_np[:, 0], 4:5]
        ttype_t    = nodes_aligned[edges_np[:, 1], 4:5]
        ea_aligned = np.hstack([diff / dist, dist, ttype_s, ttype_t]).astype(np.float32)
    else:
        ea_aligned = acdc_graph['edge_feats']

    obs_mask_t = torch.from_numpy(obs_mask_np).to(DEVICE)
    nodes_t    = torch.from_numpy(nodes_aligned).float().to(DEVICE)
    ei         = torch.from_numpy(acdc_graph['edges'].T).long().to(DEVICE)
    ea         = torch.from_numpy(ea_aligned).float().to(DEVICE)

    x_in   = torch.cat([nodes_t, obs_mask_t.float().unsqueeze(1)], dim=1)
    d_acdc = Batch.from_data_list(
        [Data(x=x_in, edge_index=ei, edge_attr=ea)]
    ).to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred_w       = model(d_acdc, epoch=0)
        pred_verts_t = reconstruct_batch(pred_w, mean_t, scaled_t, n_verts, DEVICE)[0]

    pred_verts      = pred_verts_t.cpu().numpy()
    pred_verts_acdc = acdc_inverse(pred_verts)
    pred_faces      = np.array(FACES)

    print(f'Predicted weights: {pred_w.cpu().numpy()[0].round(3)}')
    print(f'Reconstructed {len(pred_verts):,} vertices — saved visualization.')

    # Simple 2-panel plot (full 4-panel version is in the notebook)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                              subplot_kw=dict(projection='3d'))
    obs_pts = acdc_inverse(nodes_aligned[obs_mask_np, :3])
    for ax, (pts, c, title) in zip(axes, [
        (pred_verts_acdc, 'dodgerblue', 'SSM Reconstruction (ACDC space)'),
        (acdc_inverse(MEAN_VERTS), 'lightgray', 'SSM Mean Shape'),
    ]):
        ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                        triangles=pred_faces[::4], color=c, alpha=0.5, edgecolor='none')
        ax.scatter(obs_pts[:, 0], obs_pts[:, 1], obs_pts[:, 2],
                   c='gold', s=20, zorder=5)
        ax.set_title(title, fontweight='bold')
        ax.view_init(20, 45)
    plt.suptitle('ACDC → SSM Inference', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = viz_dir / 'phase2_acdc_result.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f'Saved → {out}')


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, choices=[1, 2], default=2)
    parser.add_argument('--acdc', action='store_true',
                        help='Run ACDC inference (Phase 2 only)')
    args = parser.parse_args()

    if args.phase == 1:
        eval_phase1()
    else:
        eval_phase2(run_acdc=args.acdc)
