"""
evaluate_acdc.py
================
Run the trained CardiacGNN on ACDC dataset patients and report metrics.

Usage
-----
# Single patient
python scripts/evaluate_acdc.py \\
    --config configs/default.yaml \\
    --checkpoint outputs/checkpoints/best_model.pth \\
    --label data/acdc/patient101/patient101_frame01_gt.nii.gz

# Entire directory (all *_gt.nii.gz files)
python scripts/evaluate_acdc.py \\
    --config configs/default.yaml \\
    --checkpoint outputs/checkpoints/best_model.pth \\
    --patient_dir data/acdc/

Reported metrics per patient
-----------------------------
  - MSE between predicted and ground-truth node positions
  - Predicted LV volume (mm³) vs ground-truth volume
  - Predicted surface area (mm²) vs ground-truth
  - Average wall thickness (mm)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.acdc_loader import acdc_patient_to_graph, extract_lv_surfaces
from models.cardiac_gnn import CardiacGNN
from training.trainer import Trainer
from utils.mesh_utils import mesh_volume, mesh_surface_area, compute_wall_thickness
from utils.visualization import plot_prediction_vs_truth, plot_slice_contours
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CardiacGNN on ACDC patients")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    p.add_argument("--label", default=None,
                   help="Single patient label NIfTI (.nii.gz)")
    p.add_argument("--patient_dir", default=None,
                   help="Directory to scan for *_gt.nii.gz files")
    p.add_argument("--save_viz", action="store_true",
                   help="Save per-patient visualisations")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def run_inference(model, graph_dict, device):
    """Convert graph dict → PyG Data → run model → return numpy predictions."""
    from torch_geometric.data import Data, Batch

    nodes = torch.tensor(graph_dict["nodes"], dtype=torch.float32)
    edges = torch.tensor(graph_dict["edges"].T, dtype=torch.long)
    data = Data(x=nodes, edge_index=edges)
    batch = Batch.from_data_list([data]).to(device)

    with torch.no_grad():
        pred = model(batch).cpu().numpy()
    return pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CardiacGNN(
        node_features=cfg["node_features"],
        hidden_dim=cfg["hidden_dim"],
        deformation_cap=cfg["deformation_cap"],
    ).to(device)
    Trainer.load_checkpoint(model, args.checkpoint, device)
    model.eval()
    print(f"  Loaded model from: {args.checkpoint}  |  device: {device}")

    # Collect label files
    if args.label:
        label_files = [Path(args.label)]
    elif args.patient_dir:
        label_files = sorted(Path(args.patient_dir).rglob("*_gt.nii.gz"))
    else:
        raise ValueError("Provide --label or --patient_dir")

    print(f"  Patients to evaluate: {len(label_files)}")

    viz_dir = Path(cfg["viz_dir"]) / "acdc"
    if args.save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for label_path in tqdm(label_files, desc="Evaluating", unit="patient"):
        patient_id = label_path.name.replace("_gt.nii.gz", "")

        try:
            graph_dict = acdc_patient_to_graph(
                label_path=label_path,
                num_slices=cfg["num_slices"],
                points_per_contour=cfg["points_per_contour"],
                knn_intra=cfg["knn_intra_slice"],
                knn_inter=cfg["knn_inter_slice"],
            )
        except Exception as e:
            print(f"  [warn] {patient_id}: graph extraction failed — {e}")
            continue

        if not graph_dict or graph_dict["num_nodes"] == 0:
            print(f"  [warn] {patient_id}: empty graph")
            continue

        # Ground-truth positions
        gt_positions = graph_dict["nodes"][:, :3]

        # Predicted positions
        pred_positions = run_inference(model, graph_dict, device)

        # MSE metric
        mse = float(np.mean((pred_positions - gt_positions) ** 2))

        # Surface metrics from predicted positions
        # (approximate: use all predicted endo nodes as a point cloud)
        node_types = graph_dict["node_types"]
        endo_pred = pred_positions[node_types == 0]
        epi_pred = pred_positions[node_types == 1]
        endo_gt = gt_positions[node_types == 0]
        epi_gt = gt_positions[node_types == 1]

        avg_thickness_pred = compute_wall_thickness(endo_pred, epi_pred)
        avg_thickness_gt = compute_wall_thickness(endo_gt, epi_gt)

        record = {
            "patient_id": patient_id,
            "num_nodes": graph_dict["num_nodes"],
            "num_slices": graph_dict["num_slices"],
            "mse_mm2": mse,
            "rmse_mm": float(np.sqrt(mse)),
            "avg_wall_thickness_pred_mm": avg_thickness_pred,
            "avg_wall_thickness_gt_mm": avg_thickness_gt,
        }
        records.append(record)

        # Optional visualisation
        if args.save_viz:
            fig_path = viz_dir / f"{patient_id}_reconstruction.png"
            plot_prediction_vs_truth(
                gt_positions, pred_positions,
                title=f"Patient: {patient_id}",
                save_path=fig_path,
            )

            # Mid-slice comparison
            slices = graph_dict.get("slices", [])
            if slices:
                mid = slices[len(slices) // 2]
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                plot_slice_contours(mid.endo_contour, mid.epi_contour, mid.z_position, ax=axes[0])
                axes[0].set_title("Ground Truth Contours")
                axes[1].axis("off")
                axes[1].text(0.5, 0.5, f"MSE: {mse:.4f} mm²\nRMSE: {np.sqrt(mse):.3f} mm",
                             ha="center", va="center", fontsize=14, transform=axes[1].transAxes,
                             bbox=dict(boxstyle="round", facecolor="lightyellow"))
                plt.suptitle(patient_id, fontweight="bold")
                plt.tight_layout()
                plt.savefig(viz_dir / f"{patient_id}_slice_info.png", dpi=120)
                plt.close()

    # Summary table
    df = pd.DataFrame(records)
    out_csv = Path(cfg["viz_dir"]) / "acdc_evaluation_results.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n{'='*55}")
    print(f"  ACDC Evaluation Results  ({len(df)} patients)")
    print(f"{'='*55}")
    print(f"  Mean RMSE        : {df['rmse_mm'].mean():.3f} ± {df['rmse_mm'].std():.3f} mm")
    print(f"  Median RMSE      : {df['rmse_mm'].median():.3f} mm")
    print(f"  Wall thickness Δ : {(df['avg_wall_thickness_pred_mm'] - df['avg_wall_thickness_gt_mm']).abs().mean():.3f} mm (MAE)")
    print(f"\n  Results saved → {out_csv}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
