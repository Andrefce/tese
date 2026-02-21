"""
generate_dataset.py
===================
End-to-end script: UK Digital Heart SSM → on-disk dataset ready for GNN training.

Outputs (inside data_dir):
  meshes/heart_XXX.stl          — STL mesh per sample
  slices/slices_XXX.json        — JSON endo/epi contours per sample
  graphs/graph_XXX.npz          — PyG-compatible graph per sample
  dataset_metadata.csv          — Per-sample statistics
  pca_weights.npy               — (N_samples × M) PCA weight matrix

Usage
-----
python scripts/generate_dataset.py --config configs/default.yaml
python scripts/generate_dataset.py --config configs/default.yaml --num_samples 200
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.ssm_loader import SSMLoader
from data.graph_builder import extract_slices, build_graph, slices_to_json
from utils.mesh_utils import save_mesh, mesh_volume, mesh_surface_area


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate SSM-based cardiac graph dataset")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Override num_samples in config")
    p.add_argument("--seed", type=int, default=None,
                   help="Override random seed in config")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    num_samples = args.num_samples or cfg["num_samples"]
    seed = args.seed or cfg["seed"]

    # Directories
    data_dir = Path(cfg["data_dir"])
    mesh_dir = data_dir / "meshes"
    slice_dir = data_dir / "slices"
    graph_dir = data_dir / "graphs"
    for d in [mesh_dir, slice_dir, graph_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Cardiac Shape Dataset Generator")
    print("=" * 60)
    print(f"  SSM dir    : {cfg['ssm_dir']}")
    print(f"  Output dir : {data_dir}")
    print(f"  Samples    : {num_samples}")
    print(f"  PCA modes  : {cfg['num_pca_modes']}")
    print(f"  Slices/shape: {cfg['num_slices']}")
    print(f"  Points/contour: {cfg['points_per_contour']}")
    print("=" * 60)

    # Load SSM
    ssm = SSMLoader(
        ssm_dir=cfg["ssm_dir"],
        mean_mesh_file=cfg["ssm_mean_mesh"],
        pc_file=cfg["ssm_pca_modes"],
        var_file=cfg["ssm_pca_vars"],
    )

    # Generate shapes
    shapes = ssm.sample_batch(
        n=num_samples,
        num_modes=cfg["num_pca_modes"],
        sigma_clip=cfg["pca_sigma_clip"],
        seed=seed,
    )

    records = []
    pca_weights_all = []
    t0 = time.time()

    for i, shape in enumerate(tqdm(shapes, desc="Generating", unit="shape")):
        # 1. Save mesh
        save_mesh(shape.vertices, shape.faces, mesh_dir / f"heart_{i:03d}.stl")

        # 2. Extract slices
        slices = extract_slices(
            shape.vertices,
            num_slices=cfg["num_slices"],
            points_per_contour=cfg["points_per_contour"],
            epsilon=cfg["slice_epsilon"],
        )

        # Save JSON slice data
        slice_json = slices_to_json(slices, sample_id=i)
        with open(slice_dir / f"slices_{i:03d}.json", "w") as f:
            json.dump(slice_json, f)

        # 3. Build graph
        graph = build_graph(
            slices,
            knn_intra=cfg["knn_intra_slice"],
            knn_inter=cfg["knn_inter_slice"],
        )

        if not graph:
            print(f"  [warn] Sample {i}: empty graph — skipping")
            continue

        np.savez_compressed(
            graph_dir / f"graph_{i:03d}.npz",
            nodes=graph["nodes"],
            edges=graph["edges"],
            node_types=graph["node_types"],
            slice_ids=graph["slice_ids"],
        )

        # 4. Collect metadata
        vol = mesh_volume(shape.vertices, shape.faces)
        area = mesh_surface_area(shape.vertices, shape.faces)
        bb = shape.vertices
        record = {
            "sample_id": i,
            "volume": vol,
            "surface_area": area,
            "width": float(bb[:, 0].ptp()),
            "depth": float(bb[:, 1].ptp()),
            "height": float(bb[:, 2].ptp()),
            "num_slices": len(slices),
            "num_graph_nodes": graph["num_nodes"],
            "num_graph_edges": graph["num_edges"],
        }
        # Append PCA mode weights
        for m in range(cfg["num_pca_modes"]):
            record[f"mode_{m}"] = float(shape.pca_weights[m])

        records.append(record)
        pca_weights_all.append(shape.pca_weights)

    # Save metadata CSV
    df = pd.DataFrame(records)
    df.to_csv(data_dir / "dataset_metadata.csv", index=False)

    # Save PCA weight matrix
    np.save(data_dir / "pca_weights.npy", np.array(pca_weights_all))

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Done!  {len(records)}/{num_samples} samples generated in {elapsed:.1f}s")
    print(f"  Output: {data_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
