"""
train.py
========
Train the CardiacGNN on the pre-generated graph dataset.

Usage
-----
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --epochs 200 --lr 1e-3
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import make_dataloaders
from models.cardiac_gnn import CardiacGNN
from training.trainer import Trainer
from utils.visualization import plot_training_curves


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train CardiacGNN")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    epochs = args.epochs or cfg["epochs"]
    lr = args.lr or cfg["learning_rate"]
    batch_size = args.batch_size or cfg["batch_size"]

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print(f"\n  Device: {device}")

    # ----- Data -----
    data_dir = Path(cfg["data_dir"])
    graph_dir = data_dir / "graphs"
    meta = pd.read_csv(data_dir / "dataset_metadata.csv")
    num_graphs = len(meta)

    train_loader, val_loader, (idx_train, idx_val) = make_dataloaders(
        graph_dir=graph_dir,
        num_graphs=num_graphs,
        val_split=cfg["val_split"],
        batch_size=batch_size,
        seed=cfg["seed"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"] and device.type == "cuda",
    )
    print(f"  Train graphs: {len(idx_train)}  |  Val graphs: {len(idx_val)}")

    # ----- Model -----
    model = CardiacGNN(
        node_features=cfg["node_features"],
        hidden_dim=cfg["hidden_dim"],
        deformation_cap=cfg["deformation_cap"],
    ).to(device)
    print(f"  Parameters : {model.count_parameters():,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=cfg["weight_decay"]
    )

    scheduler = None
    if cfg.get("lr_scheduler", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=cfg.get("lr_min", 1e-6)
        )

    # ----- Resume -----
    if args.resume:
        ckpt = Trainer.load_checkpoint(model, args.resume, device)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"  Resumed from: {args.resume}")

    # ----- Train -----
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=cfg["checkpoint_dir"],
        log_dir=cfg["log_dir"],
        smoothness_weight=cfg["smoothness_weight"],
        patience=cfg["patience"],
        save_every=cfg["save_every"],
        scheduler=scheduler,
    )

    trainer.fit(train_loader, val_loader, epochs=epochs)

    # ----- Visualise training curves -----
    viz_dir = Path(cfg["viz_dir"])
    viz_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(
        trainer.history,
        save_path=viz_dir / "training_curves.png",
    )


if __name__ == "__main__":
    main()
