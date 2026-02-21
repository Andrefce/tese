"""
trainer.py
==========
Training loop for CardiacGNN with:
  - Early stopping
  - Best-model checkpointing
  - Periodic checkpoint saves
  - CosineAnnealing LR schedule
  - CSV loss logging
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from models.cardiac_gnn import CardiacGNN
from training.losses import combined_loss


class Trainer:
    """
    Parameters
    ----------
    model : CardiacGNN
    optimizer : torch.optim.Optimizer
    device : torch.device
    checkpoint_dir : str or Path
    log_dir : str or Path
    smoothness_weight : float
    patience : int  — early stopping patience (epochs)
    save_every : int — save periodic checkpoint every N epochs
    scheduler : optional LR scheduler
    """

    def __init__(
        self,
        model: CardiacGNN,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str | Path,
        log_dir: str | Path,
        smoothness_weight: float = 0.05,
        patience: int = 15,
        save_every: int = 10,
        scheduler=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.smoothness_weight = smoothness_weight
        self.patience = patience
        self.save_every = save_every
        self.scheduler = scheduler

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._log_path = self.log_dir / "training_log.csv"
        self._best_val_loss = float("inf")
        self._no_improve_count = 0
        self.best_epoch = 0
        self.history: list[dict] = []

        # Write CSV header
        with open(self._log_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["epoch", "train_mse", "train_smooth",
                               "train_total", "val_mse", "val_smooth",
                               "val_total", "lr", "epoch_time_s"]
            )
            writer.writeheader()

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        """Run one epoch. Returns dict of averaged metrics."""
        self.model.train() if train else self.model.eval()

        total_mse = total_smooth = total_loss = 0.0
        n_batches = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                batch = batch.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                pred = self.model(batch)
                loss, l_mse, l_smooth = combined_loss(
                    pred, batch.y, batch.edge_index, self.smoothness_weight
                )

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                total_mse += l_mse.item()
                total_smooth += l_smooth.item()
                n_batches += 1

        denom = max(n_batches, 1)
        return {
            "total": total_loss / denom,
            "mse": total_mse / denom,
            "smooth": total_smooth / denom,
        }

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> None:
        """
        Train the model for up to `epochs` epochs with early stopping.
        """
        print(f"\n{'─'*60}")
        print(f"  CardiacGNN Training  |  device={self.device}  |  params={self.model.count_parameters():,}")
        print(f"{'─'*60}")
        print(f"  {'Epoch':>5}  {'Train MSE':>10}  {'Val MSE':>10}  {'Val Total':>10}  {'LR':>8}")
        print(f"{'─'*60}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)

            if self.scheduler is not None:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            # Log to CSV
            row = {
                "epoch": epoch,
                "train_mse": train_metrics["mse"],
                "train_smooth": train_metrics["smooth"],
                "train_total": train_metrics["total"],
                "val_mse": val_metrics["mse"],
                "val_smooth": val_metrics["smooth"],
                "val_total": val_metrics["total"],
                "lr": lr,
                "epoch_time_s": elapsed,
            }
            self.history.append(row)
            with open(self._log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=row.keys()).writerow(row)

            print(
                f"  {epoch:5d}  {train_metrics['mse']:10.4f}  "
                f"{val_metrics['mse']:10.4f}  {val_metrics['total']:10.4f}  {lr:8.2e}"
            )

            # Periodic checkpoint
            if epoch % self.save_every == 0:
                self._save(f"checkpoint_epoch{epoch:04d}.pth")

            # Best model checkpoint + early stopping
            val_loss = val_metrics["total"]
            if val_loss < self._best_val_loss - 1e-4:
                self._best_val_loss = val_loss
                self.best_epoch = epoch
                self._no_improve_count = 0
                self._save("best_model.pth")
                print(f"         ★  New best val loss: {val_loss:.4f}")
            else:
                self._no_improve_count += 1
                if self._no_improve_count >= self.patience:
                    print(f"\n  Early stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs).")
                    break

        print(f"\n  Training complete.  Best epoch: {self.best_epoch}  "
              f"Best val loss: {self._best_val_loss:.4f}")
        print(f"  Checkpoints: {self.checkpoint_dir}")

    # ------------------------------------------------------------------
    def _save(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self._best_val_loss,
                "best_epoch": self.best_epoch,
            },
            path,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def load_checkpoint(model: CardiacGNN, checkpoint_path: str | Path, device: torch.device) -> dict:
        """Load a saved checkpoint into `model` in-place. Returns the checkpoint dict."""
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        return ckpt
