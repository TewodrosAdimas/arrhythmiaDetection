import os
import time
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.engine.evaluator import evaluate_model
from src.utils.logger import get_logger


class Trainer:
    """
    Standardized PyTorch Trainer for ECG Arrhythmia Detection models.
    Supports Early Stopping, Learning Rate Scheduling, TensorBoard, and Checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_name: str = "adam",
        device: str = "auto",
        checkpoint_dir: str = "results/checkpoints",
        experiment_name: str = "experiment",
        pos_weight: Optional[float] = None,
        use_tensorboard: bool = True
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir
        self.pos_weight = pos_weight
        self.logger = get_logger("Trainer")

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Optimizer
        if optimizer_name.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        # TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_log_dir = os.path.join("results/logs", experiment_name)
                os.makedirs(tb_log_dir, exist_ok=True)
                self.writer = SummaryWriter(tb_log_dir)
            except Exception:
                self.writer = None

    def train_epoch(self) -> float:
        """Runs one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        pos_w_tensor = (
            torch.tensor([self.pos_weight], device=self.device)
            if self.pos_weight is not None
            else None
        )

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w_tensor)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

        return total_loss / max(n_samples, 1)

    def fit(
        self,
        epochs: int = 30,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Executes full training loop across epochs with validation and early stopping.
        """
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_roc_auc": [],
            "epoch_times": []
        }

        best_checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_best.pth")
        last_checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_last.pth")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            train_loss = self.train_epoch()
            val_loss, val_metrics = evaluate_model(
                self.model, self.val_loader, device=self.device, pos_weight=self.pos_weight
            )
            ep_time = time.time() - ep_start

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1_score"])
            history["val_roc_auc"].append(val_metrics["roc_auc"])
            history["epoch_times"].append(ep_time)

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Metrics/val_accuracy", val_metrics["accuracy"], epoch)
                self.writer.add_scalar("Metrics/val_f1", val_metrics["f1_score"], epoch)
                self.writer.add_scalar("Metrics/val_roc_auc", val_metrics["roc_auc"], epoch)
                self.writer.add_scalar("Params/lr", current_lr, epoch)

            if verbose:
                self.logger.info(
                    f"Epoch [{epoch:02d}/{epochs:02d}] "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
                    f"Val F1: {val_metrics['f1_score']:.4f} | "
                    f"Val AUC: {val_metrics['roc_auc']:.4f} | "
                    f"Time: {ep_time:.2f}s"
                )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "experiment_name": self.experiment_name
                }, best_checkpoint_path)
            else:
                patience_counter += 1

            # Save last checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics
            }, last_checkpoint_path)

            if patience_counter >= early_stopping_patience:
                if verbose:
                    self.logger.info(f"Early stopping triggered at epoch {epoch} (Best epoch was {best_epoch}).")
                break

        total_time = time.time() - start_time
        if self.writer is not None:
            self.writer.close()

        # Reload best weights for subsequent evaluation
        if os.path.exists(best_checkpoint_path):
            state = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model_state_dict"])

        return {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "total_train_time_sec": total_time,
            "avg_epoch_time_sec": sum(history["epoch_times"]) / max(len(history["epoch_times"]), 1),
            "history": history,
            "best_checkpoint_path": best_checkpoint_path
        }
