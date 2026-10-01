import argparse
import os
import sys
import json
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import build_model, list_models
from src.data import get_dataloaders, Compose1D, AddGaussianNoise, RandomScale, BaselineShift
from src.engine import Trainer, evaluate_model
from src.utils import get_logger, compute_classification_metrics, format_metrics_table


def main():
    parser = argparse.ArgumentParser(description="Train an ECG Arrhythmia Detection Model.")
    parser.add_argument("--model", type=str, default="enhanced_cnn", choices=list_models(), help="Model architecture to train")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay / L2 regularization")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"], help="Optimizer")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--augment", action="store_true", help="Apply 1D signal augmentations")
    parser.add_argument("--balance-weights", action="store_true", help="Use positive class weight in loss for imbalance")
    parser.add_argument("--experiment-name", type=str, default=None, help="Custom name for experiment/checkpoints")
    args = parser.parse_args()

    logger = get_logger("TrainCLI")
    exp_name = args.experiment_name or f"{args.model}_exp"

    logger.info(f"=== Starting Training: Model={args.model}, Experiment={exp_name} ===")

    # Augmentation pipeline
    train_transform = None
    if args.augment:
        train_transform = Compose1D([
            AddGaussianNoise(std=0.01, p=0.5),
            RandomScale(min_scale=0.95, max_scale=1.05, p=0.5),
            BaselineShift(max_shift=0.03, p=0.5)
        ])
        logger.info("Using 1D data augmentations during training.")

    # DataLoaders
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_transform
    )
    logger.info(f"Dataset loaded: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    logger.info(f"Train class distribution: {train_ds.get_class_counts()}")

    # Class weight
    pos_weight = None
    if args.balance_weights:
        pos_weight = train_ds.get_pos_weight().item()
        logger.info(f"Applying positive class weight: {pos_weight:.4f}")

    # Build model
    model = build_model(args.model, in_channels=1, num_classes=1)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        device=args.device,
        checkpoint_dir="results/checkpoints",
        experiment_name=exp_name,
        pos_weight=pos_weight
    )

    # Train
    train_res = trainer.fit(epochs=args.epochs, early_stopping_patience=args.patience)
    logger.info(f"Training completed in {train_res['total_train_time_sec']:.2f}s. Best epoch: {train_res['best_epoch']}")

    # Final Evaluation on Test Set
    logger.info("=== Running Final Evaluation on Test Set ===")
    test_loss, test_metrics = evaluate_model(model, test_loader, device=trainer.device, pos_weight=pos_weight)
    logger.info("\n" + format_metrics_table(test_metrics, model_name=args.model))

    # Save metrics JSON
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"{exp_name}_test_metrics.json")
    
    clean_metrics = {k: v for k, v in test_metrics.items() if not isinstance(v, (np.ndarray, list))}
    clean_metrics["test_loss"] = float(test_loss)
    with open(summary_path, "w") as f:
        json.dump(clean_metrics, f, indent=4)
    logger.info(f"Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
