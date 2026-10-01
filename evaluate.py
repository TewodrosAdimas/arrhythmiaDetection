import argparse
import os
import sys
import json
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import build_model, list_models
from src.data import get_dataloaders
from src.engine import evaluate_model
from src.utils import (
    get_logger,
    format_metrics_table,
    profile_model_efficiency
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate an ECG Arrhythmia Model Checkpoint.")
    parser.add_argument("--model", type=str, default="enhanced_cnn", choices=list_models(), help="Model architecture")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing test.csv")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    args = parser.parse_args()

    logger = get_logger("EvalCLI")
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"

    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # Build model and load weights
    model = build_model(args.model, in_channels=1, num_classes=1)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif "net" in state:
        model.load_state_dict(state["net"])
    else:
        model.load_state_dict(state)

    model.to(device)
    logger.info("Model weights loaded successfully.")

    # DataLoaders
    _, _, test_loader, _, _, test_ds = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0
    )

    # Evaluate
    logger.info(f"Evaluating model on {len(test_ds)} test samples...")
    test_loss, test_metrics = evaluate_model(model, test_loader, device=device)
    logger.info("\n" + format_metrics_table(test_metrics, model_name=args.model))

    # Profile Efficiency
    logger.info("Profiling computational efficiency...")
    eff = profile_model_efficiency(model, input_shape=(1, 1, 187), device=device)
    logger.info(f"Parameters: {eff['total_params']:,} (Trainable: {eff['trainable_params']:,})")
    logger.info(f"Single-sample Latency: {eff['single_latency_ms']:.3f} ms")
    logger.info(f"Batched Throughput: {eff['throughput_samples_per_sec']:.1f} samples/sec")
    logger.info(f"Model Disk Size: {eff['model_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
