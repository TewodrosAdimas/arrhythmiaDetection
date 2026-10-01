import argparse
import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import build_model, list_models
from src.data import get_dataloaders, Compose1D, AddGaussianNoise, RandomScale, BaselineShift
from src.engine import Trainer, evaluate_model
from src.utils import (
    get_logger,
    profile_model_efficiency,
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrices,
    plot_efficiency_tradeoff
)


DEFAULT_BENCHMARK_MODELS = [
    "baseline_cnn",
    "enhanced_cnn",
    "resnet1d",
    "bilstm",
    "transformer1d",
    "pretrained_resnet18"
]


def run_benchmark(
    models_to_benchmark=None,
    data_dir="data",
    epochs=25,
    batch_size=64,
    lr=1e-3,
    device="auto",
    output_dir="results",
    augment=True,
    skip_training=False
):
    logger = get_logger("Benchmark")
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    if models_to_benchmark is None:
        models_to_benchmark = DEFAULT_BENCHMARK_MODELS

    logger.info("================================================================")
    logger.info("   ECG ARRHYTHMIA DETECTION: MULTI-MODEL BENCHMARK SUITE       ")
    logger.info("================================================================")
    logger.info(f"Models to evaluate: {models_to_benchmark}")
    logger.info(f"Epochs per model: {epochs} | Batch size: {batch_size} | LR: {lr}")

    # DataLoaders
    train_transform = Compose1D([
        AddGaussianNoise(std=0.01, p=0.5),
        RandomScale(min_scale=0.95, max_scale=1.05, p=0.5),
        BaselineShift(max_shift=0.03, p=0.5)
    ]) if augment else None

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
        train_transform=train_transform
    )
    logger.info(f"Dataset: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    summary_rows = []
    all_eval_results = {}

    for model_name in models_to_benchmark:
        logger.info(f"\n>>> Benchmarking Model: [{model_name.upper()}] <<<")
        ckpt_path = os.path.join(checkpoints_dir, f"{model_name}_best.pth")
        
        # Build model
        model = build_model(model_name, in_channels=1, num_classes=1)

        train_time_sec = 0.0
        avg_ep_time_sec = 0.0

        if not skip_training or not os.path.exists(ckpt_path):
            logger.info(f"Training {model_name} for up to {epochs} epochs...")
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=lr,
                device=device,
                checkpoint_dir=checkpoints_dir,
                experiment_name=model_name
            )
            train_res = trainer.fit(epochs=epochs, early_stopping_patience=7, verbose=True)
            train_time_sec = train_res["total_train_time_sec"]
            avg_ep_time_sec = train_res["avg_epoch_time_sec"]
        else:
            logger.info(f"Found existing checkpoint at {ckpt_path}. Skipping training.")
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            elif "net" in state:
                model.load_state_dict(state["net"])

        # Final Evaluation on Test Set
        dev_str = "cuda" if (device == "auto" and torch.cuda.is_available()) or device == "cuda" else "cpu"
        test_loss, test_metrics = evaluate_model(model, test_loader, device=dev_str)

        # Computational Efficiency Profiling
        eff = profile_model_efficiency(model, input_shape=(1, 1, 187), device=dev_str, batch_size=batch_size)

        # Merge results
        combined_res = {**test_metrics, **eff, "train_time_sec": train_time_sec, "avg_ep_time_sec": avg_ep_time_sec}
        all_eval_results[model_name] = combined_res

        summary_rows.append({
            "Model": model_name,
            "Accuracy (%)": round(test_metrics["accuracy"] * 100, 2),
            "Sensitivity (%)": round(test_metrics["sensitivity"] * 100, 2),
            "Specificity (%)": round(test_metrics["specificity"] * 100, 2),
            "Precision (%)": round(test_metrics["precision"] * 100, 2),
            "F1-Score": round(test_metrics["f1_score"], 4),
            "ROC-AUC": round(test_metrics["roc_auc"], 4),
            "PR-AUC": round(test_metrics["pr_auc"], 4),
            "Test Loss": round(test_loss, 4),
            "Params (K)": round(eff["total_params"] / 1000, 1),
            "Latency (ms)": round(eff["single_latency_ms"], 2),
            "Throughput (sps)": round(eff["throughput_samples_per_sec"], 1),
            "Size (MB)": round(eff["model_size_mb"], 2)
        })

    # Generate Leaderboard Table
    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    try:
        md_table = df_summary.to_markdown(index=False)
    except Exception:
        # Fallback manual markdown table generator
        headers = list(df_summary.columns)
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join([":---:" for _ in headers]) + " |"
        row_lines = []
        for _, row in df_summary.iterrows():
            row_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        md_table = "\n".join([header_line, sep_line] + row_lines)
    md_report = f"""# 🏆 ECG Arrhythmia Detection Model Benchmark Summary

*Evaluated on {len(test_ds)} Test ECG Beats (Sequence Length = 187)*

{md_table}

### Metrics Legend:
- **Sensitivity / Recall**: Detection rate of abnormal cardiac arrhythmia beats (minimizing False Negatives).
- **Specificity**: Detection rate of normal sinus rhythm beats (minimizing False Positives).
- **Latency (ms)**: Inference delay per single heartbeat sample on {eff['device']}.
- **Throughput (sps)**: Inferences per second in batched mode (Batch Size = {batch_size}).
- **Params (K)**: Total model parameter count in thousands.
"""

    md_path = os.path.join(output_dir, "benchmark_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    # Save JSON summary with arrays stripped
    json_path = os.path.join(output_dir, "benchmark_summary.json")
    clean_json = {}
    for m, res in all_eval_results.items():
        clean_json[m] = {k: v for k, v in res.items() if not isinstance(v, (np.ndarray, list))}
    with open(json_path, "w") as f:
        json.dump(clean_json, f, indent=4)

    # Generate Visualizations
    logger.info("Generating publication-ready benchmark visualizations...")
    plot_roc_curves(all_eval_results, save_path=os.path.join(plots_dir, "roc_curves.png"))
    plot_pr_curves(all_eval_results, save_path=os.path.join(plots_dir, "pr_curves.png"))
    plot_confusion_matrices(all_eval_results, save_path=os.path.join(plots_dir, "confusion_matrices.png"))
    plot_efficiency_tradeoff(all_eval_results, save_path=os.path.join(plots_dir, "efficiency_tradeoff.png"))

    logger.info("\n" + md_report)
    logger.info(f"\nAll benchmark assets successfully saved to '{output_dir}/'.")
    return df_summary


def main():
    parser = argparse.ArgumentParser(description="Run ECG Arrhythmia Detection Multi-Model Benchmark.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_BENCHMARK_MODELS, help="List of model architectures to benchmark")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    parser.add_argument("--output-dir", type=str, default="results", help="Results directory")
    parser.add_argument("--skip-training", action="store_true", help="Skip training if checkpoints exist")
    parser.add_argument("--no-augment", action="store_true", help="Disable training data augmentation")
    args = parser.parse_args()

    run_benchmark(
        models_to_benchmark=args.models,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        augment=not args.no_augment,
        skip_training=args.skip_training
    )


if __name__ == "__main__":
    main()
