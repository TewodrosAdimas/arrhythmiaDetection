import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def set_plot_style():
    """Sets clean seaborn/matplotlib styling for publication-quality figures."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14
    })


def plot_roc_curves(results: Dict[str, Dict[str, Any]], save_path: str = "results/plots/roc_curves.png"):
    """Plots superimposed ROC curves for all benchmarked models."""
    set_plot_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(8, 6), dpi=300)
    
    for model_name, res in results.items():
        if "y_true" in res and "y_probs" in res:
            fpr, tpr, _ = roc_curve(res["y_true"], res["y_probs"])
            roc_auc = res.get("roc_auc", auc(fpr, tpr))
            plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pr_curves(results: Dict[str, Dict[str, Any]], save_path: str = "results/plots/pr_curves.png"):
    """Plots Precision-Recall curves for all benchmarked models."""
    set_plot_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6), dpi=300)

    for model_name, res in results.items():
        if "y_true" in res and "y_probs" in res:
            precision, recall, _ = precision_recall_curve(res["y_true"], res["y_probs"])
            pr_auc = res.get("pr_auc", 0.0)
            plt.plot(recall, precision, lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves Comparison")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrices(results: Dict[str, Dict[str, Any]], save_path: str = "results/plots/confusion_matrices.png"):
    """Plots a multi-panel grid of Confusion Matrices."""
    set_plot_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_models = len(results)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=300, squeeze=False)
    
    for idx, (model_name, res) in enumerate(results.items()):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        
        cm = res.get("confusion_matrix", None)
        if cm is not None:
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax,
                xticklabels=["Normal (0)", "Arrhythmia (1)"],
                yticklabels=["Normal (0)", "Arrhythmia (1)"]
            )
            ax.set_title(f"{model_name}\n(F1: {res.get('f1_score', 0):.3f}, Acc: {res.get('accuracy', 0):.3f})")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

    # Hide unused subplots
    for idx in range(n_models, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_efficiency_tradeoff(results: Dict[str, Dict[str, Any]], save_path: str = "results/plots/efficiency_tradeoff.png"):
    """
    Plots an Efficiency vs. Performance trade-off scatter plot:
      X-axis: Inference Latency (ms)
      Y-axis: F1-Score / Accuracy
      Bubble Size: Total Parameters
    """
    set_plot_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(9, 6), dpi=300)

    names = []
    latencies = []
    f1_scores = []
    sizes = []

    for name, res in results.items():
        lat = res.get("single_latency_ms", res.get("batched_latency_ms", 1.0))
        f1 = res.get("f1_score", 0.0)
        params = res.get("total_params", 100000)

        names.append(name)
        latencies.append(lat)
        f1_scores.append(f1)
        # Scale bubble size
        sizes.append(max(80, min(800, np.sqrt(params) * 2)))

    scatter = plt.scatter(latencies, f1_scores, s=sizes, alpha=0.7, c=range(len(names)), cmap="viridis", edgecolors="black", linewidth=1.5)

    for i, name in enumerate(names):
        plt.annotate(
            name,
            (latencies[i], f1_scores[i]),
            xytext=(6, 6),
            textcoords="offset points",
            fontweight="bold"
        )

    plt.xlabel("Single-Sample Inference Latency (ms) [Lower is Better]")
    plt.ylabel("Test F1-Score [Higher is Better]")
    plt.title("Model Efficiency vs. Performance Trade-Off\n(Bubble size proportional to Parameter Count)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
