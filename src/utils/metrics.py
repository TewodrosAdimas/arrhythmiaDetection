from typing import Dict, Any, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_logits: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes comprehensive clinical and statistical classification metrics.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred_logits: Predicted raw logits or probabilities.
        threshold: Decision threshold for probability classification.
        
    Returns:
        Dictionary containing Accuracy, Sensitivity/Recall, Specificity, Precision,
        F1, ROC-AUC, PR-AUC, and Confusion Matrix.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_logits, torch.Tensor):
        y_pred_logits = y_pred_logits.detach().cpu().numpy()

    y_true = y_true.reshape(-1).astype(int)
    y_logits = y_pred_logits.reshape(-1)

    # Convert logits to probabilities via sigmoid
    # Handle if already probabilities
    if np.any(y_logits < 0) or np.any(y_logits > 1):
        y_probs = 1.0 / (1.0 + np.exp(-np.clip(y_logits, -20.0, 20.0)))
    else:
        y_probs = y_logits

    y_pred = (y_probs >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0      # Specificity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC-AUC and PR-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_probs)
    except Exception:
        pr_auc = 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),             # Sensitivity
        "sensitivity": float(rec),
        "specificity": float(spec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_probs": y_probs,
        "y_pred": y_pred
    }


def format_metrics_table(metrics: Dict[str, Any], model_name: str = "Model") -> str:
    """Formats metrics as a clean markdown/text block."""
    table = f"""
| Metric | Value |
| :--- | :--- |
| **Model** | {model_name} |
| **Accuracy** | {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) |
| **Sensitivity (Recall)** | {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%) |
| **Specificity** | {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%) |
| **Precision** | {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%) |
| **F1-Score** | {metrics['f1_score']:.4f} |
| **ROC-AUC** | {metrics['roc_auc']:.4f} |
| **PR-AUC** | {metrics['pr_auc']:.4f} |
| **Confusion Matrix** | TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']} |
"""
    return table.strip()
