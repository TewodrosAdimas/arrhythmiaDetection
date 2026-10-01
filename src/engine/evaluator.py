import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
import numpy as np

from src.utils.metrics import compute_classification_metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    pos_weight: float = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Runs inference over a DataLoader, computes BCE loss and full medical classification metrics.
    
    Returns:
        (avg_loss, metrics_dict)
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    total_loss = 0.0
    all_targets = []
    all_logits = []

    weight_tensor = torch.tensor([pos_weight], device=dev) if pos_weight is not None else None

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(dev)
            y = y.to(dev)

            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight_tensor)
            total_loss += loss.item() * x.size(0)

            all_targets.append(y.detach().cpu())
            all_logits.append(logits.detach().cpu())

    avg_loss = total_loss / max(len(dataloader.dataset), 1)
    
    y_true = torch.cat(all_targets, dim=0).numpy()
    y_logits = torch.cat(all_logits, dim=0).numpy()

    metrics = compute_classification_metrics(y_true, y_logits)
    metrics["loss"] = float(avg_loss)

    return avg_loss, metrics
