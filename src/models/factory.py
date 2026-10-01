import torch.nn as nn
from typing import Dict, Any, List

from src.models.baseline_cnn import BaselineCNN
from src.models.enhanced_cnn import EnhancedCNN
from src.models.resnet1d import ResNet1D, resnet18_1d, resnet34_1d
from src.models.rnn1d import BiLSTMNet
from src.models.transformer1d import Transformer1D
from src.models.pretrained_cwt import PretrainedTransferModel

MODEL_REGISTRY = {
    "baseline_cnn": BaselineCNN,
    "enhanced_cnn": EnhancedCNN,
    "resnet18_1d": resnet18_1d,
    "resnet34_1d": resnet34_1d,
    "resnet1d": resnet18_1d,
    "bilstm": BiLSTMNet,
    "rnn1d": BiLSTMNet,
    "transformer1d": Transformer1D,
    "transformer": Transformer1D,
    "pretrained_resnet18": lambda **kw: PretrainedTransferModel(backbone_name="resnet18", **kw),
    "pretrained_mobilenet": lambda **kw: PretrainedTransferModel(backbone_name="mobilenet_v3_small", **kw),
}


def list_models() -> List[str]:
    """Returns list of registered model names."""
    return list(MODEL_REGISTRY.keys())


def build_model(
    model_name: str,
    in_channels: int = 1,
    num_classes: int = 1,
    **kwargs
) -> nn.Module:
    """
    Factory function to instantiate any supported model by string identifier.
    """
    name_clean = model_name.lower().strip()
    if name_clean not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name: '{model_name}'. Available options: {list_models()}"
        )

    model_cls = MODEL_REGISTRY[name_clean]
    return model_cls(in_channels=in_channels, num_classes=num_classes, **kwargs)
