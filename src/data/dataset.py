import os
from typing import Optional, Tuple, Union, List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ECGDataset(Dataset):
    """
    PyTorch Dataset for 1D ECG Arrhythmia Signals.
    
    Preserves strict chronological feature column ordering (0 to 186)
    and supports optional 1D data augmentations or 2D spectrogram/CWT transforms.
    """
    def __init__(
        self,
        data_source: Union[str, pd.DataFrame, np.ndarray],
        label_col: str = "class",
        transform = None,
        target_transform = None
    ):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.label_col = label_col

        if isinstance(data_source, str):
            if not os.path.exists(data_source):
                raise FileNotFoundError(f"Dataset CSV not found at: {data_source}")
            self.df = pd.read_csv(data_source, sep=",")
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy()
        elif isinstance(data_source, np.ndarray):
            # Assume last column is class label if 2D
            n_cols = data_source.shape[1]
            cols = [str(i) for i in range(n_cols - 1)] + [label_col]
            self.df = pd.DataFrame(data_source, columns=cols)
        else:
            raise TypeError(f"Unsupported data source type: {type(data_source)}")

        # Extract features and labels with STRICT numerical column ordering
        if self.label_col in self.df.columns:
            self.labels = self.df[self.label_col].values.astype(np.float32)
            raw_feature_cols = [c for c in self.df.columns if c != self.label_col]
        else:
            self.labels = None
            raw_feature_cols = list(self.df.columns)

        # Sort feature columns numerically if they are digits (e.g. '0', '1', ..., '186')
        try:
            self.feature_cols = sorted(raw_feature_cols, key=lambda c: int(c))
        except ValueError:
            self.feature_cols = raw_feature_cols

        # Pre-convert features and labels to torch tensor in memory for 100x faster indexing
        raw_feats = self.df[self.feature_cols].to_numpy(dtype=np.float32)
        self.x_data = torch.from_numpy(raw_feats).unsqueeze(1)  # (N, 1, 187)
        if self.labels is not None:
            self.y_data = torch.from_numpy(self.labels).unsqueeze(1)  # (N, 1)
        else:
            self.y_data = None

    def __len__(self) -> int:
        return len(self.x_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.x_data[idx]
        if self.transform is not None:
            x = self.transform(x)

        if self.y_data is not None:
            y = self.y_data[idx]
            if self.target_transform is not None:
                y = self.target_transform(y)
            return x, y
        else:
            return x, torch.tensor([-1.0], dtype=torch.float32)

    def get_class_counts(self) -> dict:
        """Returns distribution of classes."""
        if self.labels is None:
            return {}
        unique, counts = np.unique(self.labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    def get_pos_weight(self) -> torch.Tensor:
        """
        Computes positive class weight for BCEWithLogitsLoss:
        pos_weight = N_neg / N_pos
        """
        counts = self.get_class_counts()
        n_neg = counts.get(0, 1)
        n_pos = counts.get(1, 1)
        return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    train_transform = None,
    val_transform = None,
    train_csv: str = "train.csv",
    val_csv: str = "val.csv",
    test_csv: str = "test.csv"
) -> Tuple[DataLoader, DataLoader, DataLoader, ECGDataset, ECGDataset, ECGDataset]:
    """
    Constructs PyTorch DataLoaders for train, validation, and test datasets.
    """
    train_path = os.path.join(data_dir, train_csv)
    val_path = os.path.join(data_dir, val_csv)
    test_path = os.path.join(data_dir, test_csv)

    # Fallback search if data_dir has subdirectories
    if not os.path.exists(train_path):
        nested_dir = os.path.join(data_dir, "arrhythmia_dataset")
        if os.path.exists(os.path.join(nested_dir, train_csv)):
            train_path = os.path.join(nested_dir, train_csv)
            val_path = os.path.join(nested_dir, val_csv)
            test_path = os.path.join(nested_dir, test_csv)

    train_ds = ECGDataset(train_path, transform=train_transform)
    val_ds = ECGDataset(val_path, transform=val_transform)
    test_ds = ECGDataset(test_path, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if len(train_ds) > batch_size else False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
