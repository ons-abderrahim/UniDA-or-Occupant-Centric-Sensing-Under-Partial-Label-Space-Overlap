
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Optional, Tuple, Sequence




def _load_raw(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (X, y, feature_names) from a CSV file.

    Works for both occupancy CSVs (label col = 'labels') and
    activity CSVs (label col = 'activity').
    """
    df = pd.read_csv(path)

    # Detect label column
    if "labels" in df.columns:
        label_col = "labels"
    elif "activity" in df.columns:
        label_col = "activity"
    else:
        raise ValueError(f"Cannot find a label column in {path}")

    # Drop non-feature columns
    drop_cols = {"time", "year_day", label_col}
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feat_cols].values.astype(np.float32)
    y = df[label_col].values
    return X, y, feat_cols


def _make_windows(X: np.ndarray, y: np.ndarray,
                  seq_len: int = 1, stride: int = 1
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a window of `seq_len` over (X, y).

    Each window label is the label of its *last* time step.
    For seq_len=1 the function is a no-op (add a channel dim).
    """
    if seq_len == 1:
        # shape: (N, C, 1)
        return X[:, :, np.newaxis].transpose(0, 1, 2), y

    windows, labels = [], []
    for i in range(0, len(X) - seq_len + 1, stride):
        windows.append(X[i: i + seq_len])   # (seq_len, C)
        labels.append(y[i + seq_len - 1])
    W = np.stack(windows, axis=0)            # (N, seq_len, C)
    W = W.transpose(0, 2, 1)               # (N, C, seq_len)
    return W.astype(np.float32), np.array(labels)


def _encode_labels(y: np.ndarray,
                   known_classes: Sequence,
                   unknown_label: int = -1) -> np.ndarray:
    """Map raw label values to contiguous ints.

    Known classes  → 0, 1, 2, … (C_known - 1)
    Unknown classes → unknown_label  (default -1)
    """
    known_set = set(known_classes)
    le_map = {cls: idx for idx, cls in enumerate(sorted(known_set))}
    out = np.full(len(y), unknown_label, dtype=np.int64)
    for i, yi in enumerate(y):
        if yi in le_map:
            out[i] = le_map[yi]
    return out


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SmartBuildingDataset(Dataset):
    """PyTorch Dataset for one domain (source or target).

    Args:
        path         : path to the CSV file.
        known_classes: iterable of raw label values treated as known (shared).
        seq_len      : sliding window length (time steps).
        stride       : stride of the sliding window.
        scaler       : fitted StandardScaler (supply for target domain to
                       avoid data leakage; leave None to fit on this domain).
        unknown_label: integer used to mark target-private samples (-1).
        is_source    : if True, drop all unknown-class rows entirely
                       (source data should not contain unknowns).
    """

    def __init__(
        self,
        path: str,
        known_classes: Sequence,
        seq_len: int = 10,
        stride: int = 1,
        scaler: Optional[StandardScaler] = None,
        unknown_label: int = -1,
        is_source: bool = True,
    ):
        X_raw, y_raw, self.feature_names = _load_raw(path)

        # Normalise
        if scaler is None:
            self.scaler = StandardScaler()
            X_norm = self.scaler.fit_transform(X_raw)
        else:
            self.scaler = scaler
            X_norm = scaler.transform(X_raw)

        # Encode labels
        y_enc = _encode_labels(y_raw, known_classes, unknown_label)

        # For the source domain, remove rows with unknown labels
        if is_source:
            mask = y_enc != unknown_label
            X_norm = X_norm[mask]
            y_enc = y_enc[mask]

        # Build windows  →  (N, C, seq_len)
        self.X, self.y = _make_windows(X_norm, y_enc, seq_len, stride)

        self.num_features = self.X.shape[1]
        self.seq_len = self.X.shape[2]
        self.num_known_classes = len(set(known_classes))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def build_dataloaders(
    source_path: str,
    target_path: str,
    known_classes: Sequence,
    seq_len: int = 10,
    stride: int = 1,
    batch_size: int = 64,
    num_workers: int = 0,
    unknown_label: int = -1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (source_loader, target_train_loader, target_eval_loader).

    The source scaler is reused on the target to prevent leakage.
    target_train_loader has shuffle=True for training;
    target_eval_loader has shuffle=False for evaluation.
    """
    src_ds = SmartBuildingDataset(
        source_path, known_classes,
        seq_len=seq_len, stride=stride,
        is_source=True,
        unknown_label=unknown_label,
    )
    tgt_ds = SmartBuildingDataset(
        target_path, known_classes,
        seq_len=seq_len, stride=stride,
        scaler=src_ds.scaler,
        is_source=False,
        unknown_label=unknown_label,
    )
    tgt_eval_ds = SmartBuildingDataset(
        target_path, known_classes,
        seq_len=seq_len, stride=stride,
        scaler=src_ds.scaler,
        is_source=False,
        unknown_label=unknown_label,
    )

    src_loader = DataLoader(src_ds, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            drop_last=True)
    tgt_eval_loader = DataLoader(tgt_eval_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

    return src_loader, tgt_loader, tgt_eval_loader, src_ds.num_features, src_ds.seq_len
