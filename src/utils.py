"""Shared utilities: seeding, device selection, checkpoint I/O."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location)


def cycle(iterable):
    """Cycle through an iterable indefinitely (used to zip source & target loaders)."""
    while True:
        for x in iterable:
            yield x


def entropy(p: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Per-sample Shannon entropy of a probability distribution.

    Args:
        p: Tensor of shape (B, C) containing class probabilities.
    Returns:
        Tensor of shape (B,).
    """
    return -torch.sum(p * torch.log(p + eps), dim=1)
