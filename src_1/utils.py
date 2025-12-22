"""Utility helpers for PPO training."""

from __future__ import annotations

import os
import random
from typing import Iterable, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed random generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy observation to torch tensor with batch dimension."""
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute explained variance for diagnostics."""
    var_y = torch.var(y_true)
    return 0.0 if var_y == 0 else float(1 - torch.var(y_true - y_pred) / var_y)


def batch_generator(indices: Iterable[int], batch_size: int) -> Iterable[np.ndarray]:
    """Yield slices of indices."""
    indices = list(indices)
    for start in range(0, len(indices), batch_size):
        yield np.array(indices[start : start + batch_size])


__all__ = ["set_seed", "to_tensor", "ensure_dir", "explained_variance"]
