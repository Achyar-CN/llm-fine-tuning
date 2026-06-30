"""Reproducibility: set random seeds across Python, NumPy, and PyTorch."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
