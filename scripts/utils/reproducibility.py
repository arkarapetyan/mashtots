"""Utilities for reproducible training and inference runs."""

import os
import random

import lightning as L
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and Lightning worker RNGs."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    L.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Seed a data loader worker from PyTorch's worker-specific seed."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    """Create a seeded generator for deterministic data loader ordering."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
