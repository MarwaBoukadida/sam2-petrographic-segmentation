# -*- coding: utf-8 -*-
"""Logging, device selection, random seeds and directory utilities."""

import os
import logging
import random
from typing import Optional

import numpy as np
import torch

RANDOM_SEED = 42

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up console logging and optional file logging if no handlers exist."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            fh.setLevel(getattr(logging, level.upper(), logging.INFO))
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger

def get_device() -> torch.device:
    """Choose CUDA, MPS or CPU, in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy and PyTorch; set deterministic cuDNN flags when CUDA is available."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def format_time(seconds: float) -> str:
    """Format elapsed seconds as hours, minutes and seconds."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"

def ensure_dir(path: str) -> str:
    """Create a directory if needed and return its path."""
    os.makedirs(path, exist_ok=True)
    return path
