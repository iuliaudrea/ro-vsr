"""
Utility extracted from MultiVSR's `dataloader.py`.

Only `subsequent_mask` is needed for inference; the rest of the original
file contains training-related logic (Datasets, augmentations) that we
don't need here.
"""

import numpy as np
import torch


def subsequent_mask(size: int) -> torch.Tensor:
    """Upper-triangular mask for the decoder's causal attention."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0
