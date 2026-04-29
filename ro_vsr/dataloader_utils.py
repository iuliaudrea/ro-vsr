"""
Utilitare extrase din `dataloader.py` al MultiVSR.

Doar `subsequent_mask` este necesar pentru beam search; restul fișierului
original conține logică de antrenare (Dataset-uri, augmentări) de care
nu avem nevoie la inferență.
"""

import numpy as np
import torch


def subsequent_mask(size: int) -> torch.Tensor:
    """Mască upper-triangulară pentru atenția cauzală a decoder-ului."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0
