"""
Four preprocessing strategies for LRRo clips.

Each function reads a folder of LRRo grayscale frames (numbered .jpg files,
typically 64x64) and returns a tensor of shape (3, T, 96, 96) ready for
the VTP visual encoder.

The four strategies differ in how the 64x64 LRRo frames are placed onto
the 96x96 canvas expected by the encoder. They are part of an ablation
study reported in the paper.

Best-performing strategy: `load_lrro_clip_64_bottom`.
"""

import glob
import os
from typing import Optional

import numpy as np
from PIL import Image


CANVAS_SIZE = 96
GRAY_PAD_VALUE = 0.5  # mid-gray padding


def _list_jpgs(clip_dir: str):
    """List the .jpg files in a clip folder, sorted numerically."""
    return sorted(
        glob.glob(os.path.join(clip_dir, "*.jpg")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
    )


def _frames_to_tensor(frames):
    """Stack a list of 3-channel frames into the shape (3, T, 96, 96)."""
    video = np.stack(frames, axis=0)            # (T, 3, 96, 96)
    video = np.transpose(video, (1, 0, 2, 3))   # (3, T, 96, 96)
    return video



def load_lrro_clip_64_bottom(clip_dir: str) -> Optional[np.ndarray]:
    """
    Strategy '64_bottom'.
    Keep 64x64 resolution; place center-bottom on a 96x96 canvas.
    """
    img_size = 64
    x_off = (CANVAS_SIZE - img_size) // 2  # 16
    y_off = CANVAS_SIZE - img_size          # 32

    jpgs = _list_jpgs(clip_dir)
    if not jpgs:
        return None

    frames = []
    for path in jpgs:
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0

        canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), GRAY_PAD_VALUE, dtype=np.float32)
        canvas[y_off:y_off + img_size, x_off:x_off + img_size] = arr

        frames.append(np.stack([canvas, canvas, canvas], axis=0))

    return _frames_to_tensor(frames)


def load_lrro_clip_64_middle(clip_dir: str) -> Optional[np.ndarray]:
    """
    Strategy '64_middle'.
    Keep 64x64 resolution; place exactly center on a 96x96 canvas.
    """
    img_size = 64
    x_off = (CANVAS_SIZE - img_size) // 2  # 16
    y_off = (CANVAS_SIZE - img_size) // 2  # 16

    jpgs = _list_jpgs(clip_dir)
    if not jpgs:
        return None

    frames = []
    for path in jpgs:
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0

        canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), GRAY_PAD_VALUE, dtype=np.float32)
        canvas[y_off:y_off + img_size, x_off:x_off + img_size] = arr

        frames.append(np.stack([canvas, canvas, canvas], axis=0))

    return _frames_to_tensor(frames)


def load_lrro_clip_96_resize(clip_dir: str) -> Optional[np.ndarray]:
    """
    Strategy '96_resize'.
    Resize each frame directly to 96x96 (no padding, fills the whole canvas).
    """
    jpgs = _list_jpgs(clip_dir)
    if not jpgs:
        return None

    frames = []
    for path in jpgs:
        img = Image.open(path).convert("L").resize((CANVAS_SIZE, CANVAS_SIZE), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        frames.append(np.stack([arr, arr, arr], axis=0))

    return _frames_to_tensor(frames)


PREPROCESSING_FNS = {
    "64_bottom": load_lrro_clip_64_bottom,
    "64_middle": load_lrro_clip_64_middle,
    "96_resize": load_lrro_clip_96_resize,
}
