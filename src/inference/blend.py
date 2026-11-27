"""Paste a generated mouth patch into a frame with a feathered edge.

The patched Wav2Lip pastes the patch with a hard rectangle (``f[y1:y2, x1:x2] = p``),
which can leave a visible seam where the patch meets the original frame. Here the
patch is mixed in with a mask that ramps from 0 at the box edge to 1 a few pixels in.
"""
import numpy as np


def feather_mask(height, width, feather):
    """Mask of shape (height, width), 0 on the outer border, 1 from ``feather`` pixels in.

    ``feather`` is limited to half the shorter side so a small box still has a core of 1s
    where it can (the ramps meet in the middle for very small boxes).
    """
    if height < 1 or width < 1:
        raise ValueError("height and width must be at least 1")
    if feather < 0:
        raise ValueError("feather must not be negative")
    feather = int(min(feather, (min(height, width) - 1) // 2))
    if feather == 0:
        return np.ones((height, width), dtype=np.float32)

    def ramp(n):
        # distance from the nearest edge, scaled so the outer pixel is 0 and `feather` in is 1
        d = np.minimum(np.arange(n), np.arange(n)[::-1]).astype(np.float32)
        return np.clip(d / feather, 0.0, 1.0)

    return np.minimum(ramp(height)[:, None], ramp(width)[None, :])


def feather_paste(frame, patch, box, feather=8):
    """Return a copy of ``frame`` with ``patch`` mixed in over ``box`` = (y1, y2, x1, x2).

    The patch must already have the size of the box. ``feather=0`` gives the old hard paste.
    """
    y1, y2, x1, x2 = box
    frame = np.asarray(frame)
    patch = np.asarray(patch)
    h, w = y2 - y1, x2 - x1
    if h < 1 or w < 1 or y1 < 0 or x1 < 0 or y2 > frame.shape[0] or x2 > frame.shape[1]:
        raise ValueError("box must lie inside the frame and not be empty")
    if patch.shape[:2] != (h, w):
        raise ValueError(f"patch is {patch.shape[:2]} but the box is {(h, w)}")
    if patch.ndim != frame.ndim:
        raise ValueError("patch and frame must have the same number of dimensions")
    out = frame.copy()
    if feather == 0:
        out[y1:y2, x1:x2] = patch
        return out
    mask = feather_mask(h, w, feather)
    if frame.ndim == 3:
        mask = mask[:, :, None]
    mixed = mask * patch.astype(np.float32) + (1.0 - mask) * frame[y1:y2, x1:x2].astype(np.float32)
    if np.issubdtype(frame.dtype, np.integer):
        mixed = np.rint(mixed)
        mixed = np.clip(mixed, np.iinfo(frame.dtype).min, np.iinfo(frame.dtype).max)
    out[y1:y2, x1:x2] = mixed.astype(frame.dtype)
    return out
