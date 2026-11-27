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


def seam_metric(frame, box):
    """How visible the border of ``box`` = (y1, y2, x1, x2) is in ``frame``.

    ``step`` is the mean absolute difference between the pixels just inside the border and
    the pixels just outside it. ``baseline`` is the same difference one pixel further out
    (outside vs outside), which is what the frame does anyway. ``ratio`` is step / baseline:
    close to 1 means no seam beyond the picture's own texture. The box needs at least
    two pixels of frame around it.
    """
    y1, y2, x1, x2 = box
    frame = np.asarray(frame, dtype=np.float32)
    h, w = frame.shape[:2]
    if y1 < 2 or x1 < 2 or y2 > h - 2 or x2 > w - 2 or y2 - y1 < 2 or x2 - x1 < 2:
        raise ValueError("box must be at least 2 pixels away from the frame edge and 2 pixels in size")

    def mean_abs(a, b):
        return float(np.abs(a - b).mean())

    pairs = [
        (frame[y1, x1:x2], frame[y1 - 1, x1:x2], frame[y1 - 2, x1:x2]),
        (frame[y2 - 1, x1:x2], frame[y2, x1:x2], frame[y2 + 1, x1:x2]),
        (frame[y1:y2, x1], frame[y1:y2, x1 - 1], frame[y1:y2, x1 - 2]),
        (frame[y1:y2, x2 - 1], frame[y1:y2, x2], frame[y1:y2, x2 + 1]),
    ]
    step = float(np.mean([mean_abs(i, o1) for i, o1, _ in pairs]))
    baseline = float(np.mean([mean_abs(o1, o2) for _, o1, o2 in pairs]))
    ratio = step / baseline if baseline > 0 else (1.0 if step == 0 else float("inf"))
    return {"step": step, "baseline": baseline, "ratio": ratio}
