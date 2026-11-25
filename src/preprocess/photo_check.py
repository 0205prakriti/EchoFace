"""Warn when a photo is a poor input for lip sync.

    python -m src.preprocess.photo_check face.jpg

The checks are cheap heuristics on the pixels, so they can be wrong; every
result is a warning, never an error.

Warnings (codes): too_small, blurry, too_dark, overexposed, no_face, many_faces.
"""
import argparse
import sys
from dataclasses import dataclass

import cv2
import numpy as np

MIN_SIDE = 256            # shorter side below this (pixels) is too small
BLUR_VAR = 60.0           # variance of the Laplacian below this is blurry
DARK_MEAN = 60.0          # mean gray level (0-255) below this is too dark
BRIGHT_MEAN = 200.0       # mean gray level above this is overexposed
CLIP_FRACTION = 0.25      # share of near-white pixels above this is overexposed
CLIP_LEVEL = 250
BLUR_SIDE = 512           # blur is measured on a copy scaled to this shorter side


@dataclass(frozen=True)
class PhotoWarning:
    code: str
    message: str


def _gray(img):
    img = np.asarray(img)
    if img.ndim == 2:
        return img.astype(np.float64)
    if img.ndim == 3 and img.shape[2] >= 3:
        # uint8 BGR as loaded by OpenCV; weights are the usual luma ones.
        b, g, r = img[..., 0], img[..., 1], img[..., 2]
        return 0.114 * b + 0.587 * g + 0.299 * r
    raise ValueError(f"unsupported image shape: {img.shape}")


def blur_score(img):
    """Variance of the Laplacian on a size-normalised copy (low = blurry)."""
    gray = _gray(img).astype(np.float32)
    short = min(gray.shape)
    if short > BLUR_SIDE:
        scale = BLUR_SIDE / short
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def check_image(img):
    """Return a list of PhotoWarning about size, sharpness and exposure."""
    gray = _gray(img)
    out = []
    h, w = gray.shape
    if min(h, w) < MIN_SIDE:
        out.append(PhotoWarning(
            "too_small", f"Photo is {w}x{h}; the shorter side should be at least {MIN_SIDE} px."))

    score = blur_score(img)
    if score < BLUR_VAR:
        out.append(PhotoWarning(
            "blurry", f"Photo looks blurry (sharpness {score:.0f}); the mouth may look smeared."))

    mean = float(gray.mean())
    if mean < DARK_MEAN:
        out.append(PhotoWarning("too_dark", f"Photo is dark (mean level {mean:.0f} of 255)."))
    elif mean > BRIGHT_MEAN or float((gray >= CLIP_LEVEL).mean()) > CLIP_FRACTION:
        out.append(PhotoWarning("overexposed", "Photo is overexposed; details are washed out."))
    return out


def haar_detector(img):
    """Default face detector: OpenCV's Haar cascade. Returns a list of boxes.

    Returns None when the cascade file is not available, so the caller can skip
    the face check instead of reporting a false "no face".
    """
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        return None
    gray = np.clip(_gray(img), 0, 255).astype(np.uint8)
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    return [tuple(int(v) for v in b) for b in boxes]


def check_faces(img, detector=haar_detector):
    """Warn about no face or several faces. `detector(img)` returns a list of boxes
    (any items), or None if it could not run, in which case nothing is reported."""
    boxes = detector(img)
    if boxes is None:
        return []
    if len(boxes) == 0:
        return [PhotoWarning("no_face", "No face found; use a clear, front-facing photo.")]
    if len(boxes) > 1:
        return [PhotoWarning(
            "many_faces", f"{len(boxes)} faces found; only one person should be in the photo.")]
    return []


def check_photo(img, detector=haar_detector):
    """All checks: pixels first, then faces."""
    return check_image(img) + check_faces(img, detector)


def main(argv=None):
    p = argparse.ArgumentParser(description="Check a photo before lip sync.")
    p.add_argument("photo", help="path to an image file")
    args = p.parse_args(argv)
    img = cv2.imread(args.photo)
    if img is None:
        print(f"error: could not read image: {args.photo}", file=sys.stderr)
        return 2
    warnings = check_photo(img)
    for w in warnings:
        print(f"warning [{w.code}]: {w.message}")
    if not warnings:
        print("no warnings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
