"""Compare the fixed blend with the one-euro filter on the mouth region of a finished video.

The filters are applied after the fact to the mouth region of each frame, so both ideas see the same
input. This is not the same as running them inside Wav2Lip.

Usage:
  python scripts/compare_smoothing.py out.mp4
  python scripts/compare_smoothing.py out.mp4 --region 145 190 95 162   # y1 y2 x1 x2 in pixels
"""
import argparse
import pathlib
import sys

import cv2

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_mouth_jitter import load_region  # noqa: E402
from src.inference import smoothing  # noqa: E402

# (label, function) pairs; every function takes the frames and the frame rate.
SETTINGS = [
    ("none", lambda f, r: f),
    ("blend alpha 0.7", lambda f, r: smoothing.fixed_blend(f, 0.7)),
    ("blend alpha 0.5", lambda f, r: smoothing.fixed_blend(f, 0.5)),
    ("one-euro mc 1.0 b 0.05", lambda f, r: smoothing.one_euro(f, r, 1.0, 0.05)),
    ("one-euro mc 2.0 b 0.05", lambda f, r: smoothing.one_euro(f, r, 2.0, 0.05)),
    ("one-euro mc 1.0 b 0.5", lambda f, r: smoothing.one_euro(f, r, 1.0, 0.5)),
    ("one-euro mc 2.0 b 0.5", lambda f, r: smoothing.one_euro(f, r, 2.0, 0.5)),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--region", nargs=4, type=int, metavar=("Y1", "Y2", "X1", "X2"))
    args = parser.parse_args(argv)

    cap = cv2.VideoCapture(args.video)
    rate = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    frames = load_region(args.video, args.region)

    print(f"{len(frames)} frames at {rate:.2f} fps")
    print("| Filter | Jitter | Accel | Motion | Jitter / motion |")
    print("|---|---|---|---|---|")
    for label, fn in SETTINGS:
        m = smoothing.jitter_metrics(fn(frames, rate))
        print(f"| {label} | {m['jitter']:.3f} | {m['accel']:.3f} | {m['motion']:.3f} | {m['ratio']:.3f} |")


if __name__ == "__main__":
    main()
