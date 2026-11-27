"""Compare the hard paste with feathered pastes on a synthetic frame and print the seam numbers.

This uses a made-up flat frame with light noise and a flat bright patch, so it shows how the
seam metric responds to feathering, not how a real face looks.

Usage:
  python scripts/compare_blend.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.inference.blend import feather_paste, seam_metric  # noqa: E402


def main():
    rng = np.random.RandomState(0)
    frame = np.clip(120 + rng.randint(-2, 3, (96, 96, 3)), 0, 255).astype(np.uint8)
    patch = np.full((32, 40, 3), 200, dtype=np.uint8)
    box = (40, 72, 28, 68)
    print(f"{'feather':>8}{'step':>10}{'baseline':>10}{'ratio':>9}")
    for feather in (0, 2, 4, 8, 12):
        m = seam_metric(feather_paste(frame, patch, box, feather), box)
        print(f"{feather:>8}{m['step']:>10.2f}{m['baseline']:>10.2f}{m['ratio']:>9.2f}")


if __name__ == "__main__":
    main()
