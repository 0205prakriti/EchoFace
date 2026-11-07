"""Measure how much the mouth region changes from frame to frame in one or more videos.

Lower jitter means a steadier mouth. Compare it with "motion", which says how much the mouth actually moves,
because heavy smoothing lowers both.

Usage:
  python scripts/measure_mouth_jitter.py out_a.mp4 out_b.mp4
  python scripts/measure_mouth_jitter.py out.mp4 --region 145 190 95 162   # y1 y2 x1 x2 in pixels

By default the region is the lower-centre of the frame (rows 57 to 74 percent, columns 37 to 63 percent),
which suits a front-facing face that fills the frame. Pass --region for other photos.
"""
import argparse

import cv2
import numpy as np


def load_region(path, region):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        y1, y2, x1, x2 = region or (int(h * 0.57), int(h * 0.74), int(w * 0.37), int(w * 0.63))
        frames.append(frame[y1:y2, x1:x2].astype(np.float32))
    cap.release()
    if len(frames) < 3:
        raise ValueError(f"{path}: need at least 3 frames, found {len(frames)}")
    return np.array(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--region", nargs=4, type=int, metavar=("Y1", "Y2", "X1", "X2"))
    args = parser.parse_args()

    print(f"{'video':<28}{'frames':>7}{'jitter':>9}{'accel':>9}{'motion':>9}{'jitter/motion':>15}")
    for path in args.videos:
        f = load_region(path, args.region)
        jitter = np.abs(np.diff(f, axis=0)).mean()          # average change between neighbouring frames
        accel = np.abs(np.diff(f, n=2, axis=0)).mean()      # change in the change: spikes read as flicker
        motion = f.std(axis=0).mean()                       # how far the mouth moves over the clip
        print(f"{path[-27:]:<28}{len(f):>7}{jitter:>9.3f}{accel:>9.3f}{motion:>9.3f}{jitter / motion:>15.3f}")


if __name__ == "__main__":
    main()
