"""Temporal smoothing for a sequence of mouth patches or frames.

Two filters are offered, both working on NumPy arrays of any shape:

* ``fixed_blend``: the fixed blend that the patched Wav2Lip ``--smooth_alpha`` uses,
  ``out[t] = alpha * x[t] + (1 - alpha) * out[t-1]``.
* ``OneEuroFilter``: an adaptive filter (Casiez et al., 2012). It smooths hard when the
  signal is nearly still and eases off when it moves fast, so quick lip closures are
  not flattened as much as with a fixed blend.
"""
import math

import numpy as np


def fixed_blend(frames, alpha):
    """Blend each frame with the previous output. ``alpha`` of 1.0 means no smoothing."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be above 0 and at most 1.0")
    frames = np.asarray(frames, dtype=np.float32)
    out = frames.copy()
    for t in range(1, len(out)):
        out[t] = alpha * frames[t] + (1.0 - alpha) * out[t - 1]
    return out


def _alpha_for_cutoff(cutoff, rate):
    """Smoothing factor of a low-pass filter with this cutoff (Hz) at this frame rate."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau * rate)


class OneEuroFilter:
    """One-euro filter for array-valued samples arriving at a fixed frame rate.

    min_cutoff: lower values smooth a still signal more (Hz).
    beta: higher values let fast movement through with less lag.
    d_cutoff: cutoff used to smooth the speed estimate (Hz).
    """

    def __init__(self, rate=25.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        if rate <= 0 or min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("rate, min_cutoff and d_cutoff must be positive")
        if beta < 0:
            raise ValueError("beta must not be negative")
        self.rate = float(rate)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = None
        self._dx = None

    def reset(self):
        self._x = None
        self._dx = None

    def __call__(self, sample):
        x = np.asarray(sample, dtype=np.float32)
        if self._x is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            return x.copy()
        speed = (x - self._x) * self.rate
        a_d = _alpha_for_cutoff(self.d_cutoff, self.rate)
        self._dx = a_d * speed + (1.0 - a_d) * self._dx
        # One cutoff for the whole sample, from the average speed, so the patch moves together.
        cutoff = self.min_cutoff + self.beta * float(np.abs(self._dx).mean())
        a = _alpha_for_cutoff(cutoff, self.rate)
        self._x = a * x + (1.0 - a) * self._x
        return self._x.copy()


def one_euro(frames, rate=25.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
    """Run a OneEuroFilter over a whole sequence of frames."""
    frames = np.asarray(frames, dtype=np.float32)
    f = OneEuroFilter(rate, min_cutoff, beta, d_cutoff)
    return np.stack([f(x) for x in frames]) if len(frames) else frames.copy()
