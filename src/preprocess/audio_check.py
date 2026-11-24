"""Warn when an audio file is a poor input for lip sync.

    python -m src.preprocess.audio_check talk.wav

The checks are cheap heuristics, not a speech detector. They only look at the
waveform, so they can be wrong; every result is a warning, never an error.

Warnings (codes): too_long, too_quiet, clipped, noise_like, music_like.
"""
import argparse
import sys
import wave
from dataclasses import dataclass

import numpy as np

MAX_SECONDS = 30.0
QUIET_RMS_DBFS = -40.0   # overall level below this is too quiet
CLIP_LEVEL = 0.99        # |sample| at or above this counts as clipped
CLIP_FRACTION = 0.001    # more than 0.1% clipped samples -> warn
FRAME_MS = 25
HOP_MS = 10
FLATNESS_NOISE = 0.30    # mean spectral flatness above this -> noise
MOD_LOW, MOD_HIGH = 2.0, 8.0   # syllable rate band in Hz
SYLLABLE_SHARE_MIN = 0.12      # envelope power share in that band for speech


@dataclass(frozen=True)
class AudioWarning:
    code: str
    message: str


def load_wav(path):
    """Read a PCM wav file into float samples in [-1, 1] (mono) and a rate."""
    with wave.open(str(path), "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {width} bytes")
    if channels > 1:
        x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    return x, rate


def _frames(x, rate):
    n, hop = int(rate * FRAME_MS / 1000), int(rate * HOP_MS / 1000)
    if len(x) < n:
        return np.zeros((0, n))
    idx = np.arange(0, len(x) - n + 1, hop)[:, None] + np.arange(n)[None, :]
    return x[idx]


def spectral_flatness(x, rate):
    """Mean spectral flatness over frames that carry sound (1 = white noise)."""
    fr = _frames(x, rate)
    if len(fr) == 0:
        return 0.0
    energy = (fr ** 2).mean(axis=1)
    fr = fr[energy > energy.max() * 0.01]
    if len(fr) == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(fr * np.hanning(fr.shape[1]), axis=1)) ** 2 + 1e-12
    flat = np.exp(np.log(spec).mean(axis=1)) / spec.mean(axis=1)
    return float(flat.mean())


def syllable_share(x, rate):
    """Share of energy-envelope power between 2 and 8 Hz (speech rhythm)."""
    fr = _frames(x, rate)
    if len(fr) < 32:
        return 0.0
    env = np.sqrt((fr ** 2).mean(axis=1))
    env = env - env.mean()
    power = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    freqs = np.fft.rfftfreq(len(env), d=HOP_MS / 1000)
    total = power[freqs >= 0.5].sum()
    if total <= 0:
        return 0.0
    band = power[(freqs >= MOD_LOW) & (freqs <= MOD_HIGH)].sum()
    return float(band / total)


def check_audio(x, rate):
    """Return a list of AudioWarning for a mono float signal in [-1, 1]."""
    x = np.asarray(x, dtype=np.float64)
    out = []
    if len(x) == 0 or rate <= 0:
        return [AudioWarning("too_quiet", "Audio is empty.")]

    seconds = len(x) / rate
    if seconds > MAX_SECONDS:
        out.append(AudioWarning(
            "too_long",
            f"Audio is {seconds:.1f} s; only the first {MAX_SECONDS:.0f} s will be used."))

    rms = float(np.sqrt((x ** 2).mean()))
    dbfs = 20 * np.log10(rms) if rms > 0 else -120.0
    if dbfs < QUIET_RMS_DBFS:
        out.append(AudioWarning(
            "too_quiet", f"Audio is very quiet ({dbfs:.0f} dBFS); the mouth may barely move."))

    clipped = float((np.abs(x) >= CLIP_LEVEL).mean())
    if clipped > CLIP_FRACTION:
        out.append(AudioWarning(
            "clipped", f"{clipped * 100:.1f}% of samples are clipped; the audio may be distorted."))

    if rms > 0 and seconds >= 1.0:
        if spectral_flatness(x, rate) > FLATNESS_NOISE:
            out.append(AudioWarning("noise_like", "Audio looks like noise, not speech."))
        elif syllable_share(x, rate) < SYLLABLE_SHARE_MIN:
            out.append(AudioWarning(
                "music_like",
                "Audio has no speech-like rhythm; it may be music or a steady sound."))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Check a wav file before lip sync.")
    p.add_argument("audio", help="path to a PCM .wav file")
    args = p.parse_args(argv)
    x, rate = load_wav(args.audio)
    warnings = check_audio(x, rate)
    for w in warnings:
        print(f"warning [{w.code}]: {w.message}")
    if not warnings:
        print("no warnings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
