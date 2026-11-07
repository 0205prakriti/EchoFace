# Experiment: reducing mouth flicker with temporal smoothing

## Problem
Running Wav2Lip on a still photo (notebook 02) gave a mouth that looked jerky and flickered between frames.

## Why it happens
Wav2Lip predicts each frame's mouth on its own, from a short audio window (`inference.py`, the loop that calls `model(mel_batch, img_batch)`). Nothing ties frame N to frame N-1, so small differences in the prediction show up as flicker.

## Change
`--smooth_alpha` (added by the notebook's patch cell) blends each predicted mouth patch with the previous output:

```
patch[t] = alpha * prediction[t] + (1 - alpha) * patch[t-1]
```

`alpha = 1.0` is the original behaviour. Lower values are steadier.

## Method
- Model: real Wav2Lip weights, run on CPU, no fine-tuning.
- Input: one synthetic cartoon face, plus 6 seconds of synthetic speech-like audio (syllable-rate bursts with pauses).
- Face box fixed by hand, because face detection weights could not be downloaded in the test environment.
- Metric, over the mouth region of the final video (`scripts/measure_mouth_jitter.py`):
  - **jitter**: mean absolute pixel change between neighbouring frames
  - **accel**: mean absolute second difference (spikes read as flicker)
  - **motion**: standard deviation of each pixel over the clip, i.e. how much the mouth really moves

## Results (147 frames)

| Model | alpha | Jitter | Accel | Motion | Jitter / motion |
|---|---|---|---|---|---|
| GAN | 1.0 (off) | 3.025 | 4.573 | 4.927 | 0.614 |
| GAN | 0.7 | 2.238 | 3.082 | 4.177 | 0.536 |
| GAN | 0.5 | 1.617 | 2.059 | 3.557 | 0.455 |
| Non-GAN | 1.0 (off) | 1.988 | 2.882 | 3.288 | 0.604 |
| Non-GAN | 0.5 | 1.058 | 1.333 | 2.380 | 0.445 |

## What this shows
- Smoothing lowers jitter and the spikes in it: alpha 0.7 cut jitter by 26%, alpha 0.5 by 47%.
- It also flattens the mouth movement (motion fell 15% and 28%), so there is a trade-off between steady and expressive.
- The non-GAN checkpoint looks calmer mainly because it moves less. Its jitter relative to its movement (0.604) is about the same as the GAN model's (0.614), so switching checkpoints is not a fix on its own.
- Relative to movement, smoothing is what improves things (0.61 down to 0.54 at alpha 0.7).

## Limits of this experiment
- The face and the audio are synthetic. Real speech and real faces may behave differently.
- The metric measures pixel change. It does not measure what looks natural, and part of the measured change comes from video compression.
- Smoothing adds a small lag, so lip sync could drift slightly at low alpha. I did not measure sync.
- One run per setting, so no error bars.

## Next
- Repeat on real photos and recorded speech, and judge by eye as well as by the numbers.
- Measure sync (for example with the SyncNet confidence score) at each alpha.
- Check whether the intermediate DIVX video that Wav2Lip writes before the final encode adds its own flicker.
- Try smoothing in a better place, such as before pasting, with a motion-aware blend.
