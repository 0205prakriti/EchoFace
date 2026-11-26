# Experiment: adaptive (one-euro) smoothing next to the fixed blend

## Idea
The fixed blend in `--smooth_alpha` smooths every frame the same amount, so it flattens real mouth movement along with the flicker. A one-euro filter changes its strength with the speed of the signal: more smoothing when the mouth is nearly still, less when it moves fast. Both are in `src/inference/smoothing.py`.

## Method
- Input: `samples/wav2lip_baseline_result.mp4` (508 frames, 23.98 fps, 768x432), the baseline Wav2Lip output already in the repo.
- Both filters were applied after the fact to the mouth region of each frame (default region of `scripts/measure_mouth_jitter.py`), so they see identical input. This is not the same as running them inside Wav2Lip.
- Command: `python scripts/compare_smoothing.py samples/wav2lip_baseline_result.mp4`
- Metrics are the same as in note 03. `mc` is the filter's min_cutoff in Hz, `b` is beta.

## Results

| Filter | Jitter | Accel | Motion | Jitter / motion |
|---|---|---|---|---|
| none | 4.310 | 5.650 | 27.911 | 0.154 |
| blend alpha 0.7 | 3.454 | 3.669 | 27.107 | 0.127 |
| blend alpha 0.5 | 2.848 | 2.515 | 26.410 | 0.108 |
| one-euro mc 1.0 b 0.05 | 3.080 | 2.964 | 26.994 | 0.114 |
| one-euro mc 2.0 b 0.05 | 3.197 | 3.173 | 27.072 | 0.118 |
| one-euro mc 1.0 b 0.5 | 3.963 | 4.800 | 27.733 | 0.143 |
| one-euro mc 2.0 b 0.5 | 3.975 | 4.828 | 27.736 | 0.143 |

## What this shows
- At about the same mouth movement (motion 26.99 vs 27.11), one-euro with mc 1.0 and b 0.05 has lower jitter (3.080 vs 3.454) and lower accel (2.964 vs 3.669) than the fixed blend at alpha 0.7.
- Blend alpha 0.5 reaches the lowest jitter / motion (0.108) but also loses the most movement (26.410).
- A larger beta lets fast movement through but also lets most of the flicker through: with b 0.5 the filter barely helps (jitter 3.96 vs 4.31).

## Limits
- One video, one run per setting, and no parameter search beyond these seven settings.
- The filters ran on already-pasted, compressed frames. Smoothing the raw mouth patch inside Wav2Lip could behave differently.
- The metrics measure pixel change, not how natural the mouth looks, and lip sync was not measured.
- The one-euro filter is not wired into `src.inference.run` or the Wav2Lip patch yet. It is only a module and a comparison script.
