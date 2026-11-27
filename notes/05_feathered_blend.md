# Feathered mouth blend

The patched Wav2Lip pastes the generated mouth with a hard rectangle. `src/inference/blend.py`
adds a feathered paste (`feather_paste`) whose mask is 0 on the box border and ramps to 1
`feather` pixels in, plus a seam metric (`seam_metric`).

The seam metric is the mean absolute step between the pixels just inside and just outside
the box border, next to the same step one pixel further out (what the frame does anyway).
A ratio near 1 means no extra seam.

## Measured (scripts/compare_blend.py)

Synthetic frame (flat grey with light noise, 96x96) and a flat bright patch (32x40).
These numbers show how the metric reacts, not how a real face looks.

| feather (px) | step  | baseline | ratio |
|--------------|-------|----------|-------|
| 0 (hard)     | 80.02 | 1.64     | 48.76 |
| 2            | 1.60  | 1.64     | 0.98  |
| 4            | 1.60  | 1.64     | 0.98  |
| 8            | 1.60  | 1.64     | 0.98  |
| 12           | 1.60  | 1.64     | 0.98  |

Any feather above 0 removes the step at the border because the outermost mask value is 0.
A wider feather only spreads the change over more pixels, which this metric cannot tell apart.

## Not verified

- The feathered paste is not wired into the Wav2Lip `inference.py` patch yet.
- No real photo, video or model weights were used, so the visual result is unjudged.
