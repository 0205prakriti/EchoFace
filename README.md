# EchoFace

Turn one photo and one audio clip into a lip-synced talking video.

EchoFace wraps a pretrained [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) model with a
one-command runner, input checks and mouth smoothing. The longer-term goal is a custom
audio-visual model with its own lip-sync discriminator, built up in stages.

**Status:** early. The Wav2Lip baseline runs end to end. Everything else below is marked as
working, partly working or not started.

## What works today

| Part | State | Where |
|---|---|---|
| Wav2Lip baseline on a Colab T4 | Works. Result video is in the repo. | `notebooks/01_wav2lip_baseline.ipynb`, `samples/wav2lip_baseline_result.mp4` |
| Own photo and audio on current Colab | Works. Patches four lines in Wav2Lip instead of pinning old `numpy` and `librosa`. | `notebooks/02_wav2lip_own_photo.ipynb` |
| Idempotent Wav2Lip patch | Works. Fails loudly if Wav2Lip's code is not what it expects. | `src/inference/patch_wav2lip.py` |
| One-command runner | Works up to the Wav2Lip call (dry run and input cleaning are tested). The full run needs the checkpoint, which is not in the repo. | `src/inference/run.py` |
| Audio check | Works as a heuristic: warns about long, quiet, clipped, noise-like or music-like audio. | `src/preprocess/audio_check.py` |
| Photo check | Works as a heuristic: warns about small, blurry, dark or overexposed photos, and no face or several faces. | `src/preprocess/photo_check.py` |
| Fixed blend (`--smooth`) | Works, wired into the runner. | `src/inference/patch_wav2lip.py` |
| One-euro smoothing | Module and comparison script only. Not wired into the runner. | `src/inference/smoothing.py` |
| Feathered mouth blend | Module, seam metric and comparison script only. Not wired into the Wav2Lip patch. | `src/inference/blend.py` |
| Tests and CI | Tests run on CPU without model weights, on every push. | `tests/`, `.github/workflows/tests.yml` |

Not started: a better generator, a trained sync discriminator, real-time inference, a demo
app and a batch evaluation harness (see the roadmap).

## Quick start

You need Python 3.10 or newer, `git` and `ffmpeg`. On a machine without a GPU the Wav2Lip
step is slow but works.

```bash
git clone https://github.com/0205prakriti/echoface
cd echoface
pip install -r requirements.txt
```

Download the `wav2lip_gan.pth` checkpoint (the steps are in
`notebooks/02_wav2lip_own_photo.ipynb`) and place it at `Wav2Lip/checkpoints/wav2lip_gan.pth`.
Then:

```bash
# Print every command without running anything
python -m src.inference.run --photo face.jpg --audio talk.m4a --out result.mp4 --dry-run

# Run it. Clones Wav2Lip at a pinned commit if the folder is missing.
python -m src.inference.run --photo face.jpg --audio talk.m4a --out result.mp4 --smooth 0.7
```

`--smooth` is the blend weight of the new mouth patch: `1.0` is off, `0.7` is the default,
`0.5` is very steady. The runner copies the inputs to plain file names, converts the audio to
16 kHz mono, keeps the first 30 seconds and calls Wav2Lip without a shell.

Check your inputs first if the result looks bad:

```bash
python -m src.preprocess.photo_check face.jpg
python -m src.preprocess.audio_check talk.wav
```

Run the tests:

```bash
pip install pytest
python -m pytest -q
```

## How it fits together

```
 photo ──► photo_check ─┐
                        ├─► run.py ─► patch_wav2lip ─► Wav2Lip inference.py ─► mp4
 audio ──► audio_check ─┘     │         (4 notebook        │
                              │          patches +         ├─ face detection (S3FD)
                              │          --smooth_alpha)   ├─ mel-spectrogram
                              └─ clean inputs              ├─ generator (Wav2Lip's)
                                 (plain names, 16 kHz      └─ fixed blend of mouth patch
                                  mono, first 30 s)

 Not wired in yet:  smoothing.py (one-euro)   blend.py (feathered paste)
```

The checks print warnings only. They never stop a run.

## Results

All numbers come from the notes in `notes/`. The metric is the mean absolute pixel change
between neighbouring frames over the mouth region (jitter) and how much the mouth really
moves (motion). Lower jitter relative to motion is steadier.

**Fixed blend** (`notes/03_smoothing_experiment.md`): real Wav2Lip GAN weights on CPU, one
synthetic cartoon face and 6 seconds of synthetic speech-like audio, 147 frames.

| alpha | Jitter | Motion | Jitter / motion |
|---|---|---|---|
| 1.0 (off) | 3.025 | 4.927 | 0.614 |
| 0.7 | 2.238 | 4.177 | 0.536 |
| 0.5 | 1.617 | 3.557 | 0.455 |

Smoothing steadies the mouth but also flattens its movement.

**One-euro vs fixed blend** (`notes/04_adaptive_smoothing.md`): applied after the fact to the
baseline sample video (508 frames).

| Filter | Jitter | Motion | Jitter / motion |
|---|---|---|---|
| none | 4.310 | 27.911 | 0.154 |
| blend alpha 0.7 | 3.454 | 27.107 | 0.127 |
| one-euro mc 1.0 b 0.05 | 3.080 | 26.994 | 0.114 |

At about the same movement the one-euro filter has less jitter than the fixed blend.

**Feathered blend** (`notes/05_feathered_blend.md`): on a synthetic frame the seam step at the
box border drops from 80.02 (hard paste) to 1.60 with any feather above 0 pixels.

**Limits:** these runs use one video or one synthetic face each, one run per setting, and no
real photos. The metrics measure pixel change, not how natural the mouth looks, and lip sync
was not measured. Judging real-photo quality by eye has not been done.

## Roadmap
- [x] Phase 1: Wav2Lip baseline (pretrained inference)
- [ ] Phase 2: Understand and modify the architecture, single-image input. In progress:
      architecture notes, one-command runner, input checks, smoothing experiments.
- [ ] Phase 3: Improved generator (identity preservation, mouth sharpness)
- [ ] Phase 4: Train a SyncNet-style sync discriminator (needs dataset access)
- [ ] Phase 5: Real-time inference (needs measurements on a real GPU)

Next small steps: a local demo app, a batch evaluation harness, wiring the one-euro filter
and the feathered paste into the runner, and measuring lip sync at each smoothing setting.

## Repository layout
```
echoface/
├── notebooks/    # Colab notebooks
├── src/
│   ├── inference/   # runner, Wav2Lip patch, smoothing, blend
│   ├── preprocess/  # audio and photo checks
│   └── models/      # adapted from Wav2Lip (non-commercial, study only)
├── scripts/      # trace_shapes, jitter measurement, comparison scripts
├── notes/        # architecture notes and experiments
├── tests/        # pytest, CPU only, no weights needed
├── docs/         # early access landing page
├── samples/      # example outputs
└── checkpoints/  # model weights (not tracked)
```

## Licensing
`src/models/` is adapted from [Wav2Lip](https://github.com/Rudrabha/Wav2Lip), which allows
personal, research and non-commercial use only. It is included here for study. Any commercial
version of EchoFace must not use it.

## Author
[0205prakriti](https://github.com/0205prakriti)
