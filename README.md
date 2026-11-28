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
