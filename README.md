# EchoFace 🎙️

> Turn any photo into a talking video, just add audio.

EchoFace is a deep learning pipeline that animates a face to match an input audio track. The end goal is a custom audio-visual fusion model with a lip-sync discriminator, built up in stages from a pretrained Wav2Lip baseline.

## Phase 1: Wav2Lip baseline (complete)

Ran pretrained Wav2Lip (GAN checkpoint) end-to-end on a Colab T4 GPU.

**Result:** [`samples/wav2lip_baseline_result.mp4`](samples/wav2lip_baseline_result.mp4)

**Pipeline:** face video + audio → mel-spectrogram → face detection (S3FD) → Wav2Lip generator → lip-synced video

**Notebook:** [`notebooks/01_wav2lip_baseline.ipynb`](notebooks/01_wav2lip_baseline.ipynb)

**Your own photo and audio:** [`notebooks/02_wav2lip_own_photo.ipynb`](notebooks/02_wav2lip_own_photo.ipynb) runs on current Colab without pinning old `numpy` or `librosa`. It patches four lines in Wav2Lip instead.

### What I learned
- Wav2Lip was written for Python 3.6–3.8. On Colab's Python 3.12, `librosa` breaks against newer `numpy` because of removed aliases like `np.complex`, so the environment needs patching or pinning.
- Observations on sync quality and mouth sharpness: _add yours here after reviewing the output_

## Roadmap
- [x] Phase 1: Wav2Lip baseline (pretrained inference)
- [ ] Phase 2: Understand and modify the architecture, single-image input
- [ ] Phase 3: Improved generator (identity preservation, mouth sharpness)
- [ ] Phase 4: Train a SyncNet-style sync discriminator
- [ ] Phase 5: Real-time inference

## Stack
PyTorch · OpenCV · librosa · Wav2Lip · Google Colab (T4)

## Structure
```
EchoFace/
├── notebooks/    # Colab notebooks
├── src/          # preprocess / models / inference modules
├── scripts/      # trace_shapes.py: runs random tensors through the models
├── notes/        # architecture notes
├── docs/         # early access landing page
├── samples/      # example outputs
└── checkpoints/  # model weights (not tracked)
```

## Licensing
`src/models/` is adapted from [Wav2Lip](https://github.com/Rudrabha/Wav2Lip), which allows personal, research and non-commercial use only. It is included here for study. Any commercial version of EchoFace must not use it.

## Author
[0205prakriti](https://github.com/0205prakriti)
