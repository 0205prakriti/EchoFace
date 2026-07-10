> Turn any photo into a talking video — just add audio.

EchoFace is a deep learning pipeline that animates a static face image to match an input audio track. It combines audio feature extraction, face encoding, and a GAN-based generator trained with a lip-sync discriminator to produce natural, accurately synced mouth movements.

## 🚀 Project Roadmap

- [x] Phase 1: Wav2Lip baseline (pretrained inference)
- [ ] Phase 2: Custom generator
- [ ] Phase 3: Single-image + head motion (SadTalker-style)
- [ ] Phase 4: Real-time inference

## 🛠️ Stack

- Python 3.9+
- PyTorch
- OpenCV
- MediaPipe
- Wav2Lip (baseline)

## 📁 Structure

EchoFace/
├── notebooks/ # Colab notebooks
├── src/ # Source modules
│ ├── preprocess/ # Audio & video preprocessing
│ ├── models/ # Model architectures
│ └── inference/ # Inference pipeline
├── checkpoints/ # Model weights (not tracked)
├── samples/ # Test inputs
└── results/ # Generated outputs (not tracked)
