"""Trace tensor shapes through the Wav2Lip generator, quality discriminator and SyncNet.

Runs on CPU with random weights and random inputs. Nothing is trained or downloaded.
Usage (from the repo root):  python scripts/trace_shapes.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import SyncNet_color, Wav2Lip, Wav2Lip_disc_qual  # noqa: E402


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def attach_hooks(model, groups, log):
    """Record the output shape of each top-level block inside the named ModuleLists/Sequentials."""
    handles = []
    for group in groups:
        container = getattr(model, group)
        for i, block in enumerate(container):
            def hook(_m, _inp, out, name=f"{group}[{i}]"):
                log.append((name, tuple(out.shape)))
            handles.append(block.register_forward_hook(hook))
    return handles


def show(title, log):
    print(f"\n{title}")
    width = max(len(n) for n, _ in log)
    for name, shape in log:
        print(f"  {name:<{width}}  {shape}")


def trace_generator():
    model = Wav2Lip().eval()
    log = []
    handles = attach_hooks(model, ["face_encoder_blocks", "face_decoder_blocks"], log)
    handles.append(model.audio_encoder.register_forward_hook(
        lambda _m, _i, out: log.append(("audio_encoder (whole)", tuple(out.shape)))))
    handles.append(model.output_block.register_forward_hook(
        lambda _m, _i, out: log.append(("output_block", tuple(out.shape)))))

    batch = 2
    audio = torch.rand(batch, 1, 80, 16)   # one mel chunk: 80 mel bins x 16 steps
    faces = torch.rand(batch, 6, 96, 96)   # reference frame + masked target frame, stacked
    with torch.no_grad():
        out = model(audio, faces)
    for h in handles:
        h.remove()

    print(f"Wav2Lip generator  | params: {n_params(model):,}")
    print(f"  input audio {tuple(audio.shape)}   input faces {tuple(faces.shape)}")
    show("Face encoder (saved as skip connections) and decoder (audio embedding goes in first):", log)
    print(f"\n  output {tuple(out.shape)}  (range {out.min():.2f} to {out.max():.2f}, sigmoid)")
    assert out.shape == (batch, 3, 96, 96)


def trace_discriminator():
    model = Wav2Lip_disc_qual().eval()
    log = []
    handles = attach_hooks(model, ["face_encoder_blocks"], log)
    batch, frames = 2, 5
    video = torch.rand(batch, 3, frames, 96, 96)  # (B, C, T, H, W)
    with torch.no_grad():
        out = model(video)
    for h in handles:
        h.remove()
    print(f"\n\nWav2Lip_disc_qual (visual quality GAN discriminator) | params: {n_params(model):,}")
    print(f"  input {tuple(video.shape)}; it keeps only the lower half of each frame, so the encoder sees 48x96")
    show("Encoder blocks:", log)
    print(f"\n  output {tuple(out.shape)}  (one real/fake score per frame)")
    assert out.shape == (batch * frames, 1)


def trace_syncnet():
    model = SyncNet_color().eval()
    batch = 2
    audio = torch.rand(batch, 1, 80, 16)
    faces = torch.rand(batch, 15, 48, 96)  # 5 frames x 3 channels, lower half of the face
    with torch.no_grad():
        a, f = model(audio, faces)
    print(f"\n\nSyncNet_color (audio-visual sync expert) | params: {n_params(model):,}")
    print(f"  input audio {tuple(audio.shape)}   input faces {tuple(faces.shape)}")
    print(f"  audio embedding {tuple(a.shape)}   face embedding {tuple(f.shape)}   (both L2-normalised)")
    cos = (a * f).sum(dim=1)
    print(f"  cosine similarity on random inputs: {[round(c, 3) for c in cos.tolist()]}")
    print(f"  embedding norms: audio {a.norm(dim=1).tolist()}  face {f.norm(dim=1).tolist()}")
    assert a.shape == f.shape == (batch, 512)


if __name__ == "__main__":
    torch.manual_seed(0)
    trace_generator()
    trace_discriminator()
    trace_syncnet()
    print("\nAll shape checks passed.")
