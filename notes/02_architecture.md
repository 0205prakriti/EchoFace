# Phase 2: Wav2Lip architecture notes

Source: `src/models/` (adapted from Rudrabha/Wav2Lip). Shapes below come from running `python scripts/trace_shapes.py` with random weights on CPU.

## 1. Generator (`Wav2Lip`, 36.3M parameters)

A U-Net. The face encoder shrinks the image, the audio embedding is the decoder's starting point, and skip connections carry face detail back across.

| Stage | Output shape (B=2) |
|---|---|
| Input faces | 6 x 96 x 96 (reference frame + masked target frame, stacked) |
| Input audio | 1 x 80 x 16 (80 mel bins, 16 time steps) |
| Face encoder | 16x96x96, 32x48x48, 64x24x24, 128x12x12, 256x6x6, 512x3x3, 512x1x1 |
| Audio encoder | 512 x 1 x 1 |
| Decoder (before skip concat) | 512x1x1, 512x3x3, 512x6x6, 384x12x12, 256x24x24, 128x48x48, 64x96x96 |
| Output | 3 x 96 x 96, sigmoid (values in 0 to 1) |

- Audio enters the network once, at the decoder's first block. Everything after that is upsampling plus skip connections from the face encoder.
- The masked target frame hides the lower half of the face, so the model has to paint the mouth from the audio.
- The reference frame supplies identity and pose.

## 2. Visual quality discriminator (`Wav2Lip_disc_qual`, 14.1M parameters)

- Input is a video tensor (B, 3, T, 96, 96). It flattens time into the batch and keeps only the lower half of each frame (48 x 96).
- Output is one real/fake score per frame.
- It judges how realistic the mouth region looks. It does not judge sync.

## 3. Sync expert (`SyncNet_color`, 16.4M parameters)

- Inputs: a mel chunk (1x80x16) and 5 stacked lower-half frames (15 x 48 x 96).
- Outputs: a 512-d audio embedding and a 512-d face embedding, both L2-normalised.
- Sync score = cosine similarity between them. In sync means high, out of sync means low.
- It is trained first and frozen. The generator is then penalised when SyncNet says its output is out of sync.

## 4. How the three fit together in training

1. Train SyncNet on real audio and video pairs.
2. Train the generator with a reconstruction loss, a sync loss from the frozen SyncNet, and a quality loss from the discriminator.

## 5. Questions to investigate next

- Audio is injected at a single point. What happens if it is fed in at several decoder scales?
- The model outputs 96 x 96 faces, which is why results look soft after pasting back into the video. What does a super-resolution or face-restoration step do to mouth sharpness?
- How much does the output change if the reference frame is swapped for a different pose?

## 6. My observations after running the baseline

_Write these yourself after watching `samples/wav2lip_baseline_result.mp4`: sync quality, mouth sharpness, any visible seam around the mouth, anything that surprised you._
