# EchoFace feature backlog

A scheduled run picks the **first unchecked item** below, builds it, tests it, ticks it off and commits. One item per day.

## Rules for each run
- One item per run. Keep the change small enough to finish and test in a single session.
- Work on CPU only. Do not rely on a GPU, on photos of real people, or on the owner's private files.
- Write tests (`pytest`) that pass without model weights. If a feature needs weights, test the logic around them and say what was not tested.
- Our own code goes in `src/`. Do not copy more Wav2Lip code. `src/models/` is non-commercial and stays labelled that way.
- Never fake a result. If something could not be verified, write that in the commit message body and in the notes.
- Commit as the repo's usual author, with no co-author or session lines. Use a conventional message (`feat:`, `fix:`, `docs:`, `test:`, `ci:`).
- If tests fail or the item cannot be finished, push nothing. Leave the item unchecked and add one line under it saying what blocked it.
- Do not touch `docs/` (the landing page), accounts, payments, or anything outside this repo.

## Queue
- [x] **1. Patch module with tests.** Move the notebook's Wav2Lip patch code into `src/inference/patch_wav2lip.py` (idempotent, fails loudly on unexpected code). Tests run it twice on a fixture copy of the relevant lines.
- [ ] **2. One-command runner.** `python -m src.inference.run --photo P --audio A --out O [--smooth 0.7]`: pins and clones Wav2Lip if missing, applies the patch, cleans the inputs (plain file names, 16 kHz mono audio, first 30 s), runs inference without a shell. `--dry-run` prints the command. Tests cover awkward file names and the dry run.
- [ ] **3. CI.** GitHub Actions workflow that installs the CPU dependencies and runs the tests on every push.
- [ ] **4. Audio check.** `src/preprocess/audio_check.py` warns when audio looks like music or noise instead of speech, is clipped, too quiet, or too long. Tests use synthetic signals.
- [ ] **5. Photo check.** `src/preprocess/photo_check.py` warns about small, blurry, dark or overexposed photos and about no face or several faces. Tests use synthetic images and a mocked detector.
- [ ] **6. Smoothing v2.** Add an adaptive filter (for example a one-euro filter) next to the fixed blend. Compare with `scripts/measure_mouth_jitter.py` and record the table in `notes/`.
- [ ] **7. Feathered mouth blend.** Replace the hard paste of the mouth patch with a feathered mask to soften the seam. Add a seam metric and test it.
- [ ] **8. README rewrite.** Accurate, easy to skim: what works, a quick start, an architecture diagram, results, and an honest roadmap.
- [ ] **9. Local demo app.** A small Gradio app: upload a photo and audio, choose smoothing, see the video. Runs the one-command runner underneath. Document how to start it.
- [ ] **10. Evaluation harness.** Run a folder of photo and audio pairs, write a CSV of jitter and acceleration per result, and summarise it.

## Needs the owner (not for scheduled runs)
- Diffusion-based generator
- Real-time inference, which needs measurements on a real GPU
- SyncNet training and the dataset access it needs
- Speed numbers on a T4 (the cells are already in notebook 02)
- Judging real-photo quality by eye
