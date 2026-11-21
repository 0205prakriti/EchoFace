"""Small compatibility patches for a fresh Wav2Lip checkout.

Wav2Lip was written for old library versions. This module applies the few
edits needed to run it with current librosa, PyTorch and numpy, plus the
optional ``--smooth_alpha`` flag used by EchoFace.

Every patch is idempotent (running twice changes nothing the second time) and
fails loudly with a RuntimeError if the Wav2Lip code is not what we expect.

Usage:
    python -m src.inference.patch_wav2lip /path/to/Wav2Lip
"""
import argparse
import pathlib
import re
import sys

PATCHED = "patched"
ALREADY = "already"


def _regex_patch(root, rel_path, pattern, replacement, already_marker):
    """Apply one regex fix. Returns PATCHED or ALREADY, raises if neither applies."""
    f = pathlib.Path(root) / rel_path
    text = f.read_text()
    new_text, n = re.subn(pattern, replacement, text)
    if n:
        f.write_text(new_text)
        return PATCHED
    if already_marker in text:
        return ALREADY
    raise RuntimeError(
        f"Could not patch {rel_path}: the Wav2Lip code is not what EchoFace expects."
    )


def patch_librosa_mel(root):
    """librosa 0.10+ only accepts keyword arguments for the mel filterbank."""
    return _regex_patch(
        root,
        "audio.py",
        r"librosa\.filters\.mel\(hp\.sample_rate, hp\.n_fft,",
        "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,",
        "mel(sr=hp.sample_rate",
    )


def patch_torch_load(root, rel_path):
    """PyTorch 2.6+ refuses to unpickle checkpoints unless weights_only is stated."""
    f = pathlib.Path(root) / rel_path
    text = f.read_text()

    def fix(m):
        if "weights_only" in m.group(0):
            return m.group(0)
        return f"torch.load({m.group(1)}, weights_only=False)"

    new_text, _ = re.subn(r"torch\.load\(([^()]*)\)", fix, text)
    if new_text != text:
        f.write_text(new_text)
        return PATCHED
    if "weights_only" in text:
        return ALREADY
    raise RuntimeError(f"No torch.load call found in {rel_path}.")


def patch_numpy_int(root):
    """numpy 1.24+ removed np.int."""
    return _regex_patch(
        root,
        "face_detection/utils.py",
        r"dtype=np\.int\)",
        "dtype=int)",
        "dtype=int)",
    )


_SMOOTH_EDITS = [
    (
        "args = parser.parse_args()",
        "parser.add_argument('--smooth_alpha', type=float, default=1.0,\n"
        "                    help='Temporal smoothing of the generated mouth: 1.0 = off, 0.5 = strong')\n"
        "args = parser.parse_args()",
    ),
    (
        "\tbatch_size = args.wav2lip_batch_size\n",
        "\tbatch_size = args.wav2lip_batch_size\n\tprev_pred = None\n",
    ),
    (
        "\t\tpred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.\n",
        "\t\tpred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.\n"
        "\t\tif args.smooth_alpha < 1.0:\n"
        "\t\t\tfor k in range(len(pred)):\n"
        "\t\t\t\tif prev_pred is not None:\n"
        "\t\t\t\t\tpred[k] = args.smooth_alpha * pred[k] + (1.0 - args.smooth_alpha) * prev_pred\n"
        "\t\t\t\tprev_pred = pred[k].copy()\n",
    ),
]


def patch_smoothing(root):
    """Add an optional --smooth_alpha flag (1.0 = off, the default)."""
    f = pathlib.Path(root) / "inference.py"
    text = f.read_text()
    if "smooth_alpha" in text:
        return ALREADY
    for old, new in _SMOOTH_EDITS:
        if text.count(old) != 1:
            raise RuntimeError(
                "Could not add smoothing: inference.py is not what EchoFace expects."
            )
        text = text.replace(old, new)
    f.write_text(text)
    return PATCHED


def apply_all(root):
    """Apply every patch to the Wav2Lip checkout at ``root``.

    Returns a list of (name, status) pairs. Raises RuntimeError on unexpected code.
    """
    root = pathlib.Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Wav2Lip folder not found: {root}")
    steps = [
        ("audio.py (librosa mel)", lambda: patch_librosa_mel(root)),
        ("inference.py (torch.load)", lambda: patch_torch_load(root, "inference.py")),
        (
            "sfd_detector.py (torch.load)",
            lambda: patch_torch_load(root, "face_detection/detection/sfd/sfd_detector.py"),
        ),
        ("face_detection/utils.py (np.int)", lambda: patch_numpy_int(root)),
        ("inference.py (--smooth_alpha)", lambda: patch_smoothing(root)),
    ]
    return [(name, fn()) for name, fn in steps]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Patch a Wav2Lip checkout for current libraries.")
    parser.add_argument("wav2lip_dir", help="Path to the Wav2Lip folder")
    args = parser.parse_args(argv)
    try:
        results = apply_all(args.wav2lip_dir)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for name, status in results:
        print(f"{status:8} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
