"""One-command runner for EchoFace.

    python -m src.inference.run --photo face.jpg --audio talk.m4a --out result.mp4

What it does, in order:
  1. Gets Wav2Lip at a pinned commit (clones it if the folder is missing).
  2. Applies the EchoFace patches (see patch_wav2lip.py).
  3. Copies the photo and audio to plain file names, converts the audio to
     16 kHz mono and keeps only the first 30 seconds.
  4. Runs Wav2Lip's inference.py. No shell is used anywhere, so file names with
     spaces, brackets or quotes are safe.

Use --dry-run to print every command without running or changing anything.
"""
import argparse
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile

from src.inference import patch_wav2lip

WAV2LIP_URL = "https://github.com/Rudrabha/Wav2Lip.git"
WAV2LIP_COMMIT = "bac9a81e63ecc153202353372e5724b83d9e6322"
MAX_SECONDS = 30
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_CHECKPOINT = "checkpoints/wav2lip_gan.pth"


def clone_commands(wav2lip_dir):
    """Commands that fetch Wav2Lip at the pinned commit."""
    d = str(wav2lip_dir)
    return [
        ["git", "clone", "-q", WAV2LIP_URL, d],
        ["git", "-C", d, "checkout", "-q", WAV2LIP_COMMIT],
    ]


def ffmpeg_command(audio_in, audio_out, max_seconds=MAX_SECONDS):
    """Convert any audio file to 16 kHz mono wav, first ``max_seconds`` only.

    Paths are made absolute so a name that starts with '-' is not read as an option.
    """
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(pathlib.Path(audio_in).resolve()),
        "-t", str(max_seconds),
        "-ac", "1", "-ar", "16000",
        str(audio_out),
    ]


def inference_command(wav2lip_dir, face, audio, out, smooth, checkpoint=DEFAULT_CHECKPOINT):
    """The Wav2Lip command. Runs with cwd=wav2lip_dir, so ``out`` must be absolute."""
    return [
        sys.executable, "inference.py",
        "--checkpoint_path", str(checkpoint),
        "--face", str(face),
        "--audio", str(audio),
        "--outfile", str(pathlib.Path(out).resolve()),
        "--pads", "0", "10", "0", "0",
        "--face_det_batch_size", "4",
        "--smooth_alpha", str(smooth),
    ]


def plain_names(workdir, photo):
    """Plain target names inside ``workdir``. Keeps the photo's extension only."""
    suffix = pathlib.Path(photo).suffix.lower()
    if suffix not in PHOTO_SUFFIXES:
        raise ValueError(
            f"Unsupported photo type '{suffix}'. Use .jpg or .png "
            "(iPhone HEIC photos need converting first)."
        )
    workdir = pathlib.Path(workdir)
    return workdir / f"face{suffix}", workdir / "audio.wav"


def prepare_inputs(photo, audio, workdir, max_seconds=MAX_SECONDS, run=subprocess.run):
    """Copy the photo and convert the audio into ``workdir`` under plain names."""
    photo, audio = pathlib.Path(photo), pathlib.Path(audio)
    for p, label in ((photo, "photo"), (audio, "audio")):
        if not p.is_file():
            raise FileNotFoundError(f"The {label} file was not found: {p}")
    face_out, audio_out = plain_names(workdir, photo)
    pathlib.Path(workdir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(photo, face_out)
    r = run(ffmpeg_command(audio, audio_out, max_seconds), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("Could not read the audio file:\n" + (r.stderr or ""))
    return face_out, audio_out


def _show(cmd):
    return shlex.join(cmd)


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m src.inference.run",
        description="Make a lip-synced video from one photo and one audio file.",
    )
    p.add_argument("--photo", required=True, help="Face photo (.jpg or .png)")
    p.add_argument("--audio", required=True, help="Speech audio (any format ffmpeg reads)")
    p.add_argument("--out", required=True, help="Output .mp4 path")
    p.add_argument("--smooth", type=float, default=0.7,
                   help="Mouth smoothing: 1.0 = off, 0.7 = default, 0.5 = very steady")
    p.add_argument("--wav2lip-dir", default="Wav2Lip", help="Where Wav2Lip lives (default: ./Wav2Lip)")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="Checkpoint path, relative to the Wav2Lip folder or absolute")
    p.add_argument("--dry-run", action="store_true", help="Print the commands and change nothing")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 0.0 < args.smooth <= 1.0:
        print("error: --smooth must be above 0 and at most 1.0", file=sys.stderr)
        return 2

    w2l = pathlib.Path(args.wav2lip_dir)
    try:
        if args.dry_run:
            workdir = pathlib.Path(tempfile.gettempdir()) / "echoface_inputs"
            face, audio = plain_names(workdir, args.photo)
            if not w2l.is_dir():
                for c in clone_commands(w2l):
                    print(_show(c))
            print(f"# apply patches: python -m src.inference.patch_wav2lip {shlex.quote(str(w2l))}")
            print(f"# copy {shlex.quote(args.photo)} -> {shlex.quote(str(face))}")
            print(_show(ffmpeg_command(args.audio, audio)))
            print(_show(inference_command(w2l, face, audio, args.out, args.smooth, args.checkpoint)))
            return 0

        if not w2l.is_dir():
            for c in clone_commands(w2l):
                subprocess.run(c, check=True)
        for name, status in patch_wav2lip.apply_all(w2l):
            print(f"{status:8} {name}")

        ckpt = pathlib.Path(args.checkpoint)
        if not ckpt.is_absolute():
            ckpt = w2l / ckpt
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}. Download wav2lip_gan.pth first "
                "(see notebooks/02_wav2lip_own_photo.ipynb)."
            )

        with tempfile.TemporaryDirectory(prefix="echoface_") as tmp:
            face, audio = prepare_inputs(args.photo, args.audio, tmp)
            out = pathlib.Path(args.out).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = inference_command(w2l, face, audio, out, args.smooth, ckpt.resolve())
            r = subprocess.run(cmd, cwd=w2l, capture_output=True, text=True)
            if r.returncode != 0 or not out.exists():
                print(r.stderr[-3000:], file=sys.stderr)
                raise RuntimeError("Wav2Lip failed. The error is printed above.")
        print(f"Done: {out}")
        return 0
    except (RuntimeError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
