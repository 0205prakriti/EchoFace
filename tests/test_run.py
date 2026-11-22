import pathlib
import shutil
import subprocess
import wave

import pytest

from src.inference import run

AWKWARD = [
    "Recording (1).m4a",
    "my song [final] & more.wav",
    "it's \"quoted\" $HOME `x`.wav",
    "-leading-dash.wav",
    "tölvu hljóð.wav",
]


def _make_wav(path, seconds=1.0, rate=8000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x10" * int(rate * seconds))


@pytest.mark.parametrize("name", AWKWARD)
def test_ffmpeg_command_keeps_awkward_names_as_one_argument(tmp_path, name):
    cmd = run.ffmpeg_command(tmp_path / name, tmp_path / "audio.wav")
    assert str(tmp_path / name) in cmd
    i = cmd.index("-i")
    assert cmd[i + 1].startswith("/")  # absolute, so a leading '-' is not an option
    assert cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-t") + 1] == "30"


def test_plain_names_ignores_original_name(tmp_path):
    face, audio = run.plain_names(tmp_path, "Me (final) [1].JPG")
    assert face.name == "face.jpg"
    assert audio.name == "audio.wav"


def test_plain_names_rejects_unknown_photo_type(tmp_path):
    with pytest.raises(ValueError, match="HEIC"):
        run.plain_names(tmp_path, "photo.heic")


def test_prepare_inputs_with_fake_ffmpeg(tmp_path):
    photo = tmp_path / "My Photo (2).png"
    photo.write_bytes(b"not really a png")
    audio = tmp_path / "a [b].m4a"
    audio.write_bytes(b"x")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    work = tmp_path / "work"
    face, wav = run.prepare_inputs(photo, audio, work, run=fake_run)
    assert face == work / "face.png" and face.read_bytes() == photo.read_bytes()
    assert calls and calls[0][0] == "ffmpeg"
    assert str(audio.resolve()) in calls[0]


def test_prepare_inputs_reports_audio_failure(tmp_path):
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"x")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    def failing(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "bad audio")

    with pytest.raises(RuntimeError, match="bad audio"):
        run.prepare_inputs(photo, audio, tmp_path / "w", run=failing)


def test_prepare_inputs_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="photo"):
        run.prepare_inputs(tmp_path / "nope.jpg", tmp_path / "a.wav", tmp_path / "w")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_real_ffmpeg_converts_and_trims(tmp_path):
    photo = tmp_path / "p (1).jpg"
    photo.write_bytes(b"x")
    audio = tmp_path / "long [clip].wav"
    _make_wav(audio, seconds=3.0, rate=8000)
    face, wav = run.prepare_inputs(photo, audio, tmp_path / "w", max_seconds=1)
    with wave.open(str(wav)) as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getnframes() <= 16000 + 1600


def test_dry_run_prints_commands_and_changes_nothing(tmp_path, capsys):
    out = tmp_path / "out dir" / "result (1).mp4"
    w2l = tmp_path / "Wav2Lip"
    code = run.main([
        "--photo", str(tmp_path / "My Face (1).jpg"),
        "--audio", str(tmp_path / "Recording (1).m4a"),
        "--out", str(out),
        "--smooth", "0.5",
        "--wav2lip-dir", str(w2l),
        "--dry-run",
    ])
    text = capsys.readouterr().out
    assert code == 0
    assert "git clone" in text and run.WAV2LIP_COMMIT in text
    assert "ffmpeg" in text and "inference.py" in text
    assert "--smooth_alpha 0.5" in text
    assert "'Recording (1).m4a'" in text or "Recording (1).m4a'" in text  # shell-quoted for display
    assert not w2l.exists()
    assert not out.parent.exists()


def test_dry_run_skips_clone_when_folder_exists(tmp_path, capsys):
    w2l = tmp_path / "Wav2Lip"
    w2l.mkdir()
    run.main(["--photo", "a.jpg", "--audio", "a.wav", "--out", "o.mp4",
              "--wav2lip-dir", str(w2l), "--dry-run"])
    assert "git clone" not in capsys.readouterr().out


def test_bad_smooth_value_is_rejected(capsys):
    code = run.main(["--photo", "a.jpg", "--audio", "a.wav", "--out", "o.mp4",
                     "--smooth", "0", "--dry-run"])
    assert code == 2
    assert "--smooth" in capsys.readouterr().err


def test_dry_run_rejects_heic(capsys):
    code = run.main(["--photo", "a.heic", "--audio", "a.wav", "--out", "o.mp4", "--dry-run"])
    assert code == 1
    assert "HEIC" in capsys.readouterr().err


def test_missing_checkpoint_fails_clearly(tmp_path, monkeypatch, capsys):
    w2l = tmp_path / "Wav2Lip"
    w2l.mkdir()
    monkeypatch.setattr(run.patch_wav2lip, "apply_all", lambda root: [("x", "already")])
    code = run.main(["--photo", "a.jpg", "--audio", "a.wav", "--out", str(tmp_path / "o.mp4"),
                     "--wav2lip-dir", str(w2l)])
    assert code == 1
    assert "Checkpoint not found" in capsys.readouterr().err
