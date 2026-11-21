import pathlib

import pytest

from src.inference import patch_wav2lip as pw

AUDIO = (
    "def _build_mel_basis():\n"
    "    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels)\n"
)
INFERENCE = (
    "parser = argparse.ArgumentParser()\n"
    "args = parser.parse_args()\n"
    "def datagen(frames, mels):\n"
    "\tbatch_size = args.wav2lip_batch_size\n"
    "\tpass\n"
    "def main():\n"
    "\twhile True:\n"
    "\t\tpred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.\n"
    "\t\tbreak\n"
    "\tcheckpoint = torch.load(path)\n"
    "\tcheckpoint = torch.load(path, map_location=lambda storage, loc: storage)\n"
)
SFD = "model_weights = torch.load(path_to_detector)\n"
UTILS = "x = np.array(a, dtype=np.int)\n"


@pytest.fixture
def w2l(tmp_path):
    (tmp_path / "audio.py").write_text(AUDIO)
    (tmp_path / "inference.py").write_text(INFERENCE)
    d = tmp_path / "face_detection" / "detection" / "sfd"
    d.mkdir(parents=True)
    (d / "sfd_detector.py").write_text(SFD)
    (tmp_path / "face_detection" / "utils.py").write_text(UTILS)
    return tmp_path


def snapshot(root):
    return {str(p): p.read_text() for p in pathlib.Path(root).rglob("*.py")}


def test_first_run_patches_everything(w2l):
    results = pw.apply_all(w2l)
    assert all(status == pw.PATCHED for _, status in results)
    assert "mel(sr=hp.sample_rate, n_fft=hp.n_fft," in (w2l / "audio.py").read_text()
    inf = (w2l / "inference.py").read_text()
    assert "torch.load(path, weights_only=False)" in inf
    assert "map_location=lambda storage, loc: storage, weights_only=False)" in inf
    assert "--smooth_alpha" in inf and "prev_pred = None" in inf
    assert "weights_only=False" in (w2l / "face_detection/detection/sfd/sfd_detector.py").read_text()
    assert "dtype=int)" in (w2l / "face_detection/utils.py").read_text()


def test_second_run_changes_nothing(w2l):
    pw.apply_all(w2l)
    after_first = snapshot(w2l)
    results = pw.apply_all(w2l)
    assert all(status == pw.ALREADY for _, status in results)
    assert snapshot(w2l) == after_first


def test_unexpected_audio_code_fails_loudly(w2l):
    (w2l / "audio.py").write_text("def nothing():\n    pass\n")
    with pytest.raises(RuntimeError, match="audio.py"):
        pw.apply_all(w2l)


def test_unexpected_inference_code_fails_loudly(w2l):
    (w2l / "inference.py").write_text("args = parser.parse_args()\ntorch.load(x)\n")
    with pytest.raises(RuntimeError, match="smoothing"):
        pw.apply_all(w2l)


def test_missing_torch_load_fails_loudly(w2l):
    (w2l / "face_detection/detection/sfd/sfd_detector.py").write_text("pass\n")
    with pytest.raises(RuntimeError, match="torch.load"):
        pw.apply_all(w2l)


def test_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        pw.apply_all(tmp_path / "nope")


def test_cli_returns_error_code(w2l, capsys):
    (w2l / "audio.py").write_text("pass\n")
    assert pw.main([str(w2l)]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_success(w2l, capsys):
    assert pw.main([str(w2l)]) == 0
    assert "patched" in capsys.readouterr().out
