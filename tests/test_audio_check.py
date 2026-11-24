import wave

import numpy as np
import pytest

from src.preprocess import audio_check

RATE = 16000


def _codes(x, rate=RATE):
    return {w.code for w in audio_check.check_audio(x, rate)}


def _speech_like(seconds=5.0, level=0.3):
    """Harmonic tone with a moving pitch, switched on and off about 4 times a second."""
    t = np.arange(int(RATE * seconds)) / RATE
    phase = np.cumsum(2 * np.pi * (120 + 20 * np.sin(2 * np.pi * 0.7 * t)) / RATE)
    s = sum(np.sin(k * phase) / k for k in range(1, 15))
    env = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 2 * (np.sin(2 * np.pi * 0.5 * t) > -0.6)
    return level * s / np.abs(s).max() * env


def test_speech_like_signal_has_no_warnings():
    assert _codes(_speech_like()) == set()


def test_white_noise_is_flagged_as_noise():
    x = 0.3 * np.random.default_rng(0).standard_normal(RATE * 5)
    assert "noise_like" in _codes(x)


def test_steady_chord_is_flagged_as_music_like():
    t = np.arange(RATE * 5) / RATE
    x = sum(0.1 * np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0, 523.3))
    assert "music_like" in _codes(x)


def test_quiet_signal_is_flagged():
    assert "too_quiet" in _codes(_speech_like(level=0.003))


def test_normal_level_is_not_flagged_quiet():
    assert "too_quiet" not in _codes(_speech_like(level=0.3))


def test_clipped_signal_is_flagged():
    x = np.clip(_speech_like(level=0.3) * 6, -1.0, 1.0)
    assert "clipped" in _codes(x)


def test_unclipped_signal_is_not_flagged_clipped():
    assert "clipped" not in _codes(_speech_like())


def test_long_audio_is_flagged_and_30_seconds_is_not():
    assert "too_long" in _codes(np.zeros(RATE * 31) + 0.1 * _speech_like(31.0))
    assert "too_long" not in _codes(_speech_like(30.0))


def test_empty_and_silent_audio_do_not_crash():
    assert _codes(np.array([])) == {"too_quiet"}
    assert "too_quiet" in _codes(np.zeros(RATE * 2))


def test_very_short_audio_does_not_crash():
    audio_check.check_audio(_speech_like(0.1), RATE)


def _write_wav(path, x, channels=1):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    if channels == 2:
        pcm = np.repeat(pcm[:, None], 2, axis=1).reshape(-1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


@pytest.mark.parametrize("channels", [1, 2])
def test_load_wav_round_trip(tmp_path, channels):
    x = _speech_like(1.0)
    p = tmp_path / "a b (1).wav"
    _write_wav(p, x, channels)
    y, rate = audio_check.load_wav(p)
    assert rate == RATE and len(y) == len(x)
    assert np.allclose(x, y, atol=1e-3)


def test_cli_prints_warnings(tmp_path, capsys):
    p = tmp_path / "quiet.wav"
    _write_wav(p, _speech_like(level=0.003))
    assert audio_check.main([str(p)]) == 0
    assert "[too_quiet]" in capsys.readouterr().out


def test_cli_reports_clean_audio(tmp_path, capsys):
    p = tmp_path / "ok.wav"
    _write_wav(p, _speech_like())
    audio_check.main([str(p)])
    assert "no warnings" in capsys.readouterr().out
