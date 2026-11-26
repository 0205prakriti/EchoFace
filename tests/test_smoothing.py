import cv2
import numpy as np
import pytest

from src.inference import smoothing as sm


def _noisy_sine(n=200, seed=0, noise=0.3):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    clean = np.sin(2 * np.pi * t / 200.0)
    return clean, clean + rng.randn(n) * noise


def test_blend_with_alpha_one_is_identity():
    x = np.random.RandomState(1).rand(10, 3, 3)
    assert np.allclose(sm.fixed_blend(x, 1.0), x, atol=1e-6)


def test_blend_matches_formula():
    x = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    out = sm.fixed_blend(x, 0.5)
    assert np.allclose(out, [0.0, 0.5, 0.75])


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_blend_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError):
        sm.fixed_blend(np.zeros(5), alpha)


def test_one_euro_first_sample_passes_through():
    f = sm.OneEuroFilter()
    x = np.full((2, 2), 7.0)
    assert np.allclose(f(x), x)


def test_one_euro_reduces_noise():
    clean, noisy = _noisy_sine()
    out = sm.one_euro(noisy, rate=25.0, min_cutoff=2.0, beta=0.0)
    assert np.abs(out - clean).mean() < np.abs(noisy - clean).mean()


def test_one_euro_follows_a_step_faster_with_larger_beta():
    step = np.concatenate([np.zeros(20), np.ones(20) * 10.0])
    slow = sm.one_euro(step, rate=25.0, min_cutoff=0.5, beta=0.0)
    fast = sm.one_euro(step, rate=25.0, min_cutoff=0.5, beta=2.0)
    assert fast[25] > slow[25]


def test_one_euro_keeps_constant_signal_constant():
    out = sm.one_euro(np.full((30, 4), 3.0))
    assert np.allclose(out, 3.0, atol=1e-5)


def test_one_euro_reset_forgets_history():
    f = sm.OneEuroFilter()
    f(np.zeros(3))
    f(np.ones(3))
    f.reset()
    assert np.allclose(f(np.full(3, 5.0)), 5.0)


def test_one_euro_empty_sequence():
    assert len(sm.one_euro(np.zeros((0, 2, 2)))) == 0


@pytest.mark.parametrize("kwargs", [{"rate": 0}, {"min_cutoff": 0}, {"d_cutoff": -1}, {"beta": -1}])
def test_one_euro_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        sm.OneEuroFilter(**kwargs)


def test_metrics_of_constant_clip_are_zero():
    m = sm.jitter_metrics(np.ones((5, 4, 4)))
    assert m["jitter"] == 0 and m["accel"] == 0 and m["motion"] == 0


def test_metrics_need_three_frames():
    with pytest.raises(ValueError):
        sm.jitter_metrics(np.zeros((2, 4, 4)))


def test_smoothing_lowers_jitter_of_noisy_frames():
    _, noisy = _noisy_sine(60)
    frames = noisy[:, None, None] * np.ones((1, 4, 4))
    base = sm.jitter_metrics(frames)["jitter"]
    assert sm.jitter_metrics(sm.fixed_blend(frames, 0.5))["jitter"] < base
    assert sm.jitter_metrics(sm.one_euro(frames, 25.0, 1.0, 0.0))["jitter"] < base


def test_compare_script_prints_a_table(tmp_path, capsys):
    path = str(tmp_path / "clip.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48))
    rng = np.random.RandomState(0)
    for _ in range(12):
        w.write(rng.randint(0, 255, (48, 64, 3), dtype=np.uint8))
    w.release()
    import scripts.compare_smoothing as cs
    cs.main([path])
    out = capsys.readouterr().out
    assert "one-euro" in out and "blend alpha 0.7" in out and "12 frames" in out
