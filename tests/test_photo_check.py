import numpy as np

from src.preprocess import photo_check


def _codes(img):
    return {w.code for w in photo_check.check_image(img)}


def _textured(h=480, w=640, mean=128, seed=0):
    """Random texture: sharp, with a chosen average brightness."""
    rng = np.random.default_rng(seed)
    x = rng.normal(mean, 40, (h, w, 3))
    return np.clip(x, 0, 255).astype(np.uint8)


def test_normal_photo_has_no_warnings():
    assert _codes(_textured()) == set()


def test_small_photo_is_flagged():
    assert "too_small" in _codes(_textured(h=120, w=160))


def test_flat_photo_is_blurry():
    assert "blurry" in _codes(np.full((480, 640, 3), 128, np.uint8))


def test_smoothed_photo_is_blurry():
    import cv2
    soft = cv2.GaussianBlur(_textured(), (0, 0), 6)
    assert "blurry" in _codes(soft)


def test_dark_photo_is_flagged():
    assert "too_dark" in _codes(_textured(mean=20))


def test_bright_photo_is_overexposed():
    assert "overexposed" in _codes(_textured(mean=235))


def test_blown_out_patch_is_overexposed():
    img = _textured()
    img[:, :320] = 255
    assert "overexposed" in _codes(img)


def test_grayscale_input_works():
    assert _codes(_textured()[..., 0]) == set()


def test_size_does_not_change_blur_verdict():
    big = _textured(h=1500, w=2000)
    assert "blurry" not in _codes(big)
