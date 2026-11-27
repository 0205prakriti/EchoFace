import numpy as np
import pytest

from src.inference import blend


def _scene(seed=0):
    rng = np.random.RandomState(seed)
    frame = np.full((60, 80, 3), 120, dtype=np.uint8)
    frame = np.clip(frame + rng.randint(-2, 3, frame.shape), 0, 255).astype(np.uint8)
    patch = np.full((20, 30, 3), 200, dtype=np.uint8)
    return frame, patch, (20, 40, 25, 55)


def test_mask_edges_and_core():
    m = blend.feather_mask(10, 12, 3)
    assert m.shape == (10, 12)
    assert m[0].max() == 0 and m[-1].max() == 0 and m[:, 0].max() == 0 and m[:, -1].max() == 0
    assert m[5, 6] == 1.0
    assert m.min() >= 0 and m.max() <= 1


def test_mask_zero_feather_is_all_ones():
    assert blend.feather_mask(4, 5, 0).min() == 1.0


def test_mask_small_box_does_not_fail():
    m = blend.feather_mask(3, 3, 10)
    assert m.shape == (3, 3) and m[1, 1] == 1.0


@pytest.mark.parametrize("args", [(0, 5, 1), (5, 0, 1), (5, 5, -1)])
def test_mask_rejects_bad_input(args):
    with pytest.raises(ValueError):
        blend.feather_mask(*args)


def test_paste_with_zero_feather_is_hard_paste():
    frame, patch, (y1, y2, x1, x2) = _scene()
    out = blend.feather_paste(frame, patch, (y1, y2, x1, x2), feather=0)
    expected = frame.copy()
    expected[y1:y2, x1:x2] = patch
    assert np.array_equal(out, expected)


def test_paste_leaves_input_and_outside_untouched():
    frame, patch, (y1, y2, x1, x2) = _scene()
    before = frame.copy()
    out = blend.feather_paste(frame, patch, (y1, y2, x1, x2), feather=5)
    assert np.array_equal(frame, before)
    outside = np.ones(frame.shape[:2], bool)
    outside[y1:y2, x1:x2] = False
    assert np.array_equal(out[outside], frame[outside])
    assert out.dtype == frame.dtype


def test_paste_core_is_patch_and_border_is_frame():
    frame, patch, (y1, y2, x1, x2) = _scene()
    out = blend.feather_paste(frame, patch, (y1, y2, x1, x2), feather=4)
    assert np.array_equal(out[y1 + 8, x1 + 8], patch[8, 8])
    assert np.array_equal(out[y1, x1 + 10], frame[y1, x1 + 10])


def test_paste_works_on_grayscale():
    frame = np.zeros((20, 20), dtype=np.uint8)
    patch = np.full((10, 10), 255, dtype=np.uint8)
    out = blend.feather_paste(frame, patch, (5, 15, 5, 15), feather=3)
    assert out[10, 10] == 255 and out[5, 8] == 0


def test_paste_rejects_wrong_patch_size_and_box():
    frame, patch, _ = _scene()
    with pytest.raises(ValueError):
        blend.feather_paste(frame, patch, (20, 40, 25, 56))
    with pytest.raises(ValueError):
        blend.feather_paste(frame, patch, (50, 70, 25, 55))


def test_feathering_lowers_the_seam():
    frame, patch, box = _scene()
    hard = blend.seam_metric(blend.feather_paste(frame, patch, box, feather=0), box)
    soft = blend.seam_metric(blend.feather_paste(frame, patch, box, feather=8), box)
    assert hard["step"] > 50
    assert soft["step"] < hard["step"] / 10
    assert soft["ratio"] < hard["ratio"]


def test_seam_metric_on_smooth_frame_is_no_seam():
    frame = np.tile(np.linspace(0, 255, 80, dtype=np.float32), (60, 1))
    m = blend.seam_metric(frame, (20, 40, 25, 55))
    assert 0.5 < m["ratio"] < 1.5


def test_seam_metric_flat_frame_has_ratio_one():
    assert blend.seam_metric(np.full((30, 30), 7, np.uint8), (5, 20, 5, 20))["ratio"] == 1.0


def test_seam_metric_rejects_box_at_edge():
    with pytest.raises(ValueError):
        blend.seam_metric(np.zeros((30, 30)), (0, 10, 5, 20))
