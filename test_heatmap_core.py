import numpy as np
from heatmap_core import DwellHeatmap, render, save_raw, load_raw


def test_add_accumulates_seconds_on_mask_pixels_only():
    hm = DwellHeatmap(4, 4)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 255
    hm.add(mask, 2.0)
    hm.add(mask, 1.5)
    assert hm.accumulator[1, 1] == 3.5
    assert hm.accumulator[0, 0] == 0.0      # untouched pixel stays cold
    assert hm.max_seconds() == 3.5


def test_add_resizes_mismatched_mask():
    hm = DwellHeatmap(8, 8)
    mask = np.full((4, 4), 255, dtype=np.uint8)   # smaller than accumulator
    hm.add(mask, 1.0)
    assert hm.max_seconds() == 1.0                # resized to fill, no crash


def test_render_returns_same_size_bgr_image():
    hm = DwellHeatmap(8, 8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    hm.add(mask, 10.0)
    bg = np.zeros((8, 8, 3), dtype=np.uint8)
    out = render(hm.accumulator, bg)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8


def test_render_cold_map_is_just_background():
    hm = DwellHeatmap(6, 6)                        # nothing added -> all zero
    bg = np.full((6, 6, 3), 123, dtype=np.uint8)
    out = render(hm.accumulator, bg)
    assert np.array_equal(out, bg)                 # no heat -> untouched background


def test_save_and_load_raw_roundtrip(tmp_path):
    hm = DwellHeatmap(3, 3)
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[0, 0] = 255
    hm.add(mask, 5.0)
    p = str(tmp_path / "acc.npy")
    save_raw(hm.accumulator, p)
    loaded = load_raw(p)
    assert loaded is not None and loaded[0, 0] == 5.0
    assert load_raw(str(tmp_path / "missing.npy")) is None
