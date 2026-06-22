import numpy as np
from heatmap_detector import mask_from_results


class _Box:
    def __init__(self, xyxy):
        self.xyxy = [np.array(xyxy, dtype=float)]


class _R:
    def __init__(self, boxes=None, masks=None):
        self.boxes = boxes
        self.masks = masks


def test_mask_from_boxes_fills_box_region():
    r = _R(boxes=[_Box([2, 2, 5, 6])])
    m = mask_from_results([r], h=10, w=10)
    assert m.shape == (10, 10)
    assert m[3, 3] == 255           # inside box
    assert m[0, 0] == 0             # outside box


def test_mask_empty_when_no_detections():
    m = mask_from_results([_R(boxes=[])], h=4, w=4)
    assert m.max() == 0


def test_mask_clamps_box_to_frame_bounds():
    r = _R(boxes=[_Box([-5, -5, 100, 100])])   # box larger than frame
    m = mask_from_results([r], h=8, w=8)
    assert m[0, 0] == 255 and m[7, 7] == 255   # clamped, fills whole frame, no crash
