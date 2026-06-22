"""YOLO person-detection wrapper for the dwell heatmap. Produces a binary mask
(255 on people) from a frame. Prefers exact segmentation masks; falls back to
filled bounding boxes. `mask_from_results` is split out so it can be unit-tested
with fake results (no model weights required).
"""
import cv2
import numpy as np


def mask_from_results(results, h: int, w: int) -> np.ndarray:
    """Build a 255-on-person mask from ultralytics results.

    Uses segmentation masks when present (exact silhouette), otherwise fills
    each detection's bounding box, clamped to the frame.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for r in results:
        seg = getattr(r, "masks", None)
        if seg is not None and getattr(seg, "data", None) is not None:
            m = seg.data.sum(0).cpu().numpy()
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask[m > 0.5] = 255
            continue
        boxes = getattr(r, "boxes", None) or []
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
    return mask


def boxes_from_results(results):
    """Extract per-person tracked boxes as (track_id, x1, y1, x2, y2).

    track_id is the YOLO tracker id when available, else None (the caller
    supplies a fallback key). Split out for unit testing with fake results.
    """
    out = []
    for r in results:
        boxes = getattr(r, "boxes", None) or []
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            tid = None
            bid = getattr(b, "id", None)
            if bid is not None:
                try:
                    tid = int(bid[0].item())
                except Exception:
                    try:
                        tid = int(bid.item())
                    except Exception:
                        tid = None
            out.append((tid, x1, y1, x2, y2))
    return out


def mask_from_boxes(boxes, h: int, w: int) -> np.ndarray:
    """Fill a 255-on-person mask from (x1, y1, x2, y2) boxes, clamped to frame."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


class PersonDetector:
    """Runs YOLO (person class only) on a frame and returns a person mask.

    Loads ultralytics lazily so importing this module needs no model/weights.
    """

    def __init__(self, model_path: str, conf: float = 0.3, use_tracking: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf
        self.use_tracking = use_tracking

    def _run(self, frame_bgr: np.ndarray):
        if self.use_tracking:
            return self.model.track(
                frame_bgr, conf=self.conf, classes=[0], persist=True, verbose=False
            )
        return self.model.predict(
            frame_bgr, conf=self.conf, classes=[0], verbose=False
        )

    def person_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        return mask_from_results(self._run(frame_bgr), h, w)

    def detect_tracks(self, frame_bgr: np.ndarray):
        """Return tracked people as (track_id, x1, y1, x2, y2) for the dwell gate."""
        return boxes_from_results(self._run(frame_bgr))
