"""Dwell-heatmap core math: accumulate person-presence seconds per pixel and
render a colored overlay. Pure (no I/O except the .npy persistence helpers),
so it is unit-testable without RTSP, models, or S3.
"""
import os

import cv2
import numpy as np


class DwellHeatmap:
    """A per-pixel accumulator measuring how many SECONDS a person occupied
    each pixel. `add(mask, dt)` adds `dt` seconds wherever `mask` is set."""

    def __init__(self, height: int, width: int):
        self.accumulator = np.zeros((height, width), dtype=np.float32)

    def add(self, mask: np.ndarray, dt: float) -> None:
        if mask.shape != self.accumulator.shape:
            mask = cv2.resize(
                mask,
                (self.accumulator.shape[1], self.accumulator.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        self.accumulator[mask > 0] += float(dt)

    def max_seconds(self) -> float:
        return float(self.accumulator.max())


def render(accumulator, background_bgr, gamma=0.6, blur=15, alpha=0.5, thr=0.04):
    """Colorize the accumulator and blend it over a background image.

    gamma (<1) lifts mid-range dwell so equal time reads as equal color; blur
    smooths blotchiness; only pixels above `thr` (normalized) get colored, so
    cold areas show the plain background.
    """
    hm = accumulator
    if blur and blur >= 3:
        k = blur | 1                       # force odd kernel
        hm = cv2.GaussianBlur(hm, (k, k), 0)
    m = hm.max()
    norm = np.power(hm / m, gamma) if m > 0 else np.zeros_like(hm)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = (norm > thr)[..., None]
    blended = cv2.addWeighted(background_bgr, 1 - alpha, colored, alpha, 0)
    return np.where(mask, blended, background_bgr).astype(np.uint8)


def save_raw(accumulator, path: str) -> None:
    np.save(path, accumulator)


def load_raw(path: str):
    """Load a saved accumulator, or None if it does not exist."""
    if not os.path.exists(path):
        return None
    return np.load(path)
