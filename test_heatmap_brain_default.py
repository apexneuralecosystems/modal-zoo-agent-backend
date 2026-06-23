import numpy as np
import heatmap_brain_default as brain


class _FakeDet:
    tracks: list = []

    def __init__(self, *a, **k):
        pass

    def detect_tracks(self, frame):
        return _FakeDet.tracks


def _setup(monkeypatch):
    monkeypatch.setattr(brain, "PersonDetector", _FakeDet)
    brain.setup({
        "model_path": "x.pt", "conf": 0.3, "dwell_seconds": 2.0,
        "move_frac": 0.5, "width": 200, "height": 200,
    })


def test_dwelling_person_produces_mask(monkeypatch):
    _setup(monkeypatch)
    _FakeDet.tracks = [(7, 10, 10, 50, 110)]
    frame = np.zeros((200, 200, 3), np.uint8)
    assert brain.heat_mask(frame, 0.0).max() == 0      # just arrived
    m = brain.heat_mask(frame, 2.0)                      # 2s later -> dwelling
    assert m.shape == (200, 200)
    assert m[60, 30] == 255                              # inside the box


def test_walkthrough_no_mask(monkeypatch):
    _setup(monkeypatch)
    frame = np.zeros((200, 200, 3), np.uint8)
    for i in range(10):
        _FakeDet.tracks = [(7, 10 + i * 30, 10, 50 + i * 30, 110)]
        assert brain.heat_mask(frame, i * 0.3).max() == 0


def test_reset_clears_dwell(monkeypatch):
    _setup(monkeypatch)
    frame = np.zeros((200, 200, 3), np.uint8)
    _FakeDet.tracks = [(7, 10, 10, 50, 110)]
    brain.heat_mask(frame, 0.0)
    brain.heat_mask(frame, 2.0)                          # dwelling
    brain.reset()
    assert brain.heat_mask(frame, 2.1).max() == 0        # timer restarted
