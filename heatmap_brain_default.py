"""Default (baked-in) heatmap brain.

The runner (`heatmap_worker.py`) normally downloads a "brain" from the cloud, but
falls back to THIS module when no cloud brain is published or a downloaded one
fails to load — so a bad publish can never kill all heatmaps.

A brain exposes three callables:
  setup(ctx: dict) -> None        # build detector + dwell gate from ctx
  heat_mask(frame, now) -> mask   # HxW uint8, 255 where heat should be added
  reset() -> None                 # clear dwell state (called on a new day)

ctx keys: model_path, conf, dwell_seconds, move_frac, width, height, camera_id.
"""
from heatmap_detector import PersonDetector, mask_from_boxes
from dwell_gate import DwellGate

_state: dict = {}


def setup(ctx: dict) -> None:
    _state["dwell_seconds"] = float(ctx.get("dwell_seconds", 2.0))
    _state["move_frac"] = float(ctx.get("move_frac", 0.5))
    _state["wh"] = (int(ctx["width"]), int(ctx["height"]))
    _state["det"] = PersonDetector(
        ctx["model_path"], conf=float(ctx.get("conf", 0.3)), use_tracking=True
    )
    _state["gate"] = DwellGate(
        min_seconds=_state["dwell_seconds"], move_frac=_state["move_frac"]
    )


def reset() -> None:
    """Drop dwell state — used when the runner rolls over to a new day."""
    _state["gate"] = DwellGate(
        min_seconds=_state["dwell_seconds"], move_frac=_state["move_frac"]
    )


def heat_mask(frame_bgr, now_ts: float):
    w, h = _state["wh"]
    tracks = _state["det"].detect_tracks(frame_bgr)
    # Key each track by its YOLO id; fall back to a coarse position bucket so the
    # dwell timer still works if the tracker didn't assign an id.
    keyed = [
        ((tid if tid is not None else ("p", x1 // 40, y1 // 40)), x1, y1, x2, y2)
        for (tid, x1, y1, x2, y2) in tracks
    ]
    dwelling = _state["gate"].update(keyed, now_ts)
    return mask_from_boxes(dwelling, h, w)
