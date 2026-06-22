"""Decide which tracked people are 'dwelling' — i.e. have stayed within roughly
one spot for at least `min_seconds`. This is what turns the heatmap from "anyone
the camera sees" into "people who actually wait/linger" (queue, shelf, lift
lobby) while ignoring people just walking through.

Pure + unit-testable: no model, no RTSP, no clock. The caller feeds it the
tracked boxes for the current frame plus the current timestamp; it returns the
subset of boxes that have been stationary long enough to count.
"""


class DwellGate:
    """Per-track stationarity timer.

    A track gets an *anchor* (its position when first seen, or when it last
    moved). While the track stays within `move_frac` of its box-width of that
    anchor, the timer keeps running; once it has been stationary for
    `min_seconds`, the track counts as *dwelling* and its box is returned every
    frame thereafter (so heat keeps accumulating while it stays). Move beyond the
    tolerance → the anchor + timer reset (that is a walk-through, not a dwell).
    """

    def __init__(self, min_seconds: float = 2.0, move_frac: float = 0.5):
        self.min_seconds = float(min_seconds)
        self.move_frac = float(move_frac)
        self._anchors: dict = {}  # key -> (anchor_cx, anchor_cy, stationary_since_ts)

    def update(self, tracks, now: float):
        """tracks: iterable of (key, x1, y1, x2, y2). `key` is a stable per-person
        id (YOLO track id, or a position-bucket fallback). `now`: epoch seconds.

        Returns the list of (x1, y1, x2, y2) boxes that are currently dwelling.
        """
        dwelling = []
        seen = set()
        for key, x1, y1, x2, y2 in tracks:
            seen.add(key)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            radius = max(1.0, (x2 - x1) * self.move_frac)
            anchor = self._anchors.get(key)
            if anchor is None:
                # New track — drop an anchor and start its timer.
                self._anchors[key] = (cx, cy, now)
                continue
            ax, ay, since = anchor
            moved = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5
            if moved > radius:
                # Walked out of tolerance — reset the timer at the new spot.
                self._anchors[key] = (cx, cy, now)
            elif now - since >= self.min_seconds:
                # Stayed put long enough — count it (and keep counting while here).
                dwelling.append((x1, y1, x2, y2))
        # Forget tracks that disappeared so the dict doesn't grow without bound.
        for key in [k for k in self._anchors if k not in seen]:
            del self._anchors[key]
        return dwelling
