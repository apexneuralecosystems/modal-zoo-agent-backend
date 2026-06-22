from dwell_gate import DwellGate


def test_walkthrough_never_counts():
    """A person moving across the frame each sample never dwells."""
    g = DwellGate(min_seconds=2.0)
    out = []
    for i in range(20):
        # box marches right ~30px each 0.2s — always outside its own tolerance
        x = 100 + i * 30
        out = g.update([("a", x, 100, x + 40, 200)], now=i * 0.2)
        assert out == [], f"walk-through should not dwell at step {i}"


def test_standing_counts_after_two_seconds():
    """A still person starts counting only once 2s have elapsed."""
    g = DwellGate(min_seconds=2.0)
    box = ("b", 100, 100, 140, 200)
    assert g.update([box], now=0.0) == []      # just arrived
    assert g.update([box], now=1.9) == []       # not yet 2s
    res = g.update([box], now=2.0)              # hits threshold
    assert res == [(100, 100, 140, 200)]
    res2 = g.update([box], now=2.5)             # keeps counting while it stays
    assert res2 == [(100, 100, 140, 200)]


def test_small_shuffle_within_tolerance_still_dwells():
    """Tiny movements (shifting weight) stay within ~half a body width."""
    g = DwellGate(min_seconds=2.0, move_frac=0.5)  # 40px box -> 20px tolerance
    assert g.update([("c", 100, 100, 140, 200)], now=0.0) == []
    # shifts 10px (< 20px tolerance) and 2s pass -> still dwelling
    res = g.update([("c", 110, 100, 150, 200)], now=2.1)
    assert len(res) == 1


def test_leaving_resets_timer():
    """If a track vanishes then reappears, its timer starts over."""
    g = DwellGate(min_seconds=2.0)
    box = ("d", 100, 100, 140, 200)
    g.update([box], now=0.0)
    g.update([box], now=2.0)             # dwelling now
    g.update([], now=3.0)                # left frame -> forgotten
    assert g.update([box], now=3.1) == []  # reappeared -> timer restarts
