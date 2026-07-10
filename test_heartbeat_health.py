"""Heartbeat reports the running VERSION and sets healthy_event on first success."""
import importlib
import threading


class _FakeApi:
    def __init__(self):
        self.calls = []

    def heartbeat(self, payload):
        self.calls.append(payload)
        return {"ok": True}


class _FailingApi:
    def heartbeat(self, payload):
        raise RuntimeError("network down")


def test_heartbeat_sets_healthy_and_reports_version(monkeypatch):
    import heartbeat
    importlib.reload(heartbeat)
    monkeypatch.setattr(heartbeat, "local_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(heartbeat, "running_version", lambda: "9.9.9")

    api = _FakeApi()
    stop = threading.Event()
    healthy = threading.Event()
    t = heartbeat.start_heartbeat(api, {"heartbeat_interval_s": 5}, stop, healthy)
    assert healthy.wait(timeout=2.0)
    stop.set()
    t.join(timeout=2.0)
    assert api.calls[0]["agent_version"] == "9.9.9"
    assert api.calls[0]["ip_local"] == "127.0.0.1"


def test_heartbeat_does_not_signal_healthy_on_failure(monkeypatch):
    import heartbeat
    importlib.reload(heartbeat)
    monkeypatch.setattr(heartbeat, "local_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(heartbeat, "running_version", lambda: "1.0.0")

    stop = threading.Event()
    healthy = threading.Event()
    t = heartbeat.start_heartbeat(_FailingApi(), {"heartbeat_interval_s": 5}, stop, healthy)
    assert not healthy.wait(timeout=1.0)  # never becomes healthy while beats fail
    stop.set()
    t.join(timeout=2.0)


# ── Punch-list #42: back off on repeated heartbeat failures, but reset to the
# configured interval immediately on any success — never hammer a struggling
# server, and never delay detecting a server that's actually reachable. ──

def test_next_interval_doubles_on_failure_and_caps():
    import heartbeat
    importlib.reload(heartbeat)
    base = 30
    current = base
    # Each consecutive failure doubles the wait...
    current = heartbeat._next_interval(current, base, failed=True)
    assert current == 60
    current = heartbeat._next_interval(current, base, failed=True)
    assert current == 120
    current = heartbeat._next_interval(current, base, failed=True)
    assert current == 240
    # ...but never exceeds the cap, however many failures pile up.
    current = heartbeat._next_interval(current, base, failed=True)
    assert current == heartbeat.MAX_BACKOFF_INTERVAL_S
    current = heartbeat._next_interval(current, base, failed=True)
    assert current == heartbeat.MAX_BACKOFF_INTERVAL_S


def test_next_interval_resets_to_base_on_success():
    import heartbeat
    importlib.reload(heartbeat)
    base = 30
    backed_off = heartbeat._next_interval(base, base, failed=True)
    assert backed_off > base
    reset = heartbeat._next_interval(backed_off, base, failed=False)
    assert reset == base


def test_heartbeat_loop_backs_off_on_failures_and_resets_on_success(monkeypatch):
    """End-to-end: drive the real loop through fail, fail, fail, succeed,
    fail and check the actual wait() calls it made, without any real sleep."""
    import heartbeat
    importlib.reload(heartbeat)
    monkeypatch.setattr(heartbeat, "local_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(heartbeat, "running_version", lambda: "1.0.0")

    outcomes = [Exception("down"), Exception("down"), Exception("down"), None, Exception("down")]

    class _FlakyApi:
        def __init__(self):
            self.n = 0

        def heartbeat(self, payload):
            outcome = outcomes[self.n]
            self.n += 1
            if isinstance(outcome, Exception):
                raise outcome
            return {"ok": True}

    stop = threading.Event()
    waited: list[float] = []
    real_wait = stop.wait

    def fake_wait(timeout=None):
        waited.append(timeout)
        if len(waited) >= len(outcomes):
            stop.set()
        return real_wait(0)  # don't actually block the test

    monkeypatch.setattr(stop, "wait", fake_wait)

    t = heartbeat.start_heartbeat(_FlakyApi(), {"heartbeat_interval_s": 10}, stop)
    t.join(timeout=2.0)

    assert waited == [20, 40, 80, 10, 20]
