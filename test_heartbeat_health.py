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
