"""The command poller dispatches the `upgrade` type to the updater and reports
the result. (fetch_clip behaviour is covered by test_clip_recorder.)"""
import threading


class _Api:
    def __init__(self):
        self.results = []

    def post_command_result(self, payload):
        self.results.append(payload)


def test_upgrade_command_invokes_updater(monkeypatch):
    import commands
    called = {}

    def fake_handle(payload, api, stop_event):
        called["payload"] = payload
        stop_event.set()
        return {"version": payload["version"]}

    monkeypatch.setattr(commands, "handle_upgrade", fake_handle)

    api = _Api()
    stop = threading.Event()
    cmd = {"id": "c1", "type": "upgrade",
           "payload": {"version": "1.1.0", "zip_url": "u", "sha256": "s"}}
    commands._process_one(api, {"log_dir": "/tmp"}, cmd, stop)

    assert called["payload"]["version"] == "1.1.0"
    assert api.results[0]["ok"] is True
    assert api.results[0]["result"] == {"version": "1.1.0"}
    assert stop.is_set()


def test_unknown_command_reports_failure():
    import commands
    api = _Api()
    stop = threading.Event()
    cmd = {"id": "c2", "type": "frobnicate", "payload": {}}
    commands._process_one(api, {"log_dir": "/tmp"}, cmd, stop)
    assert api.results[0]["ok"] is False
    assert "unsupported" in api.results[0]["error"]
    assert not stop.is_set()
