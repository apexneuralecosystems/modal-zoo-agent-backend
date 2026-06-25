"""End-to-end self-update simulation that runs on Windows (no launchd, no Mac).

Spins up a real local HTTP server that serves a version zip, then drives the
actual download path (`updater._default_download` -> requests.get) through
verify -> unpack -> pointer swap, and finally exercises the watchdog rollback.
This mirrors what a Mac does, minus the launchd re-exec (which we can't run here).
"""
import hashlib
import importlib
import io
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer


def _make_zip(version: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("VERSION", version + "\n")
        z.writestr("main.py", f"# agent code for {version}\n")
        z.writestr("run.sh", "#!/bin/sh\n")
    return buf.getvalue()


def _serve(blob: bytes):
    """Start a one-route HTTP server returning `blob`; returns (url, shutdown)."""
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), H)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return f"http://127.0.0.1:{port}/agent.zip", httpd.shutdown


def _bootstrap(tmp_path, monkeypatch, current="1.0.0"):
    code = tmp_path / "versions" / current
    code.mkdir(parents=True)
    (code / "VERSION").write_text(current + "\n", encoding="utf-8")
    (tmp_path / "current_version").write_text(current + "\n", encoding="utf-8")
    (tmp_path / "last_good").write_text(current + "\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_CODE_DIR", str(code))
    monkeypatch.setenv("AGENT_HOME", str(tmp_path))
    import agent_paths
    importlib.reload(agent_paths)
    import updater
    importlib.reload(updater)
    import watchdog
    importlib.reload(watchdog)
    return agent_paths, updater, watchdog


def test_full_update_over_real_http_then_rollback(tmp_path, monkeypatch):
    ap, updater, watchdog = _bootstrap(tmp_path, monkeypatch, current="1.0.0")

    # --- 1. Publish 1.0.1 and apply it through the REAL download path ---------
    blob = _make_zip("1.0.1")
    url, shutdown = _serve(blob)
    try:
        payload = {
            "version": "1.0.1",
            "zip_url": url,
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
        installed = updater.apply_upgrade(payload)  # no download= -> real requests.get
    finally:
        shutdown()

    assert installed == "1.0.1"
    assert (ap.VERSIONS_DIR / "1.0.1" / "main.py").exists()
    assert ap.CURRENT_VERSION_FILE.read_text().strip() == "1.0.1"

    # --- 2. New version boots healthy: watchdog promotes it to last_good ------
    # Simulate the post-swap boot now running 1.0.1.
    monkeypatch.setenv("AGENT_CODE_DIR", str(ap.VERSIONS_DIR / "1.0.1"))
    importlib.reload(ap)
    importlib.reload(watchdog)
    assert ap.running_version() == "1.0.1"

    healthy, stop = threading.Event(), threading.Event()
    t = watchdog.start_watchdog(healthy, stop, timeout_s=2)
    healthy.set()
    t.join(timeout=3)
    assert watchdog.read_last_good() == "1.0.1"
    assert not stop.is_set()

    # --- 3. Push a BAD 1.0.2 that never goes healthy: watchdog rolls back -----
    (ap.VERSIONS_DIR / "1.0.2").mkdir()
    (ap.VERSIONS_DIR / "1.0.2" / "VERSION").write_text("1.0.2\n", encoding="utf-8")
    updater.write_current_version("1.0.2")
    monkeypatch.setenv("AGENT_CODE_DIR", str(ap.VERSIONS_DIR / "1.0.2"))
    importlib.reload(ap)
    importlib.reload(watchdog)
    assert ap.running_version() == "1.0.2"

    healthy2, stop2 = threading.Event(), threading.Event()
    t2 = watchdog.start_watchdog(healthy2, stop2, timeout_s=1)  # never healthy
    t2.join(timeout=3)
    assert stop2.is_set()
    assert ap.CURRENT_VERSION_FILE.read_text().strip() == "1.0.1"  # rolled back to last_good
