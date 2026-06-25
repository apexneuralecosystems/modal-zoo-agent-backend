"""Updater: verify checksum, unpack, switch the current_version pointer.
Pointer is a text file (not a symlink), so these run on Windows too."""
import hashlib
import importlib
import io
import zipfile

import pytest


def _make_zip(version: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("VERSION", version + "\n")
        z.writestr("main.py", "# new code\n")
    return buf.getvalue()


def _setup(tmp_path, monkeypatch, current="1.0.0"):
    code = tmp_path / "versions" / current
    code.mkdir(parents=True)
    (code / "VERSION").write_text(current + "\n", encoding="utf-8")
    (tmp_path / "current_version").write_text(current + "\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_CODE_DIR", str(code))
    monkeypatch.setenv("AGENT_HOME", str(tmp_path))
    import agent_paths
    importlib.reload(agent_paths)
    import updater
    return importlib.reload(updater), agent_paths


def test_apply_upgrade_unpacks_and_switches_pointer(tmp_path, monkeypatch):
    updater, ap = _setup(tmp_path, monkeypatch)
    blob = _make_zip("1.1.0")
    payload = {"version": "1.1.0", "zip_url": "http://x/a.zip",
               "sha256": hashlib.sha256(blob).hexdigest()}

    v = updater.apply_upgrade(payload, download=lambda url: blob)

    assert v == "1.1.0"
    assert (ap.VERSIONS_DIR / "1.1.0" / "VERSION").read_text().strip() == "1.1.0"
    assert ap.CURRENT_VERSION_FILE.read_text().strip() == "1.1.0"


def test_apply_upgrade_rejects_bad_checksum(tmp_path, monkeypatch):
    updater, ap = _setup(tmp_path, monkeypatch)
    blob = _make_zip("1.1.0")
    payload = {"version": "1.1.0", "zip_url": "http://x/a.zip", "sha256": "deadbeef"}

    with pytest.raises(ValueError):
        updater.apply_upgrade(payload, download=lambda url: blob)

    # pointer must NOT have moved and the new version must NOT exist
    assert ap.CURRENT_VERSION_FILE.read_text().strip() == "1.0.0"
    assert not (ap.VERSIONS_DIR / "1.1.0").exists()


def test_apply_upgrade_installs_deps_before_swap(tmp_path, monkeypatch):
    updater, ap = _setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(updater, "_install_deps", lambda d: calls.append(d.name))
    blob = _make_zip("1.1.0")
    payload = {"version": "1.1.0", "zip_url": "u", "sha256": hashlib.sha256(blob).hexdigest()}

    updater.apply_upgrade(payload, download=lambda url: blob)

    assert calls == ["1.1.0"]  # deps installed for the new version
    assert ap.CURRENT_VERSION_FILE.read_text().strip() == "1.1.0"


def test_apply_upgrade_aborts_if_deps_fail(tmp_path, monkeypatch):
    updater, ap = _setup(tmp_path, monkeypatch)

    def boom(_d):
        raise RuntimeError("pip install failed")
    monkeypatch.setattr(updater, "_install_deps", boom)
    blob = _make_zip("1.1.0")
    payload = {"version": "1.1.0", "zip_url": "u", "sha256": hashlib.sha256(blob).hexdigest()}

    with pytest.raises(RuntimeError):
        updater.apply_upgrade(payload, download=lambda url: blob)

    # pointer must NOT have moved — old version keeps running
    assert ap.CURRENT_VERSION_FILE.read_text().strip() == "1.0.0"


def test_apply_upgrade_with_fallback_ensures_stable_on_disk(tmp_path, monkeypatch):
    updater, ap = _setup(tmp_path, monkeypatch, current="1.0.0")
    monkeypatch.setattr(updater, "_install_deps", lambda d: None)
    latest = _make_zip("1.1.0")
    stable = _make_zip("1.0.5")

    def dl(url):
        return latest if "latest" in url else stable

    payload = {
        "version": "1.1.0", "zip_url": "http://x/latest.zip",
        "sha256": hashlib.sha256(latest).hexdigest(), "target": "latest",
        "fallback": {
            "version": "1.0.5", "zip_url": "http://x/stable.zip",
            "sha256": hashlib.sha256(stable).hexdigest(),
        },
    }
    updater.apply_upgrade(payload, download=dl)

    assert (ap.VERSIONS_DIR / "1.1.0").exists()      # latest installed
    assert (ap.VERSIONS_DIR / "1.0.5").exists()      # stable fallback fetched
    assert ap.STABLE_FILE.read_text().strip() == "1.0.5"
    assert ap.CURRENT_VERSION_FILE.read_text().strip() == "1.1.0"


def test_apply_upgrade_stable_target_records_stable(tmp_path, monkeypatch):
    updater, ap = _setup(tmp_path, monkeypatch, current="1.0.0")
    monkeypatch.setattr(updater, "_install_deps", lambda d: None)
    blob = _make_zip("1.0.5")
    payload = {"version": "1.0.5", "zip_url": "u",
               "sha256": hashlib.sha256(blob).hexdigest(), "target": "stable"}
    updater.apply_upgrade(payload, download=lambda url: blob)
    assert ap.STABLE_FILE.read_text().strip() == "1.0.5"


def test_handle_upgrade_sets_stop_event(tmp_path, monkeypatch):
    import threading
    updater, ap = _setup(tmp_path, monkeypatch)
    blob = _make_zip("1.1.0")
    payload = {"version": "1.1.0", "zip_url": "u",
               "sha256": hashlib.sha256(blob).hexdigest()}
    monkeypatch.setattr(updater, "_default_download", lambda url: blob)

    stop = threading.Event()
    result = updater.handle_upgrade(payload, api=None, stop_event=stop)

    assert result == {"version": "1.1.0"}
    assert stop.is_set()
