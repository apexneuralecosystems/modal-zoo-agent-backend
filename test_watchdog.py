"""Watchdog: promote healthy version to last_good; roll back the pointer on timeout.
Pointer is a text file, so this runs on Windows."""
import importlib
import threading


def _setup(tmp_path, monkeypatch, running="1.1.0", last_good="1.0.0"):
    versions = tmp_path / "versions"
    (versions / "1.0.0").mkdir(parents=True)
    (versions / "1.1.0").mkdir(parents=True)
    code = versions / running
    (code / "VERSION").write_text(running + "\n", encoding="utf-8")
    (tmp_path / "current_version").write_text(running + "\n", encoding="utf-8")
    if last_good is not None:
        (tmp_path / "last_good").write_text(last_good + "\n", encoding="utf-8")

    monkeypatch.setenv("AGENT_CODE_DIR", str(code))
    monkeypatch.setenv("AGENT_HOME", str(tmp_path))
    import agent_paths
    importlib.reload(agent_paths)
    import updater
    importlib.reload(updater)
    import watchdog
    return importlib.reload(watchdog), agent_paths


def test_healthy_records_last_good(tmp_path, monkeypatch):
    wd, ap = _setup(tmp_path, monkeypatch)
    healthy = threading.Event()
    stop = threading.Event()
    t = wd.start_watchdog(healthy, stop, timeout_s=2)
    healthy.set()
    t.join(timeout=3)
    assert wd.read_last_good() == "1.1.0"
    assert not stop.is_set()
    assert ap.CURRENT_VERSION_FILE.read_text().strip() == "1.1.0"


def test_timeout_rolls_back_pointer(tmp_path, monkeypatch):
    wd, ap = _setup(tmp_path, monkeypatch)
    healthy = threading.Event()
    stop = threading.Event()
    t = wd.start_watchdog(healthy, stop, timeout_s=1)  # never set healthy
    t.join(timeout=3)
    assert stop.is_set()
    assert ap.CURRENT_VERSION_FILE.read_text().strip() == "1.0.0"  # rolled back


def test_timeout_prefers_stable_over_last_good(tmp_path, monkeypatch):
    # versions: 1.0.0 (last_good), 1.0.5 (stable), 1.1.0 (running/current)
    versions = tmp_path / "versions"
    for v in ("1.0.0", "1.0.5", "1.1.0"):
        (versions / v).mkdir(parents=True)
        (versions / v / "VERSION").write_text(v + "\n", encoding="utf-8")
    (tmp_path / "current_version").write_text("1.1.0\n", encoding="utf-8")
    (tmp_path / "last_good").write_text("1.0.0\n", encoding="utf-8")
    (tmp_path / "stable_version").write_text("1.0.5\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_CODE_DIR", str(versions / "1.1.0"))
    monkeypatch.setenv("AGENT_HOME", str(tmp_path))
    import agent_paths; importlib.reload(agent_paths)
    import updater; importlib.reload(updater)
    import watchdog as wd; importlib.reload(wd)

    healthy = threading.Event(); stop = threading.Event()
    t = wd.start_watchdog(healthy, stop, timeout_s=1)  # never healthy
    t.join(timeout=3)
    assert stop.is_set()
    # rolled back to STABLE (1.0.5), not last_good (1.0.0)
    assert (tmp_path / "current_version").read_text().strip() == "1.0.5"


def test_first_boot_no_last_good_does_not_roll_back(tmp_path, monkeypatch):
    wd, ap = _setup(tmp_path, monkeypatch, running="1.0.0", last_good=None)
    healthy = threading.Event()
    stop = threading.Event()
    t = wd.start_watchdog(healthy, stop, timeout_s=1)  # no rollback target
    t.join(timeout=3)
    assert not stop.is_set()
    assert ap.CURRENT_VERSION_FILE.read_text().strip() == "1.0.0"
