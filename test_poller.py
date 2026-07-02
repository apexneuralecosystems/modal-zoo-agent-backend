"""If a worker's asset download fails (exit code 2), the poller must back off
instead of respawning it every tick forever, and must report the failure to
the backend once it gives up. This is distinct from the exit-99 (RTSP dead
12h) camera-offline path, which already existed and must keep working."""
import threading
from unittest.mock import MagicMock

from poller import JobPoller, MAX_DOWNLOAD_FAILURES, DOWNLOAD_RETRY_BACKOFF_S


def _make_poller(now_fn):
    api = MagicMock()
    cfg = {"log_dir": "/tmp", "poll_interval_s": 10}
    return JobPoller(api, cfg, threading.Event(), now_fn=now_fn), api


def _dead_proc(code):
    proc = MagicMock()
    proc.poll.return_value = code
    proc.returncode = code
    return proc


def test_download_failure_does_not_respawn_immediately():
    clock = {"t": 0.0}
    poller, api = _make_poller(now_fn=lambda: clock["t"])
    poller.running["dep1"] = _dead_proc(2)
    poller.hashes["dep1"] = "hash-a"

    poller._reap()

    assert "dep1" not in poller.running
    assert poller._should_hold_off("dep1", "hash-a") is True
    api.mark_deployment_failed.assert_not_called()


def test_download_failure_retries_after_backoff_elapses():
    clock = {"t": 0.0}
    poller, api = _make_poller(now_fn=lambda: clock["t"])
    poller.running["dep1"] = _dead_proc(2)
    poller.hashes["dep1"] = "hash-a"
    poller._reap()

    clock["t"] += DOWNLOAD_RETRY_BACKOFF_S + 1
    assert poller._should_hold_off("dep1", "hash-a") is False


def test_gives_up_after_max_failures_and_reports_to_backend():
    clock = {"t": 0.0}
    poller, api = _make_poller(now_fn=lambda: clock["t"])
    poller.hashes["dep1"] = "hash-a"

    for _ in range(MAX_DOWNLOAD_FAILURES):
        poller.running["dep1"] = _dead_proc(2)
        poller._reap()
        clock["t"] += DOWNLOAD_RETRY_BACKOFF_S + 1

    assert poller._should_hold_off("dep1", "hash-a") is True
    api.mark_deployment_failed.assert_called_once_with("dep1", reason="model_download_failed")


def test_pipeline_change_clears_download_failure_history():
    clock = {"t": 0.0}
    poller, api = _make_poller(now_fn=lambda: clock["t"])
    poller.running["dep1"] = _dead_proc(2)
    poller.hashes["dep1"] = "hash-a"
    poller._reap()

    assert poller._should_hold_off("dep1", "hash-b") is False


def test_exit_99_camera_offline_path_still_works_alongside_download_failures():
    """Regression guard: the pre-existing RTSP-dead-12h path must be unaffected
    by the new download-failure backoff logic."""
    clock = {"t": 0.0}
    poller, api = _make_poller(now_fn=lambda: clock["t"])
    poller.running["dep1"] = _dead_proc(99)
    poller.hashes["dep1"] = "hash-a"

    poller._reap()

    assert "dep1" in poller.permanently_failed
    api.mark_camera_offline.assert_called_once_with("dep1")
    api.mark_deployment_failed.assert_not_called()
