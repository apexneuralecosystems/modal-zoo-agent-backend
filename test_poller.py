"""If a worker's asset download fails (exit code 2), the poller must back off
instead of respawning it every tick forever, and must report the failure to
the backend once it gives up. This is distinct from the exit-99 (RTSP dead
12h) camera-offline path, which already existed and must keep working."""
import threading
import time
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


def test_kill_closes_stdout_and_joins_quick_tail_thread():
    """Fix #18: _kill() should close the pipe and reap a tail thread that
    exits promptly, so it doesn't linger in tracking after the job is gone."""
    poller, api = _make_poller(now_fn=lambda: 0.0)
    proc = MagicMock()
    proc.poll.return_value = None
    proc.stdout = MagicMock()
    poller.running["dep1"] = proc

    finished = threading.Event()
    finished.set()  # simulates a tail thread that has already hit EOF
    t = threading.Thread(target=finished.wait, daemon=True)
    t.start()
    poller.tail_threads["dep1"] = t

    poller._kill("dep1")

    proc.stdout.close.assert_called_once()
    assert "dep1" not in poller.tail_threads


def test_kill_leaves_stuck_tail_thread_tracked_for_sweep():
    """Fix #18: a tail thread that doesn't exit within the join timeout must
    stay tracked (not silently forgotten) so _sweep_tail_threads() keeps
    watching it."""
    poller, api = _make_poller(now_fn=lambda: 0.0)
    proc = MagicMock()
    proc.poll.return_value = None
    proc.stdout = MagicMock()
    poller.running["dep1"] = proc

    block_forever = threading.Event()  # never set — simulates a stuck reader
    t = threading.Thread(target=block_forever.wait, daemon=True)
    t.start()
    poller.tail_threads["dep1"] = t

    poller._kill("dep1")

    assert "dep1" in poller.tail_threads
    assert poller.tail_threads["dep1"].is_alive()


def test_sweep_tail_threads_cleans_up_orphan_once_it_exits():
    """Fix #18: an orphaned tail thread (job already gone from self.running)
    gets reaped by the periodic sweep as soon as it actually exits."""
    poller, api = _make_poller(now_fn=lambda: 0.0)
    finished = threading.Event()
    t = threading.Thread(target=finished.wait, daemon=True)
    t.start()
    poller.tail_threads["dep1"] = t  # note: "dep1" is NOT in poller.running

    poller._sweep_tail_threads()
    assert "dep1" in poller.tail_threads  # still alive, still being watched
    assert t.is_alive()

    finished.set()
    time.sleep(0.05)
    poller._sweep_tail_threads()
    assert "dep1" not in poller.tail_threads  # reaped once it exited


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
