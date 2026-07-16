"""Entrypoint — boot order:
  1. load config + setup logging
  2. start registration retry loop in the background (non-blocking)
  3. start heartbeat/watchdog/discovery/telemetry/etc threads immediately
  4. run poller in foreground (blocks)
Registration does not gate step 3+: heartbeat/poller/etc only need the
pre-provisioned secret_token (already valid before the agent even boots),
not a successful /agent/register call — so a cloud outage at boot no longer
leaves the whole agent idle. See register.py / _register_loop below.
On SIGTERM / SIGINT, stop_event is set; threads + workers shut down cleanly.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading

from pathlib import Path

from api_client import ApiClient
from asset_cache import prune_cache
from commands import start_commands
from config_loader import load_config, setup_logging
from discovery import start_discovery
from heartbeat import start_heartbeat
from log_shipper import start_log_shipper
from poller import start_poller
from register import register
from telemetry import start_telemetry
from updater import prune_old_versions
from watchdog import start_watchdog
from ws_streamer import start_ws_streamer


def _start_cache_pruner(cfg: dict, stop_event: threading.Event) -> threading.Thread:
    """Background thread: prunes the model + script caches once a day.
    Files older than 30d that aren't being used get deleted; if needed
    again they're just re-downloaded on next /agent/jobs poll.

    Also sweeps the fetch_clip scratch folder (`<log_dir>/clips/`) as a safety
    net: normally a clip is deleted right after its S3 upload attempt (success
    or failure — see clip_recorder.py), but any clip left behind by a crash or
    a killed process mid-upload would otherwise sit there forever.

    Also prunes superseded agent code versions under versions/<v>/ — every
    published update used to accumulate on disk forever. Only the running,
    Stable, and last-known-good versions are protected (watchdog.py's
    rollback needs them present without a re-download); everything else is
    deleted and would simply be re-fetched if ever needed again."""
    log = logging.getLogger("agent.cache")
    interval = 24 * 3600
    clips_dir = os.path.join(
        cfg.get("log_dir") or str(Path(__file__).resolve().parent / "logs"),
        "clips",
    )
    CLIPS_MAX_AGE_S = 24 * 3600  # leftover clips are scratch, not a cache — 1 day is plenty

    def loop():
        # Run once at startup, then daily.
        while not stop_event.is_set():
            for d, max_age in ((cfg["models_cache_dir"], None),
                                (cfg["scripts_cache_dir"], None),
                                (clips_dir, CLIPS_MAX_AGE_S)):
                try:
                    n = prune_cache(d) if max_age is None else prune_cache(d, max_age)
                    if n:
                        log.info("pruned %s old files from %s", n, d)
                except Exception as e:
                    log.warning("prune %s failed: %s", d, e)
            try:
                removed = prune_old_versions()
                if removed:
                    log.info("pruned %d superseded agent version(s): %s", len(removed), removed)
            except Exception as e:
                log.warning("prune old versions failed: %s", e)
            stop_event.wait(interval)

    t = threading.Thread(target=loop, name="cache-pruner", daemon=True)
    t.start()
    return t


def _register_loop(api: ApiClient, cfg: dict, stop_event: threading.Event, log: logging.Logger) -> None:
    """Keeps retrying /agent/register forever (capped backoff) in the
    background. Runs alongside every other subsystem instead of blocking
    them — see the Fix #16 note in main()."""
    backoff = 5
    while not stop_event.is_set():
        if register(api, cfg):
            log.info("registration succeeded")
            return
        log.warning("registration failed — retrying in %ss", backoff)
        if stop_event.wait(backoff):
            return
        backoff = min(300, backoff * 2)


def main() -> int:
    cfg = load_config()
    log = setup_logging(cfg["log_dir"], name="agent")
    log.info("=" * 60)
    log.info("Vision AI Mac Agent starting | branch=%s ver=%s", cfg["branch_id"], cfg["agent_version"])
    log.info("=" * 60)

    api = ApiClient(cfg["server_url"], cfg["secret_token"])

    stop_event = threading.Event()
    # Set by the heartbeat on its first successful beat; the watchdog waits on
    # it to confirm a freshly-swapped version is alive (else it rolls back).
    healthy_event = threading.Event()

    def _early_stop(signum, frame):
        log.info("signal %s during boot — aborting", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _early_stop)
    signal.signal(signal.SIGINT, _early_stop)

    # Fix #16: registration used to block every other subsystem, so a cloud
    # outage right at boot (e.g. a restart landing during cloud maintenance)
    # left the whole Mac doing nothing at all — no heartbeat, no camera
    # jobs, nothing — even though the process itself wasn't hung. Run it as
    # a background retry loop instead; everything else starts immediately.
    threading.Thread(
        target=_register_loop,
        args=(api, cfg, stop_event, log),
        name="register",
        daemon=True,
    ).start()

    start_heartbeat(api, cfg, stop_event, healthy_event)
    # Rollback watchdog: if this boot is a freshly-swapped version that never
    # heartbeats healthy within the window, revert current_version to last_good.
    # 15 minutes, not 5 — a short server-maintenance window shouldn't look
    # identical to a genuinely broken update. Repeated failures of the same
    # version also back off (see updater.failure_backoff_remaining_s).
    start_watchdog(healthy_event, stop_event, timeout_s=int(cfg.get("update_watchdog_s", 900)))
    # Both intervals are 15s now. The list_devices GET is cheap; a 5-min idle
    # poll meant that newly-added NVRs sat unprobed for up to 5 minutes, and
    # the UI's 60s "waiting for Mac" timeout fires long before that — so the
    # user sees a fake "Mac didn't respond" error even though the agent is
    # alive and well. 15s feels instantaneous to humans and costs nothing.
    start_discovery(api, stop_event, interval_idle_s=15, interval_active_s=15)
    start_log_shipper(api, cfg["log_dir"], stop_event)
    start_telemetry(api, stop_event)
    start_ws_streamer(cfg["server_url"], cfg["secret_token"], stop_event)
    start_commands(api, cfg, stop_event)
    _start_cache_pruner(cfg, stop_event)
    poller_thread, _ = start_poller(api, cfg, stop_event)

    poller_thread.join()
    log.info("agent stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
