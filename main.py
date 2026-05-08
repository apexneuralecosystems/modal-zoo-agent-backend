"""Entrypoint — boot order:
  1. load config + setup logging
  2. register with cloud
  3. start heartbeat thread
  4. start discovery thread
  5. run poller in foreground (blocks)
On SIGTERM / SIGINT, stop_event is set; threads + workers shut down cleanly.
"""
from __future__ import annotations

import logging
import signal
import sys
import threading

from api_client import ApiClient
from asset_cache import prune_cache
from config_loader import load_config, setup_logging
from discovery import start_discovery
from heartbeat import start_heartbeat
from log_shipper import start_log_shipper
from poller import start_poller
from register import register
from telemetry import start_telemetry


def _start_cache_pruner(cfg: dict, stop_event: threading.Event) -> threading.Thread:
    """Background thread: prunes the model + script caches once a day.
    Files older than 30d that aren't being used get deleted; if needed
    again they're just re-downloaded on next /agent/jobs poll."""
    log = logging.getLogger("agent.cache")
    interval = 24 * 3600

    def loop():
        # Run once at startup, then daily.
        while not stop_event.is_set():
            for d in (cfg["models_cache_dir"], cfg["scripts_cache_dir"]):
                try:
                    n = prune_cache(d)
                    if n:
                        log.info("pruned %s old files from %s", n, d)
                except Exception as e:
                    log.warning("prune %s failed: %s", d, e)
            stop_event.wait(interval)

    t = threading.Thread(target=loop, name="cache-pruner", daemon=True)
    t.start()
    return t


def main() -> int:
    cfg = load_config()
    log = setup_logging(cfg["log_dir"], name="agent")
    log.info("=" * 60)
    log.info("Vision AI Mac Agent starting | branch=%s ver=%s", cfg["branch_id"], cfg["agent_version"])
    log.info("=" * 60)

    api = ApiClient(cfg["server_url"], cfg["secret_token"])

    stop_event = threading.Event()

    def _early_stop(signum, frame):
        log.info("signal %s during boot — aborting", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _early_stop)
    signal.signal(signal.SIGINT, _early_stop)

    # Fix #6: keep retrying registration forever (with backoff) instead of
    # aborting boot. If the cloud is briefly down when the Mac powers on,
    # the agent must come up on its own once the cloud is back — not stay
    # dead until someone manually restarts it.
    backoff = 5
    while not stop_event.is_set():
        if register(api, cfg):
            break
        log.warning("registration failed — retrying in %ss", backoff)
        if stop_event.wait(backoff):
            break
        backoff = min(300, backoff * 2)

    if stop_event.is_set():
        log.info("stopped before registration completed")
        return 0

    start_heartbeat(api, cfg, stop_event)
    start_discovery(api, stop_event, interval_s=300)
    start_log_shipper(api, cfg["log_dir"], stop_event)
    start_telemetry(api, stop_event)
    _start_cache_pruner(cfg, stop_event)
    poller_thread, _ = start_poller(api, cfg, stop_event)

    poller_thread.join()
    log.info("agent stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
