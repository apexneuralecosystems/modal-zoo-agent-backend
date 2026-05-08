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
from config_loader import load_config, setup_logging
from discovery import start_discovery
from heartbeat import start_heartbeat
from log_shipper import start_log_shipper
from poller import start_poller
from register import register
from telemetry import start_telemetry


def main() -> int:
    cfg = load_config()
    log = setup_logging(cfg["log_dir"], name="agent")
    log.info("=" * 60)
    log.info("Vision AI Mac Agent starting | branch=%s ver=%s", cfg["branch_id"], cfg["agent_version"])
    log.info("=" * 60)

    api = ApiClient(cfg["server_url"], cfg["secret_token"])

    if not register(api, cfg):
        log.error("registration failed — aborting boot")
        return 1

    stop_event = threading.Event()

    def _stop(signum, frame):
        log.info("signal %s — shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    start_heartbeat(api, cfg, stop_event)
    start_discovery(api, stop_event, interval_s=300)
    start_log_shipper(api, cfg["log_dir"], stop_event)
    start_telemetry(api, stop_event)
    poller_thread, _ = start_poller(api, cfg, stop_event)

    poller_thread.join()
    log.info("agent stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
