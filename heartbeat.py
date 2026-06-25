"""Heartbeat loop — runs in a daemon thread, every N seconds."""
from __future__ import annotations

import logging
import threading

from agent_paths import running_version
from api_client import ApiClient
from net_utils import local_ip

log = logging.getLogger("agent.heartbeat")


def start_heartbeat(
    api: ApiClient,
    cfg: dict,
    stop_event: threading.Event,
    healthy_event: threading.Event | None = None,
) -> threading.Thread:
    interval = max(5, int(cfg.get("heartbeat_interval_s", 30)))

    def loop():
        # Fix #6: heartbeat is naturally a forever-loop already — every tick
        # is an independent attempt, so a failed tick just gets retried on
        # the next interval. The HTTPAdapter handles short blips (5 tries
        # with backoff inside one call); the outer loop handles longer
        # outages (backend down for minutes/hours).
        while not stop_event.is_set():
            try:
                api.heartbeat({
                    "ip_local": local_ip(),
                    # Report the version of the code actually running (from the
                    # VERSION file in CODE_DIR), not a static config value — this
                    # is how the cloud and the rollback watchdog observe a swap.
                    "agent_version": running_version(),
                })
                # First successful beat after boot signals the watchdog the new
                # version is alive and reachable.
                if healthy_event is not None and not healthy_event.is_set():
                    healthy_event.set()
            except Exception as e:
                log.warning("heartbeat failed (will retry next tick): %s", e)
            stop_event.wait(interval)

    t = threading.Thread(target=loop, name="heartbeat", daemon=True)
    t.start()
    return t
