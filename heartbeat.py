"""Heartbeat loop — runs in a daemon thread, every N seconds."""
from __future__ import annotations

import logging
import threading
import time

from api_client import ApiClient
from register import _local_ip

log = logging.getLogger("agent.heartbeat")


def start_heartbeat(api: ApiClient, cfg: dict, stop_event: threading.Event) -> threading.Thread:
    interval = max(5, int(cfg.get("heartbeat_interval_s", 30)))

    def loop():
        while not stop_event.is_set():
            try:
                api.heartbeat({
                    "ip_local": _local_ip(),
                    "agent_version": cfg.get("agent_version", "1.0.0"),
                })
            except Exception as e:
                log.warning("heartbeat failed: %s", e)
            stop_event.wait(interval)

    t = threading.Thread(target=loop, name="heartbeat", daemon=True)
    t.start()
    return t
