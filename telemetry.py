"""Telemetry push — runs in a background thread, posts host metrics every minute."""
from __future__ import annotations

import logging
import threading
import time

import psutil

from api_client import ApiClient

log = logging.getLogger("agent.telemetry")
INTERVAL_S = 60
_BOOT_TIME = psutil.boot_time()


def _snapshot() -> dict:
    try:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent
        uptime = int(time.time() - _BOOT_TIME)
        return {
            "cpu_usage": float(cpu),
            "memory_usage": float(mem),
            "disk_usage": float(disk),
            "uptime_s": uptime,
        }
    except Exception as e:
        log.warning("telemetry snapshot failed: %s", e)
        return {}


def start_telemetry(api: ApiClient, stop_event: threading.Event) -> threading.Thread:
    def loop():
        # Prime cpu_percent so first reading isn't 0.
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        while not stop_event.is_set():
            payload = _snapshot()
            if payload:
                try:
                    api.post_telemetry(payload)
                except Exception as e:
                    log.warning("telemetry post failed: %s", e)
            stop_event.wait(INTERVAL_S)

    t = threading.Thread(target=loop, name="telemetry", daemon=True)
    t.start()
    return t
