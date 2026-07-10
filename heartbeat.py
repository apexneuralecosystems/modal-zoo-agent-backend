"""Heartbeat loop — runs in a daemon thread, every N seconds."""
from __future__ import annotations

import logging
import threading

from agent_paths import running_version
from api_client import ApiClient
from net_utils import local_ip

log = logging.getLogger("agent.heartbeat")

# On a failed heartbeat, double the wait before the next attempt (capped)
# instead of hammering a struggling/unreachable server at the full configured
# rate. A SUCCESS always resets immediately back to the configured interval —
# the backoff only ever slows down retries after failures, so a genuinely
# reachable server is never detected late. This is safe for "is the Mac
# online" purposes too: the cloud's own HeartbeatSweeperService already marks
# a Mac offline based on how long it's been silent, independent of how the
# agent paces its own retries.
BACKOFF_MULTIPLIER = 2
MAX_BACKOFF_INTERVAL_S = 300


def _next_interval(current_interval: float, base_interval: float, failed: bool) -> float:
    """Given the interval just used and whether that attempt failed, return
    the interval to wait before the next attempt."""
    if not failed:
        return base_interval
    return min(current_interval * BACKOFF_MULTIPLIER, MAX_BACKOFF_INTERVAL_S)


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
        # outages (backend down for minutes/hours) via the backoff below.
        current_interval = interval
        while not stop_event.is_set():
            failed = False
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
                failed = True
                log.warning("heartbeat failed (will retry with backoff): %s", e)
            current_interval = _next_interval(current_interval, interval, failed)
            stop_event.wait(current_interval)

    t = threading.Thread(target=loop, name="heartbeat", daemon=True)
    t.start()
    return t
