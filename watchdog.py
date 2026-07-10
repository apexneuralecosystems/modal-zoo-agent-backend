"""Post-upgrade safety net. On startup, if we're running a version that isn't
the last-known-good one, give it `timeout_s` to heartbeat healthy. If it does,
promote it to last_good. If it doesn't, switch the current_version pointer back
to last_good and exit so launchd re-execs the known-good code.

Uses the same text-file pointer as updater (no symlink)."""
from __future__ import annotations

import logging
import threading

from agent_paths import LAST_GOOD_FILE, STABLE_FILE, VERSIONS_DIR, running_version
from updater import clear_failed_version, record_failed_version, write_current_version

log = logging.getLogger("agent.watchdog")


def read_last_good() -> str | None:
    try:
        v = LAST_GOOD_FILE.read_text(encoding="utf-8").strip()
        return v or None
    except OSError:
        return None


def read_stable() -> str | None:
    try:
        v = STABLE_FILE.read_text(encoding="utf-8").strip()
        return v or None
    except OSError:
        return None


def _pick_rollback_target(current: str) -> str | None:
    """Prefer the Stable channel as the fallback; else the last-known-good
    version. Must already be present on disk (we never download at rollback time)."""
    for candidate in (read_stable(), read_last_good()):
        if candidate and candidate != current and (VERSIONS_DIR / candidate).exists():
            return candidate
    return None


def mark_healthy(version: str) -> None:
    LAST_GOOD_FILE.write_text(version + "\n", encoding="utf-8")


def start_watchdog(
    healthy_event: threading.Event,
    stop_event: threading.Event,
    timeout_s: int = 300,
) -> threading.Thread:
    current = running_version()
    # Prefer Stable as the fallback, else last-known-good. Must be on disk.
    rollback_target = _pick_rollback_target(current)

    def loop():
        if healthy_event.wait(timeout=timeout_s):
            mark_healthy(current)
            clear_failed_version(current)
            log.info("version %s confirmed healthy (last_good updated)", current)
            return

        if rollback_target:
            log.error(
                "version %s never became healthy in %ss — rolling back to %s",
                current, timeout_s, rollback_target,
            )
            # Remember this failure so the next upgrade command for the same
            # version backs off instead of immediately repeating the same
            # rollback (see updater.failure_backoff_remaining_s) — e.g. if the
            # server was just down for its own maintenance during this window,
            # we don't want to thrash on every retry until it comes back.
            record_failed_version(current)
            try:
                write_current_version(rollback_target)
            except Exception as e:
                log.error("rollback pointer write failed: %s", e)
            stop_event.set()
        else:
            log.warning(
                "version %s not healthy in %ss but no rollback target available",
                current, timeout_s,
            )

    t = threading.Thread(target=loop, name="watchdog", daemon=True)
    t.start()
    return t
