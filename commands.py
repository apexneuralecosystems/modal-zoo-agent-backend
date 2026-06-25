"""Command poller — every N seconds:
  1. GET /agent/commands  (cloud marks them claimed)
  2. For each command, dispatch by type; today only 'fetch_clip'.
  3. POST /agent/command-result with the outcome.

Mirrors the discovery/heartbeat/poller thread pattern.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from api_client import ApiClient
from clip_recorder import handle_fetch_clip
from updater import handle_upgrade

log = logging.getLogger("agent.commands")


def _process_one(api: ApiClient, cfg: dict, cmd: dict, stop_event: threading.Event) -> None:
    cmd_id = cmd.get("id")
    ctype = cmd.get("type")
    payload = cmd.get("payload") or {}
    work_dir = os.path.join(
        cfg.get("log_dir") or str(Path(__file__).resolve().parent / "logs"),
        "clips",
    )
    try:
        if ctype == "fetch_clip":
            result = handle_fetch_clip(payload, work_dir)
            api.post_command_result({"command_id": cmd_id, "ok": True, "result": result})
            log.info("fetch_clip %s done: %s bytes", cmd_id, result.get("bytes"))
        elif ctype == "upgrade":
            # Applies the update then sets stop_event so the process exits and
            # launchd re-execs on the new code. We report success first; the
            # post-restart heartbeat (new version) is the real confirmation,
            # and the startup watchdog rolls back if it never arrives.
            result = handle_upgrade(payload, api, stop_event)
            api.post_command_result({"command_id": cmd_id, "ok": True, "result": result})
            log.info("upgrade %s applied -> %s (exiting for restart)", cmd_id, result.get("version"))
        else:
            # Unknown command type — report failure so the cloud doesn't wait forever.
            api.post_command_result({
                "command_id": cmd_id, "ok": False,
                "error": f"unsupported command type: {ctype}",
            })
            log.warning("unsupported command type %s (id=%s)", ctype, cmd_id)
    except Exception as e:
        log.warning("command %s (%s) failed: %s", cmd_id, ctype, e)
        try:
            api.post_command_result({"command_id": cmd_id, "ok": False, "error": str(e)})
        except Exception as e2:
            log.warning("failed to report command failure for %s: %s", cmd_id, e2)


def start_commands(api: ApiClient, cfg: dict, stop_event: threading.Event) -> threading.Thread:
    interval = max(5, int(cfg.get("poll_interval_s", 10)))

    def loop():
        log.info("commands loop starting interval=%ss", interval)
        while not stop_event.is_set():
            try:
                cmds = api.get_commands()
            except Exception as e:
                log.warning("get_commands failed: %s", e)
                stop_event.wait(interval)
                continue
            for cmd in cmds:
                if stop_event.is_set():
                    break
                _process_one(api, cfg, cmd, stop_event)
            stop_event.wait(interval)
        log.info("commands loop stopped")

    t = threading.Thread(target=loop, name="commands", daemon=True)
    t.start()
    return t
