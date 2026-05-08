"""Daily log shipper.

Every N seconds (default 1h), looks at rolled log files in `log_dir` and
uploads any that haven't been shipped yet to S3 via the cloud's presigned URL.
A `.shipped` marker file is left next to each upload so we don't re-send.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import requests

from api_client import ApiClient

log = logging.getLogger("agent.log_shipper")
SHIP_INTERVAL_S = 3600


def _ship_one(api: ApiClient, path: Path) -> bool:
    marker = path.with_suffix(path.suffix + ".shipped")
    if marker.exists():
        return False
    try:
        out = api.get_log_upload_url(path.name)
        with path.open("rb") as f:
            r = requests.put(
                out["presigned_url"], data=f.read(),
                headers={"Content-Type": "text/plain"},
                timeout=60,
            )
            r.raise_for_status()
        marker.write_text(out.get("s3_key", ""))
        log.info("shipped %s -> %s", path.name, out.get("s3_key"))
        return True
    except Exception as e:
        log.warning("ship %s failed: %s", path.name, e)
        return False


def _scan_and_ship(api: ApiClient, log_dir: str) -> None:
    p = Path(log_dir)
    if not p.exists():
        return
    # Only ship rolled files (TimedRotatingFileHandler produces e.g. agent.log.2026-05-04).
    # Skip the active .log file (still being written).
    rolled = [f for f in p.iterdir() if f.is_file() and f.suffix != "" and not f.name.endswith(".shipped")]
    rolled = [f for f in rolled if not f.name.endswith(".log")]
    for f in rolled:
        _ship_one(api, f)


def start_log_shipper(api: ApiClient, log_dir: str, stop_event: threading.Event) -> threading.Thread:
    def loop():
        # Initial delay so we don't ship on every boot before any rotation.
        stop_event.wait(60)
        while not stop_event.is_set():
            try:
                _scan_and_ship(api, log_dir)
            except Exception as e:
                log.warning("shipper tick failed: %s", e)
            stop_event.wait(SHIP_INTERVAL_S)

    t = threading.Thread(target=loop, name="log-shipper", daemon=True)
    t.start()
    return t
