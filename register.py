"""First-boot registration. Idempotent — safe to call on every start."""
from __future__ import annotations

import logging
import platform
import socket

from api_client import ApiClient

log = logging.getLogger("agent.register")


def _local_ip() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def register(api: ApiClient, cfg: dict) -> bool:
    payload = {
        "branch_id": cfg["branch_id"],
        "serial_number": cfg["mac_serial"],
        "ip_local": _local_ip(),
        "os_version": f"{platform.system()} {platform.release()}",
        "agent_version": cfg.get("agent_version", "1.0.0"),
    }
    try:
        api.register(payload)
        log.info("registered branch=%s ip=%s", cfg["branch_id"], payload["ip_local"])
        return True
    except Exception as e:
        log.error("register failed: %s", e)
        return False
