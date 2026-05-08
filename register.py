"""First-boot registration. Idempotent — safe to call on every start."""
from __future__ import annotations

import logging
import platform

from api_client import ApiClient
from net_utils import local_ip

log = logging.getLogger("agent.register")


def register(api: ApiClient, cfg: dict) -> bool:
    payload = {
        "branch_id": cfg["branch_id"],
        "serial_number": cfg["mac_serial"],
        "ip_local": local_ip(),
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
