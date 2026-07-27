"""First-boot registration. Idempotent — safe to call on every start."""
from __future__ import annotations

import logging
import platform
import re

from api_client import ApiClient
from net_utils import local_ip

log = logging.getLogger("agent.register")

# Matches config.example.json's literal template value so a forgotten edit
# fails fast locally instead of reaching the backend (which rejects it too).
_PLACEHOLDER_SERIAL_RE = re.compile(r"^REPLACE_WITH_HARDWARE_SERIAL", re.IGNORECASE)


def register(api: ApiClient, cfg: dict) -> bool:
    serial = str(cfg["mac_serial"]).strip()
    if not serial or _PLACEHOLDER_SERIAL_RE.match(serial):
        log.error(
            "config.json still has a placeholder mac_serial (%r) -- "
            "set the real hardware serial before starting the agent",
            serial,
        )
        return False

    payload = {
        "branch_id": cfg["branch_id"],
        "serial_number": serial,
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
