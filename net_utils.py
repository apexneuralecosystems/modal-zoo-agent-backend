"""Small networking helpers shared across the agent."""
from __future__ import annotations

import socket


def local_ip() -> str | None:
    """Best-effort LAN IP. Opens a UDP socket to a public address (no packets
    are actually sent) and reads back the kernel-chosen source IP. Always
    closes the socket, even on error."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return None
    finally:
        s.close()
