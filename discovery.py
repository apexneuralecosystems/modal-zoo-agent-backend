"""NVR/DVR channel discovery.

For each device the cloud reports, probe RTSP channels 1..MAX. For every
channel that returns at least one frame within a timeout, report it back to
the cloud which will auto-create a camera row.
"""
from __future__ import annotations

import logging
import os
import socket
import threading
import time
from urllib.parse import quote

# Fix #4: bound RTSP probe time. Must be set before cv2 import.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "stimeout;5000000|rw_timeout;5000000",
)

import cv2

from api_client import ApiClient

log = logging.getLogger("agent.discovery")

MAX_CHANNELS_PROBED = 16
PROBE_TIMEOUT_S = 4
COMMON_PATHS = (
    # Path templates are tried in order. {ch} is substituted with channel #.
    "/cam/realmonitor?channel={ch}&subtype=0",   # Dahua / CP Plus
    "/Streaming/Channels/{ch}01",                # Hikvision
    "/h264/ch{ch}/main/av_stream",               # Reolink-style
    "/live/ch{ch}",                              # Generic
)


def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    """True if we can open a TCP connection to host:port. Used to distinguish
    'NVR powered off / wrong IP' from 'NVR up but speaks an RTSP path we
    don't know about yet'."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _build_rtsp(host: str, port: int, user: str | None, pwd: str | None, path: str) -> str:
    auth = ""
    if user:
        auth = f"{quote(user, safe='')}:{quote(pwd or '', safe='')}@"
    return f"rtsp://{auth}{host}:{port}{path}"


def _probe_channel(host: str, port: int, user: str | None, pwd: str | None, channel: int) -> tuple[bool, str | None, str | None]:
    """Try each path template until one returns a frame. Returns (ok, resolution, path)."""
    for tpl in COMMON_PATHS:
        path = tpl.format(ch=channel)
        url = _build_rtsp(host, port, user, pwd, path)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            continue
        deadline = time.time() + PROBE_TIMEOUT_S
        ok, frame = False, None
        while time.time() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        cap.release()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            return True, f"{w}x{h}", path
    return False, None, None


def discover_device(api: ApiClient, device: dict, max_channels: int = MAX_CHANNELS_PROBED) -> dict | None:
    host, port = device["ip_address"], device["port"]
    log.info("probing device %s @ %s:%s", device["name"], host, port)

    # P2-#1: cheap TCP probe first. Lets us distinguish "device offline"
    # (post offline status) from "device online but no known RTSP path
    # matched" (don't claim it's offline — log so we know to add a path).
    if not _tcp_reachable(host, port):
        log.warning("  TCP %s:%s unreachable — marking device offline", host, port)
        try:
            api.post_device_status({"device_id": device["id"], "status": "offline"})
        except Exception as e:
            log.warning("device-status post failed: %s", e)
        return None

    channels = []
    for ch in range(1, max_channels + 1):
        ok, res, _path = _probe_channel(
            host, port,
            device.get("username"), device.get("password"),
            ch,
        )
        if ok:
            log.info("  ch %s online (%s)", ch, res)
            channels.append({"channel": ch, "resolution": res})
        else:
            # On channel 1 failure: device is reachable (TCP succeeded above)
            # but no RTSP path template matched. Don't post offline — that
            # would be misleading. Just stop probing and surface a warning
            # so an operator knows to add a path template for this vendor.
            if ch == 1:
                log.warning(
                    "  ch 1 failed but %s:%s is TCP-reachable — vendor RTSP path likely unknown. "
                    "Add a template to COMMON_PATHS.", host, port,
                )
                try:
                    api.post_device_status({"device_id": device["id"], "status": "online"})
                except Exception as e:
                    log.warning("device-status post failed: %s", e)
                return None
            # Otherwise: assume we hit the end of the channel range.
            break

    if not channels:
        return None

    try:
        result = api.post_discover({"device_id": device["id"], "channels": channels})
        log.info("discover ok: %s cameras_created=%s", device["name"], result.get("cameras_created"))
        return result
    except Exception as e:
        log.warning("discover post failed for %s: %s", device["name"], e)
        return None


def start_discovery(api: ApiClient, stop_event: threading.Event, interval_s: int = 300) -> threading.Thread:
    """Background loop. Periodically probe every device this branch knows about.
    Cheap because we hit the cloud for the device list, then rely on probe
    timeouts. Probed devices that are already known are mostly no-ops on the
    cloud (idempotent discover endpoint)."""
    seen_devices: set[str] = set()

    def loop():
        while not stop_event.is_set():
            try:
                devices = api.list_devices()
            except Exception as e:
                log.warning("list_devices failed: %s", e)
                stop_event.wait(interval_s)
                continue
            for d in devices:
                if d["id"] in seen_devices:
                    continue
                # Fix #5: only mark a device "seen" once we successfully
                # probe it. A device that's powered-off on first poll used
                # to be marked seen forever; now it gets retried each
                # interval until it answers.
                result = discover_device(api, d)
                if result is not None:
                    seen_devices.add(d["id"])
            stop_event.wait(interval_s)

    t = threading.Thread(target=loop, name="discovery", daemon=True)
    t.start()
    return t
