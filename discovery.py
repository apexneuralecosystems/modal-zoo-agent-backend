"""NVR/DVR channel discovery.

For each device the cloud reports, probe RTSP channels 1..MAX. For every
channel that returns at least one frame within a timeout, report it back to
the cloud which will auto-create a camera row.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Iterable
from urllib.parse import quote

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
    log.info("probing device %s @ %s:%s", device["name"], device["ip_address"], device["port"])
    channels = []
    for ch in range(1, max_channels + 1):
        ok, res, _path = _probe_channel(
            device["ip_address"], device["port"],
            device.get("username"), device.get("password"),
            ch,
        )
        if ok:
            log.info("  ch %s online (%s)", ch, res)
            channels.append({"channel": ch, "resolution": res})
        else:
            # If the first channel fails, the device is likely unreachable.
            if ch == 1:
                log.warning("  ch 1 failed — assuming device unreachable, stopping probe")
                try:
                    api.post_device_status({"device_id": device["id"], "status": "offline"})
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
                discover_device(api, d)
                seen_devices.add(d["id"])
            stop_event.wait(interval_s)

    t = threading.Thread(target=loop, name="discovery", daemon=True)
    t.start()
    return t
