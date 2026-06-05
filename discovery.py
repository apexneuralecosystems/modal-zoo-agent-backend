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


def _post_failed(api: ApiClient, device_id: str, reason: str, detail: str | None = None) -> None:
    """Report a probe failure to the cloud so the UI can show a real error
    instead of timing out. Failures here are best-effort — if the post itself
    fails we just log and move on; the UI's own 60s timeout still covers us."""
    try:
        api.post_discover_failed({"device_id": device_id, "reason": reason, "detail": detail or ""})
    except Exception as e:
        log.warning("discover-failed post failed for %s: %s", device_id, e)


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
        _post_failed(api, device["id"], "unreachable", f"TCP {host}:{port} not reachable from the Mac")
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
            # but no RTSP path template matched. cv2 doesn't tell us whether
            # the underlying cause was auth (wrong user/pass) or an unknown
            # vendor path — both surface as "could not open stream". We
            # default to "auth" because that's the overwhelmingly more
            # common cause when an operator is setting up an NVR; the detail
            # message tells them to also double-check the URL path template.
            if ch == 1:
                log.warning(
                    "  ch 1 failed but %s:%s is TCP-reachable — assuming auth/path failure",
                    host, port,
                )
                try:
                    api.post_device_status({"device_id": device["id"], "status": "online"})
                except Exception as e:
                    log.warning("device-status post failed: %s", e)
                _post_failed(
                    api, device["id"], "auth",
                    "NVR is reachable but channel 1 didn't open — check username/password, "
                    "or your NVR may use an unusual RTSP path",
                )
                return None
            # Otherwise: assume we hit the end of the channel range.
            break

    if not channels:
        # Shouldn't happen in practice (we return early on ch=1 fail), but
        # report it so the UI doesn't spin forever.
        _post_failed(api, device["id"], "unknown", "Probe produced no channels")
        return None

    try:
        result = api.post_discover({"device_id": device["id"], "channels": channels})
        log.info("discover ok: %s channels=%s", device["name"], len(channels))
        return result
    except Exception as e:
        log.warning("discover post failed for %s: %s", device["name"], e)
        _post_failed(api, device["id"], "unknown", f"Cloud rejected discover: {e}")
        return None


def start_discovery(
    api: ApiClient,
    stop_event: threading.Event,
    interval_idle_s: int = 300,
    interval_active_s: int = 15,
) -> threading.Thread:
    """Background loop. Periodically probe every device this branch knows about.

    Adaptive interval: when at least one device is flagged `discovery_pending`
    by the backend (user clicked Add NVR or Retry), we tighten the poll to
    `interval_active_s` (default 15s) so the UI doesn't have to wait minutes
    for the next probe. When nothing is pending we relax to `interval_idle_s`
    (default 5 minutes), which is fine because already-discovered devices
    don't need frequent re-probing.

    seen_devices tracks devices we've successfully reported to the cloud so
    we don't re-probe them on every tick. A device that turns `discovery_pending`
    again (e.g. user clicked Retry) is dropped from this set so the next
    iteration re-probes it."""
    seen_devices: set[str] = set()

    def loop():
        while not stop_event.is_set():
            try:
                devices = api.list_devices()
            except Exception as e:
                log.warning("list_devices failed: %s", e)
                stop_event.wait(interval_idle_s)
                continue

            # Drop any device that is pending again from the seen set so the
            # retry-discovery flow actually re-probes.
            for d in devices:
                if d.get("discovery_pending") and d["id"] in seen_devices:
                    seen_devices.discard(d["id"])

            for d in devices:
                if d["id"] in seen_devices:
                    continue
                # type=RTSP devices are single-stream — backend already
                # created the companion camera row from the user-supplied
                # rtsp_url at device-create time. No probing, no channel
                # discovery. Just mark the device online and move on.
                if (d.get("type") or "").upper() == "RTSP":
                    try:
                        api.post_device_status({"device_id": d["id"], "status": "online"})
                    except Exception as e:
                        log.warning("device-status post failed: %s", e)
                    seen_devices.add(d["id"])
                    continue
                # Fix #5: only mark a device "seen" once we successfully
                # probe it. A device that's powered-off on first poll used
                # to be marked seen forever; now it gets retried each
                # interval until it answers.
                result = discover_device(api, d)
                if result is not None:
                    seen_devices.add(d["id"])

            # Pick the next wait based on whether any device is still waiting
            # for discovery. If yes, sleep briefly so retries feel responsive;
            # otherwise relax to the long idle interval.
            any_pending = any(d.get("discovery_pending") for d in devices)
            wait_s = interval_active_s if any_pending else interval_idle_s
            stop_event.wait(wait_s)

    t = threading.Thread(target=loop, name="discovery", daemon=True)
    t.start()
    return t
