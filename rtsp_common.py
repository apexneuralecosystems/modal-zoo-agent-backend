"""Shared RTSP-open helpers used by both ws_streamer.py (WebRTC signaling)
and webrtc_streamer.py (the actual RTSP capture). Split out from
ws_streamer.py so webrtc_streamer.py can import these without a circular
import (ws_streamer.py also needs to reach into webrtc_streamer.py to spawn
sessions)."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from urllib.parse import unquote, urlparse

import av  # PyAV — decodes the camera H.264 that OpenCV's FFmpeg can't

# Silence PyAV/FFmpeg logging. PyAV forwards FFmpeg messages both to stderr
# (av.logging.set_level) and into Python's logging under the `libav.*` loggers;
# we mute both so the console isn't flooded with transient RTSP/decode notices.
try:
    av.logging.set_level(av.logging.CRITICAL)
except Exception:
    pass
logging.getLogger("libav").setLevel(logging.CRITICAL)

log = logging.getLogger("agent.stream")

# RTSP open options for PyAV — kept deliberately MINIMAL. This is the proven
# recipe (verified against the real NVR, which decoded a clean 1920x1080 frame):
#   - rtsp_transport=tcp : ordered packets, reliable across NAT/Wi-Fi.
#   - stimeout/rw_timeout (microseconds): give up on a stalled socket (~8s).
# We do NOT pass extra demux flags here:
#   - nobuffer/low_delay/reorder_queue_size : some Hikvision firmware 404s the
#     DESCRIBE when present. Low latency is handled at the read side instead, by
#     dropping/skipping stale frames.
#   - err_detect=ignore_err / fflags=discardcorrupt : observed to drop the
#     keyframe and make decoding WORSE. Tolerance is handled by the demux loop,
#     which skips corrupt pre-keyframe packets and waits for the first IDR.
AV_OPTS = {
    "rtsp_transport": "tcp",
    "stimeout": "8000000",
    "rw_timeout": "8000000",
}
# Seconds PyAV waits for the initial RTSP open before raising.
AV_OPEN_TIMEOUT_S = 10


def redact(url: str) -> str:
    """Mask the password in `rtsp://user:pass@host/...` for log output."""
    p = urlparse(url)
    if p.password:
        netloc = f"{p.username}:***@{p.hostname}"
        if p.port:
            netloc += f":{p.port}"
        return url.replace(p.netloc, netloc)
    return url


def nvr_now(rtsp_url: str) -> datetime:
    """Best-effort current wall-clock of the NVR, read from its own ISAPI clock.

    Hikvision RTSP playback interprets the `starttime` against the NVR's LOCAL
    wall clock (verified in testing: localTime + 'Z' returned the right footage),
    so we must anchor to the NVR's clock, not the cloud's or even the Mac's.
    Falls back to the agent's local clock if the read fails (the Mac is on-site,
    usually the same timezone). Never raises.
    """
    try:
        import requests  # already an agent dependency (used by api_client)
        from requests.auth import HTTPDigestAuth

        p = urlparse(rtsp_url)
        host = p.hostname
        user = unquote(p.username) if p.username else ""
        pwd = unquote(p.password) if p.password else ""
        r = requests.get(
            f"http://{host}/ISAPI/System/time",
            auth=HTTPDigestAuth(user, pwd),
            timeout=5,
        )
        if r.status_code == 200:
            import xml.etree.ElementTree as ET
            flat = {e.tag.split("}")[-1]: e.text for e in ET.fromstring(r.text).iter()}
            lt = flat.get("localTime")  # e.g. 2026-06-16T12:34:42+05:30
            if lt:
                return datetime.strptime(lt[:19], "%Y-%m-%dT%H:%M:%S")
    except Exception as e:
        log.warning("near-live: NVR clock read failed (%s); using local clock", e)
    return datetime.now()


def to_near_live_url(rtsp_url: str, delay_seconds: int) -> str:
    """Convert a Hikvision LIVE channel URL into a near-live PLAYBACK URL that
    starts `delay_seconds` behind the NVR's clock and plays forward, staying
    ~that far behind live.

        rtsp://…/Streaming/Channels/NN01
          → rtsp://…/Streaming/tracks/NN01?starttime=<NVR_now − delay>Z

    Open-ended (no endtime) so the NVR keeps feeding recorded data forward. If
    the URL isn't a Hikvision Channels path we can't build a playback form, so
    we return it unchanged (falls back to live) rather than emit a bad URL.
    """
    if "/Streaming/Channels/" not in rtsp_url:
        log.warning("near-live: non-Hikvision path, falling back to live: %s", redact(rtsp_url))
        return rtsp_url
    start = nvr_now(rtsp_url) - timedelta(seconds=delay_seconds)
    ts = start.strftime("%Y%m%dT%H%M%SZ")
    base = rtsp_url.replace("/Streaming/Channels/", "/Streaming/tracks/")
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}starttime={ts}"
