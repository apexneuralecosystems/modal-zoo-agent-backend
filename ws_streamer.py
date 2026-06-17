"""Live-preview WebSocket client.

Persistent connection to the cloud's /ws/agent endpoint. Listens for
control messages from the cloud and pumps JPEG frames back over the same
socket whenever a viewer is asking for a particular camera.

Wire format for binary frames sent UP to the cloud:
    [16 bytes camera_id (UUID without dashes, hex-decoded)] + [JPEG bytes]

Control messages received DOWN from the cloud (text JSON):
    { "type": "start-stream", "camera_id": "<uuid>", "rtsp_url": "rtsp://..." }
    { "type": "stop-stream",  "camera_id": "<uuid>" }

Design notes:
- One worker thread per active camera; the main WS reader stays cheap.
- We rebuild the WS connection with exponential backoff on failure, mirroring
  what the existing HTTP client does for /agent/heartbeat etc.
- Frames are dropped (not buffered) when the encoder is faster than the WS;
  this prevents memory blow-up if the cloud or browser stalls.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse, unquote

import av  # PyAV — decodes the camera H.264 that OpenCV's FFmpeg can't
import cv2  # used ONLY for JPEG encoding of decoded frames
import websocket  # websocket-client

# Silence PyAV/FFmpeg logging. PyAV forwards FFmpeg messages both to stderr
# (av.logging.set_level) and into Python's logging under the `libav.*` loggers;
# we mute both so the console isn't flooded with transient RTSP/decode notices.
try:
    av.logging.set_level(av.logging.CRITICAL)
except Exception:
    pass
logging.getLogger("libav").setLevel(logging.CRITICAL)

# RTSP open options for PyAV — kept deliberately MINIMAL. This is the proven
# recipe (verified against the real NVR, which decoded a clean 1920x1080 frame):
#   - rtsp_transport=tcp : ordered packets, reliable across NAT/Wi-Fi.
#   - stimeout/rw_timeout (microseconds): give up on a stalled socket (~8s).
# We do NOT pass extra demux flags here:
#   - nobuffer/low_delay/reorder_queue_size : some Hikvision firmware 404s the
#     DESCRIBE when present. Low latency is handled at the read side instead, by
#     dropping frames that arrive early (see the pacing in _StreamWorker._run).
#   - err_detect=ignore_err / fflags=discardcorrupt : observed to drop the
#     keyframe and make decoding WORSE. Tolerance is handled by the demux loop,
#     which skips corrupt pre-keyframe packets and waits for the first IDR.
_AV_OPTS = {
    "rtsp_transport": "tcp",
    "stimeout": "8000000",
    "rw_timeout": "8000000",
}
# Seconds PyAV waits for the initial RTSP open before raising.
_AV_OPEN_TIMEOUT_S = 10

# Silence OpenCV's logger too (we still import cv2 for imencode).
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

log = logging.getLogger("agent.stream")

JPEG_QUALITY = 70
TARGET_FPS = 12
RECONNECT_INITIAL_S = 2
RECONNECT_MAX_S = 60


def _http_to_ws(server_url: str) -> str:
    """Map http(s)://host:port → ws(s)://host:port for the same host."""
    p = urlparse(server_url)
    scheme = "wss" if p.scheme == "https" else "ws"
    return f"{scheme}://{p.netloc}{p.path or ''}".rstrip("/")


def _redact(url: str) -> str:
    """Mask the password in `rtsp://user:pass@host/...` for log output."""
    p = urlparse(url)
    if p.password:
        netloc = f"{p.username}:***@{p.hostname}"
        if p.port:
            netloc += f":{p.port}"
        return url.replace(p.netloc, netloc)
    return url


def _camera_id_to_bytes(camera_id: str) -> bytes:
    """UUID string → 16 raw bytes prefix used in binary frames."""
    return bytes.fromhex(camera_id.replace("-", ""))


def _nvr_now(rtsp_url: str) -> datetime:
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


def _to_near_live_url(rtsp_url: str, delay_seconds: int) -> str:
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
        log.warning("near-live: non-Hikvision path, falling back to live: %s", _redact(rtsp_url))
        return rtsp_url
    start = _nvr_now(rtsp_url) - timedelta(seconds=delay_seconds)
    ts = start.strftime("%Y%m%dT%H%M%SZ")
    base = rtsp_url.replace("/Streaming/Channels/", "/Streaming/tracks/")
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}starttime={ts}"


def _probe_rtsp(url: str, timeout_s: float = 10.0, near_live_delay: Optional[int] = None) -> dict:
    """Open an RTSP/HTTP URL with PyAV and try to decode one real frame.

    If `near_live_delay` is set, probe the near-live RECORDED feed (the same one
    a viewer gets in playback mode) instead of the live feed — otherwise a
    near-live deployment whose live encoder is faulty would falsely report the
    camera as offline here while the actual preview plays fine.

    Runs ON THE MAC — the camera lives on the same LAN as the agent, so this is
    the only place a probe can actually reach a private 192.168.x.x NVR (the
    cloud is on the public internet). PyAV is used because OpenCV's bundled
    FFmpeg fails to decode many real Hikvision H.264 streams.

    Returns a JSON-able dict matching what the cloud's probe endpoint expects:
        {"ok": True,  "width": W, "height": H, "elapsed_ms": N}
        {"ok": False, "error": "<short reason>", "elapsed_ms": N}

    PyAV's open can block past its own timeout on some networks, so we run the
    open+decode in a daemon thread and abandon it on the hard deadline rather
    than letting it hang the WS reader.
    """
    out: dict = {"done": False}
    started = time.monotonic()
    open_url = _to_near_live_url(url, near_live_delay) if near_live_delay else url

    def _worker() -> None:
        container = None
        try:
            container = av.open(open_url, options=_AV_OPTS, timeout=min(timeout_s, _AV_OPEN_TIMEOUT_S))
            vstream = next((s for s in container.streams if s.type == "video"), None)
            if vstream is None:
                out.update(done=True, ok=False, error="no video stream")
                return
            # Demux + per-packet decode, skipping corrupt pre-keyframe packets
            # (same reason as the live worker — plain decode() raises on them).
            # Require a REAL frame (>=160px wide): the decoder often emits a tiny
            # 48x16 parameter-set fragment first, which is not a usable picture.
            for packet in container.demux(vstream):
                try:
                    frames = packet.decode()
                except Exception:
                    continue
                for frame in frames:
                    if frame.width >= 160:
                        out.update(done=True, ok=True, width=int(frame.width), height=int(frame.height))
                        return
            out.update(done=True, ok=False, error="no frame decoded")
        except Exception as e:
            out.update(done=True, ok=False, error=f"{type(e).__name__}: {e}")
        finally:
            try:
                if container is not None:
                    container.close()
            except Exception:
                pass

    t = threading.Thread(target=_worker, name="probe", daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    if not out.get("done"):
        return {"ok": False, "error": "timeout opening stream", "elapsed_ms": elapsed_ms}
    result = {k: v for k, v in out.items() if k != "done"}
    result["elapsed_ms"] = elapsed_ms
    return result


def _capture_snapshot(url: str, timeout_s: float = 10.0, near_live_delay: Optional[int] = None,
                      max_width: int = 640) -> dict:
    """Grab ONE frame from the camera and return it as a base64 JPEG.

    Used for the camera-tile previews on the Mac detail page (replaces the
    old per-tile status probe). Same LAN-only constraint and near-live handling
    as `_probe_rtsp`: in playback mode it snapshots the clean recorded feed so
    the tile shows a real picture even when the live encoder is faulty.

    Returns:
        {"ok": True,  "image_b64": "<base64 jpeg>", "width": W, "height": H, "elapsed_ms": N}
        {"ok": False, "error": "<short reason>", "elapsed_ms": N}
    The cloud base64-decodes image_b64 and uploads the bytes to S3 (the agent
    never holds S3 credentials).
    """
    out: dict = {"done": False}
    started = time.monotonic()
    open_url = _to_near_live_url(url, near_live_delay) if near_live_delay else url

    def _worker() -> None:
        container = None
        try:
            container = av.open(open_url, options=_AV_OPTS, timeout=min(timeout_s, _AV_OPEN_TIMEOUT_S))
            vstream = next((s for s in container.streams if s.type == "video"), None)
            if vstream is None:
                out.update(done=True, ok=False, error="no video stream")
                return
            for packet in container.demux(vstream):
                try:
                    frames = packet.decode()
                except Exception:
                    continue
                for frame in frames:
                    if frame.width < 160:
                        continue
                    try:
                        img = frame.to_ndarray(format="bgr24")
                    except Exception:
                        continue
                    h, w = img.shape[:2]
                    if w > max_width:
                        nh = int(h * (max_width / w))
                        img = cv2.resize(img, (max_width, nh), interpolation=cv2.INTER_AREA)
                    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    if not ok:
                        out.update(done=True, ok=False, error="jpeg encode failed")
                        return
                    out.update(done=True, ok=True,
                               image_b64=base64.b64encode(jpg.tobytes()).decode("ascii"),
                               width=int(w), height=int(h))
                    return
            out.update(done=True, ok=False, error="no frame decoded")
        except Exception as e:
            out.update(done=True, ok=False, error=f"{type(e).__name__}: {e}")
        finally:
            try:
                if container is not None:
                    container.close()
            except Exception:
                pass

    t = threading.Thread(target=_worker, name="snapshot", daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    if not out.get("done"):
        return {"ok": False, "error": "timeout opening stream", "elapsed_ms": elapsed_ms}
    result = {k: v for k, v in out.items() if k != "done"}
    result["elapsed_ms"] = elapsed_ms
    return result


@dataclass
class _StreamWorker:
    """One per active camera. Reads RTSP frames and pushes JPEGs over the WS."""
    camera_id: str
    rtsp_url: str
    ws: websocket.WebSocket
    stop_event: threading.Event
    thread: Optional[threading.Thread] = None
    # When set, stream a near-live RECORDED feed this many seconds behind live
    # instead of the real-time stream (super-admin "playback" mode). None = live.
    near_live_delay: Optional[int] = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name=f"stream-{self.camera_id[:8]}", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _run(self) -> None:
        # PyAV path. OpenCV's bundled FFmpeg cannot decode many real-world
        # Hikvision H.264 streams (endless cabac_init_idc / PPS errors, zero
        # frames). PyAV ships a newer FFmpeg that decodes them cleanly — proven
        # against the production NVR (1920x1080 yuv420p h264 decoded on the
        # first frame). So we demux+decode with PyAV and only use cv2 for the
        # JPEG encode (which always worked).
        mode = f"near-live(-{self.near_live_delay}s)" if self.near_live_delay else "live"
        log.info("stream cam=%s: opening [%s] %s", self.camera_id, mode, _redact(self.rtsp_url))

        prefix = _camera_id_to_bytes(self.camera_id)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        period = 1.0 / TARGET_FPS
        exit_reason = "stop_event"
        # Gentle reconnect backoff. Hikvision NVRs throttle (and send
        # undecodable packets to) a client that reopens RTSP too fast, so we
        # start at 3s and back off to 30s rather than hammering on failure.
        backoff = 3.0
        started_logged = False
        try:
            while not self.stop_event.is_set():
                # Resolve the URL to open on EACH (re)connect. In near-live mode
                # this recomputes the playback start time so a reconnect re-syncs
                # to ~delay behind live instead of replaying from the old anchor.
                if self.near_live_delay:
                    open_url = _to_near_live_url(self.rtsp_url, self.near_live_delay)
                else:
                    open_url = self.rtsp_url
                try:
                    container = av.open(
                        open_url,
                        options=_AV_OPTS,
                        timeout=_AV_OPEN_TIMEOUT_S,
                    )
                except Exception as e:
                    log.warning("stream cam=%s: open failed (%s), retry in %.0fs",
                                self.camera_id, e, backoff)
                    if self.stop_event.wait(backoff):
                        return
                    backoff = min(30.0, backoff * 2)
                    continue

                backoff = 3.0  # reset to the gentle base on a successful open
                try:
                    vstream = next((s for s in container.streams if s.type == "video"), None)
                    if vstream is None:
                        log.warning("stream cam=%s: no video stream", self.camera_id)
                        container.close()
                        if self.stop_event.wait(2.0):
                            return
                        continue
                    # Decode threads + drop late frames: we want the freshest
                    # frame, not a backlog. PyAV honours this on the codec ctx.
                    vstream.thread_type = "AUTO"
                    if not started_logged:
                        log.info("stream cam=%s: started (codec=%s)",
                                 self.camera_id, vstream.codec_context.name)
                        started_logged = True

                    next_tick = time.monotonic()
                    # Demux packets and decode each in its own try, skipping any
                    # corrupt packet (the pre-keyframe garbage a Hikvision stream
                    # emits) instead of letting it abort the whole session. This
                    # is what VLC does — and the reason VLC plays this camera
                    # while a plain container.decode() loop raised InvalidData.
                    for packet in container.demux(vstream):
                        if self.stop_event.is_set():
                            exit_reason = "stop_event"
                            container.close()
                            return
                        try:
                            frames = packet.decode()
                        except Exception:
                            # Corrupt/incomplete packet — skip, wait for the
                            # next (clean) one. Do NOT reconnect on these.
                            continue
                        for frame in frames:
                            # Skip tiny parameter-set fragments (e.g. 48x16) the
                            # decoder emits before a real picture — they're not
                            # viewable frames and would just send junk JPEGs.
                            if frame.width < 160:
                                continue
                            # Pace to TARGET_FPS by DROPPING frames that arrive
                            # early — keeps latency low (live, not buffered).
                            now = time.monotonic()
                            if now < next_tick:
                                continue
                            next_tick = now + period

                            try:
                                img = frame.to_ndarray(format="bgr24")
                            except Exception:
                                continue
                            try:
                                ok, jpg = cv2.imencode(".jpg", img, encode_params)
                            except Exception as e:
                                exit_reason = f"imencode exception: {e}"
                                container.close()
                                return
                            if not ok:
                                continue
                            try:
                                self.ws.send_bytes(prefix + jpg.tobytes())
                            except Exception as e:
                                exit_reason = f"ws send failed: {e}"
                                container.close()
                                return
                    # demux() ended → the RTSP source dropped.
                    # Close and let the outer while-loop reconnect with backoff.
                    log.warning("stream cam=%s: stream ended, reconnecting", self.camera_id)
                    container.close()
                    if self.stop_event.wait(backoff):
                        return
                except av.error.EOFError:
                    container.close()
                    if self.stop_event.wait(backoff):
                        return
                except Exception as e:
                    # Mid-stream decode/transport error — close and reconnect
                    # rather than die. PyAV recovers cleanly on a fresh open.
                    log.warning("stream cam=%s: stream error (%s), reconnecting",
                                self.camera_id, e)
                    try:
                        container.close()
                    except Exception:
                        pass
                    if self.stop_event.wait(backoff):
                        return
        except Exception as e:
            exit_reason = f"unexpected exception: {e!r}"
        finally:
            log.info("stream cam=%s: stopped (reason=%s)", self.camera_id, exit_reason)


class _WsAdapter:
    """Tiny shim so workers can call .send_bytes() without knowing about websocket-client internals."""
    def __init__(self, ws: websocket.WebSocket):
        self.ws = ws
        self.lock = threading.Lock()

    def send_bytes(self, data: bytes) -> None:
        with self.lock:
            self.ws.send_binary(data)

    def send_text(self, s: str) -> None:
        with self.lock:
            self.ws.send(s)


def start_ws_streamer(server_url: str, secret_token: str, stop_event: threading.Event) -> threading.Thread:
    """Spawn the WS streamer thread. Returns the thread for tests; daemonized."""
    ws_url = _http_to_ws(server_url) + "/ws/agent"

    def loop():
        backoff = RECONNECT_INITIAL_S
        while not stop_event.is_set():
            workers: dict[str, _StreamWorker] = {}
            adapter: Optional[_WsAdapter] = None
            try:
                log.info("connecting to %s", ws_url)
                ws = websocket.create_connection(
                    ws_url,
                    header=[f"Authorization: Bearer {secret_token}"],
                    timeout=10,
                )
                ws.settimeout(60)
                adapter = _WsAdapter(ws)
                log.info("connected")
                backoff = RECONNECT_INITIAL_S
                # Optional handshake so the cloud knows we're ready.
                adapter.send_text(json.dumps({"type": "ready"}))

                while not stop_event.is_set():
                    try:
                        msg = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        # No activity is fine — keep the socket warm.
                        continue
                    if not msg:
                        log.info("server closed the stream socket")
                        break
                    if isinstance(msg, bytes):
                        # Cloud doesn't send binary down to us today; ignore.
                        continue
                    try:
                        cmd = json.loads(msg)
                    except json.JSONDecodeError:
                        log.warning("malformed control msg: %r", msg[:80])
                        continue

                    t = cmd.get("type")
                    if t == "start-stream":
                        cam = cmd.get("camera_id")
                        rtsp = cmd.get("rtsp_url")
                        if not cam or not rtsp:
                            continue
                        # If a worker exists already, leave it alone (idempotent).
                        if cam in workers:
                            continue
                        # Present only in super-admin "playback" mode; absent =
                        # live (unchanged). Ignore non-positive / bad values.
                        delay_raw = cmd.get("near_live_delay_seconds")
                        try:
                            near_live = int(delay_raw) if delay_raw else None
                            if near_live is not None and near_live <= 0:
                                near_live = None
                        except (TypeError, ValueError):
                            near_live = None
                        w = _StreamWorker(cam, rtsp, adapter, threading.Event(), near_live_delay=near_live)
                        workers[cam] = w
                        w.start()
                    elif t == "stop-stream":
                        cam = cmd.get("camera_id")
                        w = workers.pop(cam, None)
                        if w:
                            w.stop()
                    elif t == "probe":
                        # Cloud is asking us to test a stream URL from the LAN
                        # (the "Test stream health" button). Run it off the
                        # reader thread — a probe can take up to its timeout and
                        # we must not block frame fan-out or stop-stream while it
                        # runs. Reply with {type:'probe-result', request_id, ...}
                        # so the cloud can match it to the waiting HTTP request.
                        req_id = cmd.get("request_id")
                        purl = cmd.get("rtsp_url")
                        if not req_id or not purl:
                            continue
                        # Present only in playback mode → probe the near-live feed.
                        pdelay_raw = cmd.get("near_live_delay_seconds")
                        try:
                            pdelay = int(pdelay_raw) if pdelay_raw else None
                            if pdelay is not None and pdelay <= 0:
                                pdelay = None
                        except (TypeError, ValueError):
                            pdelay = None

                        def _do_probe(req_id=req_id, purl=purl, adapter=adapter, pdelay=pdelay):
                            res = _probe_rtsp(purl, near_live_delay=pdelay)
                            try:
                                adapter.send_text(json.dumps({
                                    "type": "probe-result",
                                    "request_id": req_id,
                                    **res,
                                }))
                            except Exception as e:
                                log.warning("probe-result send failed: %s", e)

                        threading.Thread(
                            target=_do_probe, name=f"probe-{req_id[:8]}", daemon=True
                        ).start()
                    elif t == "capture-snapshot":
                        # Cloud wants a fresh still for a camera tile. Grab one
                        # frame off the reader thread and reply with the JPEG.
                        req_id = cmd.get("request_id")
                        surl = cmd.get("rtsp_url")
                        if not req_id or not surl:
                            continue
                        sdelay_raw = cmd.get("near_live_delay_seconds")
                        try:
                            sdelay = int(sdelay_raw) if sdelay_raw else None
                            if sdelay is not None and sdelay <= 0:
                                sdelay = None
                        except (TypeError, ValueError):
                            sdelay = None

                        def _do_snapshot(req_id=req_id, surl=surl, adapter=adapter, sdelay=sdelay):
                            res = _capture_snapshot(surl, near_live_delay=sdelay)
                            try:
                                adapter.send_text(json.dumps({
                                    "type": "snapshot-result",
                                    "request_id": req_id,
                                    **res,
                                }))
                            except Exception as e:
                                log.warning("snapshot-result send failed: %s", e)

                        threading.Thread(
                            target=_do_snapshot, name=f"snap-{req_id[:8]}", daemon=True
                        ).start()
                    elif t == "ping":
                        adapter.send_text(json.dumps({"type": "pong"}))
                    # Unknown types are ignored — forwards-compat.
            except Exception as e:
                log.warning("ws loop error: %s", e)
            finally:
                # Tear down workers before reconnect so we don't leak captures.
                for w in workers.values():
                    w.stop()
                workers.clear()
                try:
                    if adapter is not None:
                        adapter.ws.close()
                except Exception:
                    pass

            if stop_event.is_set():
                break
            log.info("reconnecting in %ss", backoff)
            if stop_event.wait(backoff):
                break
            backoff = min(RECONNECT_MAX_S, backoff * 2)

    t = threading.Thread(target=loop, name="ws-streamer", daemon=True)
    t.start()
    return t
