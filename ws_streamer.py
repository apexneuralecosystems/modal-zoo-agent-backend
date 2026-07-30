"""Live-preview WebSocket client.

Persistent connection to the cloud's /ws/agent endpoint. Listens for
control/signaling messages from the cloud; live video itself flows over a
separate WebRTC connection per camera (see webrtc_streamer.py) — this socket
only carries the start/stop signal and the WebRTC offer/answer/ICE exchange.

Control messages received DOWN from the cloud (text JSON):
    { "type": "start-stream", "camera_id": "<uuid>", "rtsp_url": "rtsp://..." }
    { "type": "stop-stream",  "camera_id": "<uuid>" }
    { "type": "webrtc-answer", "camera_id": "<uuid>", "sdp": "..." }
    { "type": "webrtc-ice-candidate", "camera_id": "<uuid>", "candidate": {...} }

Messages sent UP to the cloud (text JSON):
    { "type": "webrtc-offer", "camera_id": "<uuid>", "sdp": "..." }

Design notes:
- One WebRTC session per active camera (WebRTCStreamerManager); the main WS
  reader stays cheap — it just dispatches signaling messages.
- We rebuild the WS connection with exponential backoff on failure, mirroring
  what the existing HTTP client does for /agent/heartbeat etc.
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
from typing import Optional
from urllib.parse import urlparse

import av  # PyAV — decodes the camera H.264 that OpenCV's FFmpeg can't
import cv2  # used ONLY for JPEG encoding of probe/snapshot frames
import websocket  # websocket-client

from rtsp_common import AV_OPEN_TIMEOUT_S as _AV_OPEN_TIMEOUT_S
from rtsp_common import AV_OPTS as _AV_OPTS
from rtsp_common import nvr_now as _nvr_now
from rtsp_common import redact as _redact
from rtsp_common import to_near_live_url as _to_near_live_url
from webrtc_streamer import WebRTCStreamerManager

# Silence OpenCV's logger too (we still import cv2 for imencode in probe/snapshot).
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

log = logging.getLogger("agent.stream")

JPEG_QUALITY = 70
RECONNECT_INITIAL_S = 2
RECONNECT_MAX_S = 60


def _http_to_ws(server_url: str) -> str:
    """Map http(s)://host:port → ws(s)://host:port for the same host."""
    p = urlparse(server_url)
    scheme = "wss" if p.scheme == "https" else "ws"
    return f"{scheme}://{p.netloc}{p.path or ''}".rstrip("/")


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


# RTSP capture + video encoding for live preview now live in webrtc_streamer.py
# (WebRTCStreamerManager / _WebRTCSession / _RtspVideoTrack) — real WebRTC
# media replaced the JPEG-per-frame relay that used to live here.


class _WsAdapter:
    """Tiny shim so webrtc_streamer can call .send_text() without knowing
    about websocket-client internals."""
    def __init__(self, ws: websocket.WebSocket):
        self.ws = ws
        self.lock = threading.Lock()

    def send_text(self, s: str) -> None:
        with self.lock:
            self.ws.send(s)


def start_ws_streamer(server_url: str, secret_token: str, stop_event: threading.Event) -> threading.Thread:
    """Spawn the WS streamer thread. Returns the thread for tests; daemonized."""
    ws_url = _http_to_ws(server_url) + "/ws/agent"

    def loop():
        backoff = RECONNECT_INITIAL_S
        while not stop_event.is_set():
            webrtc_manager = WebRTCStreamerManager()
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
                        # Present only in super-admin "playback" mode; absent =
                        # live (unchanged). Ignore non-positive / bad values.
                        delay_raw = cmd.get("near_live_delay_seconds")
                        try:
                            near_live = int(delay_raw) if delay_raw else None
                            if near_live is not None and near_live <= 0:
                                near_live = None
                        except (TypeError, ValueError):
                            near_live = None
                        # start_stream() is itself idempotent — a second
                        # start-stream for a camera already streaming (e.g. a
                        # 2nd viewer joining) is a no-op.
                        webrtc_manager.start_stream(cam, rtsp, near_live, adapter.send_text)
                    elif t == "stop-stream":
                        cam = cmd.get("camera_id")
                        webrtc_manager.stop_stream(cam)
                    elif t == "webrtc-answer":
                        cam = cmd.get("camera_id")
                        sdp = cmd.get("sdp")
                        if cam and sdp:
                            webrtc_manager.handle_answer(cam, sdp)
                    elif t == "webrtc-ice-candidate":
                        cam = cmd.get("camera_id")
                        candidate = cmd.get("candidate")
                        if cam and candidate:
                            webrtc_manager.handle_ice_candidate(cam, candidate)
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
                # Tear down all WebRTC sessions before reconnect so we don't
                # leak RTSP captures or dangling peer connections.
                webrtc_manager.stop_all()
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
