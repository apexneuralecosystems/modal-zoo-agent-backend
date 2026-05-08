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

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import cv2
import websocket  # websocket-client

# Force RTSP over TCP. Many cameras (and most local RTSP test servers like
# MediaMTX / rtsp-simple-server) drop UDP packets aggressively or block UDP
# entirely; TCP is far more reliable across NAT, Wi-Fi, and Docker bridges.
# OpenCV reads this env var when opening a stream via FFmpeg.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

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


@dataclass
class _StreamWorker:
    """One per active camera. Reads RTSP frames and pushes JPEGs over the WS."""
    camera_id: str
    rtsp_url: str
    ws: websocket.WebSocket
    stop_event: threading.Event
    thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name=f"stream-{self.camera_id[:8]}", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _run(self) -> None:
        # Log the URL we're actually using (creds masked) — when the cloud
        # splices in stored device credentials a wrong/extra `user:pass@`
        # silently breaks the connection on servers that expect anonymous
        # access (or different creds). Knowing what we tried makes debugging
        # tractable instead of just "failed to open RTSP".
        log.info("stream cam=%s: opening %s", self.camera_id, _redact(self.rtsp_url))
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            log.warning(
                "stream cam=%s: failed to open RTSP url=%s",
                self.camera_id, _redact(self.rtsp_url),
            )
            return
        # Trim internal buffering to keep latency low — we're a live preview,
        # not a recorder; an old frame is worse than a dropped one.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        prefix = _camera_id_to_bytes(self.camera_id)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        period = 1.0 / TARGET_FPS
        log.info("stream cam=%s: started", self.camera_id)
        try:
            next_tick = time.monotonic()
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    # Camera dropped — try a quick reopen rather than die.
                    log.warning("stream cam=%s: read failed, reopening", self.camera_id)
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue
                ok, jpg = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    continue
                try:
                    self.ws.send_bytes(prefix + jpg.tobytes())
                except Exception as e:
                    log.warning("stream cam=%s: ws send failed (%s) — stopping worker", self.camera_id, e)
                    return
                # Pace to TARGET_FPS — sleep only the remainder so we don't drift.
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    self.stop_event.wait(delay)
                else:
                    next_tick = time.monotonic()
        finally:
            cap.release()
            log.info("stream cam=%s: stopped", self.camera_id)


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
                        w = _StreamWorker(cam, rtsp, adapter, threading.Event())
                        workers[cam] = w
                        w.start()
                    elif t == "stop-stream":
                        cam = cmd.get("camera_id")
                        w = workers.pop(cam, None)
                        if w:
                            w.stop()
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
