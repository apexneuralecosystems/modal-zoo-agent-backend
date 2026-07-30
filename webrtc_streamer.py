"""Live-preview video over WebRTC — replaces the old per-frame JPEG-over-
WebSocket relay in ws_streamer.py with real video.

Signaling still travels over the existing persistent /ws/agent WebSocket
(ws_streamer.py's control-message loop) — this module only owns the actual
WebRTC peer connection + the RTSP capture feeding it. One _WebRTCSession per
actively-streamed camera.

Bridging note: aiortc is asyncio-based; the rest of the agent (including
ws_streamer.py's WS reader) is plain synchronous threads. A single dedicated
asyncio event loop runs in its own background thread for the lifetime of the
process (see `_loop_thread`); every aiortc call is scheduled onto it via
`asyncio.run_coroutine_threadsafe` and waited on with `.result(timeout=...)`
from the calling (synchronous) thread.

ICE note: aiortc does not trickle its own outgoing ICE candidates (no public
per-candidate event) — it only exposes a complete SDP once gathering
finishes. So this side always waits for `iceGatheringState == "complete"`
before sending its offer, and expects the cloud's answer to already be
complete too (the backend's WebrtcRelayService does the same wait on its
end). Incoming trickled candidates from the cloud are still applied
defensively via addIceCandidate in case that ever changes, but nothing
requires it today.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import av
import numpy as np
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.sdp import candidate_from_sdp
from av import VideoFrame

from rtsp_common import AV_OPEN_TIMEOUT_S as _AV_OPEN_TIMEOUT_S
from rtsp_common import AV_OPTS as _AV_OPTS
from rtsp_common import redact as _redact
from rtsp_common import to_near_live_url as _to_near_live_url

log = logging.getLogger("agent.webrtc")

ICE_SERVERS = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
ICE_GATHER_TIMEOUT_S = 5.0


# ─── Shared asyncio loop, running in its own thread ─────────────────────────

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_loop_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None:
            return _loop
        loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        t = threading.Thread(target=_run, name="webrtc-loop", daemon=True)
        t.start()
        _loop = loop
        _loop_thread = t
        return loop


def _run_coro(coro, timeout: float = 10.0):
    """Schedule `coro` on the shared loop and block the CALLING (sync) thread
    until it finishes. Safe to call from any thread except the loop's own."""
    loop = _ensure_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result(timeout=timeout)


# ─── RTSP capture feeding an aiortc video track ─────────────────────────────

class _RtspVideoTrack(VideoStreamTrack):
    """Pulls the freshest decoded frame from a background RTSP-reading
    thread. Frames are NOT queued/buffered — recv() always returns whatever
    is most recent, matching the same "drop stale frames, stay live" design
    the old JPEG streamer used (see ws_streamer.py's pacing comment)."""

    def __init__(self, rtsp_url: str, near_live_delay: Optional[int]):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.near_live_delay = near_live_delay
        self._stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._frame_seq = 0
        self._thread = threading.Thread(target=self._capture_loop, name="rtsp-capture", daemon=True)
        self._thread.start()

    def stop_capture(self) -> None:
        self._stop.set()

    def _capture_loop(self) -> None:
        backoff = 3.0
        while not self._stop.is_set():
            open_url = _to_near_live_url(self.rtsp_url, self.near_live_delay) if self.near_live_delay else self.rtsp_url
            try:
                container = av.open(open_url, options=_AV_OPTS, timeout=_AV_OPEN_TIMEOUT_S)
            except Exception as e:
                log.warning("webrtc capture: open failed (%s), retry in %.0fs", e, backoff)
                if self._stop.wait(backoff):
                    return
                backoff = min(30.0, backoff * 2)
                continue

            backoff = 3.0
            try:
                vstream = next((s for s in container.streams if s.type == "video"), None)
                if vstream is None:
                    container.close()
                    if self._stop.wait(2.0):
                        return
                    continue
                vstream.thread_type = "AUTO"
                log.info("webrtc capture: started cam-url=%s codec=%s", _redact(self.rtsp_url), vstream.codec_context.name)

                for packet in container.demux(vstream):
                    if self._stop.is_set():
                        container.close()
                        return
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
                        with self._frame_lock:
                            self._latest = img
                            self._frame_seq += 1
                container.close()
                log.warning("webrtc capture: stream ended, reconnecting")
                if self._stop.wait(backoff):
                    return
            except Exception as e:
                try:
                    container.close()
                except Exception:
                    pass
                log.warning("webrtc capture: stream error (%s), reconnecting", e)
                if self._stop.wait(backoff):
                    return

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        # Wait (briefly, non-blocking to the event loop) for the first frame
        # to arrive rather than sending blank video while RTSP is still
        # opening.
        last_seq = -1
        for _ in range(200):  # ~10s ceiling at the 50ms poll below
            with self._frame_lock:
                img = self._latest
                seq = self._frame_seq
            if img is not None and seq != last_seq:
                break
            await asyncio.sleep(0.05)
        else:
            img = np.zeros((480, 640, 3), dtype="uint8")

        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


# ─── Per-camera WebRTC session ───────────────────────────────────────────────

@dataclass
class _WebRTCSession:
    camera_id: str
    rtsp_url: str
    near_live_delay: Optional[int]
    send_text: "callable"  # ws_streamer's adapter.send_text
    pc: RTCPeerConnection = None
    track: _RtspVideoTrack = None

    async def _start_async(self) -> None:
        self.pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ICE_SERVERS))
        self.track = _RtspVideoTrack(self.rtsp_url, self.near_live_delay)
        self.pc.addTrack(self.track)

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # aiortc doesn't trickle its own candidates — wait for gathering to
        # finish so the SDP we send actually has them all.
        deadline = time.monotonic() + ICE_GATHER_TIMEOUT_S
        while self.pc.iceGatheringState != "complete" and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

        self.send_text(json.dumps({
            "type": "webrtc-offer",
            "camera_id": self.camera_id,
            "sdp": self.pc.localDescription.sdp,
        }))

    async def _on_answer_async(self, sdp: str) -> None:
        if self.pc is None:
            return
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))

    async def _add_ice_candidate_async(self, candidate: dict) -> None:
        if self.pc is None:
            return
        c = (candidate.get("candidate") or "").strip()
        if not c:
            return
        # Incoming candidates carry the full W3C "candidate:<sdp-attr>" form;
        # aiortc's parser wants just the <sdp-attr> part.
        if c.startswith("candidate:"):
            c = c[len("candidate:"):]
        try:
            parsed = candidate_from_sdp(c)
        except Exception as e:
            log.warning("cam=%s: malformed remote ICE candidate ignored: %s", self.camera_id, e)
            return
        parsed.sdpMid = candidate.get("sdpMid")
        parsed.sdpMLineIndex = candidate.get("sdpMLineIndex")
        await self.pc.addIceCandidate(parsed)

    async def _stop_async(self) -> None:
        if self.track is not None:
            self.track.stop_capture()
        if self.pc is not None:
            await self.pc.close()

    def start(self) -> None:
        _run_coro(self._start_async())

    def on_answer(self, sdp: str) -> None:
        _run_coro(self._on_answer_async(sdp))

    def add_ice_candidate(self, candidate: dict) -> None:
        try:
            _run_coro(self._add_ice_candidate_async(candidate))
        except Exception as e:
            log.warning("cam=%s: failed to apply remote ICE candidate: %s", self.camera_id, e)

    def stop(self) -> None:
        try:
            _run_coro(self._stop_async(), timeout=5.0)
        except Exception as e:
            log.warning("cam=%s: error stopping webrtc session: %s", self.camera_id, e)


class WebRTCStreamerManager:
    """Owns all active per-camera WebRTC sessions for this agent process.
    Mirrors the lifecycle ws_streamer.py's old `workers: dict[str, _StreamWorker]`
    had — one entry per camera currently being live-viewed."""

    def __init__(self) -> None:
        self.sessions: dict[str, _WebRTCSession] = {}

    def start_stream(self, camera_id: str, rtsp_url: str, near_live_delay: Optional[int], send_text) -> None:
        if camera_id in self.sessions:
            return  # already streaming — idempotent, same as the old worker dict check
        session = _WebRTCSession(camera_id=camera_id, rtsp_url=rtsp_url, near_live_delay=near_live_delay, send_text=send_text)
        self.sessions[camera_id] = session
        try:
            session.start()
        except Exception as e:
            # exc_info=True: this specific error has been hard to pin down —
            # the bare message alone hasn't been enough to find its source,
            # so log the full traceback (which frame, which library call)
            # instead of guessing again next time it happens.
            log.warning("cam=%s: failed to start webrtc session: %s", camera_id, e, exc_info=True)
            self.sessions.pop(camera_id, None)
            # _start_async() creates the RTSP-capture thread (inside
            # _RtspVideoTrack.__init__) BEFORE the parts that can raise
            # (createOffer/setLocalDescription/ICE gathering). Popping the
            # session above without also stopping it would leak that thread —
            # it keeps retrying RTSP forever, invisible, and a subsequent
            # start-stream retry would spawn a SECOND one hammering the same
            # NVR concurrently, since the failed session is no longer in
            # `self.sessions` to be caught by the idempotency check above.
            try:
                session.stop()
            except Exception:
                pass

    def stop_stream(self, camera_id: str) -> None:
        session = self.sessions.pop(camera_id, None)
        if session:
            session.stop()

    def handle_answer(self, camera_id: str, sdp: str) -> None:
        session = self.sessions.get(camera_id)
        if session:
            session.on_answer(sdp)

    def handle_ice_candidate(self, camera_id: str, candidate: dict) -> None:
        session = self.sessions.get(camera_id)
        if session:
            session.add_ice_candidate(candidate)

    def stop_all(self) -> None:
        for camera_id in list(self.sessions.keys()):
            self.stop_stream(camera_id)
