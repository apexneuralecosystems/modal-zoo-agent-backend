"""
Vision AI — Inference Script Template (library style)
=====================================================

This file defines the contract every inference script in the platform must follow.
The Mac Mini Worker imports this module, calls `set_model_path()` once, then
`set_context()` once, then loops feeding RTSP frames into `run(frame)`.

Required public functions:
    set_model_path(path: str) -> None
    set_context(ctx: dict)    -> None
    run(frame: np.ndarray)    -> np.ndarray   # returns annotated frame

Optional helpers (provided here, callable from run()):
    _dispatch(frame, label: str, details: str = "", extra: dict | None = None)
        → preferred. Routes to fire_event or fire_alert based on the
          deployment's output_kind.
    fire_event(frame, event_type: str, details: str = "")
    fire_alert(frame, alert_type: str, severity: str = ...)

═══════════════════════════════════════════════════════════════════════════════
OUTPUT KIND & LABELS — read this first
═══════════════════════════════════════════════════════════════════════════════
Every deployment is configured by the user at deploy time with two things:

  - `output_kind`: 'event' or 'alert'
        'event' → triggers POST /agent/event (passive log entry).
        'alert' → triggers POST /agent/alert (carries severity + needs ack).
  - `event_types`: a list of free-form label strings.
        The labels are used as event_type when output_kind='event', and as
        alert_type when output_kind='alert'. Same list, different endpoint.

Both fields are injected via `set_context(ctx)` as `ctx["output_kind"]` and
`ctx["event_types"]`. For alerts, `ctx["alert_severity"]` carries the chosen
severity ('critical' | 'warning' | 'info').

Rules every script MUST follow:

  1. Call _dispatch(frame, label, ...) — it picks fire_event or fire_alert
     for you. Only use fire_event / fire_alert directly if you really need to.

  2. The label must come from ctx["event_types"]. Anything else is silently
     dropped — both inside the script and again at the backend.

  3. If ctx["event_types"] is empty, the deployment fires nothing. There is
     no default fallback.

  4. Author your detection logic to map your detected classes / conditions
     onto the user's labels. Example: user typed ["ppe_violation",
     "no_helmet"] — your script decides which one fires per detection.

Push flow (cloud, NOT local backend):
    1. fire_event / fire_alert builds a metadata payload + encodes the frame
       to JPEG bytes.
    2. GET  {backend_url}/agent/presigned-url?deployment_id=...&filename=...
       -> returns { presigned_url, s3_key }
    3. PUT  presigned_url with the JPEG bytes (direct S3 upload).
    4. POST {backend_url}/agent/event  (or /agent/alert) with the payload.
    All HTTP calls carry  Authorization: Bearer <agent_token>  header (token
    is supplied via the context dict so this script stays credential-free).

    If any step fails (cloud briefly unreachable), the detection is queued
    in-memory and retried automatically (piggybacked on the frame loop,
    ~every RETRY_INTERVAL_S) until it succeeds, ages out past
    RETRY_MAX_AGE_S, or the queue fills up and the oldest entry is dropped.
    This queue is in-memory only — it does not survive a module reload
    (see "Keep this file STATELESS" below) or a worker restart.

Author guidelines:
- Keep this file STATELESS across deployments — module globals are reset by the
  worker between deployments by reloading the module.
- Do NOT block in run(); offload network I/O to threads.
- Do NOT print secrets. The Mac worker captures stdout to log files.
"""

from __future__ import annotations

import collections
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import cv2
from ultralytics import YOLO


# ─── Module state (set once via setters before run() is called) ───────────────
_model = None
_model_path: str | None = None

_ctx: dict = {
    "deployment_id":    None,   # uuid string
    "camera_id":        None,   # uuid string
    "camera_name":      "",
    "device_name":      "",
    "channel":          0,
    "branch_id":        None,
    "org_id":           None,
    "backend_url":      "https://api.yourplatform.com",
    "agent_token":      "",     # bearer token for /agent/* endpoints
    "pipeline_id":      None,
    "config":           {},     # user-tunable config (confidence_threshold, frame_interval, ...)
    "event_types":      [],     # whitelist of event_type / alert_type strings this deployment can fire
    # 'event' → triggers POST /agent/event (passive log).
    # 'alert' → triggers POST /agent/alert (needs human ack, carries severity).
    "output_kind":      "event",
    # Used by fire_alert() when the deployment is configured for alerts.
    "alert_severity":   "warning",
}

# Detection / event de-bounce
_last_person_count = 0
_last_event_ts     = 0.0
EVENT_COOLDOWN_S   = 5.0

# Retry queue for event/alert sends that fail (cloud briefly unreachable).
# Failed detections are queued here instead of dropped, and flushed one at a
# time from run() — piggybacked on the frame loop that's already running,
# rather than a dedicated background thread, so there's nothing to leak
# across module reloads between deployments (this module is intentionally
# stateless across deployments; see Author guidelines above). Bounded so a
# long outage can't grow memory without limit: the oldest queued item is
# dropped once full, and anything older than RETRY_MAX_AGE_S is dropped as
# no longer actionable.
_retry_queue: "collections.deque[dict]" = collections.deque(maxlen=200)
_retry_lock = threading.Lock()
_last_retry_attempt_ts = 0.0
RETRY_INTERVAL_S = 20.0
RETRY_MAX_AGE_S = 3600.0


# ─── Required interface ───────────────────────────────────────────────────────

def set_model_path(path: str) -> None:
    """Load the YOLO model from a local .pt file."""
    global _model, _model_path
    _model_path = path
    if not os.path.exists(path):
        print(f"[inference] model file missing: {path}")
        _model = None
        return
    try:
        _model = YOLO(path)
        print(f"[inference] model loaded: {path}")
    except Exception as e:
        print(f"[inference] model load failed: {e}")
        _model = None


def set_context(ctx: dict) -> None:
    """Inject deployment + camera + cloud context (called once by the worker)."""
    if isinstance(ctx, dict):
        _ctx.update(ctx)
        # Defensive normalisation: event_types must be a list of strings.
        _ctx["event_types"] = [
            str(t).strip()
            for t in (_ctx.get("event_types") or [])
            if isinstance(t, str) and t.strip()
        ]
        # Clamp output_kind to a known value; default to 'event'.
        if _ctx.get("output_kind") not in ("event", "alert"):
            _ctx["output_kind"] = "event"
        # Clamp alert_severity to a known value; default to 'warning'.
        if _ctx.get("alert_severity") not in ("critical", "warning", "info"):
            _ctx["alert_severity"] = "warning"
        print(
            f"[inference] context: dep={_ctx['deployment_id']} "
            f"cam={_ctx['camera_name']} ch={_ctx['channel']} "
            f"output_kind={_ctx['output_kind']} "
            f"severity={_ctx['alert_severity']} "
            f"event_types={_ctx['event_types']}"
        )


def _safe_filename(stem: str) -> str:
    """
    Sanitise a free-form label so it's safe to drop into a URL query string and
    a filesystem path. The user can type anything as an event_type (spaces,
    slashes, accents). We collapse everything that isn't [a-zA-Z0-9_-] into '_'
    so the resulting filename can be carried in URLs / S3 keys without surprises.
    """
    cleaned = re.sub(r'[^A-Za-z0-9_-]+', '_', stem).strip('_')
    return cleaned or 'event'


def _user_event_label() -> str | None:
    """
    The label this deployment should emit on each trigger. Uses the FIRST entry
    of the user's event_types list (set in the UI when creating the deployment),
    so whatever the user typed is exactly what shows up in the event/alert log.

    Returns None if the user didn't set any labels — in which case the
    deployment is intentionally muted.
    """
    types = _ctx.get("event_types") or []
    for t in types:
        if isinstance(t, str) and t.strip():
            return t
    return None


def _dispatch(frame, label: str, *, details: str = "", extra: dict | None = None) -> None:
    """
    Route a trigger to /agent/event or /agent/alert based on the deployment's
    `output_kind` (set by the user at deploy time).

    - output_kind == 'event' → calls fire_event(frame, label, ...)
    - output_kind == 'alert' → calls fire_alert(frame, label, severity=<ctx>, ...)

    Scripts should call _dispatch() instead of fire_event/fire_alert directly
    so the user's UI choice always wins. Use the underlying functions only if
    your script genuinely needs to do both (rare).
    """
    if _ctx.get("output_kind") == "alert":
        fire_alert(
            frame,
            label,
            severity=_ctx.get("alert_severity") or "warning",
            message=details or "",
            extra=extra,
        )
    else:
        fire_event(frame, label, details=details, extra=extra)


def run(frame):
    """
    Process a single RTSP frame. Must return an annotated frame (BGR ndarray).
    Called in a tight loop by the worker — keep this fast.

    The user picks the event label in the UI (event_types). This script fires
    one event with that exact label every time it sees a new person on screen,
    and one more whenever the person count changes. No hardcoded labels.
    """
    global _model, _last_person_count

    _flush_retry_queue()

    if _model is None or frame is None:
        return frame

    try:
        results = _model(frame, classes=[0], verbose=False)  # class 0 = person
        person_count = 0
        last_conf = 0.0

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                last_conf = conf
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"Person {conf:.2f}",
                    (x1, max(10, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

        cv2.putText(
            frame, f"Persons: {person_count}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )

        # Trigger logic — the LABEL comes from the user's event_types list,
        # so whatever they typed in the UI is exactly what appears in the
        # event/alert log. _dispatch() picks between /agent/event and
        # /agent/alert based on the deployment's output_kind, so the same
        # script supports both modes without any extra branching here.
        label = _user_event_label()
        if label is not None:
            if _last_person_count == 0 and person_count > 0:
                # New person appeared.
                _dispatch(
                    frame, label,
                    details=f"count={person_count}",
                    extra={"confidence": last_conf, "count": person_count, "trigger": "person_appeared"},
                )
            elif person_count > 0 and person_count != _last_person_count:
                # Number of visible persons changed.
                _dispatch(
                    frame, label,
                    details=f"prev={_last_person_count} now={person_count}",
                    extra={"prev": _last_person_count, "now": person_count, "trigger": "count_changed"},
                )

        _last_person_count = person_count
        return frame

    except Exception as e:
        print(f"[inference] run error: {e}")
        return frame


# ─── Helper: cloud event firing (presigned-URL S3 upload + /agent/event POST) ──

def fire_event(frame, event_type: str, details: str = "", extra: dict | None = None) -> None:
    """
    De-bounced. Encodes frame to JPEG, uploads to S3 via presigned URL,
    then POSTs event metadata to /agent/event. All non-blocking.

    The backend enforces a strict whitelist against the deployment's
    `event_types` list, so any event_type that wasn't in the user's UI input
    will be dropped server-side. This script generally builds `event_type`
    from that list itself, so callers don't need to gate.
    """
    global _last_event_ts
    now = time.time()
    if now - _last_event_ts < EVENT_COOLDOWN_S:
        return
    _last_event_ts = now

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return
    jpeg_bytes = buf.tobytes()

    triggered_at = datetime.now(timezone.utc).isoformat()
    # event_type is free-form user input — sanitise before using it as a
    # filename / URL fragment. We still send the original event_type in the
    # JSON body so the DB row keeps the user's exact label.
    filename = f"{_safe_filename(event_type)}_{int(now * 1000)}.jpg"

    payload = {
        "deployment_id": _ctx.get("deployment_id"),
        "camera_id":     _ctx.get("camera_id"),
        "event_type":    event_type,
        "triggered_at":  triggered_at,
        "metadata": {
            "details":     details,
            "pipeline_id": _ctx.get("pipeline_id"),
            "camera_name": _ctx.get("camera_name"),
            "device_name": _ctx.get("device_name"),
            "channel":     _ctx.get("channel"),
            **(extra or {}),
        },
    }

    threading.Thread(
        target=_upload_and_post,
        args=(jpeg_bytes, filename, payload),
        daemon=True,
    ).start()


def fire_alert(frame, alert_type: str, severity: str = "warning",
               title: str | None = None, message: str = "",
               extra: dict | None = None) -> None:
    """
    High-priority detection that needs human acknowledgement.
    Goes into the alerts table (status=active) until ack/resolve.
    Uses the same de-bounce + S3-snapshot upload as fire_event.

    severity: 'critical' | 'warning' | 'info'. If omitted, falls back to the
    deployment's `alert_severity` from ctx (set by the user at deploy time).

    Backend filter: an alert whose alert_type isn't in ctx["event_types"], OR
    whose deployment is configured for events (output_kind='event'), is
    rejected server-side.
    """
    # Honour the deployment's choice of severity if the caller didn't pick one.
    if severity not in ("critical", "warning", "info"):
        severity = _ctx.get("alert_severity") or "warning"
    global _last_event_ts
    now = time.time()
    if now - _last_event_ts < EVENT_COOLDOWN_S:
        return
    _last_event_ts = now

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return
    jpeg_bytes = buf.tobytes()

    triggered_at = datetime.now(timezone.utc).isoformat()
    filename = f"alert_{_safe_filename(alert_type)}_{int(now * 1000)}.jpg"

    payload = {
        "deployment_id": _ctx.get("deployment_id"),
        "alert_type": alert_type,
        "severity": severity if severity in ("critical", "warning", "info") else "warning",
        "title": title or alert_type.replace("_", " ").title(),
        "message": message,
        "triggered_at": triggered_at,
        "metadata": {
            "pipeline_id": _ctx.get("pipeline_id"),
            "camera_name": _ctx.get("camera_name"),
            "device_name": _ctx.get("device_name"),
            "channel": _ctx.get("channel"),
            **(extra or {}),
        },
    }

    threading.Thread(
        target=_upload_and_post_alert,
        args=(jpeg_bytes, filename, payload),
        daemon=True,
    ).start()


def _send_alert(jpeg_bytes: bytes, filename: str, payload: dict) -> bool:
    """One attempt: presigned URL -> S3 PUT -> POST /agent/alert.
    Returns True on success, False on any failure (network, timeout, bad
    response) so the caller can decide whether to queue a retry."""
    backend = _ctx["backend_url"].rstrip("/")
    headers_auth = {"Authorization": f"Bearer {_ctx.get('agent_token', '')}"}
    try:
        # Belt-and-braces: even though fire_event/fire_alert sanitise filename,
        # quote it here too so any future caller can't break the URL with a
        # raw space / unicode / '?' in the name.
        url = (
            f"{backend}/agent/presigned-url"
            f"?deployment_id={urllib.parse.quote(str(payload['deployment_id']), safe='')}"
            f"&filename={urllib.parse.quote(filename, safe='')}"
        )
        req = urllib.request.Request(url, headers=headers_auth, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body_json = json.loads(resp.read().decode("utf-8"))
        # Backend wraps responses as {status, success, data, errors}. Unwrap if present.
        data = body_json.get("data", body_json) if isinstance(body_json, dict) else body_json
        presigned_url = data["presigned_url"]
        s3_key = data["s3_key"]

        put_req = urllib.request.Request(
            presigned_url, data=jpeg_bytes, method="PUT",
            headers={"Content-Type": "image/jpeg"},
        )
        urllib.request.urlopen(put_req, timeout=10).read()

        payload["image_s3_key"] = s3_key
        body = json.dumps(payload).encode("utf-8")
        evt_req = urllib.request.Request(
            f"{backend}/agent/alert", data=body, method="POST",
            headers={**headers_auth, "Content-Type": "application/json"},
        )
        urllib.request.urlopen(evt_req, timeout=5).read()
        return True
    except Exception as e:
        print(f"[inference] alert push failed: {e}")
        return False


def _upload_and_post_alert(jpeg_bytes: bytes, filename: str, payload: dict) -> None:
    """Thread target for fire_alert: one immediate attempt; on failure,
    queues the detection for retry instead of dropping it."""
    if not _send_alert(jpeg_bytes, filename, payload):
        _enqueue_retry("alert", jpeg_bytes, filename, payload)


def _send_event(jpeg_bytes: bytes, filename: str, payload: dict) -> bool:
    """One attempt: presigned URL -> S3 PUT -> POST /agent/event.
    Returns True on success, False on any failure (network, timeout, bad
    response) so the caller can decide whether to queue a retry."""
    backend = _ctx["backend_url"].rstrip("/")
    headers_auth = {"Authorization": f"Bearer {_ctx.get('agent_token', '')}"}

    try:
        # 1. Get presigned URL (URL-encode every variable that goes into the
        #     query string so user-typed labels with spaces / unicode never
        #     break the request).
        url = (
            f"{backend}/agent/presigned-url"
            f"?deployment_id={urllib.parse.quote(str(payload['deployment_id']), safe='')}"
            f"&filename={urllib.parse.quote(filename, safe='')}"
        )
        req = urllib.request.Request(url, headers=headers_auth, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body_json = json.loads(resp.read().decode("utf-8"))
        # Backend wraps responses as {status, success, data, errors}. Unwrap if present.
        data = body_json.get("data", body_json) if isinstance(body_json, dict) else body_json
        presigned_url = data["presigned_url"]
        s3_key        = data["s3_key"]

        # 2. PUT image directly to S3
        put_req = urllib.request.Request(
            presigned_url, data=jpeg_bytes, method="PUT",
            headers={"Content-Type": "image/jpeg"},
        )
        urllib.request.urlopen(put_req, timeout=10).read()

        # 3. POST event metadata
        payload["image_s3_key"] = s3_key
        body = json.dumps(payload).encode("utf-8")
        evt_req = urllib.request.Request(
            f"{backend}/agent/event", data=body, method="POST",
            headers={**headers_auth, "Content-Type": "application/json"},
        )
        urllib.request.urlopen(evt_req, timeout=5).read()
        return True

    except Exception as e:
        print(f"[inference] event push failed: {e}")
        return False


def _upload_and_post(jpeg_bytes: bytes, filename: str, payload: dict) -> None:
    """Thread target for fire_event: one immediate attempt; on failure,
    queues the detection for retry instead of dropping it."""
    if not _send_event(jpeg_bytes, filename, payload):
        _enqueue_retry("event", jpeg_bytes, filename, payload)


def _enqueue_retry(kind: str, jpeg_bytes: bytes, filename: str, payload: dict) -> None:
    with _retry_lock:
        was_full = len(_retry_queue) >= _retry_queue.maxlen
        _retry_queue.append({
            "kind": kind,
            "jpeg_bytes": jpeg_bytes,
            "filename": filename,
            "payload": payload,
            "queued_at": time.time(),
        })
    if was_full:
        print(f"[inference] retry queue full — dropped oldest queued item to make room for this {kind}")


def _flush_retry_queue() -> None:
    """Try to resend the oldest queued item. Called from run() every frame
    but only actually acts once per RETRY_INTERVAL_S, and only handles one
    item per call (in a short-lived daemon thread) so it never blocks the
    frame loop."""
    global _last_retry_attempt_ts
    now = time.time()
    if now - _last_retry_attempt_ts < RETRY_INTERVAL_S:
        return
    _last_retry_attempt_ts = now

    with _retry_lock:
        if not _retry_queue:
            return
        item = _retry_queue[0]
        if now - item["queued_at"] > RETRY_MAX_AGE_S:
            _retry_queue.popleft()
            stale_kind = item["kind"]
            age = int(RETRY_MAX_AGE_S)
            print(f"[inference] dropping queued {stale_kind}, unsent for over {age}s")
            return
        _retry_queue.popleft()

    def _attempt():
        send_fn = _send_event if item["kind"] == "event" else _send_alert
        if send_fn(item["jpeg_bytes"], item["filename"], item["payload"]):
            return
        # Still failing — put it back at the front (oldest-first) for the next flush.
        with _retry_lock:
            _retry_queue.appendleft(item)

    threading.Thread(target=_attempt, daemon=True).start()
