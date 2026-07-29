"""Worker subprocess — one per active deployment.

Spawned by the poller with the pipeline JSON on stdin. Pulls the RTSP stream,
loads the inference module, calls run(frame) per frame.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import signal
import sys
import time

# Fix #4: set FFmpeg socket timeouts BEFORE importing cv2 so the env vars
# take effect. stimeout = TCP connect timeout (microseconds);
# rw_timeout = read/write timeout (microseconds). Without these, a wedged
# RTSP connection blocks the worker indefinitely.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "stimeout;5000000|rw_timeout;5000000",
)

import cv2

from asset_cache import fetch_to_cache
from config_loader import load_config, setup_logging


def _crop_to_zone(frame, zone):
    """zone is a normalized {x,y,w,h} rectangle (0-1, top-left origin), or
    None/falsy for "full frame" (today's default, unchanged). Returns a view
    into `frame` -- the inference script never knows a zone was involved, it
    just receives a smaller frame and detects on it like always."""
    if not zone:
        return frame
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(zone.get("x", 0) * w)))
    y1 = max(0, min(h - 1, int(zone.get("y", 0) * h)))
    x2 = max(x1 + 1, min(w, int((zone.get("x", 0) + zone.get("w", 1)) * w)))
    y2 = max(y1 + 1, min(h, int((zone.get("y", 0) + zone.get("h", 1)) * h)))
    return frame[y1:y2, x1:x2]


def _load_inference_module(path: str):
    spec = importlib.util.spec_from_file_location("inference", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load inference script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # set_model_paths is the new multi-model hook; older scripts only expose
    # set_model_path. Either one is acceptable -- we'll route at call time.
    if not (hasattr(mod, "set_model_path") or hasattr(mod, "set_model_paths")):
        raise RuntimeError("inference script missing set_model_path()/set_model_paths()")
    for fn in ("set_context", "run"):
        if not hasattr(mod, fn):
            raise RuntimeError(f"inference script missing {fn}()")
    return mod


def main():
    cfg = load_config()
    log = setup_logging(cfg["log_dir"], name="agent.worker")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, help="Pipeline JSON")
    parser.add_argument("--deployment-id", required=True)
    args = parser.parse_args()

    pipeline = json.loads(args.pipeline)
    deployment_id = args.deployment_id
    log = logging.LoggerAdapter(log, {"dep": deployment_id})

    rtsp_url = pipeline["rtsp_url"]
    inference_url = pipeline["inference_script_presigned_url"]
    cfg_block = pipeline.get("config", {}) or {}
    frame_interval = max(1, int(cfg_block.get("frame_interval", 5)))
    # Normalized {x,y,w,h} crop box (deploy-time "zone only" option) -- see
    # _crop_to_zone(). None means full frame, same as before this existed.
    zone = pipeline.get("zone") or None

    # Multi-model pipelines send a list of {node_id, url, filename}; legacy
    # single-model pipelines only send `model_presigned_url`. Normalise to a
    # list so the download loop is uniform.
    models_list = pipeline.get("models")
    if not models_list:
        models_list = [{
            "node_id": "__primary__",
            "url": pipeline["model_presigned_url"],
            "filename": "model.pt",
        }]

    log.info(
        "worker start dep=%s rtsp=%s models=%d",
        deployment_id, rtsp_url.split("@")[-1], len(models_list),
    )

    # 1. Download every model + inference script (cached). Each yolo_model
    #    node in the pipeline gets its own .pt under its own cache entry, keyed
    #    on the presigned URL so re-poll within the cache TTL is a no-op.
    try:
        model_paths_by_node: dict = {}
        for entry in models_list:
            url = entry.get("url") or entry.get("presigned_url")
            if not url:
                raise RuntimeError(f"model entry missing url: {entry}")
            mp = fetch_to_cache(url, cfg["models_cache_dir"], ".pt")
            model_paths_by_node[entry.get("node_id") or "__primary__"] = mp
        script_path = fetch_to_cache(inference_url, cfg["scripts_cache_dir"], ".py")
    except Exception as e:
        log.error("asset download failed: %s", e)
        sys.exit(2)

    # 2. Load inference script and inject model paths + context. Newer scripts
    #    expose set_model_paths({node_id: path}) which supports multi-model
    #    pipelines; older scripts only have set_model_path(path) and we route
    #    the first model into it (single-model back-compat).
    try:
        inf = _load_inference_module(script_path)
        if hasattr(inf, "set_model_paths"):
            inf.set_model_paths(model_paths_by_node)
        else:
            first_path = next(iter(model_paths_by_node.values()))
            inf.set_model_path(first_path)
        inf.set_context({
            "deployment_id": deployment_id,
            "pipeline_id": pipeline.get("pipeline_id"),
            "camera_id": pipeline.get("camera_id"),
            "branch_id": cfg["branch_id"],
            "backend_url": cfg["server_url"],
            "agent_token": cfg["secret_token"],
            "branch_timezone": pipeline.get("branch_timezone", "UTC"),
            "config": pipeline.get("config") or {},
            # User-defined list of event_type / alert_type strings the
            # deployment is allowed to emit. Scripts MUST only call
            # fire_event(...) / fire_alert(...) with one of these values; the
            # backend additionally rejects anything not in this list.
            # An empty list means the deployment fires nothing.
            "event_types": list(pipeline.get("event_types") or []),
            # 'event' → script calls fire_event() → /agent/event
            # 'alert' → script calls fire_alert() → /agent/alert
            # Default 'event' if the field is missing (older deployments).
            "output_kind": (pipeline.get("output_kind") or "event"),
            # Severity used by every alert this deployment fires. Only
            # meaningful when output_kind == 'alert'. Defaults to 'warning'.
            "alert_severity": (pipeline.get("alert_severity") or "warning"),
            # Values for this model's declared custom_fields (see
            # model.entity.ts's custom_fields column), keyed by field key.
            "custom_values": dict(pipeline.get("custom_values") or {}),
        })
    except Exception as e:
        log.error("inference module load failed: %s", e)
        sys.exit(3)

    # 3. RTSP loop with reconnect
    stopping = False

    def _stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    STREAM_FAIL_LIMIT_S = 12 * 60 * 60  # 12 hours of continuous RTSP failure
    stream_failed_since: float | None = None

    frame_idx = 0
    inferred = 0
    last_stat = time.time()
    backoff = 1
    while not stopping:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            # A failed open still allocates real handles/sockets under the
            # hood (FFmpeg backend) — release them now, not just on the
            # success path below, or a long outage leaks one per retry.
            cap.release()
            now = time.time()
            if stream_failed_since is None:
                stream_failed_since = now
            elif now - stream_failed_since >= STREAM_FAIL_LIMIT_S:
                log.error(
                    "RTSP unreachable for 12h dep=%s — requesting camera-offline mark (exit 99)",
                    deployment_id,
                )
                sys.exit(99)
            elapsed_h = (now - stream_failed_since) / 3600
            log.warning(
                "cannot open RTSP — retrying in %ss (failing %.1fh / 12h)",
                backoff, elapsed_h,
            )
            time.sleep(backoff)
            backoff = min(30, backoff * 2)
            continue
        # RTSP opened successfully — reset the failure clock.
        stream_failed_since = None
        log.info("RTSP open — frame_interval=%d", frame_interval)
        backoff = 1
        # Pipelines can register a per-camera-frame hook via the agent wrapper
        # (record_frame). It collects clip footage at the camera's real fps
        # while inference still runs at frame_interval. Pre-resolved here so
        # we don't getattr() in the hot path.
        record_frame_fn = getattr(inf, "record_frame", None)
        try:
            while not stopping:
                ok, frame = cap.read()
                if not ok or frame is None:
                    log.warning("frame read failed — reconnecting")
                    break
                frame_idx += 1
                # Per-frame hook for clip recorders (pipeline event_sink) —
                # always called, regardless of frame_interval skipping.
                if record_frame_fn is not None:
                    try:
                        record_frame_fn(frame)
                    except Exception as e:
                        log.warning("record_frame error: %s", e)
                if frame_idx % frame_interval != 0:
                    continue
                try:
                    # Crop happens only for the model's view -- record_frame_fn
                    # above already got the full, uncropped frame.
                    inf.run(_crop_to_zone(frame, zone))
                    inferred += 1
                except Exception as e:
                    log.warning("inference run error: %s", e)
                # heartbeat every 30s so we can SEE the worker is alive and processing
                now = time.time()
                if now - last_stat >= 30:
                    log.info("stats: frames_read=%d inferences=%d", frame_idx, inferred)
                    last_stat = now
        finally:
            cap.release()

    log.info("worker stop dep=%s", deployment_id)


if __name__ == "__main__":
    main()
