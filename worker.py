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


def _load_inference_module(path: str):
    spec = importlib.util.spec_from_file_location("inference", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load inference script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("set_model_path", "set_context", "run"):
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
    model_url = pipeline["model_presigned_url"]
    inference_url = pipeline["inference_script_presigned_url"]
    cfg_block = pipeline.get("config", {}) or {}
    frame_interval = max(1, int(cfg_block.get("frame_interval", 5)))

    log.info("worker start dep=%s rtsp=%s", deployment_id, rtsp_url.split("@")[-1])

    # 1. Download model + inference script (cached)
    try:
        model_path = fetch_to_cache(model_url, cfg["models_cache_dir"], ".pt")
        script_path = fetch_to_cache(inference_url, cfg["scripts_cache_dir"], ".py")
    except Exception as e:
        log.error("asset download failed: %s", e)
        sys.exit(2)

    # 2. Load inference script and inject context
    try:
        inf = _load_inference_module(script_path)
        inf.set_model_path(model_path)
        inf.set_context({
            "deployment_id": deployment_id,
            "pipeline_id": pipeline.get("pipeline_id"),
            "camera_id": pipeline.get("camera_id"),
            "branch_id": cfg["branch_id"],
            "backend_url": cfg["server_url"],
            "agent_token": cfg["secret_token"],
            "branch_timezone": pipeline.get("branch_timezone", "UTC"),
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

    frame_idx = 0
    backoff = 1
    while not stopping:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            log.warning("cannot open RTSP — retrying in %ss", backoff)
            time.sleep(backoff)
            backoff = min(30, backoff * 2)
            continue
        log.info("RTSP open")
        backoff = 1
        try:
            while not stopping:
                ok, frame = cap.read()
                if not ok or frame is None:
                    log.warning("frame read failed — reconnecting")
                    break
                frame_idx += 1
                if frame_idx % frame_interval != 0:
                    continue
                try:
                    inf.run(frame)
                except Exception as e:
                    log.warning("inference run error: %s", e)
        finally:
            cap.release()

    log.info("worker stop dep=%s", deployment_id)


if __name__ == "__main__":
    main()
