"""People dwell-heatmap worker. Runs (in a thread) while a camera's heatmap job
is active: grabs frames, detects people, accumulates dwell-seconds on the exact
people pixels, and every `interval_minutes` renders the heatmap over a clean
background and uploads it to S3. Resets each new (local) day.

All I/O is injected (open_stream / upload_jpg / upload_npy / download_model) so
the math is testable and the agent owns the RTSP/S3 specifics.
"""
import datetime as _dt
import logging
import os
import tempfile
import time

# Set FFmpeg socket timeouts BEFORE importing cv2 (same as worker.py) so a
# wedged RTSP connection can't block the heatmap worker indefinitely.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "stimeout;5000000|rw_timeout;5000000",
)

import cv2

from heatmap_core import DwellHeatmap, render, save_raw, load_raw
from heatmap_detector import PersonDetector, mask_from_boxes
from dwell_gate import DwellGate

log = logging.getLogger("heatmap_worker")


def local_day(ts: float) -> str:
    """Local-time date (store-local) as YYYY-MM-DD for the given epoch seconds."""
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


WARMUP_SECONDS = 60  # upload a first heatmap this soon after start (then every interval)


def due_to_upload(last_upload_ts: float, now_ts: float, interval_minutes: int) -> bool:
    return (now_ts - last_upload_ts) >= interval_minutes * 60


def _npy_path(cam: str, day: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"heatmap_{cam}_{day}.npy")


def run_heatmap(job, stop_flag, *, open_stream, upload_jpg, upload_npy, download_model):
    """Run the dwell-heatmap loop for one camera until stop_flag is set.

    job: dict with camera_id, rtsp_url, interval_minutes, model_presigned_url, conf.
    stop_flag: object with .is_set() (e.g. threading.Event).
    open_stream(rtsp) -> cv2.VideoCapture-like (has .read()/.release()).
    upload_jpg(camera_id, day, jpg_bytes) / upload_npy(camera_id, day, npy_bytes).
    download_model(presigned_url) -> local .pt path.
    """
    cam = job["camera_id"]
    interval = int(job.get("interval_minutes", 30))
    conf = float(job.get("conf", 0.3))
    # Only people who stay put for this long count as "dwelling" — walk-throughs
    # are ignored. Tracking must stay on so each person keeps a stable id.
    dwell_seconds = float(job.get("dwell_seconds", 2.0))

    model_path = download_model(job["model_presigned_url"])
    det = PersonDetector(model_path, conf=conf, use_tracking=True)
    gate = DwellGate(min_seconds=dwell_seconds)

    cap = open_stream(job["rtsp_url"])
    ok, frame = cap.read()
    if not ok or frame is None:
        log.warning("heatmap[%s]: could not read first frame; aborting", cam)
        cap.release()
        return

    h, w = frame.shape[:2]
    day = local_day(time.time())
    hm = DwellHeatmap(h, w)
    resumed = load_raw(_npy_path(cam, day))
    if resumed is not None and resumed.shape == hm.accumulator.shape:
        hm.accumulator = resumed
        log.info("heatmap[%s]: resumed %s from saved accumulator", cam, day)
    background = frame.copy()
    now0 = time.time()
    prev_t = now0
    # Do NOT flush at startup (the accumulator is empty). Flush a first image
    # after a short warmup so the user sees heat quickly, then every interval.
    last_upload = now0
    warmup_done = False

    log.info("heatmap[%s]: started (%dx%d, interval=%dm)", cam, w, h, interval)
    while not stop_flag.is_set():
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt <= 0 or dt > 1.0:        # guard first frame / stalls
            dt = 0.1

        today = local_day(now)
        if today != day:               # new day -> fresh background + accumulator
            day = today
            hm = DwellHeatmap(h, w)
            gate = DwellGate(min_seconds=dwell_seconds)
            background = frame.copy()
            last_upload = now
            warmup_done = False
            log.info("heatmap[%s]: new day %s, reset", cam, day)

        try:
            tracks = det.detect_tracks(frame)
        except Exception as e:
            log.warning("heatmap[%s]: detect error: %s", cam, e)
            continue
        # Key each track by its YOLO id; if the tracker didn't assign one, fall
        # back to a coarse position bucket so the dwell timer still works.
        keyed = [
            ((tid if tid is not None else ("p", x1 // 40, y1 // 40)), x1, y1, x2, y2)
            for (tid, x1, y1, x2, y2) in tracks
        ]
        dwelling = gate.update(keyed, now)     # only people stationary >= dwell_seconds
        mask = mask_from_boxes(dwelling, h, w)
        hm.add(mask, dt)

        # First image after WARMUP_SECONDS of accumulation, then every interval.
        should_flush = (
            (not warmup_done and (now - last_upload) >= WARMUP_SECONDS)
            or due_to_upload(last_upload, now, interval)
        )
        if should_flush:
            _flush(cam, day, hm, background, upload_jpg, upload_npy)
            last_upload = now
            warmup_done = True

    # final flush on stop so the latest dwell isn't lost
    _flush(cam, day, hm, background, upload_jpg, upload_npy)
    cap.release()
    log.info("heatmap[%s]: stopped", cam)


def _flush(cam, day, hm, background, upload_jpg, upload_npy):
    overlay = render(hm.accumulator, background)
    ok_jpg, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok_jpg:
        return
    try:
        upload_jpg(cam, day, buf.tobytes())
        path = _npy_path(cam, day)
        save_raw(hm.accumulator, path)
        with open(path, "rb") as f:
            upload_npy(cam, day, f.read())
    except Exception as e:
        log.warning("heatmap[%s]: upload failed: %s", cam, e)


def _main():
    """Subprocess entrypoint. Spawned by the poller:
        python heatmap_worker.py --job '<json>'
    Builds the real RTSP/model/S3 dependencies from the agent config and runs
    the dwell-heatmap loop until SIGTERM/SIGINT.
    """
    import argparse
    import json
    import signal
    import threading

    from api_client import ApiClient
    from asset_cache import fetch_to_cache
    from config_loader import load_config, setup_logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="Heatmap job JSON")
    args = parser.parse_args()
    job = json.loads(args.job)

    cfg = load_config()
    setup_logging(cfg["log_dir"], name="agent.heatmap")
    api = ApiClient(cfg["server_url"], cfg["secret_token"])

    stop_flag = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_flag.set())
    signal.signal(signal.SIGINT, lambda *_: stop_flag.set())

    def open_stream(rtsp):
        return cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)

    def download_model(url):
        return fetch_to_cache(url, cfg["models_cache_dir"], ".pt")

    def upload_jpg(cam, day, data):
        r = api.get_heatmap_upload_url(cam, day, "jpg")
        api.put_bytes(r["presigned_url"], data, "image/jpeg")

    def upload_npy(cam, day, data):
        r = api.get_heatmap_upload_url(cam, day, "npy")
        api.put_bytes(r["presigned_url"], data, "application/octet-stream")

    run_heatmap(
        job, stop_flag,
        open_stream=open_stream,
        upload_jpg=upload_jpg,
        upload_npy=upload_npy,
        download_model=download_model,
    )


if __name__ == "__main__":
    _main()
