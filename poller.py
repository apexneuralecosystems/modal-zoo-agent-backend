"""Job poller — every N seconds:
  1. GET /agent/jobs
  2. Diff against currently-running worker subprocesses
  3. Spawn new workers, kill removed ones
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path

from api_client import ApiClient

log = logging.getLogger("agent.poller")
WORKER_PATH = str(Path(__file__).resolve().parent / "worker.py")


def _pipeline_hash(pipeline: dict) -> str:
    """Stable hash of the pipeline config so we can detect changes.
    Excludes presigned-URL query strings (they rotate every poll) — we hash
    the underlying S3 path/key instead so identical assets compare equal."""
    def _strip(url: str | None) -> str:
        if not url:
            return ""
        return url.split("?", 1)[0]

    snapshot = {
        "rtsp_url": pipeline.get("rtsp_url"),
        "model": _strip(pipeline.get("model_presigned_url")),
        "inference": _strip(pipeline.get("inference_script_presigned_url")),
        "config": pipeline.get("config") or {},
        "event_types": pipeline.get("event_types") or [],
        "pipeline_id": pipeline.get("pipeline_id"),
    }
    blob = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()


class JobPoller:
    def __init__(self, api: ApiClient, cfg: dict, stop_event: threading.Event):
        self.api = api
        self.cfg = cfg
        self.stop_event = stop_event
        self.interval = max(5, int(cfg.get("poll_interval_s", 10)))
        self.running: dict[str, subprocess.Popen] = {}
        # Fix #10: track the pipeline hash for each running worker so we can
        # detect when the cloud pushed a config change (new model, new
        # frame_interval, new RTSP) and restart the worker.
        self.hashes: dict[str, str] = {}

    def _tail_worker(self, deployment_id: str, proc: subprocess.Popen, err_path: str):
        """Read worker stdout line-by-line and re-emit through the main logger."""
        dep_log = logging.getLogger("agent.worker")
        short = deployment_id[:8]
        try:
            for raw in proc.stdout:  # type: ignore[union-attr]
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                lower = line.lower()
                if "event triggered" in lower or "event sent" in lower or ">>> event" in lower:
                    dep_log.info("[dep=%s] *** %s", short, line)
                elif "[warning]" in lower or "failed" in lower or "error" in lower:
                    dep_log.warning("[dep=%s] %s", short, line)
                else:
                    dep_log.info("[dep=%s] %s", short, line)
        except Exception:
            pass

    def _spawn(self, deployment_id: str, pipeline: dict):
        log.info("spawn worker dep=%s", deployment_id)
        cmd = [
            sys.executable, WORKER_PATH,
            "--deployment-id", deployment_id,
            "--pipeline", json.dumps(pipeline),
        ]
        log_dir = self.cfg.get("log_dir") or str(Path(__file__).resolve().parent / "logs")
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass
        err_path = os.path.join(log_dir, f"worker-{deployment_id}.err")
        try:
            # On Windows, CREATE_NEW_PROCESS_GROUP isolates the worker from
            # Ctrl+C / console signals sent to the parent. Without this,
            # pressing Ctrl+C kills YOLO's BLAS threads mid-inference and the
            # worker crashes before it can POST the event.
            extra = {}
            if platform.system() == "Windows":
                extra["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).resolve().parent),
                **extra,
            )
            self.running[deployment_id] = proc
            # Tail worker output in a daemon thread — re-emits through the
            # main logger so event-fired lines appear in the terminal.
            t = threading.Thread(
                target=self._tail_worker,
                args=(deployment_id, proc, err_path),
                daemon=True,
                name=f"tail-{deployment_id[:8]}",
            )
            t.start()
        except Exception as e:
            log.error("spawn failed for %s: %s", deployment_id, e)

    def _kill(self, deployment_id: str):
        proc = self.running.pop(deployment_id, None)
        self.hashes.pop(deployment_id, None)
        if not proc:
            return
        log.info("kill worker dep=%s", deployment_id)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception as e:
            log.warning("kill error %s: %s", deployment_id, e)

    def _reap(self):
        """Drop entries whose subprocess has died so they get respawned."""
        dead = [d for d, p in self.running.items() if p.poll() is not None]
        for d in dead:
            log.warning("worker dep=%s exited code=%s", d, self.running[d].returncode)
            self.running.pop(d, None)
            self.hashes.pop(d, None)

    def _tick(self):
        try:
            jobs = self.api.get_jobs()
        except Exception as e:
            log.warning("get_jobs failed: %s", e)
            return

        wanted = {j["deployment_id"]: j["pipeline"] for j in jobs}

        for dep_id, pipeline in wanted.items():
            new_hash = _pipeline_hash(pipeline)
            if dep_id not in self.running:
                self._spawn(dep_id, pipeline)
                self.hashes[dep_id] = new_hash
            elif self.hashes.get(dep_id) != new_hash:
                # Fix #10: pipeline config changed (model, frame_interval,
                # rtsp, etc.) — restart the worker so the new config takes
                # effect immediately instead of after the worker happens to die.
                log.info("pipeline changed for dep=%s — restarting worker", dep_id)
                self._kill(dep_id)
                self._spawn(dep_id, pipeline)
                self.hashes[dep_id] = new_hash

        for dep_id in list(self.running.keys()):
            if dep_id not in wanted:
                self._kill(dep_id)

        self._reap()

    def run(self):
        log.info("poller starting interval=%ss", self.interval)
        while not self.stop_event.is_set():
            self._tick()
            self.stop_event.wait(self.interval)
        # shutdown — kill all workers
        for dep_id in list(self.running.keys()):
            self._kill(dep_id)
        log.info("poller stopped")


def start_poller(api: ApiClient, cfg: dict, stop_event: threading.Event) -> tuple[threading.Thread, JobPoller]:
    poller = JobPoller(api, cfg, stop_event)
    t = threading.Thread(target=poller.run, name="poller", daemon=True)
    t.start()
    return t, poller
