"""Job poller — every N seconds:
  1. GET /agent/jobs
  2. Diff against currently-running worker subprocesses
  3. Spawn new workers, kill removed ones
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from api_client import ApiClient

log = logging.getLogger("agent.poller")
WORKER_PATH = str(Path(__file__).resolve().parent / "worker.py")


class JobPoller:
    def __init__(self, api: ApiClient, cfg: dict, stop_event: threading.Event):
        self.api = api
        self.cfg = cfg
        self.stop_event = stop_event
        self.interval = max(5, int(cfg.get("poll_interval_s", 10)))
        self.running: dict[str, subprocess.Popen] = {}

    def _spawn(self, deployment_id: str, pipeline: dict):
        log.info("spawn worker dep=%s", deployment_id)
        cmd = [
            sys.executable, WORKER_PATH,
            "--deployment-id", deployment_id,
            "--pipeline", json.dumps(pipeline),
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(Path(__file__).resolve().parent),
            )
            self.running[deployment_id] = proc
        except Exception as e:
            log.error("spawn failed for %s: %s", deployment_id, e)

    def _kill(self, deployment_id: str):
        proc = self.running.pop(deployment_id, None)
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

    def _tick(self):
        try:
            jobs = self.api.get_jobs()
        except Exception as e:
            log.warning("get_jobs failed: %s", e)
            return

        wanted = {j["deployment_id"]: j["pipeline"] for j in jobs}

        for dep_id, pipeline in wanted.items():
            if dep_id not in self.running:
                self._spawn(dep_id, pipeline)

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
