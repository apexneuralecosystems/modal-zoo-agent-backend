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
import time
from pathlib import Path

from api_client import ApiClient

log = logging.getLogger("agent.poller")
WORKER_PATH = str(Path(__file__).resolve().parent / "worker.py")
HEATMAP_WORKER_PATH = str(Path(__file__).resolve().parent / "heatmap_worker.py")

# worker.py sys.exit(2)s specifically when the model/inference-script download
# fails. Without backoff, a bad model URL or a network blip makes the poller
# respawn the worker every tick forever, pegging the Mac's CPU. This is
# distinct from the exit(99) RTSP-dead-12h path below.
DOWNLOAD_FAILURE_EXIT_CODE = 2
# worker.py sys.exit(3)s when the inference script/model fails to *load*
# after a successful download (e.g. a corrupt .pt, a script missing
# set_context()/run(), or an exception raised while importing it). Before
# this fix, this exit code fell through to the generic "else" branch in
# _reap() and was silently retried forever with nothing ever reported to the
# backend -- a deployment could crash-loop indefinitely while its DB status
# stayed "running", with zero visibility anywhere in the platform.
SCRIPT_LOAD_FAILURE_EXIT_CODE = 3
MAX_DOWNLOAD_FAILURES = 5
DOWNLOAD_RETRY_BACKOFF_S = 30


def _heatmap_hash(job: dict) -> str:
    """Stable hash of a heatmap job so a config change restarts the worker.
    Excludes the rotating presigned model URL query string."""
    model = (job.get("model_presigned_url") or "").split("?", 1)[0]
    snapshot = {
        "camera_id": job.get("camera_id"),
        "rtsp_url": job.get("rtsp_url"),
        "interval_minutes": job.get("interval_minutes"),
        "conf": job.get("conf"),
        "dwell_seconds": job.get("dwell_seconds"),
        "model": model,
        # Bump the brain version in the cloud -> hash changes -> worker respawns
        # and re-downloads the new brain (versioned S3 path = cache-busted).
        "brain_version": job.get("brain_version"),
    }
    blob = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()


def _pipeline_hash(pipeline: dict) -> str:
    """Stable hash of the pipeline config so we can detect changes.
    Excludes presigned-URL query strings (they rotate every poll) — we hash
    the underlying S3 path/key instead so identical assets compare equal."""
    def _strip(url: str | None) -> str:
        if not url:
            return ""
        return url.split("?", 1)[0]

    # Multi-model pipelines have a list of {node_id, url, ...}; hash each entry's
    # underlying key so swapping ANY of the .pt files triggers a worker restart.
    # Single-model pipelines just have model_presigned_url -- hash that as before.
    models = pipeline.get("models") or []
    if models:
        models_hash = [
            {"node_id": m.get("node_id"), "key": _strip(m.get("url"))}
            for m in models
        ]
    else:
        models_hash = [{"node_id": "__primary__", "key": _strip(pipeline.get("model_presigned_url"))}]

    snapshot = {
        "rtsp_url": pipeline.get("rtsp_url"),
        "models": models_hash,
        "inference": _strip(pipeline.get("inference_script_presigned_url")),
        "config": pipeline.get("config") or {},
        "event_types": pipeline.get("event_types") or [],
        "pipeline_id": pipeline.get("pipeline_id"),
        "custom_values": pipeline.get("custom_values") or {},
        "zone": pipeline.get("zone"),
    }
    blob = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()


class JobPoller:
    def __init__(self, api: ApiClient, cfg: dict, stop_event: threading.Event, now_fn=time.monotonic):
        self.api = api
        self.cfg = cfg
        self.stop_event = stop_event
        self.interval = max(5, int(cfg.get("poll_interval_s", 10)))
        self.running: dict[str, subprocess.Popen] = {}
        # Fix #18: track the stdout-tailing thread for each running worker so
        # _kill() can force-close its pipe and join it instead of leaving it
        # fire-and-forget. See _kill() and _sweep_tail_threads().
        self.tail_threads: dict[str, threading.Thread] = {}
        # Fix #10: track the pipeline hash for each running worker so we can
        # detect when the cloud pushed a config change (new model, new
        # frame_interval, new RTSP) and restart the worker.
        self.hashes: dict[str, str] = {}
        # Deployments whose worker exited with code 99 (RTSP unreachable 12h).
        # The backend marks them camera_offline and removes them from /agent/jobs.
        # We keep the set here so we don't re-spawn before the backend responds.
        self.permanently_failed: set[str] = set()
        # dep_id -> {"count": int, "retry_at": float} while backing off, or
        # {"count": 5, "given_up": True} once we've stopped retrying.
        self.download_failures: dict[str, dict] = {}
        self._now = now_fn

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
            self.tail_threads[deployment_id] = t
        except Exception as e:
            log.error("spawn failed for %s: %s", deployment_id, e)

    def _spawn_heatmap(self, key: str, job: dict):
        """Spawn a heatmap worker subprocess (one per enabled camera)."""
        log.info("spawn heatmap worker %s", key)
        cmd = [sys.executable, HEATMAP_WORKER_PATH, "--job", json.dumps(job)]
        log_dir = self.cfg.get("log_dir") or str(Path(__file__).resolve().parent / "logs")
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass
        err_path = os.path.join(log_dir, f"worker-{key.replace(':', '_')}.err")
        try:
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
            self.running[key] = proc
            t = threading.Thread(
                target=self._tail_worker,
                args=(key, proc, err_path),
                daemon=True,
                name=f"tail-{key}",
            )
            t.start()
            self.tail_threads[key] = t
        except Exception as e:
            log.error("spawn heatmap failed for %s: %s", key, e)

    def _kill(self, deployment_id: str):
        proc = self.running.pop(deployment_id, None)
        self.hashes.pop(deployment_id, None)
        self.download_failures.pop(deployment_id, None)
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

        # Fix #18: the tail thread reading proc.stdout normally exits on its
        # own once the pipe closes, but force-close it here too so a thread
        # blocked on a slow-to-close pipe unblocks immediately instead of
        # lingering. Then join with a short timeout — if it's still alive
        # after that, leave it tracked so _sweep_tail_threads() keeps
        # watching it instead of it silently disappearing from accounting.
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        t = self.tail_threads.get(deployment_id)
        if t is not None:
            t.join(timeout=2)
            if t.is_alive():
                log.warning("tail thread for %s did not exit after kill — still watching it", deployment_id)
            else:
                self.tail_threads.pop(deployment_id, None)

    def _sweep_tail_threads(self):
        """Fix #18: retry cleanup for any tail thread that outlived its job.
        Runs every tick — turns what used to be a silent, unbounded leak into
        a visible, self-healing one: nothing is left tracked as 'active'
        without something actually still watching it."""
        for key in list(self.tail_threads.keys()):
            t = self.tail_threads[key]
            if not t.is_alive():
                self.tail_threads.pop(key, None)
                continue
            if key in self.running:
                continue  # job is still active; thread is supposed to be alive
            t.join(timeout=1)
            if not t.is_alive():
                self.tail_threads.pop(key, None)
            else:
                log.warning("orphaned tail thread for %s still hasn't exited", key)

    def _reap(self):
        """Drop entries whose subprocess has died so they get respawned.
        Exit code 99 means RTSP has been unreachable for 12h — mark the
        deployment camera_offline in the backend and stop respawning.
        Exit code 2 means the model/inference-script download failed; exit
        code 3 means it downloaded fine but failed to load (corrupt weights,
        a script missing set_context()/run(), or an import-time exception).
        Both back off and retry a bounded number of times, then report the
        failure to the backend (see _on_worker_failure)."""
        dead = [d for d, p in self.running.items() if p.poll() is not None]
        for d in dead:
            code = self.running[d].returncode
            log.warning("worker dep=%s exited code=%s", d, code)
            self.running.pop(d, None)
            if code == 99:
                self.hashes.pop(d, None)
                log.error("dep=%s RTSP unreachable 12h — marking camera_offline", d)
                self.permanently_failed.add(d)
                try:
                    self.api.mark_camera_offline(d)
                except Exception as e:
                    log.error("mark_camera_offline failed for %s: %s", d, e)
            elif code == DOWNLOAD_FAILURE_EXIT_CODE and not d.startswith("heatmap:"):
                # Deliberately keep self.hashes[d] intact here (unlike the
                # 99/else branches) — _tick() compares it against the freshly
                # fetched pipeline hash to tell whether anything actually
                # changed before it retries or gives up.
                self._on_worker_failure(d, reason="model_download_failed")
            elif code == SCRIPT_LOAD_FAILURE_EXIT_CODE and not d.startswith("heatmap:"):
                self._on_worker_failure(d, reason="inference_script_failed")
            else:
                self.hashes.pop(d, None)
                self.download_failures.pop(d, None)

    def _on_worker_failure(self, dep_id: str, reason: str):
        """Shared backoff + backend-reporting path for worker.py failures that
        happen before the RTSP loop ever starts (asset download, or inference
        script/model load). Without this, a bad model file or a broken
        inference script crash-loops the worker forever with the deployment's
        DB status stuck on "running" and no error visible anywhere."""
        count = self.download_failures.get(dep_id, {}).get("count", 0) + 1
        if count >= MAX_DOWNLOAD_FAILURES:
            log.error(
                "dep=%s worker failed %d times (%s) — giving up until the pipeline changes",
                dep_id, count, reason,
            )
            self.download_failures[dep_id] = {"count": count, "given_up": True}
            try:
                self.api.mark_deployment_failed(dep_id, reason=reason)
            except Exception as e:
                log.warning("failed to report worker failure for dep=%s: %s", dep_id, e)
        else:
            log.warning(
                "dep=%s worker failed (%d/%d, %s) — retrying in %ss",
                dep_id, count, MAX_DOWNLOAD_FAILURES, reason, DOWNLOAD_RETRY_BACKOFF_S,
            )
            self.download_failures[dep_id] = {
                "count": count,
                "retry_at": self._now() + DOWNLOAD_RETRY_BACKOFF_S,
            }

    def _should_hold_off(self, dep_id: str, new_hash: str) -> bool:
        """True if dep_id just failed to download and isn't due for a retry yet."""
        state = self.download_failures.get(dep_id)
        if not state:
            return False
        if self.hashes.get(dep_id) != new_hash:
            # Pipeline changed since the failure (new model URL, fixed config,
            # a redeploy) — clear the history and give it a clean attempt.
            self.download_failures.pop(dep_id, None)
            return False
        if state.get("given_up"):
            return True
        retry_at = state.get("retry_at")
        return retry_at is not None and self._now() < retry_at

    def _tick(self):
        try:
            jobs = self.api.get_jobs()
        except Exception as e:
            log.warning("get_jobs failed: %s", e)
            return

        # Deployment jobs carry deployment_id + pipeline; heatmap jobs carry
        # type=='heatmap' + camera_id. Handle them separately but share the
        # running-process map (heatmap keys are prefixed 'heatmap:').
        deploy_jobs = [j for j in jobs if j.get("type") != "heatmap" and "deployment_id" in j]
        heatmap_jobs = [j for j in jobs if j.get("type") == "heatmap"]

        wanted = {j["deployment_id"]: j["pipeline"] for j in deploy_jobs}
        wanted_heatmaps = {f"heatmap:{j['camera_id']}": j for j in heatmap_jobs}

        for dep_id, pipeline in wanted.items():
            # Skip deployments that exited with code 99 until the backend
            # removes them from /agent/jobs (status → camera_offline).
            if dep_id in self.permanently_failed:
                continue
            new_hash = _pipeline_hash(pipeline)
            if dep_id not in self.running:
                if self._should_hold_off(dep_id, new_hash):
                    continue
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

        for key, job in wanted_heatmaps.items():
            new_hash = _heatmap_hash(job)
            if key not in self.running:
                self._spawn_heatmap(key, job)
                self.hashes[key] = new_hash
            elif self.hashes.get(key) != new_hash:
                log.info("heatmap config changed for %s — restarting worker", key)
                self._kill(key)
                self._spawn_heatmap(key, job)
                self.hashes[key] = new_hash

        all_wanted = set(wanted.keys()) | set(wanted_heatmaps.keys())
        for key in list(self.running.keys()):
            if key not in all_wanted:
                self._kill(key)

        # Once the backend confirms camera_offline (deployment drops out of jobs),
        # clear the local guard so a future Restart shows up cleanly.
        self.permanently_failed -= (self.permanently_failed - set(wanted.keys()))

        self._reap()
        self._sweep_tail_threads()

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
