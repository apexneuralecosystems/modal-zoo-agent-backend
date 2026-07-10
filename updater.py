"""Self-update: download a versioned agent zip, verify it, unpack it beside the
running version, and atomically switch the `current_version` pointer to it.

The process then exits; launchd (KeepAlive:true) re-execs run.sh, which reads the
new current_version and launches the new code. A bad checksum or unpack error
aborts WITHOUT moving the pointer, so the running version keeps serving.

We use a text pointer file (current_version) instead of a symlink so the swap
works identically on macOS and on a Windows dev box, with no elevated privileges.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import zipfile
from pathlib import Path

import requests

from agent_paths import (
    AGENT_ROOT, CURRENT_VERSION_FILE, FAILED_VERSIONS_FILE, STABLE_FILE,
    VERSIONS_DIR, running_version,
)

log = logging.getLogger("agent.updater")

# Escalating cooldown before re-attempting a version that already failed the
# post-upgrade health check once (see watchdog.py). A version can fail health
# because it's genuinely broken OR because the cloud/network happened to be
# down for that whole window (e.g. planned server maintenance) — we can't
# always tell which, so instead of retrying immediately (and risking the exact
# same rollback thrash every poll tick), we back off further each time the
# same version keeps failing.
FAILURE_BACKOFF_S = [15 * 60, 60 * 60, 6 * 60 * 60]  # 15m, 1h, then 6h and holds


def _read_failed_versions() -> dict:
    try:
        return json.loads(FAILED_VERSIONS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_failed_versions(data: dict) -> None:
    tmp = FAILED_VERSIONS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, FAILED_VERSIONS_FILE)


def record_failed_version(version: str) -> None:
    """Called by the watchdog right after it rolls back `version`."""
    data = _read_failed_versions()
    entry = data.get(version, {"count": 0})
    entry["count"] = int(entry.get("count", 0)) + 1
    entry["last_failed_at"] = time.time()
    data[version] = entry
    try:
        _write_failed_versions(data)
    except OSError as e:
        log.warning("could not persist failed-version record for %s: %s", version, e)


def failure_backoff_remaining_s(version: str) -> float:
    """Seconds until `version` may be re-attempted (0 if it never failed, or
    its cooldown already elapsed)."""
    entry = _read_failed_versions().get(version)
    if not entry:
        return 0.0
    count = int(entry.get("count", 1))
    backoff = FAILURE_BACKOFF_S[min(count, len(FAILURE_BACKOFF_S)) - 1]
    elapsed = time.time() - float(entry.get("last_failed_at", 0))
    return max(0.0, backoff - elapsed)


def clear_failed_version(version: str) -> None:
    """Called once a version is confirmed healthy — forget any past failures
    so a since-fixed version isn't stuck on an old cooldown."""
    data = _read_failed_versions()
    if data.pop(version, None) is not None:
        try:
            _write_failed_versions(data)
        except OSError as e:
            log.warning("could not clear failed-version record for %s: %s", version, e)


def _venv_python() -> str | None:
    """Path to the shared venv's python (created by setup.sh), or None if absent."""
    for c in (
        AGENT_ROOT / ".venv" / "bin" / "python3",
        AGENT_ROOT / ".venv" / "bin" / "python",
        AGENT_ROOT / ".venv" / "Scripts" / "python.exe",
    ):
        if c.exists():
            return str(c)
    return None


def _install_deps(version_dir: Path) -> None:
    """Install the new version's requirements into the shared venv BEFORE we swap.
    Raises RuntimeError on failure so the caller aborts the swap (old version
    keeps running). No-op when there's no requirements.txt or no venv (dev mode)."""
    req = version_dir / "requirements.txt"
    if not req.exists():
        return
    py = _venv_python()
    if not py:
        log.warning("no venv at %s/.venv — skipping dependency install", AGENT_ROOT)
        return
    log.info("installing dependencies for %s ...", version_dir.name)
    proc = subprocess.run(
        [py, "-m", "pip", "install", "-r", str(req)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pip install failed (rc={proc.returncode}) — aborting upgrade: "
            f"{(proc.stderr or proc.stdout)[-500:]}"
        )
    log.info("dependencies installed for %s", version_dir.name)


def verify_sha256(data: bytes, expected: str) -> bool:
    return hashlib.sha256(data).hexdigest().lower() == (expected or "").lower()


def _default_download(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def write_current_version(version: str) -> None:
    """Atomically point current_version at `version` (write temp + os.replace)."""
    tmp = CURRENT_VERSION_FILE.with_name("current_version.tmp")
    tmp.write_text(version + "\n", encoding="utf-8")
    os.replace(tmp, CURRENT_VERSION_FILE)


def write_stable_version(version: str) -> None:
    """Record which version is the Stable fallback for this Mac."""
    tmp = STABLE_FILE.with_name("stable_version.tmp")
    tmp.write_text(version + "\n", encoding="utf-8")
    os.replace(tmp, STABLE_FILE)


def _unpack(data: bytes, version: str) -> Path:
    """Extract a verified zip into versions/<version>/ (atomic move into place)."""
    dest = VERSIONS_DIR / version
    staging = VERSIONS_DIR / f".{version}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(staging)
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(staging, dest)
    return dest


def _ensure_present(info: dict, download) -> str:
    """Make sure a version (the Stable fallback) is unpacked on disk, downloading
    it if missing. Returns the version. Used so a failed Latest can drop to Stable
    WITHOUT needing network at rollback time."""
    version = str(info["version"])
    if (VERSIONS_DIR / version).exists():
        return version
    data = download(info["zip_url"])
    if not verify_sha256(data, info.get("sha256", "")):
        raise ValueError(f"sha256 mismatch for fallback {version}")
    _unpack(data, version)
    return version


def apply_upgrade(payload: dict, download=None) -> str:
    """Download, verify, unpack into versions/<version>/, then switch the pointer.
    Returns the installed version. Raises ValueError on checksum mismatch (before
    the pointer is touched). `download` defaults to _default_download, resolved at
    call time so it stays patchable.

    payload: { version, zip_url, sha256, target?: 'latest'|'stable',
               fallback?: { version, zip_url, sha256 } }
    When a `fallback` (Stable) is given, it's ensured on disk and recorded so the
    watchdog can drop to it if this version fails."""
    download = download or _default_download
    version = str(payload["version"])

    data = download(payload["zip_url"])
    if not verify_sha256(data, payload.get("sha256", "")):
        raise ValueError(f"sha256 mismatch for agent {version} — refusing to install")

    dest = _unpack(data, version)

    # Record the Stable fallback this Mac should drop to if the new version fails.
    fallback = payload.get("fallback")
    if fallback:
        sv = _ensure_present(fallback, download)
        write_stable_version(sv)
    elif payload.get("target") == "stable":
        # We're installing Stable itself — it becomes this Mac's stable anchor.
        write_stable_version(version)

    # Install any new dependencies into the shared venv BEFORE switching the
    # pointer. If this fails, we DON'T swap — the old version keeps running.
    _install_deps(dest)

    write_current_version(version)
    log.info("upgrade applied: current_version -> %s", version)
    return version


def handle_upgrade(payload: dict, api, stop_event: threading.Event) -> dict:
    """Apply the upgrade then set stop_event so the process exits and launchd
    re-execs on the new code. The startup watchdog confirms health or rolls back.

    Fix #17: if we're already running the requested version (a duplicate or
    resent upgrade command), skip the download/verify/unpack/pip-install/
    restart pipeline entirely instead of redoing it — and restarting — for
    no reason."""
    target_version = str(payload["version"])
    if target_version == running_version():
        log.info("upgrade to %s requested but already running it — skipping", target_version)
        return {"version": target_version, "already_current": True}

    remaining = failure_backoff_remaining_s(target_version)
    if remaining > 0:
        log.warning(
            "upgrade to %s skipped — it failed its health check recently, "
            "retrying again in %ds", target_version, int(remaining),
        )
        return {"version": target_version, "held_off": True, "retry_after_s": int(remaining)}

    version = apply_upgrade(payload)
    stop_event.set()
    return {"version": version}
