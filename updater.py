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
import logging
import os
import shutil
import subprocess
import threading
import zipfile
from pathlib import Path

import requests

from agent_paths import AGENT_ROOT, CURRENT_VERSION_FILE, STABLE_FILE, VERSIONS_DIR

log = logging.getLogger("agent.updater")


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
    re-execs on the new code. The startup watchdog confirms health or rolls back."""
    version = apply_upgrade(payload)
    stop_event.set()
    return {"version": version}
