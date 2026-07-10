"""Resolve the agent's on-disk layout so shared state survives version swaps.

Versioned layout (production):
    <AGENT_ROOT>/versions/<version>/   <- code (CODE_DIR), contains a VERSION file
    <AGENT_ROOT>/current_version       <- text file naming the version run.sh should exec
    <AGENT_ROOT>/last_good             <- text file naming the last-known-healthy version
    <AGENT_ROOT>/run.sh                <- launchd execs this; it reads current_version
    <AGENT_ROOT>/config.json, logs/, *_cache/   <- shared, survive swaps

We switch versions by rewriting the `current_version` text file (atomic os.replace),
NOT a symlink — that works identically on macOS and on a Windows dev box and needs no
elevated privileges. launchd (KeepAlive:true) re-execs run.sh on exit, which reads the
new current_version and launches the new code.

Dev layout (running straight from the repo): AGENT_ROOT == CODE_DIR, self-update is a no-op.

Test/override hooks (env vars, read at import time):
    AGENT_CODE_DIR  -> force CODE_DIR (the folder the "running" code lives in)
    AGENT_HOME      -> force AGENT_ROOT (where shared state + version folders live)
"""
from __future__ import annotations

import os
from pathlib import Path


def _compute_code_dir() -> Path:
    override = os.environ.get("AGENT_CODE_DIR")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parent


def _compute_agent_root(code_dir: Path) -> Path:
    override = os.environ.get("AGENT_HOME")
    if override:
        return Path(override).resolve()
    if code_dir.parent.name == "versions":
        return code_dir.parent.parent
    return code_dir


CODE_DIR = _compute_code_dir()
AGENT_ROOT = _compute_agent_root(CODE_DIR)

VERSIONS_DIR = AGENT_ROOT / "versions"
CURRENT_VERSION_FILE = AGENT_ROOT / "current_version"
LAST_GOOD_FILE = AGENT_ROOT / "last_good"
# The "stable" channel version known to this Mac — the watchdog drops to this
# (not just the previous version) when a "latest" update fails to go healthy.
STABLE_FILE = AGENT_ROOT / "stable_version"
# Remembers versions the watchdog has already rolled back, with a timestamp and
# count, so a version that just failed isn't immediately re-attempted the very
# next time the poller sees it offered (e.g. the server was just down for its
# own maintenance, not because the version is actually broken) — see updater.py.
FAILED_VERSIONS_FILE = AGENT_ROOT / "failed_versions.json"


def running_version() -> str:
    """Version of the code currently executing (from CODE_DIR/VERSION)."""
    try:
        return (CODE_DIR / "VERSION").read_text(encoding="utf-8").strip() or "0.0.0"
    except OSError:
        return "0.0.0"
