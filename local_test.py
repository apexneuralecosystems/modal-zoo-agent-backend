#!/usr/bin/env python3
"""Local self-update sandbox for Windows (no Mac, no launchd needed).

This lets you watch the WHOLE self-update flow on your own machine:
it builds a proper installed layout in a sandbox folder, then runs a keep-alive
loop that behaves like macOS launchd — it starts the current version, and when
the agent swaps versions and exits, it restarts on the new version.

Usage:
    python local_test.py C:\\path\\to\\sandbox

Steps it does:
  1. Builds  <sandbox>/versions/1.0.0/  from the code in THIS folder
  2. Writes  <sandbox>/current_version  and  <sandbox>/last_good  = 1.0.0
  3. Copies  config.json (or config.example.json) into <sandbox>
  4. Runs the keep-alive loop (Ctrl+C to stop)

Then: edit <sandbox>/config.json with your cloud server_url + secret_token +
branch_id + mac_serial, restart this script, and click "Update" in your
dashboard — you'll see the agent download the new version, swap, and restart here.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Files/dirs that are shared runtime state — never copied into a version folder.
SKIP = {
    "versions", ".venv", "venv", "logs", "models_cache", "scripts_cache",
    "__pycache__", ".pytest_cache", ".ruff_cache", ".git", "config.json",
    "current_version", "last_good", "local_test.py",
}


def _version() -> str:
    return (HERE / "VERSION").read_text(encoding="utf-8").strip() or "1.0.0"


def build_layout(sandbox: Path) -> str:
    version = _version()
    vdir = sandbox / "versions" / version
    vdir.mkdir(parents=True, exist_ok=True)
    for item in HERE.iterdir():
        if item.name in SKIP or item.name.startswith("test_"):
            continue
        dest = vdir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    (sandbox / "current_version").write_text(version + "\n", encoding="utf-8")
    (sandbox / "last_good").write_text(version + "\n", encoding="utf-8")
    cfg = sandbox / "config.json"
    if not cfg.exists():
        src = HERE / "config.json"
        if not src.exists():
            src = HERE / "config.example.json"
        if src.exists():
            shutil.copy2(src, cfg)
    for d in ("logs", "models_cache", "scripts_cache"):
        (sandbox / d).mkdir(exist_ok=True)
    return version


def keepalive(sandbox: Path) -> None:
    """launchd substitute: run the current version; restart it when it exits."""
    print(f"[keepalive] sandbox = {sandbox}")
    print("[keepalive] Ctrl+C to stop.\n")
    while True:
        version = (sandbox / "current_version").read_text(encoding="utf-8").strip()
        main = sandbox / "versions" / version / "main.py"
        print(f"[keepalive] starting agent v{version}")
        rc = subprocess.call([sys.executable, str(main)], cwd=str(main.parent))
        print(f"[keepalive] agent v{version} exited (code={rc}) — restarting in 2s\n")
        time.sleep(2)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    sandbox = Path(sys.argv[1]).resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    version = build_layout(sandbox)
    print(f"Built local install for v{version} at {sandbox}")
    cfg = sandbox / "config.json"
    print(f"-> Edit {cfg} (server_url, secret_token, branch_id, mac_serial) before it can register.\n")
    try:
        keepalive(sandbox)
    except KeyboardInterrupt:
        print("\n[keepalive] stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
