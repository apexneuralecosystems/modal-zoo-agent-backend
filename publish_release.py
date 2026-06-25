#!/usr/bin/env python3
"""Publish a new Mac-agent release in one command.

It builds a zip of the agent code (with the VERSION you give), uploads it to the
cloud via a presigned S3 URL, and registers it as the new "latest" release. After
this, the Update button for that version appears on each Mac in the dashboard.

Usage:
    python publish_release.py 1.0.1 \
        --server http://localhost:3000 \
        --token  <super_admin_JWT> \
        --notes  "fix camera reconnect"

Notes:
  - --server is your backend URL (local while testing, e.g. http://localhost:3000).
  - --token is a super_admin/admin JWT (the same auth the dashboard uses).
  - The zip contains the agent .py files + run.sh + a VERSION file set to <version>.
    It excludes shared/runtime state (config.json, logs, caches, versions/, tests).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import requests

# Folders/files that must NOT go into a release zip — they are shared runtime
# state that lives at the agent root and survives version swaps, or dev-only.
EXCLUDE_DIR_PARTS = {
    "versions", "logs", "models_cache", "scripts_cache",
    "__pycache__", ".pytest_cache", ".ruff_cache", ".git", "docs", "venv",
}
EXCLUDE_FILES = {"config.json", "current_version", "last_good"}


def build_zip(src: Path, version: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src.rglob("*")):
            if p.is_dir():
                continue
            rel = p.relative_to(src)
            if set(rel.parts) & EXCLUDE_DIR_PARTS:
                continue
            if rel.name in EXCLUDE_FILES:
                continue
            if rel.name.startswith("test_") or rel.name.endswith((".pyc", ".log")):
                continue
            if rel.name == "VERSION":
                continue  # we write our own below so VERSION always matches --version
            z.writestr(str(rel).replace("\\", "/"), p.read_bytes())
        z.writestr("VERSION", version + "\n")
    return buf.getvalue()


def _unwrap(body):
    """Backend wraps some responses as { success, data }. Return the inner data."""
    if isinstance(body, dict) and "data" in body and "success" in body:
        return body["data"]
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description="Publish a new agent release.")
    ap.add_argument("version", help="e.g. 1.0.1")
    ap.add_argument("--server", required=True, help="Backend base URL, e.g. http://localhost:3000")
    ap.add_argument("--token", default=None, help="super_admin/admin JWT (or use --email/--password)")
    ap.add_argument("--email", default=None, help="Login email (alternative to --token)")
    ap.add_argument("--password", default=None, help="Login password (used with --email)")
    ap.add_argument("--source", default=str(Path(__file__).resolve().parent),
                    help="Folder holding the agent code (default: this folder)")
    ap.add_argument("--notes", default=None, help="Optional release notes")
    args = ap.parse_args()

    src = Path(args.source).resolve()
    base = args.server.rstrip("/")

    token = args.token
    if not token:
        if not (args.email and args.password):
            print("ERROR: provide --token, or --email and --password.")
            return 2
        lr = requests.post(f"{base}/auth/login",
                           json={"email": args.email, "password": args.password}, timeout=30)
        lr.raise_for_status()
        token = _unwrap(lr.json()).get("access_token")
        if not token:
            print("ERROR: login succeeded but no access_token returned.")
            return 1
        print("[auth] logged in.")
    headers = {"Authorization": f"Bearer {token}"}

    blob = build_zip(src, args.version)
    sha = hashlib.sha256(blob).hexdigest()
    print(f"[1/3] Built zip for {args.version}: {len(blob)} bytes, sha256={sha}")

    # 1. ask the cloud for a presigned upload URL
    r = requests.post(f"{base}/platform/agent-releases/upload-url",
                      json={"version": args.version}, headers=headers, timeout=30)
    r.raise_for_status()
    up = _unwrap(r.json())
    upload_url = up["upload_url"]
    print(f"[2/3] Got upload URL, uploading zip to S3 ...")

    # 2. upload the zip straight to S3
    pr = requests.put(upload_url, data=blob,
                      headers={"Content-Type": "application/zip"}, timeout=300)
    pr.raise_for_status()

    # 3. register it as the new latest release
    body = {"version": args.version, "sha256": sha}
    if args.notes:
        body["notes"] = args.notes
    rr = requests.post(f"{base}/platform/agent-releases", json=body, headers=headers, timeout=30)
    rr.raise_for_status()

    print(f"[3/3] Published {args.version} as the latest release. [OK]")
    print("Now open the dashboard - each Mac behind this version shows an Update button.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
