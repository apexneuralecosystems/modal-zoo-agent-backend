"""Download model + inference script via presigned URLs and cache locally.

The cloud regenerates presigned URLs on every /agent/jobs poll, so URLs change
frequently. We cache by deployment_id + a short hash of the URL path so we only
re-download when the underlying object actually changes.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from urllib.parse import urlparse

import requests

log = logging.getLogger("agent.cache")

# Fix #7: cached files older than this with no recent access (mtime) are
# pruned. A pruned file is just re-downloaded on next use — no harm done.
CACHE_MAX_AGE_S = 30 * 24 * 3600   # 30 days


def _key_from_url(url: str) -> str:
    # Use the S3 object path (before the ? query) as the cache key — that's the
    # part that identifies the actual file. Query params change every poll.
    path = urlparse(url).path
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


def fetch_to_cache(url: str, cache_dir: str, suffix: str) -> str:
    """Download `url` to `cache_dir` if not already present. Returns local path."""
    os.makedirs(cache_dir, exist_ok=True)
    key = _key_from_url(url)
    local = os.path.join(cache_dir, f"{key}{suffix}")
    if os.path.exists(local) and os.path.getsize(local) > 0:
        # Touch so prune_cache doesn't remove a still-in-use asset.
        try:
            os.utime(local, None)
        except OSError:
            pass
        return local

    log.info("downloading %s -> %s", suffix, local)
    tmp = local + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    os.replace(tmp, local)
    return local


def prune_cache(cache_dir: str, max_age_s: int = CACHE_MAX_AGE_S) -> int:
    """Delete files in `cache_dir` whose mtime is older than `max_age_s`.
    Returns the number of files removed. Safe to call any time — anything
    deleted will simply be re-downloaded on next fetch."""
    if not os.path.isdir(cache_dir):
        return 0
    cutoff = time.time() - max_age_s
    removed = 0
    for name in os.listdir(cache_dir):
        path = os.path.join(cache_dir, name)
        try:
            if not os.path.isfile(path):
                continue
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
                log.info("pruned stale cache file: %s", name)
        except OSError as e:
            log.warning("prune skip %s: %s", name, e)
    return removed
