"""Config + logging bootstrap for the Mac agent."""
from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from agent_paths import AGENT_ROOT

CONFIG_PATH = AGENT_ROOT / "config.json"

# Fields that must be a positive, non-boolean number of seconds. Present as a
# module-level tuple (not inline in _validate_config) so it's one place to
# extend if a new interval-style field is added later.
_POSITIVE_INT_FIELDS = ("heartbeat_interval_s", "poll_interval_s", "update_watchdog_s")

# Fields that must be non-empty strings.
_NON_EMPTY_STRING_FIELDS = (
    "server_url", "secret_token", "branch_id", "mac_serial", "agent_version",
    "log_dir", "models_cache_dir", "scripts_cache_dir",
)


def _validate_config(cfg: dict) -> list[str]:
    """Catch bad field types/formats at startup instead of letting a
    background thread (heartbeat/poller/watchdog) crash silently, hours
    later, the first time it does int(cfg["some_field"]) on a bad value.
    Returns a list of human-readable error strings — empty means valid.
    Collects every error found rather than stopping at the first one, so a
    single fix-and-rerun catches everything instead of playing whack-a-mole.
    """
    errors: list[str] = []

    for key in _NON_EMPTY_STRING_FIELDS:
        # Fields not yet defaulted (agent_version/log_dir/etc. before
        # setdefault runs) are allowed to be absent here — load_config()
        # applies defaults before calling this for the optional ones. Only
        # flag a field that's present but the wrong type or empty.
        if key not in cfg:
            continue
        value = cfg[key]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{key} must be a non-empty string, got {value!r}")

    server_url = cfg.get("server_url")
    if isinstance(server_url, str) and not server_url.startswith(("http://", "https://")):
        errors.append(f"server_url must start with http:// or https://, got {server_url!r}")

    for key in _POSITIVE_INT_FIELDS:
        if key not in cfg:
            continue
        value = cfg[key]
        # bool is a subclass of int in Python — exclude it explicitly so
        # `"heartbeat_interval_s": true` doesn't silently pass as 1.
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"{key} must be a positive number of seconds, got {value!r}")

    return errors


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        sys.stderr.write(f"config.json not found at {CONFIG_PATH}\n")
        sys.exit(1)
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = ("server_url", "secret_token", "branch_id", "mac_serial")
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        sys.stderr.write(f"config.json missing fields: {missing}\n")
        sys.exit(1)
    cfg.setdefault("agent_version", "1.0.0")
    cfg.setdefault("heartbeat_interval_s", 30)
    cfg.setdefault("poll_interval_s", 10)

    errors = _validate_config(cfg)
    if errors:
        sys.stderr.write("config.json has invalid field(s):\n")
        for e in errors:
            sys.stderr.write(f"  - {e}\n")
        sys.exit(1)

    base = AGENT_ROOT
    cfg["log_dir"] = str((base / cfg.get("log_dir", "./logs")).resolve())
    cfg["models_cache_dir"] = str((base / cfg.get("models_cache_dir", "./models_cache")).resolve())
    cfg["scripts_cache_dir"] = str((base / cfg.get("scripts_cache_dir", "./scripts_cache")).resolve())
    for k in ("log_dir", "models_cache_dir", "scripts_cache_dir"):
        os.makedirs(cfg[k], exist_ok=True)
    return cfg


class _DepFilter(logging.Filter):
    """Ensure every record has a `dep` attribute so the formatter never KeyErrors."""
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "dep") or not getattr(record, "dep", ""):
            record.dep = "-"
        return True


def setup_logging(log_dir: str, name: str = "agent") -> logging.Logger:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s [dep=%(dep)s]: %(message)s")
    dep_filter = _DepFilter()

    # Configure the ROOT logger once with stdout + rotating file handlers.
    # All named loggers (agent, agent.worker, inference.person_count, ...)
    # propagate to root, so adding handlers ONLY to root gives every logger
    # output without duplicating lines.
    root = logging.getLogger()
    if not getattr(root, "_vision_configured", False):
        root.setLevel(logging.INFO)

        # backupCount=0 — never auto-delete a rolled log file here. Deletion is
        # log_shipper's job, and only AFTER a file is confirmed uploaded to the
        # cloud (see log_shipper.py). If we let the handler purge on a fixed
        # day count, a rolled-but-not-yet-shipped file (e.g. cloud unreachable
        # for a while) would be silently destroyed before it ever shipped.
        fh = TimedRotatingFileHandler(
            os.path.join(log_dir, f"{name}.log"),
            when="midnight", backupCount=0, encoding="utf-8",
        )
        fh.setFormatter(fmt)
        fh.addFilter(dep_filter)
        root.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.addFilter(dep_filter)
        root.addHandler(sh)

        root._vision_configured = True  # type: ignore[attr-defined]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Important: do not add handlers to the named logger — let propagation
    # to root carry the records, otherwise every line prints twice.
    return logger
