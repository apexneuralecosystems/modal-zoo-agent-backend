"""Config + logging bootstrap for the Mac agent."""
from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


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

    base = Path(__file__).resolve().parent
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

        fh = TimedRotatingFileHandler(
            os.path.join(log_dir, f"{name}.log"),
            when="midnight", backupCount=14, encoding="utf-8",
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
