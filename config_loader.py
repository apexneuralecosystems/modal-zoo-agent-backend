"""Config + logging bootstrap for the Mac agent."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date
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


def setup_logging(log_dir: str, name: str = "agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fh = TimedRotatingFileHandler(
        os.path.join(log_dir, f"{name}.log"),
        when="midnight", backupCount=14, encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger
