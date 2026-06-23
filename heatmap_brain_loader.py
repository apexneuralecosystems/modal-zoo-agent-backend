"""Load the heatmap 'brain' module.

Returns the cloud-published brain when the job carries a `brain_presigned_url`
AND it downloads + imports + exposes the right interface; otherwise returns the
baked-in default. A broken cloud brain must NEVER kill the heatmap — every
failure path falls back to the default. Mirrors worker.py's inference-module
loader (importlib exec of a downloaded .py).
"""
import importlib.util
import logging

import heatmap_brain_default as default

log = logging.getLogger("heatmap_worker")


def _load_module_from_path(path: str):
    spec = importlib.util.spec_from_file_location("heatmap_brain_cloud", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not (hasattr(mod, "setup") and hasattr(mod, "heat_mask")):
        raise AttributeError("brain missing setup/heat_mask")
    return mod


def load_brain(job: dict, *, download_brain):
    """download_brain(url) -> local .py path. Falls back to the default brain on
    no URL or any download/import error."""
    url = job.get("brain_presigned_url")
    if not url:
        return default
    try:
        path = download_brain(url)
        mod = _load_module_from_path(path)
        log.info("heatmap[%s]: loaded cloud brain v%s",
                 job.get("camera_id"), job.get("brain_version"))
        return mod
    except Exception as e:
        log.warning("heatmap[%s]: cloud brain failed (%s) — using default brain",
                    job.get("camera_id"), e)
        return default
