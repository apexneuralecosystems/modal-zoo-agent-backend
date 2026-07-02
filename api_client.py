"""Thin HTTP client around the cloud /agent/* endpoints."""
from __future__ import annotations

import logging
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("agent.api")


class ApiClient:
    def __init__(self, server_url: str, secret_token: str, timeout: int = 10):
        self.base = server_url.rstrip("/")
        self.token = secret_token
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {secret_token}",
            "Content-Type": "application/json",
        })
        # Fix #6: retry transient network/server errors so a 1-2s blip doesn't
        # drop a heartbeat or a poll. 5 tries with exponential backoff
        # (~0s, 1s, 2s, 4s, 8s) on connect errors and 429/5xx responses.
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST", "PATCH", "PUT", "DELETE"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def _request(self, method: str, path: str, **kwargs) -> Any:
        kwargs.setdefault("timeout", self.timeout)
        try:
            r = self.session.request(method, self._url(path), **kwargs)
            r.raise_for_status()
            if r.content:
                ct = r.headers.get("Content-Type", "")
                if "application/json" in ct:
                    body = r.json()
                    if isinstance(body, dict) and "data" in body and "success" in body:
                        return body.get("data")
                    return body
                return r.content
            return None
        except requests.HTTPError as e:
            body = ""
            try:
                body = e.response.text[:300]
            except Exception:
                pass
            log.warning("HTTP %s %s -> %s %s", method, path, e.response.status_code, body)
            raise
        except requests.RequestException as e:
            log.warning("Request error %s %s: %s", method, path, e)
            raise

    # ─── /agent/* endpoints ─────────────────────────────────────────────────
    def register(self, payload: dict) -> dict:
        return self._request("POST", "/agent/register", json=payload)

    def heartbeat(self, payload: dict) -> dict:
        return self._request("POST", "/agent/heartbeat", json=payload)

    def get_jobs(self) -> list[dict]:
        return self._request("GET", "/agent/jobs") or []

    def get_commands(self) -> list[dict]:
        """Fetch queued commands for this agent (the cloud marks them claimed).
        Returns [] when there's nothing queued."""
        return self._request("GET", "/agent/commands") or []

    def post_command_result(self, payload: dict) -> dict:
        """Report a command outcome.
        payload: {"command_id": "...", "ok": bool, "result": {...}?, "error": "..."?}"""
        return self._request("POST", "/agent/command-result", json=payload)

    def get_presigned_url(self, deployment_id: str, filename: str) -> dict:
        return self._request(
            "GET", "/agent/presigned-url",
            params={"deployment_id": deployment_id, "filename": filename},
        )

    def post_event(self, payload: dict) -> dict:
        return self._request("POST", "/agent/event", json=payload)

    def list_devices(self) -> list[dict]:
        return self._request("GET", "/agent/devices") or []

    def post_discover(self, payload: dict) -> dict:
        return self._request("POST", "/agent/discover", json=payload)

    def post_discover_failed(self, payload: dict) -> dict:
        """Tell the cloud that probing this NVR/DVR failed so the user-facing
        Add NVR flow can show a real reason instead of timing out.
        payload: {"device_id": "...", "reason": "auth"|"unreachable"|"timeout"|"unknown", "detail": "..."}"""
        return self._request("POST", "/agent/discover-failed", json=payload)

    def get_log_upload_url(self, filename: str) -> dict:
        return self._request("GET", "/agent/log-upload-url", params={"filename": filename})

    def get_heatmap_upload_url(self, camera_id: str, date: str, ext: str) -> dict:
        return self._request(
            "GET", "/agent/heatmap-upload-url",
            params={"camera_id": camera_id, "date": date, "ext": ext},
        )

    def put_bytes(self, url: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """PUT raw bytes to a presigned S3 URL (bypasses the JSON session headers)."""
        r = requests.put(url, data=data, headers={"Content-Type": content_type}, timeout=30)
        r.raise_for_status()

    def post_telemetry(self, payload: dict) -> dict:
        return self._request("POST", "/agent/telemetry", json=payload)

    def post_alert(self, payload: dict) -> dict:
        return self._request("POST", "/agent/alert", json=payload)

    def post_device_status(self, payload: dict) -> dict:
        return self._request("POST", "/agent/device-status", json=payload)

    def post_camera_status(self, payload: dict) -> dict:
        """Report per-camera live status (batched) so the UI can show
        online/offline/degraded per camera instead of a black tile.
        payload: {"cameras": [{"camera_id": "...", "status": "online"|"offline"|"degraded"}, ...]}"""
        return self._request("POST", "/agent/camera-status", json=payload)

    def mark_camera_offline(self, deployment_id: str) -> dict:
        """Tell the backend that RTSP has been unreachable for 12h on this
        deployment. Sets status=camera_offline; the deployment is removed from
        /agent/jobs until a user clicks Restart in the portal."""
        return self._request("POST", f"/agent/deployments/{deployment_id}/camera-offline")

    def mark_deployment_failed(self, deployment_id: str, reason: str) -> dict:
        """Tell the backend the poller gave up retrying this deployment's
        asset download (see MAX_DOWNLOAD_FAILURES in poller.py). Sets
        status=error so it shows up in the dashboard instead of silently
        never running."""
        return self._request(
            "POST", f"/agent/deployments/{deployment_id}/mark-failed", json={"reason": reason},
        )
