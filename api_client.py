"""Thin HTTP client around the cloud /agent/* endpoints."""
from __future__ import annotations

import logging
from typing import Any

import requests

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

    def get_log_upload_url(self, filename: str) -> dict:
        return self._request("GET", "/agent/log-upload-url", params={"filename": filename})

    def post_telemetry(self, payload: dict) -> dict:
        return self._request("POST", "/agent/telemetry", json=payload)

    def post_alert(self, payload: dict) -> dict:
        return self._request("POST", "/agent/alert", json=payload)

    def post_device_status(self, payload: dict) -> dict:
        return self._request("POST", "/agent/device-status", json=payload)
