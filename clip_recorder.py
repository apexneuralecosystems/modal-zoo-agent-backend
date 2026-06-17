"""Fetch an exact recorded clip from a Hikvision NVR via ISAPI download (by
time range — NVR playback, not live recording), then upload it to S3 via a
presigned URL. Mechanism proven in nvr-test/07_download_clip.py.
"""
from __future__ import annotations

import datetime
import logging
import os

import requests
from requests.auth import HTTPDigestAuth

log = logging.getLogger("agent.clip")

_HIK_NS = "http://www.hikvision.com/ver20/XMLSchema"


def hik_ts(wallclock_iso: str) -> str:
    """Branch/NVR-local wall-clock ISO string -> Hikvision '%Y%m%dT%H%M%SZ'.
    Accepts '2026-06-16T10:00:00' or '...Z'; the trailing Z is the Hikvision
    local-wall-clock convention (NOT a UTC marker)."""
    dt = datetime.datetime.strptime(wallclock_iso[:19], "%Y-%m-%dT%H:%M:%S")
    return dt.strftime("%Y%m%dT%H%M%SZ")


def track_id(channel: int) -> str:
    """Hikvision playback track id for a channel: channel 2 -> '201'."""
    return f"{channel}01"


def build_playback_uri(host: str, channel: int, start_iso: str, end_iso: str) -> str:
    return (
        f"rtsp://{host}/Streaming/tracks/{track_id(channel)}"
        f"?starttime={hik_ts(start_iso)}&endtime={hik_ts(end_iso)}"
    )


def build_download_body(playback_uri: str) -> str:
    """ISAPI /ContentMgmt/download request XML. The inner playbackURI's query
    separators must be XML-escaped (&amp;)."""
    escaped = playback_uri.replace("&", "&amp;")
    return (
        f'<downloadRequest version="1.0" xmlns="{_HIK_NS}">'
        f"<playbackURI>{escaped}&amp;name=clip&amp;size=0</playbackURI>"
        f"</downloadRequest>"
    )


def download_clip(host: str, port: int, username: str | None, password: str | None,
                  channel: int, start_iso: str, end_iso: str, out_path: str,
                  timeout: int = 180) -> int:
    """Export the exact recorded range to out_path (MP4) via ISAPI download.
    Returns bytes written. Raises on HTTP/transport failure."""
    auth = HTTPDigestAuth(username or "", password or "")
    playback_uri = build_playback_uri(host, channel, start_iso, end_iso)
    body = build_download_body(playback_uri)
    # ISAPI lives on the HTTP port (usually 80), NOT the RTSP port (554). The
    # device's HTTP port is normally 80; if the NVR uses a custom HTTP port the
    # cloud should send it. We default to 80 when port is the RTSP default.
    http_port = port if port and port != 554 else 80
    url = f"http://{host}:{http_port}/ISAPI/ContentMgmt/download"
    log.info("clip download: %s", playback_uri)
    r = requests.post(url, data=body, auth=auth,
                      headers={"Content-Type": "application/xml"},
                      stream=True, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"ISAPI download HTTP {r.status_code}: {r.text[:200]}")
    total = 0
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    if total == 0:
        raise RuntimeError("ISAPI download returned 0 bytes (no footage for range?)")
    return total


def upload_to_s3(presigned_put_url: str, file_path: str, timeout: int = 300) -> None:
    """PUT the file to a presigned S3 URL. No auth header — the signature is in
    the URL; sending the agent's Bearer token would break the S3 signature."""
    with open(file_path, "rb") as f:
        r = requests.put(presigned_put_url, data=f,
                         headers={"Content-Type": "video/mp4"}, timeout=timeout)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"S3 upload HTTP {r.status_code}: {r.text[:200]}")


def handle_fetch_clip(payload: dict, work_dir: str) -> dict:
    """Run the full fetch_clip flow. Returns the result dict for command-result.
    Raises on failure (caller reports error)."""
    os.makedirs(work_dir, exist_ok=True)
    job_id = payload.get("job_id", "clip")
    out_path = os.path.join(work_dir, f"{job_id}.mp4")
    bytes_written = download_clip(
        host=payload["host"], port=payload.get("port", 554),
        username=payload.get("username"), password=payload.get("password"),
        channel=int(payload["channel"]),
        start_iso=payload["start"], end_iso=payload["end"],
        out_path=out_path,
    )
    upload_to_s3(payload["presigned_put_url"], out_path)
    try:
        os.remove(out_path)
    except OSError:
        pass
    return {"clip_s3_key": payload.get("clip_s3_key"), "bytes": bytes_written}
