"""Fetch an exact recorded clip from an NVR/DVR by time range (recorder
playback / file download, NOT live recording), then upload it to S3 via a
presigned URL.

Vendor-aware — picks the fastest method the device actually supports:

* Dahua / CP-Plus DVR  -> HTTP file download via /cgi-bin/loadfile.cgi
  (disk-speed, faster than realtime). This is the proven mechanism from the
  vision-ds reference (apps/api/src/streams/streams.service.ts).

* Hikvision NVR        -> FAST ISAPI download (disk-speed, ~15x realtime),
  the 2-step flow from FAST_CLIP_DOWNLOAD.md: (1) POST /ISAPI/ContentMgmt/search
  to get the recording segment(s) + their real boundaries, (2) POST
  /ISAPI/ContentMgmt/download with the exact playbackURI to pull the whole
  segment file fast. The download returns the *entire* segment (endtime is not
  honored), so we trim it locally to the exact requested window using the
  segment's real start time (from a wide search). Falls back to RTSP stream-copy
  (nvr-test/06_save_clip.py; ~realtime) if search/download/trim fails.

The vendor is detected by a cheap HTTP probe; unknown vendors fall back to RTSP.
"""
from __future__ import annotations

import datetime
import logging
import os
import re
import time
from urllib.parse import quote

import requests
from requests.auth import HTTPDigestAuth

import av
import av.logging

log = logging.getLogger("agent.clip")

try:
    av.logging.set_level(av.logging.ERROR)
except Exception:
    pass

# Proven RTSP open options (nvr-test/06_save_clip.py).
_OPEN_OPTS = {
    "rtsp_transport": "tcp",
    "stimeout": "10000000",
    "rw_timeout": "10000000",
    "analyzeduration": "5000000",
    "probesize": "5000000",
}


def hik_ts(wallclock_iso: str) -> str:
    """Branch/NVR-local wall-clock ISO -> Hikvision '%Y%m%dT%H%M%SZ'."""
    dt = datetime.datetime.strptime(wallclock_iso[:19], "%Y-%m-%dT%H:%M:%S")
    return dt.strftime("%Y%m%dT%H%M%SZ")


def track_id(channel: int) -> str:
    """Fallback Hikvision track id from a channel number: channel 2 -> '201'.
    Only used when the camera's saved rtsp_url has no parseable track."""
    return f"{channel}01"


def hik_track_from_path(rtsp_path: str | None, channel: int) -> str:
    """Derive the REAL Hikvision track from the camera's saved rtsp_url — the
    same stream the live view uses (source of truth). e.g. '/Streaming/Channels/201'
    or '/Streaming/tracks/201' -> '201'. Falls back to {channel}01 only if the
    url has no track. This avoids guessing a non-existent track from the channel
    number (e.g. a camera labelled CH4 whose real recording track is 201)."""
    if rtsp_path:
        m = re.search(r"/Streaming/(?:Channels|tracks)/(\d+)", rtsp_path)
        if m:
            return m.group(1)
    return track_id(channel)


def dahua_channel_from_path(rtsp_path: str | None, channel: int) -> int:
    """Derive the Dahua channel number from the camera's saved rtsp_url
    (e.g. '/cam/realmonitor?channel=4&subtype=0' -> 4). Falls back to channel."""
    if rtsp_path:
        m = re.search(r"channel=(\d+)", rtsp_path)
        if m:
            return int(m.group(1))
    return channel


def build_rtsp_playback_url(host: str, port: int, username: str | None, password: str | None,
                            track: str, start_iso: str, end_iso: str) -> str:
    """RTSP playback URL for a recorded time range (Hikvision). `track` is the
    real track id (e.g. '201') taken from the camera's saved url. Creds inline."""
    auth = ""
    if username:
        auth = f"{quote(username, safe='')}:{quote(password or '', safe='')}@"
    return (
        f"rtsp://{auth}{host}:{port or 554}/Streaming/tracks/{track}"
        f"?starttime={hik_ts(start_iso)}&endtime={hik_ts(end_iso)}"
    )


def build_dahua_loadfile_url(host: str, http_port: int, channel: int,
                             start_iso: str, end_iso: str) -> str:
    """Dahua/CP-Plus HTTP file-download URL (disk-speed). Matches the reference:
    space is encoded as %20, colons kept literal, channel is the 1-based cam #."""
    def fmt(iso: str) -> str:
        dt = datetime.datetime.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S")
        return dt.strftime("%Y-%m-%d") + "%20" + dt.strftime("%H:%M:%S")
    return (
        f"http://{host}:{http_port or 80}/cgi-bin/loadfile.cgi?action=startLoad"
        f"&channel={channel}&startTime={fmt(start_iso)}&endTime={fmt(end_iso)}&type=dav"
    )


def detect_vendor(host: str, username: str | None, password: str | None,
                  http_port: int = 80, timeout: int = 4) -> str | None:
    """Cheap HTTP probe -> 'dahua' | 'hikvision' | None. Best-effort, never raises."""
    auth = HTTPDigestAuth(username or "", password or "")
    # Dahua / CP-Plus: magicBox returns "type=..." lines.
    try:
        r = requests.get(f"http://{host}:{http_port}/cgi-bin/magicBox.cgi?action=getDeviceType",
                         auth=auth, timeout=timeout)
        body = (r.text or "").lower()
        if r.status_code == 200 and ("type=" in body or "dahua" in body):
            return "dahua"
    except Exception:
        pass
    # Hikvision: ISAPI deviceInfo exists (200 or 401-with-digest).
    try:
        r = requests.get(f"http://{host}:{http_port}/ISAPI/System/deviceInfo",
                         auth=auth, timeout=timeout)
        body = (r.text or "").lower()
        wwwauth = (r.headers.get("WWW-Authenticate") or "").lower()
        if r.status_code in (200, 401) and (
            "hikvision" in body or "<devicetype" in body or "<devicename" in body
            or ("digest" in wwwauth and "realm=" in wwwauth)
        ):
            return "hikvision"
    except Exception:
        pass
    return None


def download_dav_dahua(host: str, http_port: int, username: str | None, password: str | None,
                       channel: int, start_iso: str, end_iso: str, out_path: str,
                       timeout: int = 180) -> int:
    """Disk-speed HTTP file download from a Dahua/CP-Plus DVR. Returns bytes."""
    url = build_dahua_loadfile_url(host, http_port, channel, start_iso, end_iso)
    log.info("clip dahua loadfile: channel=%s %s..%s", channel, start_iso[:19], end_iso[:19])
    r = requests.get(url, auth=HTTPDigestAuth(username or "", password or ""),
                     stream=True, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Dahua loadfile HTTP {r.status_code}: {r.text[:200]}")
    total = 0
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    if total == 0:
        raise RuntimeError("Dahua loadfile returned 0 bytes (no footage for the range?)")
    return total


def download_clip_rtsp(host: str, port: int, username: str | None, password: str | None,
                       track: str, start_iso: str, end_iso: str, out_path: str,
                       timeout: int = 15) -> int:
    """Stream-copy the recorded range to out_path (MP4) via RTSP playback
    (Hikvision). `track` is the real track id. Returns bytes. Streams ~realtime."""
    url = build_rtsp_playback_url(host, port, username, password, track, start_iso, end_iso)
    log.info("clip rtsp stream-copy: tracks/%s %s..%s", track,
             hik_ts(start_iso), hik_ts(end_iso))
    try:
        inp = av.open(url, options=_OPEN_OPTS, timeout=timeout)
    except Exception as e:
        raise RuntimeError(
            f"NVR playback unavailable for track {track} / this time range "
            f"(no recording for the range): {e}"
        )
    packets = 0
    try:
        ivs = inp.streams.video[0]
        out = av.open(out_path, "w")
        try:
            ovs = out.add_stream_from_template(ivs)
            try:
                for packet in inp.demux(ivs):
                    if packet.dts is None:
                        continue
                    packet.stream = ovs
                    out.mux(packet)
                    packets += 1
            except av.error.ExitError:
                pass  # reached endtime — normal end of a bounded playback clip
        finally:
            out.close()
    finally:
        inp.close()
    size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    if packets == 0 or size == 0:
        raise RuntimeError("playback produced no data (no footage for the selected range?)")
    return size


# ── Hikvision fast ISAPI download (search -> download -> local trim) ────────
_HIK_NS = "http://www.hikvision.com/ver20/XMLSchema"


def _hik_strip(tag: str) -> str:
    return tag.split("}")[-1]


def _hik_time_offset(host: str, http_port: int, auth) -> str:
    """Read the NVR's local clock offset suffix (e.g. '+05:30') from
    /ISAPI/System/time. Hikvision search needs the request times in the NVR's
    own offset (using 'Z' finds 0 segments)."""
    import xml.etree.ElementTree as ET
    r = requests.get(f"http://{host}:{http_port}/ISAPI/System/time", auth=auth, timeout=8)
    flat = {_hik_strip(e.tag): e.text for e in ET.fromstring(r.text).iter()}
    lt = flat.get("localTime") or ""
    return lt[19:] if len(lt) > 19 else "Z"


def _hik_search_segments(host: str, http_port: int, auth, track: str,
                         start_dt: "datetime.datetime", end_dt: "datetime.datetime",
                         offset: str) -> list[tuple]:
    """POST /ISAPI/ContentMgmt/search. Returns [(segStart_dt, segEnd_dt, playbackURI)].
    Hikvision result times use the local-wall-clock-with-Z convention, so we
    parse them as naive local datetimes (matching the requested wall-clock)."""
    import uuid
    import xml.etree.ElementTree as ET
    iso = lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%S") + offset
    body = (
        f'<?xml version="1.0" encoding="utf-8"?>'
        f'<CMSearchDescription xmlns="{_HIK_NS}">'
        f"<searchID>{{{uuid.UUID(int=0x5644)}}}</searchID>"
        f"<trackList><trackID>{track}</trackID></trackList>"
        f"<timeSpanList><timeSpan><startTime>{iso(start_dt)}</startTime>"
        f"<endTime>{iso(end_dt)}</endTime></timeSpan></timeSpanList>"
        f"<maxResults>100</maxResults><searchResultPosition>0</searchResultPosition>"
        f"<metadataList><metadataDescriptor>//recordType.meta.std-cgi.com</metadataDescriptor></metadataList>"
        f"</CMSearchDescription>"
    )
    r = requests.post(f"http://{host}:{http_port}/ISAPI/ContentMgmt/search",
                      data=body, auth=auth, headers={"Content-Type": "application/xml"}, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"ISAPI search HTTP {r.status_code}: {r.text[:200]}")
    segs = []
    for m in ET.fromstring(r.text).iter():
        if _hik_strip(m.tag) in ("searchMatchItem", "matchItem"):
            k = {_hik_strip(e.tag): e.text for e in m.iter()}
            try:
                ss = datetime.datetime.strptime(k["startTime"][:19], "%Y-%m-%dT%H:%M:%S")
                se = datetime.datetime.strptime(k["endTime"][:19], "%Y-%m-%dT%H:%M:%S")
            except Exception:
                continue
            uri = k.get("playbackURI")
            if uri:
                msz = re.search(r"size=(\d+)", uri)
                size = int(msz.group(1)) if msz else 0
                segs.append((ss, se, uri, size))
    return segs


# Estimated ISAPI disk-download rate on these NVRs (measured ~4 MB/s) and the
# fixed per-stream overhead of an RTSP playback open.
_ISAPI_RATE_BPS = 4 * 1024 * 1024
_RTSP_OVERHEAD_S = 25


def choose_fetch_method(total_segment_bytes: int, requested_seconds: float) -> str:
    """Pick 'isapi' (fast whole-segment download + trim) vs 'rtsp' (realtime
    stream of just the window). ISAPI downloads the ENTIRE segment regardless of
    the window, so it only wins when the window is a large fraction of the
    segment. Returns 'isapi' or 'rtsp'."""
    est_isapi = (total_segment_bytes / _ISAPI_RATE_BPS) if total_segment_bytes else 1e9
    est_rtsp = requested_seconds + _RTSP_OVERHEAD_S
    return "isapi" if est_isapi < est_rtsp else "rtsp"


def _isapi_download(host: str, http_port: int, auth, playback_uri: str, out_path: str,
                    timeout: int = 300, attempts: int = 3) -> int:
    """POST /ISAPI/ContentMgmt/download with the exact playbackURI from search.
    Streams the whole segment file at disk speed. Returns bytes written.

    Retries on connection resets: Hikvision NVRs drop connections under
    rapid-fire requests, and large segment files can be interrupted mid-stream."""
    body = (f'<?xml version="1.0" encoding="utf-8"?>'
            f'<downloadRequest xmlns="{_HIK_NS}"><playbackURI>{playback_uri}</playbackURI></downloadRequest>')
    last_err = None
    for attempt in range(1, attempts + 1):
        total = 0
        try:
            r = requests.post(f"http://{host}:{http_port}/ISAPI/ContentMgmt/download",
                              data=body, auth=auth, headers={"Content-Type": "application/xml"},
                              stream=True, timeout=(15, timeout))
            if r.status_code != 200:
                raise RuntimeError(f"ISAPI download HTTP {r.status_code}: {r.text[:200]}")
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
            if total == 0:
                raise RuntimeError("ISAPI download returned 0 bytes")
            return total
        except Exception as e:
            last_err = e
            log.warning("ISAPI download attempt %d/%d failed: %s", attempt, attempts, e)
            time.sleep(2 * attempt)  # back off; lets the NVR clear its connection lockout
    raise RuntimeError(f"ISAPI download failed after {attempts} attempts: {last_err}")


def _trim_concat(parts: list[tuple], out_path: str) -> None:
    """parts = [(segment_file, overlap_start_off_s, overlap_end_off_s)] in order.
    Stream-copy the requested window out of each segment file and concat into one
    MP4, re-basing timestamps so playback/extraction is continuous."""
    out = None
    ovs = None
    base_out = 0  # running output pts (in out stream time_base units)
    for seg_file, oss, oee in parts:
        inp = av.open(seg_file)
        ivs = inp.streams.video[0]
        tb = ivs.time_base
        base = ivs.start_time or 0
        if out is None:
            out = av.open(out_path, "w")
            ovs = out.add_stream_from_template(ivs)
        try:
            inp.seek(int(oss / tb) + base, stream=ivs, backward=True, any_frame=False)
        except Exception:
            pass
        seg_first = None
        for p in inp.demux(ivs):
            if p.dts is None:
                continue
            ft = float((p.pts - base) * tb)
            if ft < oss:
                continue
            if ft > oee:
                break
            if seg_first is None:
                seg_first = p.pts
            newpts = base_out + (p.pts - seg_first)
            p.pts = newpts
            p.dts = newpts
            p.stream = ovs
            try:
                out.mux(p)
            except Exception:
                pass
        inp.close()
        base_out += int((oee - oss) / tb)
    if out is not None:
        out.close()


def _hik_overlapping_segments(host, http_port, auth, track, start_dt, end_dt):
    """Wide-search and return the segments overlapping [start_dt, end_dt] as
    (seg_start, seg_end, uri, size_bytes), sorted by start. Searches from well
    before the window so the containing segment reports its REAL start."""
    offset = _hik_time_offset(host, http_port, auth)
    ws = start_dt - datetime.timedelta(hours=2)
    segs = _hik_search_segments(host, http_port, auth, track, ws, end_dt, offset)
    return sorted(
        [(s, e, uri, sz) for (s, e, uri, sz) in segs if s < end_dt and e > start_dt],
        key=lambda x: x[0],
    )


def fetch_hikvision(host: str, http_port: int, rtsp_port: int,
                    username: str | None, password: str | None,
                    channel: int, rtsp_path: str | None,
                    start_iso: str, end_iso: str, out_path: str,
                    work_dir: str) -> tuple[int, str]:
    """Fetch the exact window from a Hikvision NVR. Auto-picks the faster of:
      - ISAPI: download the whole segment(s) at disk speed, trim to the window.
      - RTSP : stream just the window at realtime.
    The track is taken from the camera's saved url (rtsp_path), NOT guessed from
    the channel number. Returns (bytes, method). Falls back to RTSP if ISAPI fails."""
    auth = HTTPDigestAuth(username or "", password or "")
    track = hik_track_from_path(rtsp_path, channel)
    start_dt = datetime.datetime.strptime(start_iso[:19], "%Y-%m-%dT%H:%M:%S")
    end_dt = datetime.datetime.strptime(end_iso[:19], "%Y-%m-%dT%H:%M:%S")
    requested_s = (end_dt - start_dt).total_seconds()

    overlapping = []
    try:
        overlapping = _hik_overlapping_segments(host, http_port, auth, track, start_dt, end_dt)
    except Exception as e:
        log.warning("ISAPI search failed (%s) — using RTSP", e)

    if not overlapping:
        # No search support / no segments found — let RTSP try (it will error if
        # there is genuinely no footage).
        return download_clip_rtsp(host, rtsp_port, username, password, track, start_iso, end_iso, out_path), "hikvision-rtsp"

    total_bytes = sum(sz for (_, _, _, sz) in overlapping)
    method = choose_fetch_method(total_bytes, requested_s)
    log.info("hikvision: track=%s %d segment(s), %.0f MB total, window %.0fs -> method=%s",
             track, len(overlapping), total_bytes / 1024 / 1024, requested_s, method)

    if method == "rtsp":
        return download_clip_rtsp(host, rtsp_port, username, password, track, start_iso, end_iso, out_path), "hikvision-rtsp"

    # ISAPI: download each overlapping segment, trim to its overlap, concat.
    try:
        parts = []
        for i, (seg_start, seg_end, uri, _sz) in enumerate(overlapping):
            seg_file = os.path.join(work_dir, f"seg_{i}.mp4")
            _isapi_download(host, http_port, auth, uri, seg_file)
            oss = (max(start_dt, seg_start) - seg_start).total_seconds()
            oee = (min(end_dt, seg_end) - seg_start).total_seconds()
            parts.append((seg_file, oss, oee))
        _trim_concat(parts, out_path)
        for seg_file, _, _ in parts:
            try:
                os.remove(seg_file)
            except OSError:
                pass
        size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        if size == 0:
            raise RuntimeError("trim produced no data")
        return size, "hikvision-isapi-fast"
    except Exception as e:
        log.warning("ISAPI download/trim failed (%s) — falling back to RTSP", e)
        return download_clip_rtsp(host, rtsp_port, username, password, track, start_iso, end_iso, out_path), "hikvision-rtsp-fallback"


def upload_to_s3(presigned_put_url: str, file_path: str, timeout: int = 300,
                  attempts: int = 3) -> None:
    """PUT the file to a presigned S3 URL. No auth header — the signature is in
    the URL; sending the agent's Bearer token would break the S3 signature.
    Retries on transient failures (flaky wifi, brief network drop) since most
    upload failures are temporary."""
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with open(file_path, "rb") as f:
                r = requests.put(presigned_put_url, data=f,
                                 headers={"Content-Type": "video/mp4"}, timeout=timeout)
            if r.status_code in (200, 201):
                return
            raise RuntimeError(f"S3 upload HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e
            if attempt < attempts:
                wait = 5 * attempt
                log.warning("clip upload attempt %s/%s failed (%s) — retrying in %ss",
                            attempt, attempts, e, wait)
                time.sleep(wait)
    raise last_err


def handle_fetch_clip(payload: dict, work_dir: str) -> dict:
    """Vendor-aware clip fetch + S3 upload. Returns the command-result dict.
    Raises on failure (caller reports the error)."""
    os.makedirs(work_dir, exist_ok=True)
    job_id = payload.get("job_id", "clip")
    host = payload["host"]
    rtsp_port = payload.get("port", 554)
    http_port = payload.get("http_port", 80)
    user = payload.get("username")
    pw = payload.get("password")
    channel = int(payload["channel"])
    # The camera's saved stream url (the one live view uses) is the source of
    # truth for the track/channel — NOT the channel number. Backend sends it.
    rtsp_path = payload.get("rtsp_path") or payload.get("rtsp_url")
    start_iso = payload["start"]
    end_iso = payload["end"]
    out_path = os.path.join(work_dir, f"{job_id}.mp4")

    vendor = detect_vendor(host, user, pw, http_port)
    log.info("fetch_clip %s vendor=%s channel=%s", job_id, vendor, channel)

    t0 = time.time()
    try:
        if vendor == "dahua":
            dahua_ch = dahua_channel_from_path(rtsp_path, channel)
            bytes_written = download_dav_dahua(host, http_port, user, pw, dahua_ch, start_iso, end_iso, out_path)
            method = "dahua-loadfile"
        elif vendor == "hikvision":
            # Auto-picks ISAPI fast-download+trim vs realtime RTSP, whichever is
            # faster for this window/segment, with RTSP fallback on error.
            bytes_written, method = fetch_hikvision(host, http_port, rtsp_port, user, pw,
                                                    channel, rtsp_path, start_iso, end_iso, out_path, work_dir)
        else:
            track = hik_track_from_path(rtsp_path, channel)
            bytes_written = download_clip_rtsp(host, rtsp_port, user, pw, track, start_iso, end_iso, out_path)
            method = "rtsp"

        log.info("clip %s via %s: %.0f KB in %.0fs", job_id, method, bytes_written / 1024, time.time() - t0)
        upload_to_s3(payload["presigned_put_url"], out_path)
    finally:
        # Delete the local copy no matter where things failed — a bad/partial
        # download (e.g. a muxer error mid-write) or a failed upload should
        # never leave a file sitting on disk; the whole job just gets
        # retried fresh from scratch next time.
        try:
            os.remove(out_path)
        except OSError:
            pass
    return {"clip_s3_key": payload.get("clip_s3_key"), "bytes": bytes_written,
            "vendor": vendor, "method": method}
