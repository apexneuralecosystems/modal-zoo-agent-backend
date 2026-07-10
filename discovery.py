"""NVR/DVR channel discovery.

For each device the cloud reports, probe RTSP channels 1..MAX. For every
channel that returns at least one frame within a timeout, report it back to
the cloud which will auto-create a camera row.
"""
from __future__ import annotations

import logging
import os
import socket
import threading
import time
from urllib.parse import quote

import base64
import io

import av  # PyAV — decodes Hikvision H.264 that OpenCV's bundled FFmpeg can't

# Silence PyAV/FFmpeg logging. Two layers:
#  1. av.logging.set_level mutes libav's own stderr writer.
#  2. PyAV ALSO forwards FFmpeg messages into Python's logging under the
#     `libav.*` loggers (that's where the "libav.rtsp: method DESCRIBE failed:
#     404" ERROR lines come from). A 404 on an empty channel is expected during
#     discovery, so we raise the `libav` logger to CRITICAL to keep it quiet.
try:
    av.logging.set_level(av.logging.CRITICAL)
except Exception:
    pass
logging.getLogger("libav").setLevel(logging.CRITICAL)

# RTSP open options for the discovery probe.
#
# CRITICAL: VLC plays this camera fine but PyAV with minimal options throws
# InvalidDataError at av.open() — because PyAV's stream probe reads the first
# packets and chokes on the corrupt leading bytes a Hikvision stream emits
# before its first keyframe. VLC tolerates that; we must tell FFmpeg to as well:
#   - err_detect=ignore_err : don't abort on the initial corrupt packets.
#   - fflags=discardcorrupt : drop corrupt packets instead of erroring.
#   - analyzeduration/probesize : give FFmpeg time+bytes to find a clean
#     keyframe during open instead of giving up on the first garbage it sees.
#   - rtsp_flags=prefer_tcp : negotiate TCP but tolerate the server's quirks.
#   NOTE: do NOT add fflags=discardcorrupt / err_detect=ignore_err here — they
#   can drop the keyframe and were observed to make decoding WORSE. The proven
#   working recipe is minimal options + a tolerant demux loop that simply skips
#   the corrupt pre-keyframe packets and waits for the first IDR (which decodes
#   fine — verified: keyframes arrive ~every 50 packets and decode cleanly).
_AV_PROBE_OPTS = {
    "rtsp_transport": "tcp",
    "stimeout": "8000000",
    "rw_timeout": "8000000",
}

from api_client import ApiClient

log = logging.getLogger("agent.discovery")

MAX_CHANNELS_PROBED = 128
# Scanned in batches of CHANNEL_BATCH_SIZE (see discover_device): probe a
# whole batch, and only continue into the next batch if that batch found at
# least one camera. If a batch comes back completely empty, stop there
# instead of continuing to the 128 ceiling. This is a deliberate tradeoff: a
# camera wired at, say, channel 20 on a device with nothing at all in
# channels 1-16 will be missed. Real NVR channel maps are contiguous from
# near channel 1, so this is the right tradeoff for typical hardware —
# scanning all 128 channels on every misconfigured/empty NVR would make
# discovery ~8x slower than today's 16-channel ceiling for no benefit.
CHANNEL_BATCH_SIZE = 16
# Total per-path budget. Raised from 8s to 12s because we now spend the first
# 1.5s draining corrupt mid-GOP frames (cap.grab only) before we start
# measuring; the decoder needs that window to lock onto an I-frame cleanly.
PROBE_TIMEOUT_S = 15
# Drain period at the start of a fresh capture: we read & discard frames
# without decoding them, giving the FFmpeg layer time to find the next
# I-frame. Without this, the first decoded frame is usually a mid-GOP
# P-frame that produces the cabac_init_idc / non-existing PPS errors.
PROBE_SETTLE_S = 1.5
# A real video frame is at least 160 px wide. Below this, the decoder is
# almost certainly emitting parameter-set fragments rather than a decoded
# I/P frame.
MIN_VALID_WIDTH = 160
# Pause between RTSP open attempts. Hikvision NVRs lock out the source IP after
# a burst of rapid connections (illegal-login protection); a small gap keeps
# discovery under that threshold. ~0.4s × ~16 channels ≈ 6s extra, acceptable.
INTER_PROBE_DELAY_S = 0.4
# Thumbnail width (px) for the snapshot we send with each discovered channel so
# the mapping UI can show a preview image instead of just "Channel N".
THUMBNAIL_WIDTH = 320


def _frame_to_thumbnail(frame) -> str | None:
    """PyAV video frame → small JPEG → base64 data URL (or None on failure).
    Lets the cloud/UI show a preview of each channel during camera mapping
    without opening a live stream. Best-effort: returns None if anything fails."""
    try:
        img = frame.to_image()  # PIL Image (PyAV bundles PIL)
        w, h = img.size
        if w > THUMBNAIL_WIDTH:
            img = img.resize((THUMBNAIL_WIDTH, max(1, int(h * THUMBNAIL_WIDTH / w))))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None
COMMON_PATHS = (
    # Path templates are tried in order. {ch} is substituted with channel #.
    # IMPORTANT: every path here is a LIVE feed. /Streaming/tracks/* used to be
    # listed as a last-resort fallback, but on some Hikvision firmware it serves
    # the *archived playback track* — which surfaces in the preview as footage
    # from earlier days. A camera that got auto-saved on a tracks path would
    # then replay old video forever. We removed it entirely: a camera that only
    # answers on tracks would show stale footage anyway, so refusing to discover
    # it (and surfacing an auth/path failure) is the safer, more honest outcome.
    "/Streaming/Channels/{ch}01",                # Hikvision live main stream
    "/Streaming/Channels/{ch}02",                # Hikvision live sub stream
    "/cam/realmonitor?channel={ch}&subtype=0",   # Dahua / CP Plus
    "/h264/ch{ch}/main/av_stream",               # Reolink-style
    "/live/ch{ch}",                              # Generic
)


# Map each RTSP path template to the vendor it belongs to, so once we know the
# NVR brand we can try that brand's path FIRST (fewer RTSP opens = less chance
# of tripping the NVR's connection lockout). Unknown-vendor templates ("generic")
# are always kept as a fallback.
_PATH_VENDOR = {
    "/Streaming/Channels/{ch}01": "hikvision",
    "/Streaming/Channels/{ch}02": "hikvision",
    "/cam/realmonitor?channel={ch}&subtype=0": "dahua",
    "/h264/ch{ch}/main/av_stream": "reolink",
    "/live/ch{ch}": "generic",
}


def _detect_vendor(host: str, port: int, timeout: float = 2.0) -> str | None:
    """Cheap brand detection over HTTP BEFORE we scan RTSP channels.

    Tries the NVR's web port (80) and its configured port, reads the HTTP
    Server header and a couple of vendor-specific endpoints. Returns
    'hikvision' / 'dahua' / 'reolink', or None if unknown. Best-effort and fast
    — never raises. Knowing the brand lets us probe only that vendor's RTSP path
    (≈1 open/channel) instead of all templates (≈5/channel), which is what trips
    Hikvision's illegal-login lockout. Unknown → we fall back to trying all.
    """
    import requests  # already a dependency (used by api_client)
    candidate_ports = [80]
    if port not in candidate_ports:
        candidate_ports.append(port)
    for p in candidate_ports:
        base = f"http://{host}:{p}"
        try:
            r = requests.get(base, timeout=timeout)
            server = (r.headers.get("Server") or "").lower()
            body = (r.text[:500] or "").lower()
            # Hikvision NVRs serve their web UI from "DNVRS-Webs" / "App-webs"
            # (verified against a real DS-7W04NI-Q1). "webs" alone is too broad.
            if any(s in server for s in ("hikvision", "dnvrs-webs", "app-webs", "dvrdvs-webs")):
                return "hikvision"
            if "hikvision" in body:
                return "hikvision"
            if "dahua" in server or "dahua" in body:
                return "dahua"
            if "reolink" in server or "reolink" in body:
                return "reolink"
        except Exception:
            pass
        # Hikvision's ISAPI endpoint is a strong signal: it exists (200/401 with
        # a Digest WWW-Authenticate and a <userCheck>/deviceType XML body) only
        # on Hikvision (and OEM clones using the same firmware).
        try:
            r = requests.get(f"{base}/ISAPI/System/deviceInfo", timeout=timeout)
            wwwauth = (r.headers.get("WWW-Authenticate") or "").lower()
            if r.status_code in (200, 401) and (
                "hikvision" in r.text.lower()
                or "<devicetype" in r.text.lower()
                or "<usercheck" in r.text.lower()
                or ("digest" in wwwauth and "realm=" in wwwauth)
            ):
                return "hikvision"
        except Exception:
            pass
    return None


def _ordered_paths(vendor: str | None) -> tuple[str, ...]:
    """Return COMMON_PATHS reordered so the detected vendor's templates come
    first, with everything else kept as fallback. Unknown vendor → original
    order unchanged."""
    if not vendor:
        return COMMON_PATHS
    preferred = [p for p in COMMON_PATHS if _PATH_VENDOR.get(p) == vendor]
    rest = [p for p in COMMON_PATHS if _PATH_VENDOR.get(p) != vendor]
    return tuple(preferred + rest)


def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    """True if we can open a TCP connection to host:port. Used to distinguish
    'NVR powered off / wrong IP' from 'NVR up but speaks an RTSP path we
    don't know about yet'."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _build_rtsp(host: str, port: int, user: str | None, pwd: str | None, path: str) -> str:
    auth = ""
    if user:
        auth = f"{quote(user, safe='')}:{quote(pwd or '', safe='')}@"
    return f"rtsp://{auth}{host}:{port}{path}"


def _probe_channel(host: str, port: int, user: str | None, pwd: str | None, channel: int,
                   only_template: str | None = None,
                   paths: tuple[str, ...] | None = None) -> tuple[bool, str | None, str | None, str | None]:
    """Try each path template until one returns a real video frame.

    "Real" = at least two consecutive frames whose width meets MIN_VALID_WIDTH.
    Single tiny frames are typically SPS/PPS parameter-set artifacts that the
    decoder emits before the first I-frame arrives — accepting them would
    falsely mark a stream as working when nothing decodable is being delivered.
    Returns (ok, "WxH", working_path_template). The third value is the TEMPLATE
    (e.g. "/Streaming/Channels/{ch}01"), not the formatted path — so the caller
    can reuse it for the remaining channels and avoid trying every vendor path
    on every channel. Probing all 5 templates × 16 channels = up to 80 rapid
    RTSP opens, which trips Hikvision's illegal-login lockout; reusing the known
    template cuts that to ~1 open per channel.
    """
    # If we already know the vendor's template (learned from a prior hit), only
    # try that one. Otherwise try the given ordered `paths` (vendor-detected
    # order if provided), falling back to the full COMMON_PATHS list.
    templates = (only_template,) if only_template else (paths or COMMON_PATHS)
    for tpl in templates:
        path = tpl.format(ch=channel)
        url = _build_rtsp(host, port, user, pwd, path)
        container = None
        try:
            container = av.open(url, options=_AV_PROBE_OPTS, timeout=5)
        except Exception:
            # Open failed (wrong path → 404, bad creds → 401, etc.). Try next.
            if container is not None:
                try: container.close()
                except Exception: pass
            # Small breather so we don't hammer the NVR's RTSP server, which
            # can lock out the source IP after a burst of rapid connections.
            time.sleep(INTER_PROBE_DELAY_S)
            continue
        try:
            vstream = next((s for s in container.streams if s.type == "video"), None)
            if vstream is None:
                continue
            good_frames = 0
            last_dims: tuple[int, int] | None = None  # (w, h)
            thumbnail: str | None = None
            deadline = time.time() + PROBE_TIMEOUT_S
            # Demux packets and decode each one in ITS OWN try, skipping any
            # that error. This is the key: a Hikvision stream sends corrupt
            # mid-GOP packets before its first keyframe, and PyAV raises
            # InvalidDataError on them. VLC just skips those and waits for the
            # keyframe — we must do the same. The OLD code let one bad packet
            # abort the whole channel (so a perfectly good camera looked dead).
            for packet in container.demux(vstream):
                if time.time() > deadline:
                    break
                try:
                    frames = packet.decode()
                except Exception:
                    # Corrupt packet (pre-keyframe garbage). Skip, keep going.
                    continue
                for frame in frames:
                    w, h = frame.width, frame.height
                    if w < MIN_VALID_WIDTH:
                        continue
                    good_frames += 1
                    last_dims = (w, h)
                    # Capture a preview thumbnail from the first good frame so
                    # the mapping UI can show what each channel sees.
                    if thumbnail is None:
                        thumbnail = _frame_to_thumbnail(frame)
                # One cleanly-decoded full frame is proof the channel works.
                # (We used to require 2; the first IDR can take ~100 packets to
                # arrive and a single decoded frame is already unambiguous.)
                if good_frames >= 1:
                    break
            if good_frames >= 1 and last_dims is not None:
                w, h = last_dims
                return True, f"{w}x{h}", tpl, thumbnail
        except Exception:
            # Transport/open-level error on this path — try the next template.
            continue
        finally:
            try:
                container.close()
            except Exception:
                pass
            time.sleep(INTER_PROBE_DELAY_S)
    return False, None, None, None


def _post_failed(api: ApiClient, device_id: str, reason: str, detail: str | None = None) -> None:
    """Report a probe failure to the cloud so the UI can show a real error
    instead of timing out. Failures here are best-effort — if the post itself
    fails we just log and move on; the UI's own 60s timeout still covers us."""
    try:
        api.post_discover_failed({"device_id": device_id, "reason": reason, "detail": detail or ""})
    except Exception as e:
        log.warning("discover-failed post failed for %s: %s", device_id, e)


def discover_device(api: ApiClient, device: dict, max_channels: int = MAX_CHANNELS_PROBED) -> dict | None:
    host, port = device["ip_address"], device["port"]
    log.info("probing device %s @ %s:%s", device["name"], host, port)

    # P2-#1: cheap TCP probe first. Lets us distinguish "device offline"
    # (post offline status) from "device online but no known RTSP path
    # matched" (don't claim it's offline — log so we know to add a path).
    if not _tcp_reachable(host, port):
        log.warning("  TCP %s:%s unreachable — marking device offline", host, port)
        try:
            api.post_device_status({"device_id": device["id"], "status": "offline"})
        except Exception as e:
            log.warning("device-status post failed: %s", e)
        _post_failed(api, device["id"], "unreachable", f"TCP {host}:{port} not reachable from the Mac")
        return None

    # Detect the NVR brand FIRST (cheap HTTP check) so we probe that vendor's
    # RTSP path before the others — fewer connections, less chance of tripping
    # the NVR's lockout. Unknown brand → try all paths in the default order.
    vendor = _detect_vendor(host, port)
    ordered_paths = _ordered_paths(vendor)
    if vendor:
        log.info("  detected vendor=%s — trying its RTSP path first", vendor)

    # Probe EVERY channel 1..max and keep whichever ones return real video.
    #
    # We deliberately do NOT bail when channel 1 fails. Real NVRs commonly have
    # an empty channel 1 (camera plugged into port 2, 3, …) or gaps in the
    # channel list — channel 1 returning 404 just means "nothing on port 1",
    # not "auth failed". The old code quit the whole device on a ch-1 miss,
    # which is exactly why a camera wired to channel 2 (path /Streaming/
    # Channels/201) was never discovered and the preview stayed black.
    #
    # A 404 (path/channel not present) is normal and expected for empty ports;
    # we just move to the next channel. We only conclude the device failed if
    # NONE of the probed channels yielded a frame.
    channels = []
    known_template: str | None = None  # vendor path learned from the first hit
    for batch_start in range(1, max_channels + 1, CHANNEL_BATCH_SIZE):
        batch_end = min(batch_start + CHANNEL_BATCH_SIZE - 1, max_channels)
        found_in_batch = False
        consecutive_misses = 0
        for ch in range(batch_start, batch_end + 1):
            ok, res, tpl, thumbnail = _probe_channel(
                host, port,
                device.get("username"), device.get("password"),
                ch,
                only_template=known_template,
                paths=ordered_paths,
            )
            if ok:
                # tpl is the matching TEMPLATE; format it into the real path.
                path = tpl.format(ch=ch) if tpl else None
                log.info("  ch %s online (%s) path=%s", ch, res, path)
                entry = {"channel": ch, "resolution": res, "path": path}
                if thumbnail:
                    entry["thumbnail"] = thumbnail  # base64 JPEG data URL for the mapping UI
                channels.append(entry)
                found_in_batch = True
                consecutive_misses = 0
                # Lock onto this vendor's path for the remaining channels so we
                # make ~1 RTSP open per channel instead of trying all 5 templates.
                known_template = tpl
            else:
                consecutive_misses += 1
                # Once this batch has already found a camera, we can stop
                # probing the rest of it early after a run of misses — we've
                # already walked past the populated range within this batch.
                # This doesn't change the batch-level continue/stop decision
                # below, it just avoids wasted probes within a batch that's
                # already proven non-empty.
                if found_in_batch and consecutive_misses >= 3:
                    break

        if not found_in_batch:
            log.info(
                "  channels %d-%d found nothing — stopping discovery here "
                "(channels found so far: %d)", batch_start, batch_end, len(channels),
            )
            break

    if not channels:
        # Device is reachable (TCP ok) but no channel returned video on any
        # known path. Most likely wrong credentials, or a vendor RTSP layout
        # we don't have a template for yet. Report it so the UI shows a real
        # error instead of spinning.
        log.warning(
            "  %s:%s reachable but no channel (1..%d) returned video — "
            "auth or unknown RTSP path", host, port, max_channels,
        )
        try:
            api.post_device_status({"device_id": device["id"], "status": "online"})
        except Exception as e:
            log.warning("device-status post failed: %s", e)
        _post_failed(
            api, device["id"], "auth",
            "NVR is reachable but no channel returned video — check the username/password, "
            "or the NVR may use an unusual RTSP path",
        )
        return None

    try:
        result = api.post_discover({"device_id": device["id"], "channels": channels})
        log.info("discover ok: %s channels=%s", device["name"], len(channels))
        return result
    except Exception as e:
        log.warning("discover post failed for %s: %s", device["name"], e)
        _post_failed(api, device["id"], "unknown", f"Cloud rejected discover: {e}")
        return None


def start_discovery(
    api: ApiClient,
    stop_event: threading.Event,
    interval_idle_s: int = 300,
    interval_active_s: int = 15,
) -> threading.Thread:
    """Background loop. Periodically probe every device this branch knows about.

    Adaptive interval: when at least one device is flagged `discovery_pending`
    by the backend (user clicked Add NVR or Retry), we tighten the poll to
    `interval_active_s` (default 15s) so the UI doesn't have to wait minutes
    for the next probe. When nothing is pending we relax to `interval_idle_s`
    (default 5 minutes), which is fine because already-discovered devices
    don't need frequent re-probing.

    seen_devices tracks devices we've successfully reported to the cloud so
    we don't re-probe them on every tick. A device that turns `discovery_pending`
    again (e.g. user clicked Retry) is dropped from this set so the next
    iteration re-probes it."""
    seen_devices: set[str] = set()

    def loop():
        log.info("discovery loop starting (idle=%ss active=%ss)", interval_idle_s, interval_active_s)
        iteration = 0
        while not stop_event.is_set():
            iteration += 1
            try:
                devices = api.list_devices()
            except Exception as e:
                log.warning("list_devices failed: %s", e)
                stop_event.wait(interval_idle_s)
                continue

            # Only log when there's ACTUAL discovery work (a pending device),
            # plus once at startup. The loop polls the cloud every 15s to notice
            # newly-added devices, but with pending=0 it never touches the NVR —
            # logging an idle heartbeat each time just looks like (and gets
            # mistaken for) a re-check, so we stay quiet when nothing is pending.
            pending_count = sum(1 for d in devices if d.get("discovery_pending"))
            if iteration == 1 or pending_count > 0:
                log.info(
                    "discovery tick #%d: total=%d pending=%d seen=%d",
                    iteration, len(devices), pending_count, len(seen_devices),
                )

            # Drop any device that is pending again from the seen set so the
            # retry-discovery flow actually re-probes.
            for d in devices:
                if d.get("discovery_pending") and d["id"] in seen_devices:
                    seen_devices.discard(d["id"])

            for d in devices:
                if d["id"] in seen_devices:
                    continue
                # type=RTSP devices are single-stream — backend already
                # created the companion camera row from the user-supplied
                # rtsp_url at device-create time. No probing, no channel
                # discovery. Just mark the device online and move on.
                if (d.get("type") or "").upper() == "RTSP":
                    try:
                        api.post_device_status({"device_id": d["id"], "status": "online"})
                    except Exception as e:
                        log.warning("device-status post failed: %s", e)
                    seen_devices.add(d["id"])
                    continue
                # Skip NVR/DVR devices already in 'failed' state — the user
                # needs to fix creds and explicitly click Retry, which flips
                # the device back to 'pending' and drops it from seen_devices
                # via the discovery_pending check above. Without this guard
                # we re-probe every 15s and spam the log forever.
                if d.get("discovery_status") == "failed":
                    continue
                # Probe the device. discover_device posts either success
                # (post_discover) or failure (_post_failed) to the backend,
                # which flips discovery_status to 'ready' or 'failed' in DB.
                # Either way, mark the device seen so we don't re-probe on
                # the *current* tick — the next list_devices call will then
                # surface the new discovery_status and either:
                #   - 'ready':  stays in seen_devices (no need to re-probe).
                #   - 'failed': caught by the `discovery_status == "failed"`
                #               skip above (no re-probe until Retry flips it).
                #   - 'pending' again (user clicked Retry): the discovery_pending
                #               loop above drops the entry from seen_devices.
                # Without adding failed probes to seen_devices, the agent would
                # hammer a locked-out NVR multiple times before the DB write
                # propagates, which is exactly how Hikvision NVRs hit their
                # auth-rate-limit and stop accepting RTSP entirely.
                discover_device(api, d)
                seen_devices.add(d["id"])

            # Pick the next wait based on whether any device is still waiting
            # for discovery. If yes, sleep briefly so retries feel responsive;
            # otherwise relax to the long idle interval.
            any_pending = any(d.get("discovery_pending") for d in devices)
            wait_s = interval_active_s if any_pending else interval_idle_s
            stop_event.wait(wait_s)

    t = threading.Thread(target=loop, name="discovery", daemon=True)
    t.start()
    return t
