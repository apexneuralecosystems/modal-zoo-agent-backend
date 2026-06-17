from clip_recorder import (
    hik_ts, track_id, build_rtsp_playback_url, build_dahua_loadfile_url,
    choose_fetch_method, hik_track_from_path, dahua_channel_from_path,
)


def test_hik_track_from_path_uses_saved_url_not_channel():
    # Camera labelled CH4 but its saved stream is track 201 -> use 201, NOT 401.
    assert hik_track_from_path("/Streaming/Channels/201", 4) == "201"
    assert hik_track_from_path("rtsp://h:554/Streaming/Channels/201", 4) == "201"
    assert hik_track_from_path("/Streaming/tracks/201", 4) == "201"
    # No parseable track -> fall back to {channel}01.
    assert hik_track_from_path(None, 2) == "201"
    assert hik_track_from_path("/cam/realmonitor?channel=4", 4) == "401"


def test_dahua_channel_from_path():
    assert dahua_channel_from_path("/cam/realmonitor?channel=4&subtype=0", 9) == 4
    assert dahua_channel_from_path(None, 3) == 3


def test_choose_fetch_method():
    # Small 2-min window inside a huge 1 GB segment -> RTSP wins (don't download 1GB).
    assert choose_fetch_method(1_065_000_000, 120) == "rtsp"
    # 4-min window from a small 100 MB segment -> ISAPI wins (tiny download).
    assert choose_fetch_method(100_000_000, 240) == "isapi"
    # 15-min window covering most of a 1 GB segment -> ISAPI wins (disk-speed beats realtime).
    assert choose_fetch_method(1_065_000_000, 900) == "isapi"
    # No size info -> default to RTSP (safe, exact).
    assert choose_fetch_method(0, 300) == "rtsp"


def test_hik_ts_formats_wallclock_with_z():
    assert hik_ts("2026-06-16T10:00:00") == "20260616T100000Z"
    assert hik_ts("2026-06-16T10:15:00Z") == "20260616T101500Z"


def test_track_id_from_channel():
    assert track_id(2) == "201"
    assert track_id(1) == "101"
    assert track_id(13) == "1301"


def test_build_rtsp_playback_url_with_creds():
    url = build_rtsp_playback_url(
        "192.168.0.7", 554, "admin", "Apex@321", "201",
        "2026-06-16T10:00:00", "2026-06-16T10:15:00",
    )
    assert url == (
        "rtsp://admin:Apex%40321@192.168.0.7:554/Streaming/tracks/201"
        "?starttime=20260616T100000Z&endtime=20260616T101500Z"
    )


def test_build_rtsp_playback_url_no_creds():
    url = build_rtsp_playback_url(
        "192.168.0.7", 554, None, None, "201",
        "2026-06-16T10:00:00", "2026-06-16T10:15:00",
    )
    assert url == (
        "rtsp://192.168.0.7:554/Streaming/tracks/201"
        "?starttime=20260616T100000Z&endtime=20260616T101500Z"
    )


def test_build_dahua_loadfile_url():
    url = build_dahua_loadfile_url(
        "192.168.0.50", 80, 4, "2026-06-17T09:43:00", "2026-06-17T09:47:00",
    )
    assert url == (
        "http://192.168.0.50:80/cgi-bin/loadfile.cgi?action=startLoad"
        "&channel=4&startTime=2026-06-17%2009:43:00&endTime=2026-06-17%2009:47:00&type=dav"
    )
