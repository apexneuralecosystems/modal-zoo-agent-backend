from clip_recorder import hik_ts, build_playback_uri, build_download_body, track_id


def test_hik_ts_formats_wallclock_with_z():
    assert hik_ts("2026-06-16T10:00:00") == "20260616T100000Z"
    assert hik_ts("2026-06-16T10:15:00Z") == "20260616T101500Z"


def test_track_id_from_channel():
    assert track_id(2) == "201"
    assert track_id(1) == "101"
    assert track_id(13) == "1301"


def test_build_playback_uri():
    uri = build_playback_uri("192.168.0.7", 2, "2026-06-16T10:00:00", "2026-06-16T10:15:00")
    assert uri == (
        "rtsp://192.168.0.7/Streaming/tracks/201"
        "?starttime=20260616T100000Z&endtime=20260616T101500Z"
    )


def test_build_download_body_escapes_ampersand():
    body = build_download_body(
        "rtsp://192.168.0.7/Streaming/tracks/201?starttime=20260616T100000Z&endtime=20260616T101500Z"
    )
    assert "<downloadRequest" in body
    # the inner playback URI's & must be XML-escaped, plus the name/size suffix
    assert "&amp;endtime=" in body
    assert "&amp;name=clip&amp;size=0" in body
