"""Tests for discovery.py's channel-probing loop, specifically the
128-channel ceiling and the batch-of-16 continue/stop logic added
alongside it."""
from unittest.mock import MagicMock, patch

import discovery


def _device(ip="10.0.0.5", port=554):
    return {
        "id": "dev-1",
        "name": "Test NVR",
        "ip_address": ip,
        "port": port,
        "username": "admin",
        "password": "pass",
    }


def _make_api():
    api = MagicMock()
    api.post_discover.return_value = {"ok": True}
    return api


def _posted_channels(api):
    call = api.post_discover.call_args
    payload = call.args[0] if call.args else call.kwargs["payload"]
    return payload["channels"]


@patch("discovery._detect_vendor", return_value=None)
@patch("discovery._tcp_reachable", return_value=True)
@patch("discovery._probe_channel")
def test_default_16_channels_unaffected(mock_probe, _tcp, _vendor):
    """max_channels=16 (today's default) is a single batch — must probe
    exactly 16 times when every channel misses, proving the new batch logic
    doesn't change existing 16-channel behavior at all."""
    mock_probe.return_value = (False, None, None, None)
    api = _make_api()
    discovery.discover_device(api, _device(), max_channels=16)
    assert mock_probe.call_count == 16


@patch("discovery._detect_vendor", return_value=None)
@patch("discovery._tcp_reachable", return_value=True)
@patch("discovery._probe_channel")
def test_128_ceiling_stops_after_first_empty_batch_of_16(mock_probe, _tcp, _vendor):
    """With max_channels=128 and zero cameras found, discovery should stop
    after the first batch (16 probes) instead of scanning all 128."""
    mock_probe.return_value = (False, None, None, None)
    api = _make_api()
    result = discovery.discover_device(api, _device(), max_channels=128)
    assert mock_probe.call_count == discovery.CHANNEL_BATCH_SIZE
    assert mock_probe.call_count < 128
    assert result is None  # no channels found -> _post_failed path


@patch("discovery._detect_vendor", return_value=None)
@patch("discovery._tcp_reachable", return_value=True)
@patch("discovery._probe_channel")
def test_hit_in_batch_continues_to_next_batch_which_is_empty(mock_probe, _tcp, _vendor):
    """A camera found at channel 5 (batch 1: channels 1-16) should let
    discovery continue into batch 2 (channels 17-32) — but batch 2 is fully
    empty here, so discovery stops after that batch instead of continuing to
    128. Also proves the existing 'stop after 3 consecutive misses following
    a hit' shortcut still applies within a batch that already found something."""
    def side_effect(host, port, user, pwd, channel, only_template=None, paths=None):
        if channel == 5:
            return True, "1920x1080", "/Streaming/Channels/{ch}01", None
        return False, None, None, None
    mock_probe.side_effect = side_effect
    api = _make_api()
    discovery.discover_device(api, _device(), max_channels=128)
    # batch 1: ch1-4 miss, ch5 hits, ch6-8 miss (3 consecutive) -> early exit at ch8 (8 probes)
    # batch 2 (17-32): all miss, no shortcut (nothing found yet in this batch) -> full 16 probes, then stop
    assert mock_probe.call_count == 8 + discovery.CHANNEL_BATCH_SIZE
    channels = _posted_channels(api)
    assert len(channels) == 1
    assert channels[0]["channel"] == 5


@patch("discovery._detect_vendor", return_value=None)
@patch("discovery._tcp_reachable", return_value=True)
@patch("discovery._probe_channel")
def test_camera_beyond_first_empty_batch_is_not_found(mock_probe, _tcp, _vendor):
    """Documents the accepted tradeoff: a camera wired at channel 20 (batch 2)
    on an NVR with nothing at all in batch 1 (channels 1-16) is NOT
    discovered, because discovery stops after the first fully-empty batch
    and never reaches batch 2."""
    def side_effect(host, port, user, pwd, channel, only_template=None, paths=None):
        if channel == 20:
            return True, "1920x1080", "/Streaming/Channels/{ch}01", None
        return False, None, None, None
    mock_probe.side_effect = side_effect
    api = _make_api()
    result = discovery.discover_device(api, _device(), max_channels=128)
    assert mock_probe.call_count == discovery.CHANNEL_BATCH_SIZE
    assert result is None


@patch("discovery._detect_vendor", return_value=None)
@patch("discovery._tcp_reachable", return_value=True)
@patch("discovery._probe_channel")
def test_scans_all_8_batches_to_128_when_every_batch_has_a_hit(mock_probe, _tcp, _vendor):
    """One camera at the first channel of each of the 8 batches (1, 17, 33,
    49, 65, 81, 97, 113) — every batch finds something, so discovery must
    keep continuing all the way to the 128-channel ceiling instead of
    stopping early."""
    hit_channels = {1, 17, 33, 49, 65, 81, 97, 113}

    def side_effect(host, port, user, pwd, channel, only_template=None, paths=None):
        if channel in hit_channels:
            return True, "1920x1080", "/Streaming/Channels/{ch}01", None
        return False, None, None, None
    mock_probe.side_effect = side_effect
    api = _make_api()
    discovery.discover_device(api, _device(), max_channels=128)
    # Each batch: 1 hit then 3 consecutive misses -> early exit after 4 probes/batch.
    assert mock_probe.call_count == 8 * 4
    channels = _posted_channels(api)
    assert {c["channel"] for c in channels} == hit_channels
