"""config_loader must catch bad field types/formats at startup (punch-list
#40) instead of letting a background thread crash silently hours later when
it first tries to use the bad value (e.g. int(cfg["heartbeat_interval_s"])
in heartbeat.py)."""
import importlib
import json

import pytest

import config_loader


def _valid_cfg(**overrides):
    cfg = {
        "server_url": "http://localhost:3001",
        "secret_token": "REPLACE_WITH_PLAINTEXT_TOKEN",
        "branch_id": "REPLACE_WITH_BRANCH_UUID",
        "mac_serial": "REPLACE_WITH_HARDWARE_SERIAL",
        "agent_version": "1.0.0",
        "heartbeat_interval_s": 30,
        "poll_interval_s": 10,
        "log_dir": "./logs",
        "models_cache_dir": "./models_cache",
        "scripts_cache_dir": "./scripts_cache",
    }
    cfg.update(overrides)
    return cfg


def test_valid_config_has_no_errors():
    assert config_loader._validate_config(_valid_cfg()) == []


def test_minimal_config_from_existing_path_test_has_no_errors():
    """Matches the minimal shape used by test_config_paths.py's existing
    test (server_url='http://x', branch_id='b', etc.) — proves the new
    validation doesn't break that already-passing test."""
    cfg = {"server_url": "http://x", "secret_token": "t", "branch_id": "b", "mac_serial": "s"}
    assert config_loader._validate_config(cfg) == []


def test_server_url_missing_scheme_is_rejected():
    errors = config_loader._validate_config(_valid_cfg(server_url="localhost:3001"))
    assert any("server_url" in e for e in errors)


def test_server_url_wrong_type_is_rejected():
    errors = config_loader._validate_config(_valid_cfg(server_url=12345))
    assert any("server_url" in e for e in errors)


def test_heartbeat_interval_as_string_is_rejected():
    """The exact case from the punch list: a numeric field given as a string."""
    errors = config_loader._validate_config(_valid_cfg(heartbeat_interval_s="30"))
    assert any("heartbeat_interval_s" in e for e in errors)


def test_poll_interval_zero_or_negative_is_rejected():
    errors = config_loader._validate_config(_valid_cfg(poll_interval_s=0))
    assert any("poll_interval_s" in e for e in errors)
    errors = config_loader._validate_config(_valid_cfg(poll_interval_s=-5))
    assert any("poll_interval_s" in e for e in errors)


def test_update_watchdog_s_bad_type_is_rejected_when_present():
    """update_watchdog_s is optional (defaults to 900 in main.py), but if the
    user did set it, a bad type should still be caught at startup."""
    errors = config_loader._validate_config(_valid_cfg(update_watchdog_s="900"))
    assert any("update_watchdog_s" in e for e in errors)


def test_update_watchdog_s_absent_is_fine():
    cfg = _valid_cfg()
    assert "update_watchdog_s" not in cfg
    assert config_loader._validate_config(cfg) == []


def test_branch_id_wrong_type_is_rejected():
    errors = config_loader._validate_config(_valid_cfg(branch_id=12345))
    assert any("branch_id" in e for e in errors)


def test_empty_secret_token_is_rejected():
    errors = config_loader._validate_config(_valid_cfg(secret_token=""))
    assert any("secret_token" in e for e in errors)


def test_log_dir_wrong_type_is_rejected():
    errors = config_loader._validate_config(_valid_cfg(log_dir=123))
    assert any("log_dir" in e for e in errors)


def test_multiple_errors_are_all_reported_together():
    errors = config_loader._validate_config(_valid_cfg(
        server_url="ftp://bad",
        heartbeat_interval_s="not-a-number",
    ))
    assert len(errors) == 2
    assert any("server_url" in e for e in errors)
    assert any("heartbeat_interval_s" in e for e in errors)


def test_load_config_exits_with_clear_message_on_bad_config(tmp_path, monkeypatch, capsys):
    """Integration check: load_config() itself must fail fast (sys.exit) with
    a message naming the bad field, not silently pass a broken value through."""
    root = tmp_path
    code = root / "versions" / "1.0.0"
    code.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({
        "server_url": "not-a-url",
        "secret_token": "t",
        "branch_id": "b",
        "mac_serial": "s",
    }), encoding="utf-8")

    monkeypatch.setenv("AGENT_CODE_DIR", str(code))
    monkeypatch.setenv("AGENT_HOME", str(root))
    import agent_paths
    importlib.reload(agent_paths)
    importlib.reload(config_loader)

    with pytest.raises(SystemExit) as exc_info:
        config_loader.load_config()
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "server_url" in captured.err
