"""config_loader must read config.json and resolve caches/logs against AGENT_ROOT,
so shared state lives outside the per-version code folder and survives swaps."""
import importlib
import json


def test_config_loaded_from_agent_root(tmp_path, monkeypatch):
    root = tmp_path
    code = root / "versions" / "1.0.0"
    code.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({
        "server_url": "http://x", "secret_token": "t",
        "branch_id": "b", "mac_serial": "s",
    }), encoding="utf-8")

    monkeypatch.setenv("AGENT_CODE_DIR", str(code))
    monkeypatch.setenv("AGENT_HOME", str(root))
    import agent_paths
    importlib.reload(agent_paths)
    import config_loader
    importlib.reload(config_loader)

    cfg = config_loader.load_config()

    assert cfg["server_url"] == "http://x"
    assert cfg["log_dir"].startswith(str(root))
    assert not cfg["log_dir"].startswith(str(code))
    # shared dirs created under the root, not the version folder
    assert (root / "logs").is_dir()
    assert (root / "models_cache").is_dir()
