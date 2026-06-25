"""Tests for agent_paths — uses AGENT_CODE_DIR / AGENT_HOME env overrides so the
layout can be exercised without creating real symlinks (Windows-friendly)."""
import importlib


def _reload(monkeypatch, code_dir, home=None):
    monkeypatch.setenv("AGENT_CODE_DIR", str(code_dir))
    if home is None:
        monkeypatch.delenv("AGENT_HOME", raising=False)
    else:
        monkeypatch.setenv("AGENT_HOME", str(home))
    import agent_paths
    return importlib.reload(agent_paths)


def test_versioned_layout(tmp_path, monkeypatch):
    code = tmp_path / "versions" / "1.2.0"
    code.mkdir(parents=True)
    (code / "VERSION").write_text("1.2.0\n", encoding="utf-8")

    mod = _reload(monkeypatch, code)  # AGENT_HOME unset -> derive from versions/ layout

    assert mod.CODE_DIR == code
    assert mod.AGENT_ROOT == tmp_path
    assert mod.VERSIONS_DIR == tmp_path / "versions"
    assert mod.CURRENT_VERSION_FILE == tmp_path / "current_version"
    assert mod.LAST_GOOD_FILE == tmp_path / "last_good"
    assert mod.running_version() == "1.2.0"


def test_dev_layout_root_is_code_dir(tmp_path, monkeypatch):
    code = tmp_path / "modal-zoo-agent-backend"
    code.mkdir()
    mod = _reload(monkeypatch, code)  # parent is not "versions" -> root == code dir
    assert mod.AGENT_ROOT == code
    assert mod.running_version() == "0.0.0"  # no VERSION file


def test_agent_home_override_wins(tmp_path, monkeypatch):
    code = tmp_path / "versions" / "9.9.9"
    code.mkdir(parents=True)
    home = tmp_path / "elsewhere"
    home.mkdir()
    mod = _reload(monkeypatch, code, home=home)
    assert mod.AGENT_ROOT == home
    assert mod.VERSIONS_DIR == home / "versions"
