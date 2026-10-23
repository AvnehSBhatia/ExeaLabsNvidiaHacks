"""DEM-4: scripts/demo_all.py runs the whole story offline and reports honestly."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytestmark = pytest.mark.render  # preflight probes OpenGL in a child process

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "demo_all.py"


@pytest.fixture(scope="module")
def demo_all():
    spec = importlib.util.spec_from_file_location("demo_all", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_quick(demo_all, tmp_path):
    assert demo_all.main(["--quick", "--out", str(tmp_path)]) == 0
    s = json.loads((tmp_path / "demo_summary.json").read_text())
    assert s["physically_valid"] is True
    assert s["dataset"]["joints_are_placeholder"] is False
    assert s["successes"] >= 1
    assert [r["status"] for r in s["stages"]] == ["PASS"] * 7
    assert [r["stage"] for r in s["stages"]][3:5] == ["validate", "loader check"]
    assert s["per_family"] and all(v["episodes"] == 2 for v in s["per_family"].values())
    assert all(v is None for v in s["cost"].values())
    assert demo_all.HONEST_LABEL in (tmp_path / "demo_summary.md").read_text()
    assert (tmp_path / "renders" / "hero_kitchen_top.png").is_file()
    page = (tmp_path / "index.html").read_text()
    assert "data:image/png;base64," in page and "Overall <b>PASS</b>" in page
    assert "https://" not in page


def test_quick_leaves_sockets_and_keys_as_it_found_them(demo_all, tmp_path, monkeypatch):
    import socket

    monkeypatch.setenv("TOKEN_FACTORY_API_KEY", "sk-test")
    real = socket.socket.connect
    seen = {}
    real_run = demo_all.run

    def spy(*a, **k):
        seen["key"] = "TOKEN_FACTORY_API_KEY" in demo_all.os.environ
        try:
            socket.socket().connect(("203.0.113.7", 80))
            seen["blocked"] = False
        except OSError as e:
            seen["blocked"] = "offline" in str(e)
        return real_run(*a, **k)

    monkeypatch.setattr(demo_all, "run", spy)
    assert demo_all.main(["--quick", "--out", str(tmp_path)]) == 0
    assert seen == {"key": False, "blocked": True}
    assert socket.socket.connect is real and demo_all.os.environ["TOKEN_FACTORY_API_KEY"] == "sk-test"


def test_missing_room_fails_with_one_line(demo_all, tmp_path, capsys):
    assert demo_all.main(["--quick", "--room", "no_such_room", "--out", str(tmp_path)]) == 1
    err = capsys.readouterr().err.strip()
    assert err.startswith("demo_all: FAIL") and "no_such_room" in err and "\n" not in err
    s = json.loads((tmp_path / "demo_summary.json").read_text())
    assert s["overall"] == "FAIL" and s["stages"][0]["status"] == "FAIL"


def test_a_failed_loader_check_fails_the_demo(demo_all, tmp_path, monkeypatch):
    from arrival.export import loader_check

    monkeypatch.setattr(loader_check, "check_loader", lambda root: {
        "ok": False, "errors": [{"code": "missing_meta", "msg": "missing meta/stats.json"}], "warnings": [], "info": {}})
    assert demo_all.main(["--quick", "--out", str(tmp_path)]) == 1
    s = json.loads((tmp_path / "demo_summary.json").read_text())
    assert s["overall"] == "FAIL" and s["stages"][-1]["stage"] == "loader check"
    assert "missing_meta: missing meta/stats.json" in s["error"]
