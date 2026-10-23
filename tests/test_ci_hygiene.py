"""Repo hygiene tooling: link checker, packaging smoke helpers, offline demo guards, CI file shape.

None of this needs mujoco, a network or a GPU.
"""
import importlib.util
import io
import json
import os
import socket
import sys
import tarfile
import threading
import zipfile

import pytest
import yaml

from conftest import REPO


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


check_links = _load("check_links")
smoke_package = _load("smoke_package")
demo_all = _load("demo_all")


# ---- check_links -------------------------------------------------------------------------------

def _tree(tmp_path, files):
    for name, text in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def test_repo_links_are_clean():
    assert check_links.check(REPO) == []


def test_good_links_and_anchors_pass(tmp_path):
    root = _tree(tmp_path, {
        "README.md": "# Top\n[a](docs/a.md) [b](docs/a.md#second-part) [self](#top) ![img](docs/pic.png)\n"
                     "[dup](docs/a.md#same-1) [ext](https://example.com/x) [mail](mailto:a@b.c)\n",
        "docs/a.md": "# A\n## Second part\n## Same\n## Same\n[up](../README.md#top)\n",
        "docs/pic.png": "x",
    })
    assert check_links.check(root) == []


def test_broken_file_anchor_and_escape_are_reported(tmp_path, capsys):
    root = _tree(tmp_path / "repo", {
        "README.md": "[gone](docs/nope.md)\n[bad](docs/a.md#no-such-heading)\n[out](../outside.md)\n![i](x.png)\n",
        "docs/a.md": "# A\n",
    })
    (tmp_path / "outside.md").write_text("# O\n")
    probs = check_links.check(root)
    assert len(probs) == 4
    assert any("README.md:1: broken link docs/nope.md" in p for p in probs)
    assert any("README.md:2: no heading anchor #no-such-heading in docs/a.md" in p for p in probs)
    assert any("README.md:3: link leaves the repo" in p for p in probs)
    assert any("README.md:4: broken link x.png" in p for p in probs)
    assert check_links.main(["--root", str(root)]) == 1
    assert capsys.readouterr().out.count("\n") == 4


def test_links_inside_code_are_ignored(tmp_path):
    root = _tree(tmp_path, {"README.md": "Use `[x](missing.md)` like so.\n```\n[y](missing2.md)\n```\n"})
    assert check_links.check(root) == []
    assert check_links.main(["--root", str(root)]) == 0


def test_slug_matches_github():
    assert check_links.slug("Join in: team plan for 30 people") == "join-in-team-plan-for-30-people"
    assert check_links.slug("The `arrival_physically_valid` flag") == "the-arrival_physically_valid-flag"
    assert check_links.slug("A [link](x.md) here") == "a-link-here"


# ---- check_docs: planned files under a to-do heading --------------------------------------------

def test_check_docs_skips_paths_only_under_a_todo_heading(tmp_path):
    check_docs = _load("check_docs")
    doc = tmp_path / "README.md"
    text = ("## Code to-do before submission\n\nAdd `arrival/jobs/shards.py` later.\n\n"
            "## Status\n\nSee `arrival/jobs/shards.py` now.\n")
    probs = check_docs.check_paths(tmp_path, doc, text, {})
    assert probs == ["README.md:7: path does not exist `arrival/jobs/shards.py`"]


# ---- smoke_package: archive checks ---------------------------------------------------------------

def _archives(tmp_path, sdist_files, wheel_files):
    sd, wh = tmp_path / "a-0.tar.gz", tmp_path / "a-0-py3-none-any.whl"
    with tarfile.open(sd, "w:gz") as t:
        for n in sdist_files:
            info = tarfile.TarInfo("a-0/" + n)
            t.addfile(info, io.BytesIO(b""))
    with zipfile.ZipFile(wh, "w") as z:
        for n, body in wheel_files.items():
            z.writestr(n, body)
    return sd, wh


GOOD_SDIST = ["pyproject.toml", "README.md", "arrival/cli.py", "arrival/__init__.py"]
GOOD_WHEEL = {"arrival/cli.py": "", "arrival/__init__.py": "",
              "a-0.dist-info/entry_points.txt": "[console_scripts]\narrival = arrival.cli:main\n"}


def test_smoke_accepts_a_good_package(tmp_path):
    assert smoke_package.check_archives(*_archives(tmp_path, GOOD_SDIST, GOOD_WHEEL)) == []


def test_smoke_flags_missing_files_script_and_stray_paths(tmp_path):
    bad_wheel = {"arrival/__init__.py": "", "tests/test_x.py": "", "scans/hero_kitchen/scan.json": "",
                 "a-0.dist-info/entry_points.txt": ""}
    probs = smoke_package.check_archives(*_archives(tmp_path, ["README.md"], bad_wheel))
    text = "\n".join(probs)
    assert "sdist is missing pyproject.toml" in text and "sdist is missing arrival/cli.py" in text
    assert "wheel is missing arrival/cli.py" in text
    assert "no `arrival` console script" in text
    assert "repo-only paths" in text


# ---- demo_all: GL backend choice, preflight, offline guard, watchdog, index --------------------------

def test_gl_candidates_order_per_platform():
    assert demo_all.gl_candidates("linux") == [None, "egl", "osmesa", "glfw"]
    assert demo_all.gl_candidates("darwin") == [None, "glfw"]
    assert demo_all.gl_candidates("win32") == [None, "glfw", "egl"]
    assert demo_all.gl_candidates("linux", forced="osmesa") == ["osmesa"]


def test_pick_gl_backend_takes_the_first_that_renders():
    seen = []

    def probe(b):
        seen.append(b)
        return b == "osmesa"

    assert demo_all.pick_gl_backend([None, "egl", "osmesa", "glfw"], probe) == (True, "osmesa",
                                                                                 ["default", "egl", "osmesa"])
    assert seen == [None, "egl", "osmesa"]
    assert demo_all.pick_gl_backend([None, "egl"], lambda b: False) == (False, None, ["default", "egl"])
    assert demo_all.pick_gl_backend([None], lambda b: True) == (True, None, ["default"])


def test_preflight_reports_missing_mujoco(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "mujoco" else real(n, *a))
    why = demo_all.preflight(probe=lambda b: True)
    assert why and "mujoco is not installed" in why and "pip install" in why and "\n" not in why


def test_preflight_reports_no_gl_and_sets_the_backend(monkeypatch):
    pytest.importorskip("mujoco")
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ.pop("MUJOCO_GL", None)
    why = demo_all.preflight(probe=lambda b: False)
    assert why and "no working OpenGL backend" in why and "default" in why and "\n" not in why
    assert demo_all.preflight(probe=lambda b: b == "glfw", log=lambda *_: None) is None
    assert os.environ["MUJOCO_GL"] == "glfw"
    os.environ.pop("MUJOCO_GL")
    # a forced MUJOCO_GL is tried alone
    os.environ["MUJOCO_GL"] = "egl"
    tried = []
    demo_all.preflight(probe=lambda b: tried.append(b) or True, log=lambda *_: None)
    assert tried == ["egl"]


def test_main_exits_2_with_one_line_when_the_machine_cannot_run_it(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(demo_all, "preflight", lambda *a, **k: "mujoco is not installed. pip install -e '.[sim]'")
    assert demo_all.main(["--quick", "--out", str(tmp_path / "o")]) == demo_all.EXIT_ENV == 2
    err = capsys.readouterr().err
    assert err.startswith("demo_all: mujoco is not installed") and err.count("\n") == 1
    assert not (tmp_path / "o").exists()


def test_offline_guard_refuses_remote_hosts_allows_loopback_and_restores(monkeypatch):
    for k in demo_all.NETWORK_ENV:
        monkeypatch.setenv(k, "secret")
    real = socket.socket.connect
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        with demo_all.offline_guard():
            assert not any(k in os.environ for k in demo_all.NETWORK_ENV)
            c = socket.socket()
            with pytest.raises(OSError, match="offline"):
                c.connect(("203.0.113.7", 443))
            with pytest.raises(OSError, match="offline"):
                c.connect_ex(("203.0.113.7", 443))
            c.close()
            ok = socket.create_connection(srv.getsockname(), timeout=2)  # loopback still works
            ok.close()
    finally:
        srv.close()
    assert socket.socket.connect is real
    assert all(os.environ[k] == "secret" for k in demo_all.NETWORK_ENV)


def test_offline_guard_restores_after_an_exception():
    real = socket.socket.connect
    with pytest.raises(RuntimeError):
        with demo_all.offline_guard():
            raise RuntimeError("boom")
    assert socket.socket.connect is real


def test_watchdog_writes_a_fail_summary_and_exits_1(tmp_path, capsys):
    done, codes = threading.Event(), []
    t = demo_all.start_watchdog(0.05, tmp_path / "o", exit_fn=lambda c: (codes.append(c), done.set()))
    assert done.wait(5)
    t.cancel()
    assert codes == [1]
    s = json.loads((tmp_path / "o" / "demo_summary.json").read_text(encoding="utf-8"))
    assert s["overall"] == "FAIL" and "timed out" in s["error"]
    assert capsys.readouterr().err.strip() == "demo_all: FAIL timed out after 0.05 s"


def test_watchdog_can_be_cancelled_or_disabled(tmp_path):
    assert demo_all.start_watchdog(0, tmp_path) is None
    fired = []
    t = demo_all.start_watchdog(0.2, tmp_path, exit_fn=fired.append)
    t.cancel()
    threading.Event().wait(0.4)
    assert fired == [] and not (tmp_path / "demo_summary.json").exists()


def _summary(tmp_path, room="hero_kitchen", error=None):
    sheet = tmp_path / "sheet.png"
    sheet.write_bytes(b"\x89PNG fake")
    return {"room": room, "mode": "quick", "overall": "FAIL" if error else "PASS", "seconds": 1.5, "error": error,
            "stages": [{"stage": "sim dataset", "status": "PASS", "seconds": 1.0, "detail": None}],
            "dataset": {"path": "out/datasets/x", "episodes": 2, "frames": 10, "physically_valid": True,
                        "joints_are_placeholder": False},
            "artifacts": {"contact_sheet": str(sheet)}}


def test_index_html_is_self_contained_honest_and_escaped(tmp_path):
    page = demo_all._index_html(_summary(tmp_path, room="<b>x</b>", error="a < b"))
    assert "data:image/png;base64," in page
    assert demo_all.HONEST_LABEL in page
    assert "arrival_physically_valid: true" in page and "joints_are_placeholder: false" in page
    assert "<b>x</b>" not in page and "&lt;b&gt;x&lt;/b&gt;" in page and "a &lt; b" in page
    assert "http://" not in page and "https://" not in page and "src=\"/" not in page
    assert "<script" not in page


def test_index_html_without_a_contact_sheet(tmp_path):
    s = _summary(tmp_path)
    s["artifacts"] = {}
    s["dataset"] = None
    page = demo_all._index_html(s)
    assert "<img" not in page and "PASS" in page


# ---- CI workflow and repo files ----------------------------------------------------------------------

def _ci():
    return yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))


def test_ci_gates_portable_checks_on_every_os_and_rendering_on_linux():
    jobs = _ci()["jobs"]
    job = jobs["test"]
    assert job["strategy"]["matrix"]["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]
    assert job["strategy"]["matrix"]["python-version"] == ["3.11", "3.12"]
    assert not job.get("continue-on-error")
    assert any("--no-render" in step.get("run", "") for step in job["steps"])
    render = jobs["render"]
    assert render["runs-on"] == "ubuntu-latest" and not render.get("continue-on-error")
    commands = [step.get("run", "") for step in render["steps"]]
    assert "python -m pytest -q -p no:cacheprovider -n auto" in commands  # the full suite, split over the CPUs
    assert not any("--no-render" in command for command in commands)
    assert any("preflight()" in command and "raise SystemExit(why)" in command for command in commands)


def test_ci_has_lint_package_and_docker_jobs():
    jobs = _ci()["jobs"]
    assert {"test", "lint", "package", "docker"} <= set(jobs)
    lint = "\n".join(s.get("run", "") for s in jobs["lint"]["steps"])
    assert "ruff check" in lint and "scripts/check_docs.py" in lint and "scripts/check_links.py" in lint
    assert any("smoke_package.py" in s.get("run", "") for s in jobs["package"]["steps"])
    assert any("docker build" in s.get("run", "") for s in jobs["docker"]["steps"])
    assert jobs["package"]["strategy"]["matrix"]["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]


def test_ci_runs_the_offline_demo_and_uploads_its_summary():
    steps = _ci()["jobs"]["render"]["steps"]
    assert any("scripts/demo_all.py --quick" in s.get("run", "") for s in steps)
    up = next(s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact@"))
    assert up["if"] == "always()"
    assert "out/demo/demo_summary.json" in up["with"]["path"] and "out/demo/index.html" in up["with"]["path"]
    setup = next(s for s in steps if str(s.get("uses", "")).startswith("actions/setup-python@"))
    assert setup["with"]["cache"] == "pip"


def test_rendering_shards_split_the_test_files_exactly_once():
    import conftest

    render = _ci()["jobs"]["render"]
    n = len(render["strategy"]["matrix"]["shard"])
    assert render["strategy"]["matrix"]["shard"] == list(range(1, n + 1))
    step = next(s for s in render["steps"] if "-n auto" in s.get("run", ""))
    assert step["env"]["ARRIVAL_TEST_SHARD"] == f"${{{{ matrix.shard }}}}/{n}"
    files = sorted(str(p.relative_to(REPO).as_posix()) for p in (REPO / "tests").glob("test_*.py"))
    weights = json.loads(conftest.SHARD_WEIGHTS.read_text(encoding="utf-8"))
    groups = conftest.shard_files(files, n, weights)
    assert sorted(f for g in groups for f in g) == files  # every file in exactly one shard
    loads = [sum(weights.get(f, 1.0) for f in g) for g in groups]
    assert max(loads) <= 1.25 * min(loads)  # roughly even, or the slowest shard sets the wall time again
    assert conftest.shard_files(files, n, weights) == groups  # the same split in every job and xdist worker


def test_dependabot_and_ruff_config_exist():
    d = yaml.safe_load((REPO / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    assert {u["package-ecosystem"] for u in d["updates"]} == {"github-actions", "pip", "docker"}
    py = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.ruff.lint]" in py and 'lint = ["ruff==' in py


def test_gitignore_and_contributing_cover_the_basics():
    ignored = (REPO / ".gitignore").read_text(encoding="utf-8").split()
    for p in ("out/", ".venv/", ".env", "build/", "dist/", ".ruff_cache/", "*.egg-info/"):
        assert p in ignored
    text = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")
    for needle in ("make test", "make demo", "make lint", "make docs-check", "make package-smoke"):
        assert needle in text


def test_make_demo_is_the_quick_offline_demo():
    import re

    mk = (REPO / "Makefile").read_text(encoding="utf-8")
    body = mk.split("\ndemo:", 1)[1].split("\n\n", 1)[0]
    assert "scripts/demo_all.py --quick" in body
    assert {"lint", "docs-check", "package-smoke"} <= set(re.findall(r"^([A-Za-z_][\w-]*):", mk, re.M))


def test_every_test_file_has_a_shard_weight():
    import conftest

    weights = json.loads(conftest.SHARD_WEIGHTS.read_text(encoding="utf-8"))
    files = {p.relative_to(REPO).as_posix() for p in (REPO / "tests").glob("test_*.py")}
    assert sorted(files - set(weights)) == []
    assert sorted(set(weights) - files) == []
