import argparse

import pytest

from arrival import cli


def _ns(tmp_path, **kw):
    d = dict(room="nope_room", scans="scans", out=str(tmp_path), families="counter_to_sink", episodes=1, people=0,
             policy="scripted", res=32, seed=0, save_preview=False, planner="rule", split="train",
             instructions="template", tavily=False, holdout=False, query=None)
    d.update(kw)
    return argparse.Namespace(**d)


def test_sim_missing_room_is_usage_error(tmp_path, capsys):
    pytest.importorskip("mujoco")
    assert cli.cmd_sim(_ns(tmp_path)) == cli.EXIT_USAGE
    assert "no scan for room" in capsys.readouterr().err


def test_research_missing_room_is_usage_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("arrival.tasks.tavily_research.TavilyResearchAgent", lambda *a, **k: object())
    assert cli.cmd_research(_ns(tmp_path)) == cli.EXIT_USAGE


def test_sim_vla_not_configured_is_usage_error(tmp_path, capsys, monkeypatch):
    pytest.importorskip("mujoco")
    monkeypatch.delenv("GR00T_POLICY_URL", raising=False)
    monkeypatch.chdir(tmp_path)  # resolve_scans falls back to the repo's scans/
    assert cli.cmd_sim(_ns(tmp_path, room="hero_kitchen", policy="vla")) == cli.EXIT_USAGE
    assert "policy vla" in capsys.readouterr().err


@pytest.mark.parametrize("cmd", ["cmd_sim", "cmd_eval"])
def test_mobile_base_with_a_served_policy_is_a_usage_error(tmp_path, capsys, monkeypatch, cmd):
    """The policy wire format is 8-float arm only (no base pose), so a served policy cannot drive the base."""
    pytest.importorskip("mujoco")
    monkeypatch.setenv("GR00T_POLICY_URL", "http://127.0.0.1:9")  # never contacted: refused before any run
    monkeypatch.chdir(tmp_path)
    ns = _ns(tmp_path, room="hero_kitchen", policy="vla", base="mobile")
    assert getattr(cli, cmd)(ns) == cli.EXIT_USAGE
    assert "--base mobile needs --policy scripted" in capsys.readouterr().err


def _room(tmp_path):
    import json
    import os
    import shutil

    from conftest import REPO

    scans = tmp_path / "scans"
    shutil.copytree(REPO / "scans" / "hero_kitchen", scans / "hero_kitchen")
    twin = tmp_path / "out" / "twins" / "hero_kitchen" / "twin.json"
    twin.parent.mkdir(parents=True)
    twin.write_text(json.dumps({"cached": True}))
    return scans, twin, scans / "hero_kitchen" / "scan.json", os


def _twin(tmp_path, scans):
    pytest.importorskip("mujoco")
    fams, twin = cli._sim_args(_ns(tmp_path / "out", base="fixed", arm="procedural", room="hero_kitchen", scans=str(scans)))
    return twin


def test_cached_twin_is_used_while_it_is_newer_than_the_scan(tmp_path):
    scans, twin, scan, os = _room(tmp_path)
    os.utime(scan, (1_000_000, 1_000_000))
    assert _twin(tmp_path, scans) == {"cached": True}


def test_cached_twin_is_recompiled_after_the_scan_changes(tmp_path):
    scans, twin, scan, os = _room(tmp_path)
    os.utime(twin, (1_000_000, 1_000_000))
    assert "cached" not in _twin(tmp_path, scans)
