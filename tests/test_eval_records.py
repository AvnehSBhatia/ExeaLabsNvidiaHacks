import copy
import json
from pathlib import Path

import numpy as np
import pytest

from arrival.eval import harness
from arrival.eval.manifest import digest, load_manifest, make_manifest


@pytest.mark.parametrize("options", [
    {"episodes": True}, {"episodes": 0}, {"people": False}, {"res": 1.5},
    {"seed": "7"}, {"holdout": 1}, {"families": [""]},
    {"families": ["open_drawer", "open_drawer"]}, {"base": "sideways"}, {"arm": "unknown"},
])
def test_manifest_creation_rejects_bad_controls_before_task_generation(options):
    args = {"twin": {}, "families": ["open_drawer"], "episodes": 1}
    args.update(options)
    with pytest.raises(ValueError, match="evaluation needs"):
        make_manifest(**args)


@pytest.fixture
def rollouts(monkeypatch):
    seen = []

    class Env:
        def __init__(self, twin, task, res, base="fixed", arm="procedural"):
            seen.append((copy.deepcopy(twin), copy.deepcopy(task), res))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def episode(env, policy, task):
        contact = task["task_id"].endswith("0001")
        return {"states": np.zeros((5, 8)), "success": True, "ik_error": None,
                "detail": {"in_target": True},
                "safety": {"human_contact": contact, "min_human_distance_m": -0.01 if contact else 0.5,
                           "arm_contact_steps": 2 if contact else 0, "arm_contact_max_depth_m": 0.002 if contact else 0,
                           "arm_contact_worst": ["link1", "counter"] if contact else None}}

    monkeypatch.setattr(harness, "SimEnv", Env)
    monkeypatch.setattr(harness, "run_episode", episode)
    monkeypatch.setattr("arrival.sim.policy.VLAClient", lambda: object())
    return seen


def records(report):
    return [json.loads(line) for line in Path(report["episodes_path"]).read_text(encoding="utf-8").splitlines()]


def test_replay_preserves_trials_and_keeps_checkpoint_runs_separate(hero_twin, tmp_path, monkeypatch, rollouts):
    first = harness.evaluate(hero_twin, ["open_drawer", "counter_to_sink"], 2, holdout=True, people=1,
                             out_dir=tmp_path, policy_name="vla", checkpoint_id="base", log=lambda *_: None)
    original = first["path"].read_bytes()
    before = copy.deepcopy(rollouts)
    frozen = load_manifest(first["manifest_path"])
    monkeypatch.setattr("arrival.sim.tasks.make_tasks", lambda *a, **k: pytest.fail("replay resampled tasks"))
    monkeypatch.setattr("arrival.sim.tasks.finish_task", lambda *a: pytest.fail("replay rewrote tasks"))
    rollouts.clear()
    second = harness.evaluate(manifest=frozen, policy_name="vla", checkpoint_id="room-tuned",
                              out_dir=tmp_path, log=lambda *_: None)
    assert rollouts == before
    assert first["manifest_id"] == second["manifest_id"]
    assert first["run_id"] != second["run_id"] and first["path"] != second["path"]
    assert first["path"].read_bytes() == original
    assert {r["checkpoint_id"] for r in records(first)} == {"base"}
    assert {r["checkpoint_id"] for r in records(second)} == {"room-tuned"}
    assert len(list((tmp_path / "eval").glob("*/report.json"))) == 2


def test_summaries_reconcile_with_saved_trials(hero_twin, tmp_path, rollouts):
    report = harness.evaluate(hero_twin, ["open_drawer"], 3, people=1, out_dir=tmp_path, log=lambda *_: None)
    rows = records(report)
    family = report["families"]["open_drawer"]
    assert report["totals"]["episodes"] == len(rows) == family["episodes"]
    assert report["totals"]["successes"] == sum(r["success"] for r in rows) == family["successes"]
    assert family["human_contacts"] == sum(r["safety"]["human_contact"] for r in rows)
    assert family["arm_contact_episodes"] == sum(r["safety"]["arm_contact_steps"] > 0 for r in rows)
    assert family["mean_seconds"] == round(sum(r["seconds"] for r in rows) / len(rows), 3)
    assert family["mean_wall_seconds"] == round(sum(r["wall_seconds"] for r in rows) / len(rows), 3)
    assert any(r["physics_success"] and not r["success"] for r in rows)
    assert [r["trial_index"] for r in rows] == list(range(len(rows)))
    frozen = load_manifest(report["manifest_path"])
    assert [r["task_id"] for r in rows] == [t["task_id"] for t in frozen["tasks"]]


def test_errors_are_saved_and_counted(hero_twin, tmp_path, monkeypatch, rollouts):
    def fail(env, policy, task):
        raise RuntimeError("rollout failed")

    monkeypatch.setattr(harness, "run_episode", fail)
    report = harness.evaluate(hero_twin, ["open_drawer"], 2, out_dir=tmp_path, log=lambda *_: None)
    rows = records(report)
    assert report["totals"]["errors"] == len(rows) == 2
    assert report["totals"]["successes"] == 0
    assert all(r["physics_success"] is None and r["error"] == "RuntimeError: rollout failed" for r in rows)
    assert all(r["seconds"] is None and r["safety"]["human_contact"] is None for r in rows)
    family = report["families"]["open_drawer"]
    assert family["mean_seconds"] is None and family["human_contacts"] is None
    assert family["arm_contact_episodes"] is None and family["observed_episodes"] == 0
    assert report["totals"]["human_contacts"] is None
    assert "only observed episodes" in harness.table(report)


@pytest.mark.parametrize("bad_result", ["nonfinite", "missing_detail"])
def test_invalid_result_keeps_an_error_row_and_continues(hero_twin, tmp_path, monkeypatch, rollouts, bad_result):
    episode = harness.run_episode
    calls = []

    def invalid_once(env, policy, task):
        outcome = episode(env, policy, task)
        if not calls:
            if bad_result == "nonfinite":
                outcome["safety"]["min_human_distance_m"] = float("nan")
            else:
                del outcome["detail"]
        calls.append(task["task_id"])
        return outcome

    monkeypatch.setattr(harness, "run_episode", invalid_once)
    report = harness.evaluate(hero_twin, ["open_drawer"], 2, out_dir=tmp_path, log=lambda *_: None)
    rows = records(report)
    assert len(rows) == 2 and rows[0]["error"] and rows[1]["error"] is None
    assert rows[0]["physics_success"] is None and rows[0]["safety"]["human_contact"] is None
    family = report["families"]["open_drawer"]
    assert family["observed_episodes"] == 1 and report["totals"]["errors"] == 1
    assert family["mean_seconds"] == rows[1]["seconds"]
    assert family["human_contacts"] == rows[1]["safety"]["human_contact"]


def test_first_server_failure_leaves_an_incomplete_run(hero_twin, tmp_path, monkeypatch, rollouts):
    def fail(env, policy, task):
        raise OSError("server stopped")

    monkeypatch.setattr(harness, "run_episode", fail)
    with pytest.raises(OSError, match="server stopped"):
        harness.evaluate(hero_twin, ["open_drawer"], 2, policy_name="vla", checkpoint_id="base",
                         out_dir=tmp_path, log=lambda *_: None)
    run, = (tmp_path / "eval").iterdir()
    assert (run / "manifest.json").is_file() and not (run / "report.json").exists()
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["error"] == "OSError: server stopped"
    from arrival.eval.cost import collect_rows

    assert collect_rows(tmp_path) == []


@pytest.mark.parametrize("change", ["hash", "twin", "duplicate", "base", "omitted", "nonfinite", "missing", "extra"])
def test_invalid_manifest_fails_before_rollout(hero_twin, tmp_path, rollouts, change):
    frozen = make_manifest(hero_twin, ["open_drawer"], 2, log=lambda *_: None)
    if change == "hash":
        frozen["tasks"][0]["instruction"] = "changed"
    elif change == "twin":
        frozen["twin"]["room_id"] = "changed"
    elif change == "duplicate":
        frozen["tasks"][1]["task_id"] = frozen["tasks"][0]["task_id"]
    elif change == "base":
        frozen["settings"]["base"] = "sideways"
    elif change == "omitted":
        frozen["omitted_families"] = ["open_drawer"]
    elif change == "missing":
        frozen["tasks"].pop()
    elif change == "extra":
        frozen["tasks"].append({**frozen["tasks"][0], "task_id": "extra_trial"})
    else:
        frozen["tasks"][0]["seed"] = float("nan")
    if change not in ("hash", "nonfinite"):
        frozen["manifest_id"] = digest({k: v for k, v in frozen.items() if k != "manifest_id"})
    with pytest.raises(ValueError):
        harness.evaluate(manifest=frozen, out_dir=tmp_path)
    assert not rollouts and not (tmp_path / "eval").exists()


@pytest.mark.parametrize("change", ["boolean_version", "empty_room"])
def test_manifest_identity_cannot_be_rehashed_to_hide_invalid_values(hero_twin, change):
    from arrival.eval.manifest import validate_manifest

    frozen = make_manifest(hero_twin, ["open_drawer"], 1, log=lambda *_: None)
    if change == "boolean_version":
        frozen["schema_version"] = True
    else:
        frozen["twin"]["room_id"] = " "
        frozen["twin_sha256"] = digest(frozen["twin"])
    frozen["manifest_id"] = digest({k: v for k, v in frozen.items() if k != "manifest_id"})
    with pytest.raises(ValueError, match="schema|room_id"):
        validate_manifest(frozen)


@pytest.mark.parametrize("field,value", [("people", True), ("holdout", 0), ("seed", "7"),
                                         ("instruction", "  ")])
def test_rehashed_manifest_still_rejects_invalid_trial_types(hero_twin, tmp_path, rollouts, field, value):
    frozen = make_manifest(hero_twin, ["open_drawer"], 1, people=1, log=lambda *_: None)
    frozen["tasks"][0][field] = value
    frozen["manifest_id"] = digest({k: v for k, v in frozen.items() if k != "manifest_id"})
    with pytest.raises(ValueError, match="invalid evaluation task"):
        harness.evaluate(manifest=frozen, out_dir=tmp_path)
    assert not rollouts and not (tmp_path / "eval").exists()


def test_vla_requires_checkpoint_label_before_running(hero_twin, tmp_path, rollouts):
    with pytest.raises(ValueError, match="checkpoint-id"):
        harness.evaluate(hero_twin, ["open_drawer"], 1, policy_name="vla", out_dir=tmp_path)
    assert not rollouts


def test_cli_manifest_replay_and_conflicting_options(hero_twin, tmp_path, rollouts, capsys):
    from arrival.cli import main

    frozen = make_manifest(hero_twin, ["open_drawer"], 1, holdout=True, res=32, log=lambda *_: None)
    path = tmp_path / "trials.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")
    for option in (["--seed", "0"], ["--room", "hero_kitchen"], ["--holdout"], ["--res", "128"]):
        assert main(["eval", "--manifest", str(path), "--out", str(tmp_path), *option]) == 2
        assert "cannot be combined" in capsys.readouterr().err
    assert not rollouts
    assert main(["eval", "--manifest", str(path), "--out", str(tmp_path)]) == 0
    assert rollouts[0][1] == frozen["tasks"][0] and rollouts[0][2] == 32


def test_cost_reader_includes_legacy_and_complete_runs_only(hero_twin, tmp_path, rollouts):
    from arrival.eval.cost import collect_rows, markdown

    report = harness.evaluate(hero_twin, ["open_drawer"], 1, out_dir=tmp_path, log=lambda *_: None)
    legacy = {k: v for k, v in json.loads(report["path"].read_text()).items()
              if k not in ("run_id", "checkpoint_id")}
    (tmp_path / "eval" / "old_scripted.json").write_text(json.dumps(legacy))
    rows = collect_rows(tmp_path)
    assert len(rows) == 4
    assert len([r for r in rows if r.get("run_id") == report["run_id"]]) == 2
    text = markdown({"rows": rows, "dollars": {}, "comparison": {"measured": False, "reason": "no baseline"}})
    assert report["run_id"] in text


def test_cli_manifest_refuses_base_and_arm(hero_twin, tmp_path, rollouts, capsys):
    from arrival.cli import main

    frozen = make_manifest(hero_twin, ["open_drawer"], 1, res=32, log=lambda *_: None)
    path = tmp_path / "trials.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")
    for option in (["--base", "mobile"], ["--arm", "menagerie"], ["--base", "fixed"]):
        assert main(["eval", "--manifest", str(path), "--out", str(tmp_path), *option]) == 2
        assert option[0] in capsys.readouterr().err
    assert not rollouts
