import json
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.eval.harness import evaluate, wilson  # noqa: E402
from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim import collect as collect_mod  # noqa: E402
from arrival.sim.collect import SimEnv, collect, run_episode  # noqa: E402
from arrival.sim.people import CLEAR, LANE, Walkers  # noqa: E402
from arrival.sim.policy import ScriptedExpert, VLANotConfigured  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402
from conftest import REPO  # noqa: E402

SINK_TASK = {"family": "counter_to_sink", "object_id": "mug_1", "object_label": "mug", "source": "counter_a",
             "target": "sink_1", "target_x": 1.6, "start": [1.0, 0.35, 0.97], "lighting": "neutral",
             "instruction": "Put the mug in the sink.", "task_id": "t0", "seed": 0}


@pytest.fixture(scope="module")
def twin():
    return compile_twin(load_room("hero_kitchen", "scans"))


def test_wilson_known_value():
    lo, hi = wilson(8, 10)
    assert abs(lo - 0.490) < 5e-4 and abs(hi - 0.943) < 5e-4
    assert wilson(0, 0) == [0.0, 1.0]
    assert wilson(5, 5)[1] == 1.0 and wilson(0, 5)[0] == 0.0


def test_eval_doc_quotes_wilson_widths():
    text = (REPO / "docs" / "07-eval.md").read_text(encoding="utf-8")
    for n in (4, 50):
        widths = [hi - lo for lo, hi in (wilson(k, n) for k in range(n + 1))]
        assert f"{min(widths):.2f} to {max(widths):.2f} wide" in text
    lo, hi = wilson(190, 381)
    assert hi - lo <= 0.1 < wilson(190, 380)[1] - wilson(190, 380)[0]
    assert "about 380 episodes" in text


def test_scripted_eval_shape(twin, tmp_path):
    rep = evaluate(twin, ["counter_to_sink", "open_drawer"], 2, res=32, people=1, out_dir=tmp_path,
                   log=lambda *_: None)
    saved = json.loads(rep["path"].read_text(encoding="utf-8"))
    assert set(saved["families"]) == {"counter_to_sink", "open_drawer"}
    for f in saved["families"].values():
        assert f["episodes"] == 2 and 0.0 <= f["success_rate"] <= 1.0
        lo, hi = f["wilson_95"]
        assert lo <= f["success_rate"] <= hi
        assert f["mean_seconds"] > 0 and f["mean_wall_seconds"] > 0
        assert f["human_contacts"] + f["successes"] <= f["episodes"]
        assert math.isfinite(f["min_human_distance_m"]) and f["min_human_distance_m"] > 0
        assert f["ik_error_max"] is not None
    assert saved["totals"]["episodes"] == 4 and saved["policy"] == "scripted" and saved["holdout"] is False
    assert rep["path"] == tmp_path / "eval" / rep["run_id"] / "report.json"


def test_eval_reports_arm_contacts(twin, tmp_path):
    fams = ["counter_to_sink", "open_door"]
    rep = evaluate(twin, fams, 1, res=32, people=0, out_dir=tmp_path, log=lambda *_: None)
    saved = json.loads(rep["path"].read_text(encoding="utf-8"))
    for f in saved["families"].values():
        assert f["episodes"] == 1 and f["arm_contact_episodes"] in (0, 1) and f["arm_contact_max_depth_m"] >= 0
    assert saved["totals"]["arm_contact_episodes"] == sum(f["arm_contact_episodes"] for f in saved["families"].values())
    from arrival.eval.harness import table

    assert "arm_hit" in table(rep).splitlines()[0]


def test_arm_contact_counts_link_into_wall(twin):
    from arrival.sim import robot
    from arrival.sim.safety import SafetyMonitor
    from arrival.sim.scene import build_mjcf

    def at(base):  # arm at HOME, facing the south wall from `base`
        m = mujoco.MjModel.from_xml_string(build_mjcf(twin, base, 90.0))
        d = mujoco.MjData(m)
        d.qpos[[int(m.joint(j).qposadr[0]) for j in robot.ARM_JOINTS]] = robot.HOME
        mujoco.mj_forward(m, d)
        mon = SafetyMonitor(m)
        mon.update(m, d)
        return mon.summary()

    s = at((2.0, 2.85, 0.75))  # 0.15 m from the wall: the forearm reaches into it
    assert s["arm_contact_steps"] == 1 and s["arm_contact_worst"] == ["link5", "wall_south"]
    assert s["arm_contact_max_depth_m"] > 0.03
    s = at((2.0, 2.75, 0.75))
    assert s["arm_contact_steps"] == 0 and s["arm_contact_worst"] is None


def test_holdout_uses_template_4_and_y_offset(twin, tmp_path, monkeypatch):
    import arrival.eval.harness as harness

    ran = []
    monkeypatch.setattr(harness, "run_episode", lambda env, pol, task: ran.append(task) or run_episode(env, pol, task))
    rep = evaluate(twin, ["counter_to_sink"], 1, res=32, people=0, holdout=True, out_dir=tmp_path,
                   log=lambda *_: None)
    assert rep["holdout"] is True and rep["families"]["counter_to_sink"]["min_human_distance_m"] is None
    (task,) = ran
    obj = next(o for o in twin["objects"] if o["id"] == task["object_id"])
    assert task["holdout"] is True and task["wording_index"] == 3
    assert task["instruction"] == f"Take the {obj['label']} and drop it in the sink."
    assert abs(task["start"][0] - obj["position"][0]) < 1e-9
    assert abs(task["start"][1] - obj["position"][1] - 0.04) < 1e-9  # the sim halves the y offset


def test_training_tasks_exclude_the_holdout(twin):
    from arrival.sim.tasks import make_tasks

    def offsets(tasks):
        out = set()
        for t in tasks:
            if "start" not in t:
                continue
            obj = next(o for o in twin["objects"] if o["id"] == t["object_id"])
            out.add((round(t["start"][0] - obj["position"][0], 3), round(t["start"][1] - obj["position"][1], 3)))
        return out

    train = make_tasks(twin, ["counter_to_sink", "open_drawer"], 40, seed=0, log=lambda *_: None)
    assert len(train) == 40 and all(t["holdout"] is False for t in train)
    assert {t["wording_index"] for t in train if t["family"] == "counter_to_sink"} == {0, 1, 2}
    assert {t["wording_index"] for t in train if t["family"] == "open_drawer"} == {0, 1, 2}
    assert offsets(train) == {(0.0, 0.0), (0.08, 0.0), (-0.08, 0.0)}
    held = make_tasks(twin, ["counter_to_sink", "open_drawer"], 40, seed=0, holdout=True, log=lambda *_: None)
    assert {t["wording_index"] for t in held if t["family"] == "counter_to_sink"} == {3}
    assert {t["wording_index"] for t in held if t["family"] == "open_drawer"} == {0, 1, 2}
    assert offsets(held) == {(0.0, 0.04)}


def test_eval_vla_needs_server(twin, monkeypatch, tmp_path):
    monkeypatch.delenv("GR00T_POLICY_URL", raising=False)
    with pytest.raises(VLANotConfigured):
        evaluate(twin, ["counter_to_sink"], 1, policy_name="vla", checkpoint_id="test-mock", out_dir=tmp_path, log=lambda *_: None)


def test_cli_eval_vla_unreachable_exits_2(monkeypatch, capsys):
    from arrival.cli import main

    monkeypatch.setenv("GR00T_POLICY_URL", "http://127.0.0.1:9")  # discard port, nothing listens
    assert main(["eval", "--policy", "vla", "--checkpoint-id", "test-mock", "--episodes", "1", "--families", "counter_to_sink", "--people", "0",
                 "--res", "32"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("policy vla: no answer from http://127.0.0.1:9:") and "Traceback" not in err


def test_person_on_base_is_contact(twin):
    env = SimEnv(twin, dict(SINK_TASK, people=1), res=32)
    env.walkers = None  # hold the person still
    env.safety.reset()
    base = env.body_pos("robot_base")
    env.data.mocap_pos[0] = [base[0], base[1], 0.0]
    env.step(np.array(list(env.data.ctrl[:7]) + [0.05]))
    s = env.safety.summary()
    assert s["human_contact"] is True and s["min_human_distance_m"] <= 0.0 and s["min_distance_step"] == 0


def test_contact_episode_is_dropped(twin, tmp_path, monkeypatch):
    class Standing:
        def __init__(self, n, floor, seed=0, **kw):
            self.n = n

        def step(self, model, data, dt, clearance=None):
            b = data.body("robot_base").xpos
            data.mocap_pos[0] = [b[0], b[1], 0.0]
            return [[float(b[0]), float(b[1])]] * self.n

    monkeypatch.setattr(collect_mod, "Walkers", Standing)
    lines = []
    res = collect(twin, ["counter_to_sink"], 1, tmp_path, people=1, res=32, log=lines.append)
    r = res["report_data"]
    assert r["human_contacts"] == 1 and r["successes"] == 0 and r["dropped"] == 1
    assert r["families"]["counter_to_sink"]["human_contacts"] == 1
    assert any("DROP human_contact" in ln for ln in lines)
    assert res["info"]["total_episodes"] == 0


def test_normal_run_has_finite_positive_distance(twin, tmp_path):
    env = SimEnv(twin, dict(SINK_TASK, people=2), res=32)
    ep = run_episode(env, ScriptedExpert(), dict(SINK_TASK, people=2))
    s = ep["safety"]
    assert s["people"] == 2 and 0.0 < s["min_human_distance_m"] < 10.0 and s["human_contact"] is False
    assert 0 <= s["min_distance_step"] < len(ep["states"])
    res = collect(twin, ["counter_to_sink"], 1, tmp_path, people=2, res=32, log=lambda *_: None)
    row = json.loads((res["dataset"] / "meta" / "episodes.jsonl").read_text().splitlines()[0])
    assert row["arrival_human_contact"] is False and row["arrival_min_human_distance_m"] > 0
    assert row["arrival_people"] == 2


@pytest.mark.parametrize("base", [(1.3, 1.0), (3.2, 1.445)])  # sink run; counter_b drawer near the east wall
def test_walkers_interact_and_avoid_base(base):
    from arrival.sim.people import _seg_dist

    floor, front = (0.4, 1.65, 3.6, 2.7), 0.75
    for seed in range(6):
        w = Walkers(2, floor, seed=seed, base_xy=base, front_y=front)
        for path, pause in zip(w.paths, w.pauses):
            assert len(path) == 6 and 2.0 <= pause[4] <= 4.0
            assert abs(path[4][1] - (front + 0.35)) < 1e-9 and abs(path[4][0] - base[0]) >= 0.8  # counter stop
            assert path[2][1] == path[3][1] and path[2][1] > base[1] + 0.4  # lane behind the base
            assert np.linalg.norm(path[0] - np.array(base)) >= CLEAR  # spawn clear of the base column
            for a, b in zip(path, path[1:] + path[:1]):
                lane = a is path[2] and b is path[3]
                assert _seg_dist(a, b, np.array(base)) >= (LANE[0] if lane else CLEAR), (seed, a, b)
    once, again = (Walkers(2, floor, seed=3, base_xy=base, front_y=front).paths for _ in range(2))
    assert all(np.allclose(a, b) for pa, pb in zip(once, again) for a, b in zip(pa, pb))


def test_finish_task_covers_close_and_put_families():
    from arrival.sim.tasks import finish_task

    for fam in ("open_drawer", "close_drawer", "put_in_drawer"):
        t = finish_task({"family": fam, "fixture": "cab"})
        assert t["handle"] == "cab_drawer0_handle" and t["joint"] == "cab_drawer0"
    for fam in ("open_door", "close_door", "open_fridge"):
        assert finish_task({"family": fam, "fixture": "cab"})["joint"] == "cab_door"


def test_eval_isolates_a_failing_episode_and_offsets_seeds(twin, tmp_path, monkeypatch):
    import arrival.eval.harness as harness

    seen = []

    def flaky(env, pol, task):
        seen.append(task)
        if len(seen) == 2:
            raise RuntimeError("ik exploded")
        return run_episode(env, pol, task)

    monkeypatch.setattr(harness, "run_episode", flaky)
    rep = evaluate(twin, ["counter_to_sink"], 2, res=32, people=0, out_dir=tmp_path, log=lambda *_: None)
    f = rep["families"]["counter_to_sink"]
    assert f["episodes"] == 2 and f["errors"] == ["RuntimeError: ik exploded"] and f["successes"] <= 1
    assert rep["totals"]["errors"] == 1 and rep["path"].is_file()
    assert all(t["seed"] >= harness.EVAL_SEED_BASE * 1000 for t in seen)
