import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim import ik, robot  # noqa: E402
from arrival.sim.collect import SimEnv, collect  # noqa: E402
from arrival.sim.policy import VLAClient, VLANotConfigured, VLMPlanner  # noqa: E402
from arrival.sim.scene import build_mjcf  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402


@pytest.fixture(scope="module")
def twin():
    return compile_twin(load_room("hero_kitchen", "scans"))


def test_scene_has_moving_parts_and_people(twin):
    m = mujoco.MjModel.from_xml_string(build_mjcf(twin, (1.3, 1.0, 0.75), n_people=2))
    names = {m.joint(i).name for i in range(m.njnt)}
    assert {"counter_a_drawer0", "counter_a_door", "fridge_1_door", "mug_1"} <= names
    assert m.joint("counter_a_drawer0").type == mujoco.mjtJoint.mjJNT_SLIDE
    assert m.joint("fridge_1_door").type == mujoco.mjtJoint.mjJNT_HINGE
    assert m.nmocap == 2
    assert {m.camera(i).name for i in range(m.ncam)} >= {"ego_view", "wrist", "overview"}


def test_ik_reaches_counter_point(twin):
    task = {"family": "counter_to_sink", "object_id": "mug_1", "source": "counter_a", "target": "sink_1",
            "target_x": 1.6, "start": [1.0, 0.35, 0.97], "lighting": "neutral", "people": 0}
    env = SimEnv(twin, task, res=32)
    target = np.array([1.0, 0.35, 1.0])
    q, err = ik.solve(env.model, env.data, target, ik.down(0.0), np.array(robot.HOME), env.qadr, env.dofadr)
    assert err < 0.02
    d = mujoco.MjData(env.model)
    d.qpos[env.qadr] = q
    mujoco.mj_kinematics(env.model, d)
    assert np.linalg.norm(d.site("eef").xpos - target) < 0.02


def test_successive_episodes_have_nonblank_camera_video(twin, tmp_path):
    import imageio.v2 as imageio

    run = collect(twin, ["open_drawer"], 3, tmp_path, people=0, res=32, log=lambda *_: None)
    assert run["report_data"]["successes"] == 3
    for camera in ("ego_view", "wrist"):
        videos = sorted((run["dataset"] / "videos" / "chunk-000" / f"observation.images.{camera}").glob("*.mp4"))
        assert len(videos) == 3
        for video in videos:
            with imageio.get_reader(video, format="ffmpeg") as reader:
                frames = [frame for frame in reader]
            assert frames, video
            assert all(frame.max() > 0 and frame.std() > 1 for frame in frames), video


def test_collect_writes_valid_dataset(twin, tmp_path):
    from arrival.export.validate import validate

    res = collect(twin, ["counter_to_sink"], 2, tmp_path, people=1, res=64, log=lambda *_: None)
    r = res["report_data"]
    assert r["attempts"] == 2 and r["successes"] >= 1
    info = json.loads((res["dataset"] / "meta" / "info.json").read_text())
    assert info["arrival_generator"] == "mujoco_scripted_ik" and info["arrival_physically_valid"] is True
    emb = json.loads((res["dataset"] / "meta" / "embodiment.json").read_text())
    assert emb["joints_are_placeholder"] is False
    import pyarrow.parquet as pq

    t = pq.read_table(res["dataset"] / "data" / "chunk-000" / "episode_000000.parquet")
    s = np.array(t.column("observation.state").to_pylist())
    assert s.shape[1] == 11 and np.ptp(s[:, :7], axis=0).max() > 0.3  # real joint motion
    assert np.ptp(s[:, 8:11], axis=0).max() == 0.0  # fixed base: the parked pose on every frame
    assert list((res["dataset"] / "videos" / "chunk-000" / "observation.images.wrist").glob("*.mp4"))
    errors = validate(res["dataset"])
    assert not errors, errors


@pytest.mark.parametrize("fixture", ["counter_a", "counter_b"])
def test_scripted_expert_opens_cabinet_door(twin, fixture):
    from arrival.sim.collect import run_episode
    from arrival.sim.policy import ScriptedExpert

    task = {"family": "open_door", "fixture": fixture, "joint": f"{fixture}_door", "lighting": "neutral", "people": 0,
            "seed": 0}
    env = SimEnv(twin, task, res=32)
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"], ep["detail"]
    assert ep["detail"]["qpos"] > np.radians(62)  # 63 to 66 deg with joint 7 held off its stop (MAN-2 had 66.6 with it pinned)
    assert ep["ik_error"] < 0.02


@pytest.mark.parametrize("fixture", ["counter_a", "counter_b"])
def test_door_pull_keeps_joint7_off_its_stop(twin, fixture):
    from arrival.sim.collect import run_episode
    from arrival.sim.policy import ScriptedExpert

    task = {"family": "open_door", "fixture": fixture, "joint": f"{fixture}_door", "lighting": "neutral", "people": 0,
            "seed": 0}
    ep = run_episode(SimEnv(twin, task, res=32), ScriptedExpert(), task)
    assert ep["success"], ep["detail"]
    assert np.abs(ep["states"][:, 6]).max() < robot.RANGES[6][1] - 0.3


def test_scripted_expert_opens_fridge(twin):
    from arrival.sim.collect import FRIDGE_OPEN, run_episode
    from arrival.sim.policy import ScriptedExpert

    task = {"family": "open_fridge", "fixture": "fridge_1", "joint": "fridge_1_door", "lighting": "neutral",
            "people": 0, "seed": 0}
    env = SimEnv(twin, task, res=32)
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"], ep["detail"]
    assert ep["detail"]["qpos"] > FRIDGE_OPEN + np.radians(10)  # measured 79 deg against the 60 deg bar
    assert ep["ik_error"] < 0.01  # every waypoint, the hooked pull and the push, is in reach


def test_open_fridge_is_one_task_per_fridge_and_not_a_default(twin):
    from arrival.cli import build_parser
    from arrival.sim.tasks import SIM_FAMILIES, make_tasks

    tasks = make_tasks(twin, ["open_fridge"], 4, seed=0, log=lambda *_: None)
    assert len(tasks) == 4 and {t["fixture"] for t in tasks} == {"fridge_1"}
    assert "open_fridge" in SIM_FAMILIES
    assert "open_fridge" not in build_parser().parse_args(["sim"]).families


@pytest.mark.parametrize("family,fixture", [("close_drawer", "counter_a"), ("close_drawer", "counter_b"),
                                            ("close_door", "counter_a"), ("close_door", "counter_b")])
def test_scripted_expert_closes_open_parts(twin, family, fixture):
    from arrival.sim.collect import DOOR_OPEN, DRAWER_OPEN, run_episode
    from arrival.sim.policy import ScriptedExpert

    joint = f"{fixture}_door" if family == "close_door" else f"{fixture}_drawer0"
    task = {"family": family, "fixture": fixture, "joint": joint, "handle": f"{fixture}_drawer0_handle",
            "lighting": "neutral", "people": 0, "seed": 0}
    env = SimEnv(twin, task, res=32)
    start = float(env.data.joint(joint).qpos[0])
    assert start == pytest.approx(DOOR_OPEN if family == "close_door" else DRAWER_OPEN)  # it really starts open
    expert = ScriptedExpert()
    ep = run_episode(env, expert, task)
    assert ep["success"], ep["detail"]
    assert ep["ik_error"] < 0.03
    if family == "close_door":
        # pushing the panel above the bar keeps joint 2 off its stop for less than the per-episode validator's
        # LIMIT_STEPS, where it used to be flagged
        from arrival.export.quality import LIMIT_FRAC, LIMIT_STEPS

        lo, hi = robot.RANGES[1]
        q2 = np.array(expert.plan)[:, 1]
        near = (q2 < lo + LIMIT_FRAC * (hi - lo)) | (q2 > hi - LIMIT_FRAC * (hi - lo))
        run = longest = 0
        for x in near:
            run = run + 1 if x else 0
            longest = max(longest, run)
        assert longest < LIMIT_STEPS
        assert ep["ik_error"] < 0.002


def test_scripted_expert_puts_mug_in_drawer_and_wipes(twin):
    from arrival.sim.collect import run_episode
    from arrival.sim.policy import ScriptedExpert
    from arrival.sim.tasks import make_tasks

    for family in ("put_in_drawer", "wipe_counter"):
        task = make_tasks(twin, [family], 1, seed=0)[0]
        if family == "put_in_drawer":
            task["handle"], task["joint"] = f'{task["fixture"]}_drawer0_handle', f'{task["fixture"]}_drawer0'
        env = SimEnv(twin, task, res=32)
        ep = run_episode(env, ScriptedExpert(), task)
        assert ep["success"], (family, ep["detail"])
    assert ep["detail"]["wiped_m"] >= 0.4  # the sponge travelled pressed to the counter, not carried above it


def test_wrist_camera_frames_the_held_object(twin):
    from arrival.sim.policy import ScriptedExpert
    from arrival.sim.tasks import make_tasks

    task = make_tasks(twin, ["counter_to_sink"], 1, seed=0)[0]
    env = SimEnv(twin, task, res=32)
    seg = mujoco.Renderer(env.model, 32, 32)
    seg.enable_segmentation_rendering()
    obj_geoms = {g for g in range(env.model.ngeom) if env.model.body(env.model.geom_bodyid[g]).name == task["object_id"]}
    policy = ScriptedExpert()
    policy.reset(env, task)
    states, closed = [], None
    while not policy.done():
        a = policy.act(env.obs())
        states.append(env.state())
        env.step(a)
        if closed is None and a[7] < 0.02:
            closed = len(states)
        if closed is not None and len(states) == closed + 8:
            seg.update_scene(env.data, "wrist")
            ids = seg.render()[..., 0]
            held = np.isin(ids, list(obj_geoms)).mean()
    seg.close()
    env.close()
    assert held > 0.2  # measured 35% on the hero mug; 0.2 leaves room for the seeded size variation
    assert np.abs(np.array(states)[:, 6]).max() < robot.RANGES[6][1] - 0.3  # joint 7 used to sit at +2.90 at the sink


def test_bowl_is_hollow_and_goes_to_sink(twin):
    from arrival.sim.collect import run_episode
    from arrival.sim.policy import ScriptedExpert
    from arrival.sim.tasks import make_tasks

    task = next(t for t in make_tasks(twin, ["counter_to_sink"], 40, seed=1) if t["object_id"] == "bowl_1")
    env = SimEnv(twin, task, res=32)
    walls = [env.model.geom(i).name for i in range(env.model.ngeom) if env.model.geom(i).name.startswith("bowl_1_wall")]
    assert len(walls) == 8
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"], ep["detail"]


@pytest.mark.parametrize("family,fixture", [("open_door", "counter_a"), ("open_door", "counter_b"),
                                            ("close_door", "counter_a"), ("close_door", "counter_b"),
                                            ("put_in_drawer", "counter_a")])
def test_arm_stays_out_of_drawn_fronts(twin, family, fixture):
    """MAN-2: a physics-checked success must not pass the arm through the fronts the cameras see."""
    from arrival.sim.collect import run_episode
    from arrival.sim.policy import ScriptedExpert

    task = {"family": family, "fixture": fixture, "lighting": "neutral", "people": 0, "seed": 0}
    if family == "put_in_drawer":
        from arrival.sim.tasks import make_tasks

        task = next(t for t in make_tasks(twin, [family], 10, seed=0) if t["fixture"] == fixture)
        task["handle"], task["joint"] = f"{fixture}_drawer0_handle", f"{fixture}_drawer0"
    else:
        task["joint"] = f"{fixture}_door"
    env = SimEnv(twin, task, res=32)
    m, d = env.model, env.data
    arm = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(("link", "hand", "finger"))]
    fixture_ids = tuple(f["id"] for f in twin["fixtures"])
    drawn = [g for g in range(m.ngeom) if m.geom_group[g] == 2 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
             and m.body(m.geom_bodyid[g]).name.startswith(fixture_ids)]
    assert arm and drawn
    deepest = [0.0, None]
    step = env.step

    def checked_step(a):
        step(a)
        for ga in arm:
            for gd in drawn:
                dist = mujoco.mj_geomDistance(m, d, ga, gd, 0.02, None)
                if dist < deepest[0]:
                    deepest[:] = dist, (m.body(m.geom_bodyid[ga]).name, m.body(m.geom_bodyid[gd]).name)

    env.step = checked_step
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"], ep["detail"]
    assert deepest[0] > -0.01, f"{deepest[1]} penetrates {-deepest[0] * 100:.1f} cm"  # contact depth, not a pass-through


def test_vla_requires_server(monkeypatch):
    monkeypatch.delenv("GR00T_POLICY_URL", raising=False)
    with pytest.raises(VLANotConfigured):
        VLAClient()


def test_vlm_planner_falls_back(monkeypatch):
    monkeypatch.delenv("TOKEN_FACTORY_API_KEY", raising=False)
    p = VLMPlanner(log=lambda *_: None)
    plan = p.plan("Put the mug in the sink.", ["mug_1", "bowl_1"], ["sink_1", "counter_a"])
    assert p.source == "rule_based"
    assert plan == {"family": "counter_to_sink", "object": "mug_1", "target": "sink_1"}


def test_rule_plan_knows_every_sim_family():
    from arrival.sim.policy import rule_plan

    objs, fxs = ["mug_1", "sponge_1"], ["sink_1", "counter_a"]
    assert rule_plan("Open the fridge.", objs, fxs)["family"] == "open_fridge"
    assert rule_plan("Pull the fridge door open.", objs, fxs)["family"] == "open_fridge"
    assert rule_plan("Wipe the counter with the sponge.", objs, fxs) == {
        "family": "wipe_counter", "object": "sponge_1", "target": None}
    assert rule_plan("Close the drawer.", objs, fxs)["family"] == "close_drawer"
    assert rule_plan("Close the cabinet door.", objs, fxs)["family"] == "close_door"
    assert rule_plan("Put the mug in the open drawer.", objs, fxs)["family"] == "put_in_drawer"
    assert rule_plan("Open the top drawer.", objs, fxs)["family"] == "open_drawer"


def test_planner_cache_key_is_stable_across_processes():
    import hashlib

    p = VLMPlanner(log=lambda *_: None, cache_path=None)
    key = p._cache_key("Open it.", ["a"], ["b"], research_context="manual says pull")
    assert key.endswith("::tavily=" + hashlib.sha1(b"manual says pull").hexdigest()[:12])  # not the salted hash()


def test_planner_never_rewrites_the_tasks_family(twin, tmp_path, monkeypatch):
    # a planner that answers with a family lacking the task's fields must not break or relabel the episode
    monkeypatch.setattr(VLMPlanner, "plan", lambda self, *a, **k: setattr(self, "source", "stub") or
                        {"family": "counter_to_counter", "object": None, "target": None})
    res = collect(twin, ["open_fridge"], 1, tmp_path, people=0, res=32, planner_name="vlm",
                  cache_path=tmp_path / "c.json", log=lambda *_: None)
    assert list(res["report_data"]["families"]) == ["open_fridge"]


def test_collect_skips_a_task_whose_policy_raises_and_closes_its_env(twin, tmp_path, monkeypatch):
    from arrival.sim.policy import ScriptedExpert

    closed = []
    real = SimEnv.close
    monkeypatch.setattr(SimEnv, "close", lambda self: (closed.append(1), real(self))[1])
    monkeypatch.setattr(ScriptedExpert, "reset", lambda self, env, task: (_ for _ in ()).throw(ValueError("boom")))
    res = collect(twin, ["open_drawer"], 2, tmp_path, people=0, res=32, log=lambda *_: None)
    r = res["report_data"]
    assert closed
    assert (r["skipped_error"], r["attempts"], r["successes"]) == (2, 0, 0)  # skipped tasks are not attempts
    assert [x["stage"] for x in r["skip_reasons"]] == ["rollout", "rollout"] and "boom" in r["skip_reasons"][0]["reason"]
    assert res["info"]["total_episodes"] == 0 and res["info"]["arrival_skipped_error"] == 2  # nothing was written


def test_free_floor_survives_fixtures_only_in_the_far_half(twin):
    import types

    far = max(w["start"][1] for w in twin["walls"])
    t = {**twin, "fixtures": [{**f, "position": [f["position"][0], far, f["position"][2]]} for f in twin["fixtures"]]}
    anchor = t["fixtures"][0]
    (x0, y0, x1, y1), *_ = SimEnv._free_floor(types.SimpleNamespace(twin=t), anchor, (2.0, 1.0, 0.75))
    assert x0 < x1 and y0 < y1


def test_base_pose_leaves_the_task_dict_alone(twin):
    from arrival.sim.tasks import make_tasks

    task = make_tasks(twin, ["counter_to_sink"], 1, people=0, log=lambda *_: None)[0]
    before = dict(task)
    SimEnv(twin, task, res=32).close()
    assert task == before


def test_generated_instructions_do_not_claim_a_template_index(twin):
    from arrival.sim.tasks import make_tasks

    tasks = make_tasks(twin, ["open_drawer"], 4, split="eval", use_generated_instructions=True, log=lambda *_: None)
    assert tasks and {t["wording_index"] for t in tasks} == {-1}


def test_sim_dataset_has_stats_and_shared_video_info(twin, tmp_path):
    res = collect(twin, ["open_drawer"], 1, tmp_path, people=0, res=32, log=lambda *_: None)
    meta = res["dataset"] / "meta"
    stats = json.loads((meta / "stats.json").read_text(), parse_constant=lambda c: pytest.fail(c))
    assert {"observation.state", "action", "timestamp", "next.done"} <= set(stats)
    feat = json.loads((meta / "info.json").read_text())["features"]["observation.images.ego_view"]
    assert feat["info"]["video.height"] == 32 and feat["info"]["video.pix_fmt"] == "yuv420p"


def test_collect_rejects_res_that_mp4_would_resize(twin, tmp_path):
    with pytest.raises(ValueError, match="multiples of 16"):
        collect(twin, ["open_drawer"], 1, tmp_path, people=0, res=40, log=lambda *_: None)


def test_run_episode_closes_env_when_the_policy_raises(twin, monkeypatch):
    from arrival.sim.collect import run_episode
    from arrival.sim.policy import ScriptedExpert

    task = {"family": "open_fridge", "fixture": "fridge_1", "joint": "fridge_1_door", "lighting": "neutral",
            "people": 0, "seed": 0}
    monkeypatch.setattr(ScriptedExpert, "reset", lambda self, env, task: (_ for _ in ()).throw(ValueError("boom")))
    env = SimEnv(twin, task, res=32)
    with pytest.raises(ValueError):
        run_episode(env, ScriptedExpert(), task)
    assert env.renderer is None
