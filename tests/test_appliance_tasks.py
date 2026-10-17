"""MAN-5 appliance families: the scripted expert opens the oven and dishwasher doors, lifts the faucet lever, and
loads and unloads the dishwasher's upper rack; the dishwasher is hollow with a rack that rolls out over the door."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.sim.collect import DOOR_FLAT, FAUCET_ON, FLAP_OPEN, RACK_OUT, SimEnv, run_episode  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.scene import RACK_Z, build_mjcf, part_names  # noqa: E402
from arrival.sim.tasks import SIM_FAMILIES, make_tasks  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

WALLS = [{"id": "n", "start": [0.0, 0.0], "end": [4.0, 0.0], "height": 2.4, "thickness": 0.02},
         {"id": "e", "start": [4.0, 0.0], "end": [4.0, 3.0], "height": 2.4, "thickness": 0.02},
         {"id": "s", "start": [4.0, 3.0], "end": [0.0, 3.0], "height": 2.4, "thickness": 0.02},
         {"id": "w", "start": [0.0, 3.0], "end": [0.0, 0.0], "height": 2.4, "thickness": 0.02}]
JOINTS = {"open_oven": "{f}_oven_door", "open_dishwasher": "{f}_door", "load_dishwasher": "{f}_door",
          "unload_dishwasher": "{f}_door", "turn_on_faucet": "{f}_faucet"}


@pytest.fixture(scope="module")
def dishwasher_twin():
    """A counter with a sink, a cup on it, and a dishwasher at the counter's end (the hero scan has none)."""
    fixtures = [{"id": "c_1", "category": "counter", "size": [2.0, 0.65, 0.92], "position": [1.0, 0.40, 0.46], "yaw": 0.0},
                {"id": "sink_1", "category": "sink", "size": [0.6, 0.5, 0.25], "position": [0.8, 0.4, 0.85], "yaw": 0.0},
                {"id": "dw_1", "category": "dishwasher", "size": [0.6, 0.6, 0.86], "position": [2.3, 0.40, 0.43], "yaw": 0.0}]
    cup = {"id": "cup_1", "label": "cup", "size": [0.07, 0.07, 0.09], "position": [1.7, 0.40, 0.965], "yaw": 0.0}
    return compile_twin({"schema_version": "1", "room_id": "t", "units": "meters", "frame": "z_up",
                         "geometry_source": "authored_sample", "blocks_claims": [], "walls": WALLS,
                         "fixtures": fixtures, "objects": [cup]})


def _task(twin, family, seed=1, people=0):
    t = make_tasks(twin, [family], 1, seed=seed, people=people, log=lambda *_: None)[0]
    t["joint"] = JOINTS[family].format(f=t["fixture"])
    t["handle"] = t["joint"] + "_handle"
    return t


def test_families_are_registered_and_pooled(hero_twin, dishwasher_twin):
    assert {"open_oven", "open_dishwasher", "turn_on_faucet", "load_dishwasher", "unload_dishwasher"} <= set(SIM_FAMILIES)
    hero = make_tasks(hero_twin, ["open_oven", "turn_on_faucet", "open_dishwasher"], 6, seed=0, log=lambda *_: None)
    assert {t["family"] for t in hero} == {"open_oven", "turn_on_faucet"}  # no dishwasher in the hero scan
    assert {t["fixture"] for t in hero} == {"stove_1", "sink_1"}
    dw = make_tasks(dishwasher_twin, ["open_dishwasher", "load_dishwasher", "unload_dishwasher"], 6, seed=0,
                    log=lambda *_: None)
    assert {t["family"] for t in dw} == {"open_dishwasher", "load_dishwasher", "unload_dishwasher"}
    assert all(t["object_id"] == "cup_1" and t["source"] == "c_1" for t in dw if "object_id" in t)


def test_dishwasher_is_hollow_with_a_rack_that_rolls_out(dishwasher_twin):
    m = mujoco.MjModel.from_xml_string(build_mjcf(dishwasher_twin, (1.3, 1.2, 0.75)))
    d = mujoco.MjData(m)
    names = {m.geom(i).name for i in range(m.ngeom)}
    assert {"dw_1_bottom", "dw_1_top", "dw_1_back", "dw_1_left", "dw_1_right", "dw_1_rack_tray"} <= names
    assert "dw_1_box" not in names
    rack = next(p for p in part_names(dishwasher_twin) if p["joint"] == "dw_1_rack")
    assert rack["kind"] == "slide" and rack["open"] == RACK_OUT
    mujoco.mj_forward(m, d)
    y0 = d.geom("dw_1_rack_tray").xpos[1]
    d.qpos[m.joint("dw_1_rack").qposadr[0]] = RACK_OUT
    mujoco.mj_forward(m, d)
    assert d.geom("dw_1_rack_tray").xpos[1] - y0 == pytest.approx(RACK_OUT, abs=1e-6)
    assert d.geom("dw_1_rack_tray").xpos[2] == pytest.approx(RACK_Z, abs=1e-6)


@pytest.mark.parametrize("family", ["open_oven", "turn_on_faucet"])
def test_hero_kitchen_oven_and_faucet(hero_twin, family):
    task = _task(hero_twin, family)
    env = SimEnv(hero_twin, task, res=32)
    assert float(env.data.joint(task["joint"]).qpos[0]) < 0.01  # starts closed / off
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"], ep["detail"]
    assert ep["detail"]["qpos"] > (FLAP_OPEN if family == "open_oven" else FAUCET_ON)
    assert ep["ik_error"] < 0.03
    a = np.array(list(env.data.ctrl[:7]) + [env.data.ctrl[7]])
    for _ in range(400):  # 20 s with the arm parked: the door stays down and the lever stays up
        env.step(a)
    assert float(env.data.joint(task["joint"]).qpos[0]) > (FLAP_OPEN if family == "open_oven" else FAUCET_ON)


def test_faucet_lever_is_pushed_without_driving_it_through_its_stop(hero_twin):
    task = _task(hero_twin, "turn_on_faucet")
    env = SimEnv(hero_twin, task, res=32)
    pol = ScriptedExpert()
    pol.reset(env, task)
    qs = []
    for _ in range(len(pol.plan) + 10):
        env.step(pol.act(None))
        qs.append(float(env.data.joint(task["joint"]).qpos[0]))
    hi = float(env.model.jnt_range[env.model.joint(task["joint"]).id][1])
    assert max(qs) < hi + 0.05, max(qs)  # never forced past the 1.2 rad range by the fingers
    assert FAUCET_ON < qs[-1] <= hi + 0.01


def test_oven_door_is_pulled_before_it_falls(hero_twin):
    """The expert pulls the door through the first part of its arc and only then lets gravity finish; the door
    must be moving with the hand while the gripper is closed, not knocked open."""
    task = _task(hero_twin, "open_oven")
    env = SimEnv(hero_twin, task, res=32)
    pol = ScriptedExpert()
    pol.reset(env, task)
    closed_q = []
    for _ in range(len(pol.plan)):
        a = pol.act(None)
        env.step(a)
        if a[7] < 0.005:
            closed_q.append(float(env.data.joint(task["joint"]).qpos[0]))
    assert closed_q and max(closed_q) > 0.4  # pulled well past 20 deg while held
    for _ in range(20):  # the door is still falling when the plan ends; run_episode's tail covers this
        env.step(a)
    assert float(env.data.joint(task["joint"]).qpos[0]) > FLAP_OPEN


@pytest.mark.parametrize("family", ["open_dishwasher", "load_dishwasher", "unload_dishwasher"])
def test_dishwasher_families(dishwasher_twin, family):
    task = _task(dishwasher_twin, family, people=2)
    env = SimEnv(dishwasher_twin, task, res=32)
    if family != "open_dishwasher":  # door flat and rack out at the start, as a person would leave it
        assert float(env.data.joint("dw_1_door").qpos[0]) == pytest.approx(DOOR_FLAT, abs=0.01)  # settled on its stop
        assert float(env.data.joint("dw_1_rack").qpos[0]) == pytest.approx(RACK_OUT, abs=0.01)
    if family == "unload_dishwasher":
        p = env.body_pos("cup_1")
        assert abs(p[2] - (RACK_Z + 0.01 + 0.045)) < 0.01 and p[1] > 0.7  # resting in the rolled-out part of the rack
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"], ep["detail"]
    if family == "unload_dishwasher":
        assert abs(ep["detail"]["object_pos"][2] - 0.965) < 0.01  # back on the counter top
    assert ep["ik_error"] < 0.07
    assert all(int(w) == 0 for w in env.data.warning.number)


def test_walkers_and_people_stream_still_work_with_the_rack_out(dishwasher_twin):
    task = _task(dishwasher_twin, "load_dishwasher", people=2)
    env = SimEnv(dishwasher_twin, task, res=32)
    env.step(np.array(list(env.data.ctrl[:7]) + [0.05]))
    assert len(env.people_xy) == 2
