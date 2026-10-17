"""SIM-3 moving parts and SIM-4 objects: every part is a named joint with a handle site that opens and holds,
new fixtures build from a scan, and every catalog object rests on a counter with a plausible mass."""

import copy
import random

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.catalog.objects import GRASPABLE, HANDLE_HELD, OBJECT_SPECS, PICKABLE, RIM_HELD, object_spec, sample_size  # noqa: E402
from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim.scene import DOOR_FLAT, UPPER_Z0, build_mjcf, part_names  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

WALLS = [{"id": "n", "start": [0.0, 0.0], "end": [4.0, 0.0], "height": 2.4, "thickness": 0.02},
         {"id": "e", "start": [4.0, 0.0], "end": [4.0, 3.0], "height": 2.4, "thickness": 0.02},
         {"id": "s", "start": [4.0, 3.0], "end": [0.0, 3.0], "height": 2.4, "thickness": 0.02},
         {"id": "w", "start": [0.0, 3.0], "end": [0.0, 0.0], "height": 2.4, "thickness": 0.02}]
COUNTER = {"id": "c_1", "category": "counter", "size": [2.0, 0.65, 0.92], "position": [1.2, 0.40, 0.46], "yaw": 0.0}


@pytest.fixture(scope="module")
def twin():
    return compile_twin(load_room("hero_kitchen", "scans"))


def _bundle(fixtures, objects=()):
    return {"schema_version": "1", "room_id": "t", "units": "meters", "frame": "z_up",
            "geometry_source": "authored_sample", "blocks_claims": [], "walls": WALLS,
            "fixtures": fixtures, "objects": list(objects)}


def _model(twin, **kw):
    return mujoco.MjModel.from_xml_string(build_mjcf(twin, (1.3, 1.0, 0.75), **kw))


def _settle(m, d, n):
    for _ in range(n):
        mujoco.mj_step(m, d)


def _no_warnings(d):
    return all(int(w) == 0 for w in d.warning.number)


def test_hero_parts_exist_open_and_hold(twin):
    m = _model(twin)
    d = mujoco.MjData(m)
    parts = part_names(twin)
    names = [p["joint"] for p in parts]
    assert {"stove_1_oven_door", "sink_1_faucet", "counter_a_upper_door0", "counter_b_upper_door1"} <= set(names)
    assert sum("upper_door" in n for n in names) == 5  # three doors over counter_a's run, two over counter_b's
    for p in parts:
        j = m.joint(p["joint"])
        want = mujoco.mjtJoint.mjJNT_SLIDE if p["kind"] == "slide" else mujoco.mjtJoint.mjJNT_HINGE
        assert j.type == want and j.range[0] <= p["open"] <= j.range[1], p["joint"]
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        closed = d.site(p["site"]).xpos.copy()
        d.qpos[j.qposadr[0]] = p["open"]
        mujoco.mj_forward(m, d)
        assert np.linalg.norm(d.site(p["site"]).xpos - closed) > 0.03, p["joint"]
        _settle(m, d, 150)
        q = float(d.qpos[j.qposadr[0]])
        assert np.isfinite(d.qpos).all() and j.range[0] - 0.05 <= q <= j.range[1] + 0.05, p["joint"]
    assert _no_warnings(d)


def test_parts_stay_closed_at_rest(twin):
    m = _model(twin)
    d = mujoco.MjData(m)
    _settle(m, d, 300)
    for p in part_names(twin):  # the bottom-hinged oven door must not fall open under gravity
        assert abs(float(d.qpos[m.joint(p["joint"]).qposadr[0]])) < 0.01, p["joint"]
    assert _no_warnings(d)


def test_upper_cabinets_start_above_the_measured_wrist_height_and_skip_tables(twin):
    assert UPPER_Z0 >= 1.55  # the wrist reaches 1.50 m over the back of the counter while dropping into the sink
    t = copy.deepcopy(twin)
    next(f for f in t["fixtures"] if f["id"] == "counter_b")["default_config_name"] = "dining"
    assert not any(p["joint"].startswith("counter_b_upper") for p in part_names(t))
    m = _model(t)
    assert not any("counter_b_upper" in m.joint(i).name for i in range(m.njnt))
    assert any("counter_a_upper" in m.joint(i).name for i in range(m.njnt))


def test_dishwasher_and_microwave_from_scan():
    fixtures = [COUNTER,
                {"id": "dw_1", "category": "dishwasher", "size": [0.6, 0.6, 0.86], "position": [2.1, 0.40, 0.43], "yaw": 0.0},
                {"id": "mw_1", "category": "microwave", "size": [0.5, 0.4, 0.3], "position": [0.6, 0.33, 1.07], "yaw": 0.0}]
    apple = {"id": "apple_1", "label": "apple", "size": [0.076, 0.076, 0.076], "position": [1.4, 0.40, 0.96], "yaw": 0.0}
    twin = compile_twin(_bundle(fixtures, [apple]))
    types = {f["id"]: f["type"] for f in twin["fixtures"]}
    assert types["dw_1"] == "dishwasher" and types["mw_1"] == "microwave"
    m = _model(twin)
    d = mujoco.MjData(m)
    joints = {m.joint(i).name for i in range(m.njnt)}
    assert {"dw_1_door", "mw_1_door", "apple_1"} <= joints
    assert {p["joint"] for p in part_names(twin)} >= {"dw_1_door", "mw_1_door"}
    # the counter run is trimmed where it meets the dishwasher (local span -1.0 .. 0.6)
    assert m.geom("c_1_top").size[0] == pytest.approx(0.8, abs=1e-3)
    # microwave door swings open on its left edge
    mujoco.mj_forward(m, d)
    h0 = d.site("mw_1_door_handle").xpos.copy()
    d.qpos[m.joint("mw_1_door").qposadr[0]] = 1.2
    mujoco.mj_forward(m, d)
    assert np.linalg.norm(d.site("mw_1_door_handle").xpos - h0) > 0.2
    # the dishwasher door swings down to nearly flat and an apple set on it stays there: objects collide with the
    # moving parts (conaffinity 3), not only with the fixtures
    mujoco.mj_resetData(m, d)
    d.qpos[m.joint("dw_1_door").qposadr[0]] = DOOR_FLAT
    mujoco.mj_forward(m, d)
    panel = d.geom("dw_1_door_panel")
    rest = panel.xpos[2] + 0.0075 + 0.038
    a = m.joint("apple_1").qposadr[0]
    d.qpos[a:a + 3] = [panel.xpos[0], panel.xpos[1], rest + 0.01]
    _settle(m, d, 300)
    assert abs(float(d.qpos[a + 2]) - rest) < 0.02 and float(d.qpos[a + 2]) > 0.08
    assert _no_warnings(d)


def test_every_catalog_object_rests_on_the_counter_with_a_plausible_mass():
    objects = []
    for k, (cid, (_, size, *_)) in enumerate(OBJECT_SPECS.items()):
        objects.append({"id": f"{cid}_1", "label": cid, "size": list(size),
                        "position": [0.35 + 0.19 * k, 0.40, 0.92 + size[2] / 2 + 0.003], "yaw": 0.0})
    twin = compile_twin(_bundle([COUNTER], objects))
    assert {o["catalog_id"] for o in twin["objects"]} == set(OBJECT_SPECS)
    m = _model(twin)
    d = mujoco.MjData(m)
    _settle(m, d, 400)
    mass = {}
    for o in objects:
        b = d.body(o["id"])
        z0 = 0.92 + o["size"][2] / 2
        assert abs(float(b.xpos[2]) - z0) < 0.012, (o["id"], b.xpos)
        assert np.linalg.norm(b.xpos[:2] - np.array(o["position"][:2])) < 0.02, o["id"]
        mass[o["label"]] = float(m.body_mass[b.id])
    assert all(0.02 < v < 1.2 for v in mass.values()), mass
    assert mass["pan"] > mass["bottle"] > mass["mug"] > mass["apple"] > mass["sponge"]
    assert mass["mug"] == pytest.approx(0.30, abs=1e-3)
    assert _no_warnings(d)


def test_size_variation_is_seeded_and_bounded():
    for cid, (_, nominal, mass, *_) in OBJECT_SPECS.items():
        assert sample_size(cid, random.Random(5)) == sample_size(cid, random.Random(5))
        for k in range(20):
            s = sample_size(cid, random.Random(k))
            assert all(0.92 * n - 1e-6 <= v <= 1.08 * n + 1e-6 for v, n in zip(s, nominal)), (cid, s)
            if nominal[0] == nominal[1]:
                assert s[0] == s[1]  # round things stay round
        assert object_spec(cid, nominal)["mass"] == pytest.approx(mass, abs=1e-3)
    assert object_spec("mug", [0.08, 0.08, 0.05])["mass"] == pytest.approx(0.15, abs=1e-3)  # mass follows volume
    assert set(PICKABLE) <= set(OBJECT_SPECS) and "bowl" not in PICKABLE and "pan" not in PICKABLE
    assert set(GRASPABLE) == set(PICKABLE) | set(RIM_HELD) | set(HANDLE_HELD) and "spatula" not in GRASPABLE


@pytest.mark.parametrize("cid", ["cup", "apple", "bottle", "plate", "pan"])
def test_new_objects_are_picked_into_the_sink(twin, cid):
    from arrival.sim.collect import SimEnv, run_episode
    from arrival.sim.policy import ScriptedExpert
    from arrival.sim.tasks import make_tasks

    t = copy.deepcopy(twin)
    o = next(o for o in t["objects"] if o["id"] == "mug_1")
    o.update({"id": f"{cid}_1", "label": cid, "catalog_id": cid, "size": list(OBJECT_SPECS[cid][1])})
    o["position"][2] = 0.92 + o["size"][2] / 2
    t["objects"] = [o]
    task = make_tasks(t, ["counter_to_sink"], 1, seed=2, people=0, log=lambda *_: None)[0]
    assert task["object_id"] == f"{cid}_1"
    ep = run_episode(SimEnv(t, task, res=32), ScriptedExpert(), task)
    assert ep["success"], ep["detail"]


def _wide_object_twin(twin, cid):
    t = copy.deepcopy(twin)
    o = next(o for o in t["objects"] if o["id"] == "mug_1")
    o.update({"id": f"{cid}_1", "label": cid, "catalog_id": cid, "size": list(OBJECT_SPECS[cid][1])})
    o["position"][2] = 0.92 + o["size"][2] / 2
    t["objects"] = [o]
    return t


def test_plate_is_a_deep_dish_the_pads_can_pinch(twin):
    from arrival.sim.scene import PLATE_FLOOR, PLATE_RIM

    m = _model(_wide_object_twin(twin, "plate"))
    walls = [m.geom(i) for i in range(m.ngeom) if m.geom(i).name.startswith("plate_1_wall")]
    assert len(walls) == 8 and all(abs(2 * w.size[1] - PLATE_RIM) < 1e-6 for w in walls)
    floor = m.geom("plate_1")
    assert abs(2 * floor.size[1] - PLATE_FLOOR) < 1e-6 and PLATE_RIM > 0.012  # thicker than the closed pads' 12 mm gap


def test_pan_is_held_by_its_handle_only(twin):
    """While the gripper is closed, the fingers touch the pan's handle and nothing else on it: the pan is carried by
    the handle (hanging from it, the clamp cannot hold 0.9 kg level), not knocked or scooped by the ring."""
    from arrival.sim.collect import SimEnv
    from arrival.sim.policy import ScriptedExpert
    from arrival.sim.tasks import make_tasks

    t = _wide_object_twin(twin, "pan")
    task = make_tasks(t, ["counter_to_sink"], 1, seed=2, people=0, log=lambda *_: None)[0]
    env = SimEnv(t, task, res=32)
    pol = ScriptedExpert()
    pol.reset(env, task)
    m, d = env.model, env.data
    pan = {i: m.geom(i).name for i in range(m.ngeom) if m.geom(i).name.startswith("pan_1")}
    fingers = {i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name in ("finger_l", "finger_r")}
    touched = set()
    for _ in range(len(pol.plan) + 10):
        a = pol.act(None)
        env.step(a)
        if a[7] < 0.005:
            for c in d.contact[:d.ncon]:
                if c.geom1 in pan and c.geom2 in fingers or c.geom2 in pan and c.geom1 in fingers:
                    touched.add(pan[c.geom1 if c.geom1 in pan else c.geom2])
    assert touched == {"pan_1_handle"}, touched
    assert env.success(task)[0]
