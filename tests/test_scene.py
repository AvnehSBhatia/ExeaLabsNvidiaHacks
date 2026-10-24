"""SIM-3 moving parts and SIM-4 objects: every part is a named joint with a handle site that opens and holds,
new fixtures build from a scan, and every catalog object rests on a counter with a plausible mass."""

import copy
import random

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.catalog.objects import (GRASPABLE, HANDLE_HELD, OBJECT_SPECS, PICKABLE, RIM_HELD, THIN_HELD,  # noqa: E402
                                     object_spec, sample_size)
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


def test_sink_in_yawed_counter_height_storage_reaches_the_scene():
    import math

    base = {"id": "base", "category": "storage", "size": [1.0, 0.6, 0.9],
            "position": [2.0, 1.5, 0.45], "yaw": -math.pi / 2}
    sink = {"id": "sink", "category": "sink", "size": [0.6, 0.5, 0.25],
            "position": [2.0, 1.4, 0.85], "yaw": -math.pi / 2}
    t = compile_twin(_bundle([sink, base]))
    assert not any(f["id"].startswith("counter_synth") for f in t["fixtures"])
    m = _model(t)
    assert m.geom("sink_bottom").pos[0] == pytest.approx(0.1, abs=1e-4)
    assert m.geom("base_carcass") is not None
    assert m.joint("sink_faucet") is not None
    assert {p["joint"] for p in part_names(t)} >= {"base_drawer0", "base_door", "sink_faucet"}


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
    objects, x = [], 0.3
    for cid, (_, size, *_) in OBJECT_SPECS.items():  # laid out in a row, each clear of the last (a pan has a handle)
        reach = max(size[:2]) / 2 + (0.28 if cid == "pan" else 0.0)
        x += max(size[:2]) / 2
        objects.append({"id": f"{cid}_1", "label": cid, "size": list(size),
                        "position": [x, 0.40, 0.92 + size[2] / 2 + 0.003], "yaw": 0.0})
        x += reach + 0.05
    twin = compile_twin(_bundle([{**COUNTER, "size": [x + 0.3, 0.65, 0.92], "position": [(x + 0.3) / 2, 0.40, 0.46]}],
                                objects))
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
    assert all(0.02 < v < 1.5 for v in mass.values()), mass  # the closed laptop is the heavy one, 1.2 kg
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
    assert set(GRASPABLE) == set(PICKABLE) | set(RIM_HELD) | set(HANDLE_HELD) | set(THIN_HELD)
    assert THIN_HELD == ("spatula",) and "laptop" not in GRASPABLE


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


@pytest.mark.parametrize("arm", ["procedural", "menagerie"])
@pytest.mark.parametrize("family, replaces", [("counter_to_sink", "mug_1"), ("wipe_counter", "sponge_1")])
def test_spatula_is_pinched_with_the_pads_off_the_counter(twin, family, replaces, arm):
    """L26: the 1 cm spatula is held between both pads for every closed step, from the pick to the release, and no
    finger touches the counter it lies on (the old body grasp put the pad tips 1 to 3 mm into the top)."""
    from arrival.sim import menagerie
    from arrival.sim.collect import PAD_BELOW_EEF, SimEnv
    from arrival.sim.policy import ScriptedExpert
    from arrival.sim.tasks import make_tasks

    if arm == "menagerie" and not menagerie.available():
        pytest.skip("run python scripts/fetch_menagerie.py")
    t = copy.deepcopy(twin)
    o = next(o for o in t["objects"] if o["id"] == replaces)
    o.update({"id": "spatula_1", "label": "spatula", "catalog_id": "spatula", "size": list(OBJECT_SPECS["spatula"][1])})
    o["position"][2] = 0.92 + o["size"][2] / 2
    t["objects"] = [o]
    task = make_tasks(t, [family], 1, seed=3, people=0, log=lambda *_: None)[0]
    assert task["object_id"] == "spatula_1"
    env = SimEnv(t, task, res=32, arm=arm)
    if arm == "procedural":  # the pad reach read from the model is the hand geometry robot.py builds
        assert env.pad_reach() == pytest.approx(PAD_BELOW_EEF, abs=0.001)
    pol = ScriptedExpert()
    pol.reset(env, task)
    m, d = env.model, env.data
    side = {i: m.body(m.geom_bodyid[i]).name for i in range(m.ngeom)
            if m.body(m.geom_bodyid[i]).name in ("finger_l", "finger_r") and m.geom_contype[i]}
    counter = {i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name == task["source"]}
    sp = m.geom("spatula_1").id
    closed = pinched = 0
    z0, top = float(env.body_pos("spatula_1")[2]), 0.0
    for _ in range(len(pol.plan) + 10):
        a = pol.act(None)
        env.step(a)
        pairs = [(c.geom1, c.geom2) for c in d.contact[:d.ncon]] + [(c.geom2, c.geom1) for c in d.contact[:d.ncon]]
        assert not any(g in side and h in counter for g, h in pairs), "a finger touched the counter"
        if a[7] < 0.005:
            closed += 1
            pinched += {side[g] for g, h in pairs if g in side and h == sp} == {"finger_l", "finger_r"}
        top = max(top, float(env.body_pos("spatula_1")[2]) - z0)
    assert closed > 50 and pinched == closed, (pinched, closed)
    ok, detail = env.success(task)
    env.close()
    assert ok, detail
    if family == "counter_to_sink":
        assert top > 0.2  # carried, not pushed
    else:
        assert detail["wiped_m"] > 1.0  # it followed the hand through all four strokes


def test_overview_camera_stands_behind_a_robot_on_any_wall():
    # a run on another wall used to keep the north-wall camera and film the empty floor behind the robot
    rotated = compile_twin(load_room("rotated_kitchen", "tests/fixtures/scans"))
    xs = [p for w in rotated["walls"] for p in (w["start"][0], w["end"][0])]
    ys = [p for w in rotated["walls"] for p in (w["start"][1], w["end"][1])]
    for yaw in (0.0, 90.0, 180.0):
        base = (sum(xs) / len(xs), sum(ys) / len(ys), 0.75)
        m = mujoco.MjModel.from_xml_string(build_mjcf(rotated, base, yaw))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)  # mj_kinematics leaves the camera frames unset
        cam = d.camera("overview")
        f = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
        look = -cam.xmat.reshape(3, 3)[:2, 2]  # the camera looks along its -z
        assert look @ f / np.linalg.norm(look) > 0.99  # it looks the way the robot faces
        assert (np.array(base[:2]) - cam.xpos[:2]) @ f > 0.5  # from behind the robot
        assert min(xs) < cam.xpos[0] < max(xs) and min(ys) < cam.xpos[1] < max(ys)  # inside the room
