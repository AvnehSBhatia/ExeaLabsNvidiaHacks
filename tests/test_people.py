import copy

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim import robot  # noqa: E402
from arrival.sim.collect import SimEnv  # noqa: E402
from arrival.sim.people import HEIGHT, Walkers, person_spec  # noqa: E402
from arrival.sim.scene import build_mjcf  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

TASK = {"family": "counter_to_sink", "object_id": "mug_1", "source": "counter_a", "target": "sink_1",
        "target_x": 1.6, "start": [1.0, 0.35, 0.97], "lighting": "neutral", "people": 2, "seed": 3}
HOLD = np.array(list(robot.HOME) + [robot.FINGER_OPEN])


@pytest.fixture(scope="module")
def twin():
    return compile_twin(load_room("hero_kitchen", "scans"))


def test_people_are_seeded_and_varied():
    assert person_spec(5, 2) == person_spec(5, 2)
    specs = [person_spec(5, i) for i in range(12)]
    heights = [s["height"] for s in specs]
    assert all(HEIGHT[0] <= h <= HEIGHT[1] for h in heights) and max(heights) - min(heights) > 0.1
    assert len({s["shirt"] for s in specs}) > 2 and len({s["skin"] for s in specs}) > 1
    assert person_spec(6, 2) != person_spec(5, 2)
    floor = (0.4, 1.65, 3.6, 2.7)
    a, b = Walkers(3, floor, seed=7), Walkers(3, floor, seed=7)
    assert all(np.array_equal(p, q) for p, q in zip(a.paths, b.paths))


def _thigh_x(env, i):
    return float(env.model.geom_pos[env.model.geom(f"person_{i}_thigh_r").id][0])


def test_limbs_swing_while_walking_and_hang_when_standing(twin):
    env = SimEnv(twin, TASK, res=32)
    m = env.model
    assert m.geom(f"person_0_torso").size[0] > m.geom(f"person_0_thigh_r").size[0]  # torso stays the widest geom
    xs = []
    for _ in range(12):
        env.step(HOLD)
        xs.append(_thigh_x(env, 0))
    assert max(abs(x) for x in xs) > 0.05 and env.walkers.amp[0] > 0.5
    r, l = m.geom_pos[m.geom("person_0_thigh_r").id][0], m.geom_pos[m.geom("person_0_thigh_l").id][0]
    assert r * l < 0  # legs alternate
    ua = m.geom_pos[m.geom("person_0_uarm_r").id][0]
    assert ua * r < 0  # the arm swings against the leg on its side
    env.walkers.speed = 0.0
    for _ in range(30):
        env.step(HOLD)
    assert env.walkers.amp[0] == 0.0 and abs(_thigh_x(env, 0)) < 1e-9
    assert env.data.mocap_pos[0][2] == 0.0


def _inside(twin, xy, z):
    for f in twin["fixtures"]:
        yaw = f.get("yaw", 0.0)
        dx, dy = xy[0] - f["position"][0], xy[1] - f["position"][1]
        lx = np.cos(yaw) * dx + np.sin(yaw) * dy
        ly = -np.sin(yaw) * dx + np.cos(yaw) * dy
        if abs(lx) < f["size"][0] / 2 and abs(ly) < f["size"][1] / 2 and z < f["position"][2] + f["size"][2] / 2:
            return f["id"]
    return None


def test_people_never_clip_through_fixtures(twin):
    t = copy.deepcopy(twin)
    island = {**copy.deepcopy(t["fixtures"][2]), "id": "island_1", "position": [2.0, 2.0, 0.46], "size": [0.6, 0.5, 0.92]}
    t["fixtures"].append(island)
    m = mujoco.MjModel.from_xml_string(build_mjcf(t, (1.3, 1.0, 0.75), n_people=3, people_seed=11))
    d = mujoco.MjData(m)
    walkers = Walkers(3, (0.4, 1.65, 3.6, 2.7), seed=11, fixtures=t["fixtures"])
    geoms = [g for g in range(m.ngeom) if m.geom(g).name.startswith("person_")]
    near_island = False
    for _ in range(600):  # 30 s of walking
        xy = walkers.step(m, d, 0.05)
        mujoco.mj_kinematics(m, d)
        for g in geoms:
            c, half = d.geom_xpos[g], m.geom_size[g][1]
            axis = d.geom_xmat[g].reshape(3, 3)[:, 2] * (half if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)
            for p in (c - axis, c, c + axis):
                hit = _inside(t, p[:2], p[2])
                assert hit is None, f"{m.geom(g).name} inside {hit} at {np.round(p, 3)}"
        for x, y in xy:
            assert 0.0 < x < 4.0 and 0.0 < y < 3.0
            near_island = near_island or np.hypot(x - 2.0, y - 2.0) < 1.0
    assert near_island  # the paths do go round the island, not just away from it
    assert all(len({tuple(np.round(p, 3)) for p in path}) >= 4 for path in walkers.paths)  # via corners repeat


def test_feet_stay_out_of_the_counter_at_the_stop(twin):
    """With the robot base given, the loop stops 0.35 m from the counter front: the swinging toe reaches further
    than that at full stride, so the walker has to arrive with the swing faded."""
    base, floor = (2.6, 1.0, 0.75), (0.4, 1.65, 3.6, 2.7)
    for seed in (2, 5, 7, 16, 18):
        m = mujoco.MjModel.from_xml_string(build_mjcf(twin, base, n_people=2, people_seed=seed))
        d = mujoco.MjData(m)
        walkers = Walkers(2, floor, seed=seed, base_xy=base[:2], front_y=0.75, fixtures=twin["fixtures"])
        feet = [g for g in range(m.ngeom) if "_foot_" in m.geom(g).name]
        (cx, _, _), (sx, _, _) = next(f for f in twin["fixtures"] if f["id"] == "counter_a")["position"],             next(f for f in twin["fixtures"] if f["id"] == "counter_a")["size"]
        stopped = set()
        for _ in range(400):
            walkers.step(m, d, 0.05)
            mujoco.mj_kinematics(m, d)
            for g in feet:
                c, half = d.geom_xpos[g], m.geom_size[g][1]
                axis = d.geom_xmat[g].reshape(3, 3)[:, 2] * half
                for p in (c - axis, c + axis):
                    assert _inside(twin, p[:2], p[2]) is None, (seed, m.geom(g).name, np.round(p, 3))
            for i in range(2):  # a pause starting in front of counter_a; the fridge visit's pauses sit left of it or deeper
                x, y = walkers.pos[i]
                if (walkers.wait[i] > 0 and abs(walkers.wait[i] - walkers.total[i]) < 1e-9
                        and abs(x - cx) < sx / 2 and y < 1.2):
                    stopped.add(i)
        assert stopped == {0, 1}, (seed, stopped)


def _door_geoms(m):
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("fridge_1_door").id]


def test_person_opens_and_closes_the_fridge(twin):
    base = (2.6, 1.0, 0.75)
    m = mujoco.MjModel.from_xml_string(build_mjcf(twin, base, n_people=2, people_seed=3))
    d = mujoco.MjData(m)
    walkers = Walkers(2, (0.4, 1.65, 3.6, 2.7), seed=3, base_xy=base[:2], front_y=0.75, fixtures=twin["fixtures"],
                      fridge_visit=True)
    assert walkers.visit is not None
    who = walkers.visit["person"]
    door, qadr = _door_geoms(m), int(m.joint("fridge_1_door").qposadr[0])
    people = [g for g in range(m.ngeom) if m.geom(g).name.startswith("person_")]
    hand = m.geom(f"person_{who}_farm_r").id
    site = m.site("fridge_1_door_handle").id
    qs, held, on_bar = [], False, False
    for _ in range(240):  # 12 s: open, look, close, walk away
        walkers.step(m, d, 0.05)
        mujoco.mj_kinematics(m, d)
        q = float(d.qpos[qadr])
        qs.append(q)
        held = held or (q > 0.05 and bool(walkers.driven_geoms))
        if 0.05 < q < 0.2 and len(qs) < 60:  # pulling: the hand is on the bar, 9 cm out from the panel
            tip = d.geom_xpos[hand] + d.geom_xmat[hand].reshape(3, 3)[:, 2] * m.geom_size[hand][1]
            on_bar = on_bar or np.linalg.norm(tip - d.site_xpos[site]) < 0.12
        for g in people:
            dist = min(mujoco.mj_geomDistance(m, d, g, h, 1.0, None) for h in door)
            assert dist >= (-0.01 if g == hand else 0.0), (m.geom(g).name, round(q, 3), round(dist, 4))
            c, half = d.geom_xpos[g], m.geom_size[g][1]
            axis = d.geom_xmat[g].reshape(3, 3)[:, 2] * (half if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)
            for p in (c - axis, c, c + axis):
                assert _inside(twin, p[:2], p[2]) is None, (m.geom(g).name, np.round(p, 3))
    assert max(qs) > 1.0 and abs(max(qs) - walkers.visit["q_open"]) < 1e-6  # opened to the planned angle (57 deg plus)
    assert qs[-1] == 0.0 and not walkers.driven_geoms  # and shut again
    assert max(abs(b - a) for a, b in zip(qs, qs[1:])) < 0.12  # a swing, not a jump
    assert held and on_bar


def test_fridge_visit_is_seeded_and_optional(twin):
    floor, fx = (0.4, 1.65, 3.6, 2.7), twin["fixtures"]
    a = Walkers(2, floor, seed=3, base_xy=(2.6, 1.0), front_y=0.75, fixtures=fx)
    b = Walkers(2, floor, seed=3, base_xy=(2.6, 1.0), front_y=0.75, fixtures=fx)
    assert (a.visit is None) == (b.visit is None)
    if a.visit:
        assert a.visit["person"] == b.visit["person"] and a.visit["q_open"] == b.visit["q_open"]
    none = Walkers(2, floor, seed=3, base_xy=(2.6, 1.0), front_y=0.75, fixtures=fx, fridge_visit=False)
    assert none.visit is None
    visits = sum(Walkers(2, floor, seed=s, base_xy=(2.6, 1.0), front_y=0.75, fixtures=fx).visit is not None
                 for s in range(20))
    assert 5 <= visits <= 18  # P_VISIT is 0.6 and the plan can decline
    near = Walkers(2, floor, seed=3, base_xy=(1.2, 1.0), front_y=0.75, fixtures=fx, fridge_visit=True)
    assert near.visit is None  # the base is inside the door's swing plus CLEAR: nobody opens it


def test_episode_with_a_fridge_visit_still_succeeds(twin):
    from arrival.sim.collect import run_episode
    from arrival.sim.policy import ScriptedExpert

    for seed in range(12):
        task = {"family": "open_drawer", "fixture": "counter_b", "handle": "counter_b_drawer0_handle",
                "joint": "counter_b_drawer0", "lighting": "neutral", "people": 2, "seed": seed}
        env = SimEnv(twin, task, res=32)
        if env.walkers.visit is not None:
            break
    else:
        pytest.fail("no seed in 0..11 plans a fridge visit for counter_b")
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["fridge_visit"] and env.walkers.door_max > 1.0
    assert ep["success"], ep["detail"]
    assert not ep["safety"]["human_contact"] and not ep["safety"]["door_contact"]
    assert "door_contact" in ep["safety"]


def test_yaw_does_not_spike_on_the_frame_a_walker_arrives_at_a_pause(twin):
    m = mujoco.MjModel.from_xml_string(build_mjcf(twin, (2.6, 1.0, 0.75), n_people=2, people_seed=3))
    d = mujoco.MjData(m)
    walkers = Walkers(2, (0.4, 1.65, 3.6, 2.7), seed=3, base_xy=(2.6, 1.0), front_y=0.75,
                      fixtures=twin["fixtures"], fridge_visit=True)
    for _ in range(240):
        walkers.step(m, d, 0.05)
        for i in range(2):
            if walkers.wait[i] > 0:  # stopped, including the frame it arrives: already facing the stop yaw
                assert walkers.yaw[i] == walkers._pause_yaw(i), i


def test_summary_is_strict_json_before_the_first_update(twin):
    import json

    from arrival.sim.safety import SafetyMonitor

    m = mujoco.MjModel.from_xml_string(build_mjcf(twin, (2.6, 1.0, 0.75), n_people=2, people_seed=3))
    s = SafetyMonitor(m).summary()
    assert s["min_human_distance_m"] is None and s["people"] == 2
    json.dumps(s, allow_nan=False)


def test_driven_door_contact_is_not_an_arm_contact(twin):
    from types import SimpleNamespace

    from arrival.sim.safety import SafetyMonitor

    m = mujoco.MjModel.from_xml_string(build_mjcf(twin, (2.6, 1.0, 0.75), n_people=1, people_seed=3))
    mon = SafetyMonitor(m)
    door = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("fridge_1_door").id)
    link = next(iter(mon.link_geoms))
    data = SimpleNamespace(ncon=1, contact=[SimpleNamespace(geom1=link, geom2=door, dist=-0.01)])
    mon._arm_contacts(m, data)
    assert mon.arm_steps == 1
    mon.reset()
    mon._arm_contacts(m, data, {door})
    assert mon.arm_steps == 0


def test_walker_yields_to_the_hand_and_resumes(twin):
    from arrival.sim.safety import YIELD, SafetyMonitor, yield_factor

    assert yield_factor(YIELD[0]) == 0.0 and yield_factor(YIELD[1]) == 1.0 and 0 < yield_factor(sum(YIELD) / 2) < 1
    env = SimEnv(twin, TASK, res=32)
    w, mon = env.walkers, SafetyMonitor(env.model)
    p0 = w.pos[0].copy()
    for _ in range(20):
        w.step(env.model, env.data, 0.05, lambda p, r: 0.1)  # something sits at the walker's elbow
    assert np.allclose(w.pos[0], p0) and w.amp[0] < 0.1
    for _ in range(20):
        w.step(env.model, env.data, 0.05, lambda p, r: 1.0)  # it moves away
    assert np.linalg.norm(w.pos[0] - p0) > 0.3 and mon.yield_geoms


def test_walker_yields_to_arm_links(twin):
    """Yield uses the closest of the base and the arm-link bodies: a walker stands when a stretched
    link is near even if the hand and base sit farther, and walks when the arm is far (same as before)."""
    from arrival.sim.safety import YIELD, SafetyMonitor

    base = (2.6, 1.0, 0.75)
    m = mujoco.MjModel.from_xml_string(build_mjcf(twin, base, n_people=1, people_seed=3))
    d = mujoco.MjData(m)
    mon = SafetyMonitor(m)
    old = [g for g in mon.geoms if m.body(m.geom_bodyid[g]).name in ("robot_base", "hand", "finger_l", "finger_r")]
    for j, v in zip(robot.ARM_JOINTS, (1.5, 1.0, 0.0, -0.2, 0.0, 0.2, 0.0)):
        d.qpos[int(m.joint(j).qposadr[0])] = v
    mujoco.mj_forward(m, d)
    w = Walkers(1, (0.4, 1.65, 3.6, 2.7), seed=3, base_xy=base[:2], front_y=0.75,
                fixtures=twin["fixtures"], fridge_visit=False)
    r, far, give = float(w.specs[0]["torso_r"]), np.array([3.4, 2.5]), mon.yield_clearance(d)
    assert give(far, r) == pytest.approx(mon.clearance(d, far, r, old)) and give(far, r) > YIELD[1]
    near = d.body("link4").xpos[:2].copy()
    assert give(near, r) < mon.clearance(d, near, r, old) and give(near, r) < YIELD[0]
    w.wait[0] = 0.0
    w.pos[0] = near
    p0 = w.pos[0].copy()
    for _ in range(20):
        w.step(m, d, 0.05, give)
    assert np.allclose(w.pos[0], p0)
    w.pos[0] = far
    p1 = w.pos[0].copy()
    for _ in range(20):
        w.step(m, d, 0.05, give)
    assert np.linalg.norm(w.pos[0] - p1) > 0.3


def test_counter_pause_is_held_while_the_hand_is_close(twin):
    w = Walkers(1, (0.4, 1.65, 3.6, 2.7), seed=3, base_xy=(2.6, 1.0), front_y=0.75, fixtures=twin["fixtures"],
                fridge_visit=False)
    m = mujoco.MjModel.from_xml_string(build_mjcf(twin, (2.6, 1.0, 0.75), n_people=1, people_seed=3))
    d = mujoco.MjData(m)
    for _ in range(2000):
        w.step(m, d, 0.05)
        if w.wait[0] > 0.5 and w.pauses[0][w.at[0]] > 0:
            break
    before = w.wait[0]
    for _ in range(10):
        w.step(m, d, 0.05, lambda p, r: 0.0)
    assert w.wait[0] == before
    w.step(m, d, 0.05, lambda p, r: 1.0)
    assert w.wait[0] < before


def test_heads_and_arms_clear_wall_cabinets_and_the_fridge_door(twin):
    from arrival.sim.safety import overhead_clearance

    m = mujoco.MjModel.from_xml_string(build_mjcf(twin, (2.6, 1.0, 0.75), n_people=2, people_seed=3))
    d = mujoco.MjData(m)
    w = Walkers(2, (0.4, 1.65, 3.6, 2.7), seed=3, base_xy=(2.6, 1.0), front_y=0.75, fixtures=twin["fixtures"],
                fridge_visit=True)
    lo = float("inf")
    for _ in range(300):
        w.step(m, d, 0.05)
        mujoco.mj_kinematics(m, d)
        lo = min(lo, overhead_clearance(m, d))
    assert lo > 0.0, lo


def test_open_fridge_with_walkers_succeeds_without_contact(twin):
    from arrival.sim.collect import run_episode
    from arrival.sim.policy import ScriptedExpert

    for seed in (3, 5):
        task = {"family": "open_fridge", "fixture": "fridge_1", "joint": "fridge_1_door", "lighting": "neutral",
                "people": 2, "seed": seed}
        env = SimEnv(twin, task, res=32)
        ep = run_episode(env, ScriptedExpert(), task)
        assert ep["success"], (seed, ep["detail"])
        assert not ep["safety"]["human_contact"] and not ep["safety"]["door_contact"], seed


def test_walkers_keep_out_of_the_door_the_robot_opens(twin):
    """`keep` discs (the swing of the cabinet door the arm works) hold no stop, free point or via point of a walker's
    loop; without them the same seeds do put a waypoint there (open_door lost episodes to a walker standing at the
    counter stop in front of counter_a_door)."""
    hinge, radius, base, floor = np.array([2.18, 0.72]), 0.75, (1.455, 1.375), (0.4, 1.3, 3.6, 2.7)
    inside = lambda w: [p for path in w.paths for p in path if np.linalg.norm(p - hinge) < radius]
    without = 0
    for seed in range(40):
        kw = dict(seed=seed, base_xy=base, front_y=1.024, fixtures=twin["fixtures"], fridge_visit=False)
        without += bool(inside(Walkers(2, floor, **kw)))
        assert not inside(Walkers(2, floor, keep=[(hinge, radius)], **kw)), seed
    assert without > 0


def test_people_never_phase_through_each_other(twin):
    """Multiple walkers maintain interpersonal separation (min dist >= 0.40 m) across all frames."""
    floor = (0.4, 1.65, 3.6, 2.7)
    base = (2.6, 1.0, 0.75)
    for n in (2, 3):
        for seed in (2, 5, 7, 16, 18):
            m = mujoco.MjModel.from_xml_string(build_mjcf(twin, base, n_people=n, people_seed=seed))
            d = mujoco.MjData(m)
            w = Walkers(n, floor, seed=seed, base_xy=base[:2], front_y=0.75, fixtures=twin["fixtures"])
            for _ in range(300):
                w.step(m, d, 0.05)
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = float(np.linalg.norm(w.pos[i] - w.pos[j]))
                        assert dist >= 0.40, f"Walkers {i} and {j} phased through each other: dist={dist:.3f} m (seed {seed}, n={n})"
