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
