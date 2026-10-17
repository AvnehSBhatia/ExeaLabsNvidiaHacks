import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim import menagerie, robot  # noqa: E402
from arrival.sim.collect import SimEnv, run_episode  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.scene import build_mjcf  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

needs_asset = pytest.mark.skipif(not menagerie.available(), reason="run python scripts/fetch_menagerie.py")
SINK_TASK = {"family": "counter_to_sink", "object_id": "mug_1", "source": "counter_a", "target": "sink_1",
             "target_x": 1.6, "start": [1.0, 0.35, 0.97], "lighting": "neutral", "people": 0, "seed": 0}


@pytest.fixture(scope="module")
def twin():
    return compile_twin(load_room("hero_kitchen", "scans"))


def test_procedural_arm_is_the_default_and_unknown_arms_are_rejected(twin):
    xml = build_mjcf(twin, (1.3, 1.0, 0.75))
    assert xml == build_mjcf(twin, (1.3, 1.0, 0.75), arm="procedural")
    assert "meshdir" not in xml and "panda_" not in xml
    with pytest.raises(ValueError):
        build_mjcf(twin, (1.3, 1.0, 0.75), arm="ur5")


def test_missing_asset_is_a_clear_error(monkeypatch, twin):
    monkeypatch.setattr(menagerie, "PANDA_DIR", menagerie.PANDA_DIR / "nowhere")
    with pytest.raises(FileNotFoundError, match="fetch_menagerie"):
        build_mjcf(twin, (1.3, 1.0, 0.75), arm="menagerie")


@needs_asset
def test_menagerie_arm_has_every_name_the_sim_relies_on(twin):
    m = mujoco.MjModel.from_xml_string(build_mjcf(twin, (1.3, 1.0, 0.75), n_people=1, arm="menagerie"))
    joints = {m.joint(i).name for i in range(m.njnt)}
    assert set(robot.ARM_JOINTS) | {"finger_l", "finger_r", "counter_a_drawer0", "mug_1"} <= joints
    bodies = {m.body(i).name for i in range(m.nbody)}
    assert {"robot_base", "link1", "link7", "hand", "finger_l", "finger_r"} <= bodies
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "eef") >= 0
    assert {m.camera(i).name for i in range(m.ncam)} >= {"ego_view", "wrist", "overview"}
    assert [m.actuator(i).name for i in range(m.nu)] == [f"a{i}" for i in range(1, 8)] + ["g_l", "g_r"]
    assert m.nmesh > 50
    for j, (lo, hi) in zip(robot.ARM_JOINTS, robot.RANGES):
        assert abs(m.joint(j).range[0] - lo) < 0.01 and abs(m.joint(j).range[1] - hi) < 0.01
    assert list(m.joint("finger_l").range) == [0, menagerie.FINGER_OPEN]
    assert m.actuator_ctrlrange[m.actuator("g_l").id][1] == menagerie.FINGER_OPEN


@needs_asset
def test_menagerie_collision_geoms_use_the_robot_classes(twin):
    m = mujoco.MjModel.from_xml_string(build_mjcf(twin, (1.3, 1.0, 0.75), arm="menagerie"))
    arm_bodies = {m.body(b).id for b in ("link1", "link4", "link7", "hand", "finger_l", "finger_r")}
    classes = {(int(m.geom_contype[g]), int(m.geom_conaffinity[g]), int(m.geom_group[g]))
               for g in range(m.ngeom) if m.geom_bodyid[g] in arm_bodies}
    assert classes == {(2, 1, 3), (0, 0, 2)}  # hidden collision meshes in the robot class, visual meshes only


@needs_asset
def test_menagerie_arm_picks_the_mug_and_opens_the_drawer(twin):
    env = SimEnv(twin, dict(SINK_TASK), res=32, arm="menagerie")
    ep = run_episode(env, ScriptedExpert(), dict(SINK_TASK))
    assert ep["success"], ep["detail"]
    s = np.array(ep["states"])
    assert s.shape[1] == 8 and s[:, 7].min() < 0.036 < s[:, 7].max() <= menagerie.FINGER_OPEN + 5e-4  # soft joint limit
    task = {"family": "open_drawer", "fixture": "counter_a", "handle": "counter_a_drawer0_handle",
            "joint": "counter_a_drawer0", "lighting": "neutral", "people": 0, "seed": 0}
    env = SimEnv(twin, task, res=32, arm="menagerie")
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"] and ep["detail"]["qpos"] > 0.25, ep["detail"]
