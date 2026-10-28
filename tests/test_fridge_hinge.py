"""Right-hinged fridge: the twin field `hinge: "right"` mirrors the door in the MJCF, the expert's pull and push,
the stand spot and the walker's visit. Without the field nothing changes."""
import copy
import hashlib

import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.export.quality import LIMIT_STEPS  # noqa: E402
from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim.collect import collect  # noqa: E402
from arrival.sim.scene import build_mjcf, part_names  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

QUIET = lambda *a, **k: None  # noqa: E731
WIDTH = 4.0  # the hero kitchen's room width: mirroring it in x puts the fridge at the east wall


@pytest.fixture(scope="module")
def hero():
    return compile_twin(load_room("hero_kitchen", "scans"))


@pytest.fixture(scope="module")
def mirrored(hero):
    """The hero kitchen reflected in x with a right-hinged fridge: the free edge faces the room, as the left-hinged
    fridge's does in the original. (The original fridge's right-hinged free edge would be in the room corner.)"""
    t = copy.deepcopy(hero)
    for w in t["walls"]:
        w["start"][0], w["end"][0] = WIDTH - w["start"][0], WIDTH - w["end"][0]
    for o in t["fixtures"] + t["objects"]:
        o["position"][0] = WIDTH - o["position"][0]
        if o["id"] == "fridge_1":
            o["hinge"] = "right"
    return t


def test_left_hinge_mjcf_is_unchanged(hero):
    xml = build_mjcf(hero, (2.6, 1.0, 0.75), n_people=2, people_seed=3)
    # measured on the code before the hinge field existed, then with the 16 mm bowl wall (only the bowl geoms moved),
    # then with the 30 mm finger pads (only the four finger geoms moved), then with drawer enclosure walls
    assert hashlib.sha256(xml.encode()).hexdigest()[:16] == "f3c102a2012d5915"
    explicit = copy.deepcopy(hero)
    next(f for f in explicit["fixtures"] if f["id"] == "fridge_1")["hinge"] = "left"
    assert build_mjcf(explicit, (2.6, 1.0, 0.75), n_people=2, people_seed=3) == xml


def test_right_hinge_mirrors_the_door(hero, mirrored):
    right = mujoco.MjModel.from_xml_string(build_mjcf(mirrored, (2.6, 1.0, 0.75)))
    left = mujoco.MjModel.from_xml_string(build_mjcf(hero, (2.6, 1.0, 0.75)))
    for m, sign in ((left, 1.0), (right, -1.0)):
        assert m.joint("fridge_1_door").axis[2] == sign
    cx = 4.0 - 0.375  # the mirrored fridge's centre
    assert right.body("fridge_1_door").pos[0] == pytest.approx(0.375)  # hinge on the fixture's right edge
    assert left.body("fridge_1_door").pos[0] == pytest.approx(-0.375)
    hr = right.site("fridge_1_door_handle").pos[0]
    hl = left.site("fridge_1_door_handle").pos[0]
    assert hr == pytest.approx(-hl) and cx > 3.0
    assert any(p["joint"] == "fridge_1_door" for p in part_names(mirrored))


def _summary(twin, families, episodes, people, seed, tmp_path):
    return collect(twin, families, episodes, tmp_path, people=people, res=32, seed=seed, log=QUIET)["report_data"]


@pytest.mark.parametrize("people,seed", [(0, 0), (2, 3), (2, 5)])
def test_expert_opens_a_right_hinged_fridge(mirrored, tmp_path, people, seed):
    r = _summary(mirrored, ["open_fridge"], 2, people, seed, tmp_path)
    assert (r["attempts"], r["successes"], r["skipped_unreachable"]) == (2, 2, 0)
    assert r["human_contacts"] == 0 and r["door_contacts"] == 0


@pytest.mark.parametrize("seed", [3, 5])
def test_walkers_open_a_right_hinged_fridge_without_contact(mirrored, tmp_path, seed):
    # open_fridge parks the robot beside the free edge, where the walker's stand point is, so that family plans no
    # visit; these families park it at the counter and the visit is planned (the logged fridge_visit stat)
    r = _summary(mirrored, ["open_drawer", "open_door", "wipe_counter"], 6, 2, seed, tmp_path)
    assert r["fridge_visits"] >= 1
    assert r["attempts"] == r["successes"] == 6
    assert r["human_contacts"] == 0 and r["door_contacts"] == 0


def _door_angle(twin, seed):
    """open_fridge once with no people: (final door angle in degrees, longest run of any arm joint inside the quality
    check's band at its limit, in steps)."""
    import numpy as np

    from arrival.export.quality import LIMIT_FRAC, longest_run
    from arrival.sim.collect import SimEnv, run_episode
    from arrival.sim.policy import ScriptedExpert
    from arrival.sim.tasks import finish_task, make_tasks

    task = make_tasks(twin, ["open_fridge"], 1, seed=seed, people=0, log=QUIET)[0]
    finish_task(task)
    env = SimEnv(twin, task, res=16)
    lo, hi = env.model.jnt_range[[env.model.dof_jntid[i] for i in env.dofadr]].T
    ep = run_episode(env, ScriptedExpert(), task)
    env.close()
    assert ep["success"], ep["detail"]
    q, band = np.asarray(ep["states"])[:, :7], LIMIT_FRAC * (hi - lo)
    run = max(longest_run((q[:, j] <= lo[j] + band[j]) | (q[:, j] >= hi[j] - band[j])) for j in range(7))
    return float(np.degrees(ep["detail"]["qpos"])), run


def test_both_hinges_open_the_fridge_as_far(hero, mirrored):
    # the push is planned to 80 deg on either hinge. The fingers used to slide off the bar closed, dragging it about
    # 10 deg shut, and a right-hinged door then stopped at 66.8 deg (the hand pushed its free edge, not its inside)
    (left, run_l), (right, run_r) = _door_angle(hero, 0), _door_angle(mirrored, 0)
    assert left > 75.0 and right > 75.0, (left, right)
    assert abs(left - right) < 1.0, (left, right)
    # the hold before the angle is read used to keep joint 4 on its stop for 43 steps: every episode flagged.
    # Keep a margin under the check's limit, not just under it
    assert max(run_l, run_r) <= LIMIT_STEPS - 5, (run_l, run_r)


def test_a_fridge_on_a_side_wall_opens():
    # u_kitchen_42's fridge faces +x (yaw pi/2). The push heading left out the fixture's yaw, so the edge rounding and
    # push were planned 90 deg off (1.70 m of IK error) and the door never opened
    u = compile_twin(load_room("u_kitchen_42", "tests/fixtures/scans"))
    angle, run = _door_angle(u, 0)
    assert angle > 75.0 and run <= LIMIT_STEPS - 5, (angle, run)


# ---- hinge side inferred from the walls when the scan does not give one ----

def _fridge(twin):
    return next(f for f in twin["fixtures"] if f["type"].startswith("fridge"))


def test_a_fridge_with_its_right_side_in_the_corner_is_hinged_right():
    # rotated_kitchen's fridge has 6.5 cm of floor in front of its right side (the room's corner) and 2 m on its left
    from arrival.twin.compile import _front_side_room

    twin = compile_twin(load_room("rotated_kitchen", "tests/fixtures/scans"))
    f = _fridge(twin)
    left, right = _front_side_room(f, twin["fixtures"], twin["walls"])
    assert right < 0.1 < 2.0 < left
    assert (f["hinge"], f["hinge_source"]) == ("right", "inferred_from_walls")


@pytest.mark.parametrize("room,scans", [("hero_kitchen", "scans"), ("u_kitchen_42", "tests/fixtures/scans")])
def test_a_fridge_with_its_left_side_against_a_wall_keeps_the_default(room, scans):
    # left is the default and stays unwritten, so these twins (and their scenes) are unchanged
    f = _fridge(compile_twin(load_room(room, scans)))
    assert "hinge" not in f and "hinge_source" not in f


def test_a_hinge_given_by_the_scan_wins_over_the_walls():
    import json
    from pathlib import Path

    from arrival.scan.bundle import validate

    scan = json.loads(Path("tests/fixtures/scans/rotated_kitchen/scan.json").read_text())
    next(f for f in scan["fixtures"] if f["id"] == "fridge_w")["hinge"] = "left"
    f = _fridge(compile_twin(validate(scan)))
    assert f["hinge"] == "left" and "hinge_source" not in f


@pytest.mark.parametrize("people,seed", [(0, 5), (2, 3)])
def test_expert_opens_the_fridge_in_the_corner(tmp_path, people, seed):
    # before: the stand spot beside the left-hinged door's free edge was outside the room, so open_fridge never ran
    twin = compile_twin(load_room("rotated_kitchen", "tests/fixtures/scans"))
    r = _summary(twin, ["open_fridge"], 2, people, seed, tmp_path)
    assert (r["attempts"], r["successes"], r["skipped_unreachable"]) == (2, 2, 0)
    assert r["human_contacts"] == 0 and r["door_contacts"] == 0
