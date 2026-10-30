"""A column the writer recorded as constant must stay constant in the loader check.

`check_loader` recomputed std as sqrt(E[x^2] - mean^2). On a column of identical values that
difference is rounding noise: 10 episodes of 150 frames holding the parked base x = 2.1
(float32) come out at 2.9802322387695312e-08. That is above CONST_STD (1e-8), so the check
said stats.json std 0 was a moving dimension (stats_std_zero) and dropped dim 8 from the
fixed_base_constant warning. The writer (`numpy` std, the same values) stores 0.
"""

import json

import numpy as np

from arrival.export.lerobot import LeRobotWriter
from arrival.export.loader_check import check_loader

BASE = (2.1, 1.35, 1.5707964)
N_EP, N = 10, 150
MODALITY = {
    "state": {"single_arm": {"start": 0, "end": 7}, "gripper": {"start": 7, "end": 8}, "base": {"start": 8, "end": 11}},
    "action": {
        "single_arm": {"start": 0, "end": 7, "absolute": True},
        "gripper": {"start": 7, "end": 8, "absolute": True},
        "base": {"start": 8, "end": 11, "absolute": True},
    },
    "annotation": {"human.task_description": {}},
}
EMBODIMENT = {
    "embodiment": "arrival_sim_panda",
    "dof_arm": 7,
    "dof_gripper": 1,
    "dof_base": 3,
    "base_actuated": False,
    "joints_are_placeholder": False,
    "control": "absolute_joint_position",
}


def test_constant_base_column_is_not_reported_as_moving(tmp_path):
    rng = np.random.default_rng(0)
    writer = LeRobotWriter(
        tmp_path / "ds",
        "room",
        "none",
        state_dim=11,
        action_dim=11,
        modality=MODALITY,
        embodiment=EMBODIMENT,
        extra_info={"arrival_physically_valid": True},
        video_keys=[],
    )
    for _ in range(N_EP):
        arm = rng.normal(0, 0.05, size=(N, 8))
        base = np.tile(np.array(BASE), (N, 1))
        row = np.concatenate([arm, base], axis=1)
        writer.add_episode(row, row.copy(), np.arange(N) / 20, "carry", True)
    writer.finish(N_EP)
    root = tmp_path / "ds"
    stored = json.loads((root / "meta" / "stats.json").read_text())
    assert stored["observation.state"]["std"][8:] == [0.0, 0.0, 0.0]
    assert stored["action"]["std"][8:] == [0.0, 0.0, 0.0]

    res = check_loader(root)
    codes = {e["code"] for e in res["errors"]}
    assert "stats_std_zero" not in codes, res["errors"]
    assert "stats_mismatch" not in codes, res["errors"]
    assert res["stats"]["observation.state"]["std"][8:] == [0.0, 0.0, 0.0]
    assert res["stats"]["action"]["std"][8:] == [0.0, 0.0, 0.0]
    parked = [w["msg"] for w in res["warnings"] if w["code"] == "fixed_base_constant"]
    assert len(parked) == 2
    assert all("[8, 9, 10]" in msg for msg in parked)
