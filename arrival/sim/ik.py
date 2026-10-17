"""Damped least-squares IK on the eef site (position + gripper-down orientation), nullspace-biased to a rest pose."""

from __future__ import annotations

import mujoco
import numpy as np

from arrival.sim.robot import HOME


def solve(model, data, target_pos, R_t, q0, qadr: list[int], dofadr: list[int],
          iters: int = 200, damping: float = 0.05, tol: float = 2e-3,
          rest=HOME, k_rest: float = 0.2, fixed: tuple[list[int], np.ndarray] | None = None) -> tuple[np.ndarray, float]:
    """Returns (q, position error). Works on a scratch MjData; R_t is the target eef rotation matrix.

    The redundant DoF are pulled toward `rest` (gain k_rest) through the Jacobian nullspace, so the elbow/wrist
    stay near the ready posture; iteration stops once the task is met and the posture has settled.
    `fixed` = (qpos addresses, values) pins other joints, e.g. the mobile base where it will stand."""
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    d.qpos[qadr] = q0
    if fixed is not None:
        d.qpos[fixed[0]] = fixed[1]
    sid = model.site("eef").id
    R_t = np.asarray(R_t, dtype=float)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    lo, hi = model.jnt_range[[model.dof_jntid[i] for i in dofadr]].T
    err = 1.0
    for _ in range(iters):
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        R = d.site_xmat[sid].reshape(3, 3)
        ep = np.asarray(target_pos) - d.site_xpos[sid]
        er = 0.5 * (np.cross(R[:, 0], R_t[:, 0]) + np.cross(R[:, 1], R_t[:, 1]) + np.cross(R[:, 2], R_t[:, 2]))
        err = float(np.linalg.norm(ep))
        mujoco.mj_jacSite(model, d, jp, jr, sid)
        J = np.vstack([jp[:, dofadr], jr[:, dofadr]])
        e = np.concatenate([ep, 0.5 * er])
        q = d.qpos[qadr].copy()
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
        dq += k_rest * (np.eye(len(qadr)) - np.linalg.pinv(J) @ J) @ (np.asarray(rest) - q)
        q_new = np.clip(q + dq, lo, hi)
        if err < tol and np.linalg.norm(er) < 0.02 and np.abs(q_new - q).max() < 1e-4:
            break
        d.qpos[qadr] = q_new
    return d.qpos[qadr].copy(), err


def down(yaw: float = 0.0) -> np.ndarray:
    """Gripper pointing straight down, fingers closing along (sin yaw, -cos yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]])


def tilted(normal_xy, along_xy, theta: float) -> np.ndarray:
    """Gripper pointing down, tilted `theta` rad from vertical toward -normal_xy (into a vertical panel whose
    outward normal is normal_xy); fingers close horizontally along along_xy (a bar handle sits between them)."""
    n = np.array([normal_xy[0], normal_xy[1], 0.0])
    n /= np.linalg.norm(n)
    z = -np.sin(theta) * n + np.array([0, 0, -np.cos(theta)])
    y = np.array([along_xy[0], along_xy[1], 0.0])
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


def facing(direction_xy) -> np.ndarray:
    """Gripper pointing horizontally along direction_xy, fingers closing horizontally (vertical bar handles)."""
    z = np.array([direction_xy[0], direction_xy[1], 0.0])
    z /= np.linalg.norm(z)
    y = np.cross(z, [0, 0, 1.0])
    return np.column_stack([np.cross(y, z), y, z])
