"""Robot-human distance per control step (PPL-2 safety metric), plus arm-fixture contacts (POL-4).

People are mocap bodies with non-colliding geoms, so MuJoCo never reports a robot-person contact pair.
The metric is geometric instead: horizontal clearance between every robot collision geom (capsule axis
or box/cylinder centre, minus its radius) and a vertical cylinder around each person (radius = the widest
person geom, the torso). Contact is clearance <= 0 or, should person geoms ever collide, a contact pair.
Height is ignored: the arm works between 0.5 m and 1.6 m, where the person cylinder always is.
"""

from __future__ import annotations

import numpy as np

YIELD = (0.25, 0.50)  # walkers: clearance (m) to the closest robot body at which they stand still, and above which they walk freely
ROBOT_BODIES = ("robot_base",) + tuple(f"link{i}" for i in range(1, 8)) + ("hand", "finger_l", "finger_r")
YIELD_BODIES = ROBOT_BODIES
MOBILE_YIELD_BODIES = ("robot_base", "hand", "finger_l", "finger_r")


def _seg_point_dist(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    ab = b - a
    n = float(ab @ ab)
    t = 0.0 if n < 1e-12 else float(np.clip((p - a) @ ab / n, 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - p))


def yield_factor(clearance: float) -> float:
    """Walking speed share for a person `clearance` metres from the closest of the base and arm-link bodies: 0 at YIELD[0] or less, 1 at YIELD[1] or more."""
    return float(np.clip((clearance - YIELD[0]) / (YIELD[1] - YIELD[0]), 0.0, 1.0))


class SafetyMonitor:
    def __init__(self, model):
        import mujoco

        body_ids = {model.body(n).id for n in ROBOT_BODIES}
        self.geoms = [g for g in range(model.ngeom)
                      if model.geom_bodyid[g] in body_ids and (model.geom_contype[g] or model.geom_conaffinity[g])]
        self.capsule = {g: model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE for g in self.geoms}
        self.radius = {g: float(np.hypot(model.geom_size[g][0], model.geom_size[g][1]))
                       if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else float(model.geom_size[g][0])
                       for g in self.geoms}
        self.half = {g: float(model.geom_size[g][1]) for g in self.geoms}
        is_mobile = any(model.joint(j).name == "base_x" for j in range(model.njnt))
        bodies = MOBILE_YIELD_BODIES if is_mobile else YIELD_BODIES
        yb = {model.body(n).id for n in bodies}
        self.yield_geoms = [g for g in self.geoms if model.geom_bodyid[g] in yb]
        self.people = []
        self.person_geoms = set()
        for b in range(model.nbody):
            if model.body(b).name.startswith("person_"):
                gs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
                self.people.append((b, max(float(model.geom_size[g][0]) for g in gs)))
                self.person_geoms.update(gs)
        # Arm-fixture contacts: the arm links (not the hand or fingers, which are meant to touch handles and
        # objects) against anything but the robot itself, people and free objects (bodies with a free joint).
        links = {model.body(f"link{i}").id for i in range(1, 8)}
        free = {model.jnt_bodyid[j] for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
        self.link_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in links}
        self.robot_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in body_ids}
        self.free_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in free}
        self.arm_ignore = self.robot_geoms | self.person_geoms | self.free_geoms
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.min_d = float("inf")
        self.min_step = None
        self.contact = False
        self.arm_steps = 0
        self.arm_depth = 0.0
        self.arm_worst = None
        self.door_contact = False

    def _arm_contacts(self, model, data, driven=()) -> None:
        hit = False
        for i in range(data.ncon):
            c = data.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self.link_geoms and b not in self.arm_ignore and b not in driven:  # a door a person is swinging is human contact, not arm contact
                    hit = True
                    if -c.dist > self.arm_depth:
                        self.arm_depth = -c.dist
                        self.arm_worst = (model.body(model.geom_bodyid[a]).name,
                                          model.geom(b).name or model.body(model.geom_bodyid[b]).name)
        self.arm_steps += hit

    def clearance(self, data, p, r: float, geoms=None) -> float:
        """Horizontal clearance from a person cylinder (centre p, radius r) to the robot geoms (all, or `geoms`)."""
        d = float("inf")
        for g in (self.geoms if geoms is None else geoms):
            c = data.geom_xpos[g]
            if self.capsule[g]:
                axis = data.geom_xmat[g].reshape(3, 3)[:, 2] * self.half[g]
                dist = _seg_point_dist((c - axis)[:2], (c + axis)[:2], p)
            else:
                dist = float(np.linalg.norm(c[:2] - p))
            d = min(d, dist - self.radius[g] - r)
        return d

    def yield_clearance(self, data):
        """Callable (xy, radius) -> clearance to the closest of the base, arm links, hand and fingers, for Walkers.step to yield on."""
        return lambda p, r: self.clearance(data, p, r, self.yield_geoms)

    def update(self, model, data, driven=()) -> None:
        """`driven` are geoms a person is moving right now (the fridge door they hold open): a contact between
        one of them and the robot counts as human contact."""
        self._arm_contacts(model, data, driven)
        if not self.people:
            return
        d = min(self.clearance(data, data.xpos[b][:2], r) for b, r in self.people)
        for i in range(data.ncon):
            g1, g2 = data.contact[i].geom1, data.contact[i].geom2
            if (g1 in self.person_geoms and g2 in self.geoms) or (g2 in self.person_geoms and g1 in self.geoms):
                d = min(d, 0.0)
            elif driven and ((g1 in driven and g2 in self.geoms) or (g2 in driven and g1 in self.geoms)):
                d = min(d, 0.0)
                self.door_contact = True
        if d < self.min_d:
            self.min_d, self.min_step = d, self.n
        self.contact = self.contact or d <= 0.0
        self.n += 1

    def summary(self) -> dict:
        arm = {"arm_contact_steps": self.arm_steps, "arm_contact_max_depth_m": round(self.arm_depth, 4),
               "arm_contact_worst": list(self.arm_worst) if self.arm_worst else None}
        if not self.people:
            return {"min_human_distance_m": None, "min_distance_step": None, "human_contact": False, "people": 0,
                    "door_contact": False, **arm}
        if self.min_d == float("inf"):  # update() never ran; inf is not valid JSON
            return {"min_human_distance_m": None, "min_distance_step": None, "human_contact": False,
                    "people": len(self.people), "door_contact": False, **arm}
        return {"min_human_distance_m": round(self.min_d, 4), "min_distance_step": self.min_step,
                "human_contact": bool(self.contact), "people": len(self.people), "door_contact": bool(self.door_contact), **arm}


def overhead_clearance(model, data) -> float:
    """Smallest 3D distance between a person's head, hair or arm geoms and a wall cabinet (its carcass or an upper
    door) or the fridge door: height-aware, unlike the footprints paths are planned with. Not in the episode summary."""
    import mujoco

    def name(g):
        return model.geom(g).name

    mine = [g for g in range(model.ngeom) if name(g).startswith("person_") and name(g).split("_", 2)[2] in
            ("head", "hair", "uarm_r", "uarm_l", "farm_r", "farm_l")]
    body = lambda g: model.body(model.geom_bodyid[g]).name
    other = [g for g in range(model.ngeom) if "_upper" in name(g) or "_upper_door" in body(g) or body(g).startswith("fridge") and body(g).endswith("_door")]
    return min((float(mujoco.mj_geomDistance(model, data, a, b, 1.0, None)) for a in mine for b in other), default=float("inf"))
