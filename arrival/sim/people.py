"""Kinematic walking people: proportioned capsule bodies on mocap roots, limbs swung by a gait phase,
on seeded waypoint loops through free floor that keep clear of fixture footprints."""

from __future__ import annotations

import math

import mujoco
import numpy as np

CLEAR = 0.6  # min distance (m) from the base column for any path segment: 0.12 pedestal + 0.17 torso + margin
LANE = (0.5, 0.75)  # the crossing lane runs this far behind the base, closer than CLEAR on purpose

HEIGHT = (1.55, 1.90)
NC = 'contype="0" conaffinity="0"'
SHIRTS = ["0.70 0.20 0.20 1", "0.20 0.40 0.70 1", "0.30 0.60 0.30 1", "0.85 0.75 0.25 1", "0.55 0.30 0.60 1",
          "0.90 0.90 0.90 1", "0.20 0.20 0.22 1", "0.90 0.55 0.25 1"]
TROUSERS = ["0.20 0.22 0.32 1", "0.12 0.12 0.14 1", "0.45 0.40 0.33 1", "0.30 0.30 0.32 1", "0.55 0.55 0.58 1"]
SKINS = ["0.95 0.80 0.69 1", "0.87 0.68 0.52 1", "0.76 0.55 0.38 1", "0.60 0.42 0.28 1", "0.42 0.28 0.18 1"]
HAIRS = ["0.10 0.08 0.06 1", "0.35 0.22 0.12 1", "0.60 0.45 0.25 1", "0.75 0.70 0.65 1", "0.15 0.15 0.15 1"]
STRIDE = 1.30  # metres per full gait cycle (two steps)
HIP_SWING, KNEE_FLEX, ARM_SWING, ELBOW = 0.40, 0.70, 0.30, 0.35  # radians
BOB = 0.012
AMP_RATE = 4.0  # 1/s, how fast the swing fades in and out when a person starts or stops
PATH_SAMPLE = 0.05


def person_spec(seed: int, i: int) -> dict:
    """Body proportions and colours for person i of a seed; the same (seed, i) always gives the same person."""
    rng = np.random.default_rng([seed, i])
    h = float(rng.uniform(*HEIGHT))
    return {"height": h, "hip_z": 0.53 * h, "shoulder_z": 0.818 * h, "hip_w": 0.055 * h, "shoulder_w": 0.12 * h,
            "thigh": 0.245 * h, "shank": 0.245 * h, "foot": 0.13 * h,
            "upper_arm": 0.188 * h, "forearm": 0.19 * h, "torso_r": 0.075 * h, "limb_r": 0.038 * h,
            "head_r": 0.065 * h, "head_z": 0.93 * h,
            "shirt": SHIRTS[int(rng.integers(len(SHIRTS)))], "trousers": TROUSERS[int(rng.integers(len(TROUSERS)))],
            "skin": SKINS[int(rng.integers(len(SKINS)))], "hair": HAIRS[int(rng.integers(len(HAIRS)))],
            "radius": 0.12 * h + 0.038 * h + 0.03}  # shoulder half-width plus a limb radius plus margin, for paths


def _down(theta: float) -> np.ndarray:
    """Unit vector of a limb hanging straight down, pitched forward (+x) by theta."""
    return np.array([math.sin(theta), 0.0, -math.cos(theta)])


def limb_segments(s: dict, phase: float, amp: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Body-frame endpoints of every limb capsule for a gait phase (radians) at swing amplitude amp (0..1).
    Legs alternate, the arm on each side swings against the leg, the knee flexes during the swing."""
    out = {}
    for side, sg, ph in (("r", -1.0, phase), ("l", 1.0, phase + math.pi)):
        hip = np.array([0.0, sg * s["hip_w"], s["hip_z"]])
        th = amp * HIP_SWING * math.sin(ph)
        kn = amp * KNEE_FLEX * max(0.0, math.cos(ph))
        knee = hip + s["thigh"] * _down(th)
        ankle = knee + s["shank"] * _down(th - kn)
        a = th - kn
        toe = ankle + s["foot"] * np.array([math.cos(a), 0.0, math.sin(a)])
        sh = np.array([0.0, sg * s["shoulder_w"], s["shoulder_z"]])
        ts = -amp * ARM_SWING * math.sin(ph)
        el = ELBOW * (0.6 + 0.4 * amp) + amp * 0.25 * max(0.0, -math.sin(ph))
        elbow = sh + s["upper_arm"] * _down(ts)
        wrist = elbow + s["forearm"] * _down(ts + el)
        out[f"thigh_{side}"], out[f"shank_{side}"], out[f"foot_{side}"] = (hip, knee), (knee, ankle), (ankle, toe)
        out[f"uarm_{side}"], out[f"farm_{side}"] = (sh, elbow), (elbow, wrist)
    return out


def _cap(name: str, r: float, a, b, rgba: str) -> str:
    f = " ".join(f"{v:.4f}" for v in (*a, *b))
    return f'<geom name="{name}" type="capsule" size="{r:.4f}" fromto="{f}" rgba="{rgba}" {NC}/>'


def person_xml(i: int, seed: int = 0) -> str:
    s = person_spec(seed, i)
    p = f"person_{i}"
    limbs = limb_segments(s, 0.0, 0.0)
    col = {"thigh": s["trousers"], "shank": s["trousers"], "foot": "0.15 0.12 0.10 1", "uarm": s["shirt"],
           "farm": s["skin"]}
    body = (f'<body name="{p}" mocap="true" pos="0 0 0">'
            + _cap(f"{p}_pelvis", 0.06 * s["height"], (0, -s["hip_w"], s["hip_z"]), (0, s["hip_w"], s["hip_z"]),
                   s["trousers"])
            + _cap(f"{p}_torso", s["torso_r"], (0, 0, s["hip_z"] + 0.03), (0, 0, s["shoulder_z"] - 0.03), s["shirt"])
            + _cap(f"{p}_neck", 0.03 * s["height"], (0, 0, s["shoulder_z"]), (0, 0, s["head_z"] - s["head_r"]),
                   s["skin"])
            + f'<geom name="{p}_head" type="sphere" size="{s["head_r"]:.4f}" pos="0 0 {s["head_z"]:.4f}" '
              f'rgba="{s["skin"]}" {NC}/>'
            + f'<geom name="{p}_hair" type="sphere" size="{s["head_r"] * 0.96:.4f}" '
              f'pos="{-0.015 * s["height"]:.4f} 0 {s["head_z"] + 0.025 * s["height"]:.4f}" rgba="{s["hair"]}" {NC}/>')
    for name, (a, b) in limbs.items():
        r = s["limb_r"] * (1.25 if name.startswith("thigh") else 0.8 if name.startswith("f") else 1.0)
        body += _cap(f"{p}_{name}", r, a, b, col[name.split("_")[0]])
    return body + "</body>"


class Footprints:
    """Fixture footprints as arrays, with point and segment tests against the footprints grown by r."""

    def __init__(self, fixtures):
        rows = [(f["position"][0], f["position"][1], f["size"][0] / 2, f["size"][1] / 2,
                 math.cos(f.get("yaw", 0.0)), math.sin(f.get("yaw", 0.0))) for f in fixtures or []]
        self.a = np.array(rows, dtype=float).reshape(-1, 6)

    def blocked(self, pts, r: float) -> bool:
        if not len(self.a):
            return False
        pts = np.atleast_2d(pts)
        dx = pts[:, 0, None] - self.a[None, :, 0]
        dy = pts[:, 1, None] - self.a[None, :, 1]
        co, si = self.a[:, 4], self.a[:, 5]
        lx, ly = co * dx + si * dy, -si * dx + co * dy
        return bool(np.any((np.abs(lx) < self.a[:, 2] + r) & (np.abs(ly) < self.a[:, 3] + r)))

    def segment_blocked(self, a, b, r: float) -> bool:
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / PATH_SAMPLE)))
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        return self.blocked(a + (b - a) * t, r)

    def corners(self, r: float, floor) -> list[np.ndarray]:
        """Corners of every footprint grown by r plus a little, the via candidates for routing round it."""
        x0, y0, x1, y1 = floor
        out = []
        for cx, cy, hx, hy, co, si in self.a:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    lx, ly = sx * (hx + r + 0.05), sy * (hy + r + 0.05)
                    c = np.array([cx + co * lx - si * ly, cy + si * lx + co * ly])
                    if x0 <= c[0] <= x1 and y0 <= c[1] <= y1 and not self.blocked(c, r):
                        out.append(c)
        return out


def route(a, b, prints: Footprints, r: float, floor) -> list[np.ndarray] | None:
    """Via points from a to b (excluding both) that keep r clear of every footprint, by shortest path over
    the grown footprint corners; [] when the straight line is free, None when no route exists."""
    if not prints.segment_blocked(a, b, r):
        return []
    nodes = [np.asarray(a, float), np.asarray(b, float)] + prints.corners(r, floor)
    n = len(nodes)
    dist, prev, done = [math.inf] * n, [-1] * n, [False] * n
    dist[0] = 0.0
    for _ in range(n):
        u = min((i for i in range(n) if not done[i]), key=lambda i: dist[i], default=-1)
        if u < 0 or dist[u] == math.inf or u == 1:
            break
        done[u] = True
        for v in range(n):
            if done[v] or prints.segment_blocked(nodes[u], nodes[v], r):
                continue
            d = dist[u] + float(np.linalg.norm(nodes[v] - nodes[u]))
            if d < dist[v]:
                dist[v], prev[v] = d, u
    if dist[1] == math.inf:
        return None
    path, k = [], prev[1]
    while k > 0:
        path.append(nodes[k])
        k = prev[k]
    return path[::-1]


def free_path(rng, floor, prints: Footprints, r: float, n: int = 4, tries: int = 40) -> list[np.ndarray]:
    """n random goals in the floor rectangle joined into a closed loop that keeps r clear of every footprint,
    routed round fixtures through via points; a goal with no route from the previous one is resampled."""
    x0, y0, x1, y1 = floor
    goals: list[np.ndarray] = []
    legs: list[list[np.ndarray]] = []
    while len(goals) < n:
        for _ in range(tries):
            p = rng.uniform([x0, y0], [x1, y1])
            if prints.blocked(p, r):
                continue
            via = route(goals[-1], p, prints, r, floor) if goals else []
            if via is not None:
                goals.append(p)
                legs.append(via)
                break
        else:
            break
    if not goals:
        return [np.array([(x0 + x1) / 2, (y0 + y1) / 2])]
    back = route(goals[-1], goals[0], prints, r, floor) if len(goals) > 1 else []
    while back is None:
        goals.pop()
        legs.pop()
        back = route(goals[-1], goals[0], prints, r, floor) if len(goals) > 1 else []
    pts: list[np.ndarray] = []
    for g, via in zip(goals, legs):
        pts += via + [g]
    return pts + back


def _seg_dist(a, b, p) -> float:
    ab = b - a
    n = float(ab @ ab)
    t = 0.0 if n < 1e-12 else float(np.clip((p - a) @ ab / n, 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - p))


def _interaction_loop(rng, floor, base_xy, front_y, prints: Footprints | None = None, r: float = 0.0):
    """Free point -> approach from behind -> walk a lane behind the base (crossing the robot's work strip)
    -> stop at the counter on the far side -> free point. Free points are resampled until no segment comes
    within CLEAR of the base; after 50 misses the point directly behind the stop or the lane is used."""
    x0, y0, x1, y1 = floor
    base = np.asarray(base_xy, dtype=float)
    room = {1.0: x1 - base[0], -1.0: base[0] - x0}  # the stop needs 0.8 m beside the base; pick a side that has it
    side = rng.choice([-1.0, 1.0]) if min(room.values()) >= 1.3 else max(room, key=room.get)
    lane_y = min(base[1] + rng.uniform(*LANE), y1 - 0.4)
    lane = [np.array([np.clip(base[0] - side, x0, x1), lane_y]), np.array([np.clip(base[0] + side, x0, x1), lane_y])]
    approach = np.array([lane[0][0], y1 - 0.2])
    stop = np.array([base[0] + side * rng.uniform(0.8, min(1.3, room[side])), front_y + 0.35])

    def free(prev, nxt, fallback):
        for _ in range(50):
            p = rng.uniform([x0, y0], [x1, y1])
            if (np.linalg.norm(p - base) >= CLEAR and _seg_dist(prev, p, base) >= CLEAR
                    and _seg_dist(p, nxt, base) >= CLEAR and not (prints and prints.blocked(p, r))):
                return p
        return fallback

    a = free(stop, approach, approach.copy())
    b = free(stop, a, np.array([stop[0], y1 - 0.2]))
    pts, pauses = [a, approach, lane[0], lane[1], stop, b], [0.0, 0.0, 0.0, 0.0, float(rng.uniform(2.0, 4.0)), 0.0]
    if prints is None:
        return pts, pauses
    routed, waits = [], []  # a leg that cuts through a fixture goes round it; the lane and stop stay where they are
    for k, p in enumerate(pts):
        via = route(pts[k - 1], p, prints, r, floor) or []
        routed += via + [p]
        waits += [0.0] * len(via) + [pauses[k]]
    return routed, waits


class Walkers:
    """Each person loops through random waypoints in a free-floor rectangle at ~0.8 m/s, steering clear of
    the fixtures. Given the robot base and the counter front, the loop instead crosses the strip behind the base
    and pauses at the counter. Limbs follow a gait phase driven by distance walked; the swing fades out when standing."""

    def __init__(self, n: int, floor: tuple[float, float, float, float], seed: int = 0, speed: float = 0.8,
                 base_xy=None, front_y: float | None = None, fixtures=None):
        rng = np.random.default_rng(seed)
        self.specs = [person_spec(seed, i) for i in range(n)]
        prints = Footprints(fixtures)
        self.paths, self.pauses = [], []
        for s in self.specs:
            if base_xy is None:
                pts = free_path(rng, floor, prints, s["radius"])
                pause = [0.0] * len(pts)
            else:
                pts, pause = _interaction_loop(rng, floor, base_xy, front_y, prints, s["radius"])
            self.paths.append(pts)
            self.pauses.append(pause)
        self.pos = [p[0].copy() for p in self.paths]
        self.idx = [1] * n
        self.wait = [0.0] * n
        self.phase = [0.0] * n
        self.amp = [0.0] * n
        self.speed = speed
        self._gids: list[dict[str, int]] | None = None

    def _bind(self, model) -> None:
        self._gids = [{k: model.geom(f"person_{i}_{k}").id for k in limb_segments(s, 0.0, 0.0)}
                      for i, s in enumerate(self.specs)]

    def pose_limbs(self, model, i: int) -> None:
        """Write the limb capsules' body-frame pose for person i's current gait phase into the model."""
        if self._gids is None:
            self._bind(model)
        q = np.zeros(4)
        for k, (a, b) in limb_segments(self.specs[i], self.phase[i], self.amp[i]).items():
            g = self._gids[i][k]
            model.geom_pos[g] = (a + b) / 2
            mujoco.mju_quatZ2Vec(q, b - a)
            model.geom_quat[g] = q

    def step(self, model, data, dt: float) -> list[list[float]]:
        out = []
        for i, path in enumerate(self.paths):
            moved = 0.0
            if self.wait[i] > 0:
                self.wait[i] -= dt
                yaw = -np.pi / 2  # face the counter while stopped
            else:
                k = self.idx[i] % len(path)
                d = path[k] - self.pos[i]
                dist = float(np.linalg.norm(d))
                if dist < 0.05:  # at a waypoint: pause there, or turn to the next one without breaking stride
                    self.wait[i] = self.pauses[i][k]
                    self.idx[i] += 1
                    d = path[self.idx[i] % len(path)] - self.pos[i]
                    dist = float(np.linalg.norm(d))
                if self.wait[i] <= 0 and dist > 1e-9:
                    moved = min(dist, self.speed * dt)
                    self.pos[i] += d / dist * moved
                yaw = np.arctan2(d[1], d[0])
            self.phase[i] += 2 * math.pi * moved / STRIDE
            target = 1.0 if moved > 0 else 0.0
            self.amp[i] += float(np.clip(target - self.amp[i], -AMP_RATE * dt, AMP_RATE * dt))
            self.pose_limbs(model, i)
            mid = model.body(f"person_{i}").mocapid[0]
            bob = BOB * self.amp[i] * (1 - abs(math.cos(self.phase[i])))
            data.mocap_pos[mid] = [self.pos[i][0], self.pos[i][1], bob]
            data.mocap_quat[mid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            out.append([float(self.pos[i][0]), float(self.pos[i][1])])
        return out
