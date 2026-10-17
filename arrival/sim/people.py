"""Kinematic walking people (mocap capsules) on seeded waypoint loops through free floor."""

from __future__ import annotations

import numpy as np

CLEAR = 0.6  # min distance (m) from the base column for any path segment: 0.12 pedestal + 0.17 torso + margin
LANE = (0.5, 0.75)  # the crossing lane runs this far behind the base, closer than CLEAR on purpose


def person_xml(i: int) -> str:
    shirt = ["0.7 0.2 0.2 1", "0.2 0.4 0.7 1", "0.3 0.6 0.3 1"][i % 3]
    nc = 'contype="0" conaffinity="0"'
    return (f'<body name="person_{i}" mocap="true" pos="0 0 0">'
            f'<geom type="capsule" size="0.06" fromto="0.0 0.08 0.05 0.0 0.08 0.85" rgba="0.2 0.2 0.3 1" {nc}/>'
            f'<geom type="capsule" size="0.06" fromto="0.0 -0.08 0.05 0.0 -0.08 0.85" rgba="0.2 0.2 0.3 1" {nc}/>'
            f'<geom type="capsule" size="0.17" fromto="0 0 1.0 0 0 1.4" rgba="{shirt}" {nc}/>'
            f'<geom type="capsule" size="0.045" fromto="0 0.22 1.45 0 0.24 0.95" rgba="{shirt}" {nc}/>'
            f'<geom type="capsule" size="0.045" fromto="0 -0.22 1.45 0 -0.24 0.95" rgba="{shirt}" {nc}/>'
            f'<geom type="sphere" size="0.11" pos="0 0 1.7" rgba="0.85 0.68 0.55 1" {nc}/></body>')


def _seg_dist(a, b, p) -> float:
    ab = b - a
    n = float(ab @ ab)
    t = 0.0 if n < 1e-12 else float(np.clip((p - a) @ ab / n, 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - p))


def _interaction_loop(rng, floor, base_xy, front_y, base_to=None):
    """Free point -> approach from behind -> walk a lane behind the base (crossing the robot's work strip)
    -> stop at the counter on the far side -> free point. Free points are resampled until no segment comes
    within CLEAR of the base; after 50 misses the point directly behind the stop or the lane is used.
    base_to (MAN-3) is where a mobile base drives to: the lane, the stop and the clearances then cover the whole
    straight drive from base_xy to base_to, so nobody stands on the route."""
    x0, y0, x1, y1 = floor
    base = np.asarray(base_xy, dtype=float)
    end = base if base_to is None else np.asarray(base_to, dtype=float)
    lo, hi = min(base[0], end[0]), max(base[0], end[0])
    route = [base + k / 8 * (end - base) for k in range(9)]  # points the loop must keep CLEAR from
    room = {1.0: x1 - hi, -1.0: lo - x0}  # the stop needs 0.8 m beside the base; pick a side that has it
    side = rng.choice([-1.0, 1.0]) if min(room.values()) >= 1.3 else max(room, key=room.get)
    lane_y = min(max(base[1], end[1]) + rng.uniform(*LANE), y1 - 0.4)
    near, far = (hi, lo) if side > 0 else (lo, hi)  # the stop sits beyond the route's end on its side
    lane = [np.array([np.clip(far - side, x0, x1), lane_y]), np.array([np.clip(near + side, x0, x1), lane_y])]
    approach = np.array([lane[0][0], y1 - 0.2])
    stop = np.array([near + side * rng.uniform(0.8, min(1.3, room[side])), front_y + 0.35])

    def free(prev, nxt, fallback):
        for _ in range(50):
            p = rng.uniform([x0, y0], [x1, y1])
            if all(np.linalg.norm(p - q) >= CLEAR and _seg_dist(prev, p, q) >= CLEAR
                   and _seg_dist(p, nxt, q) >= CLEAR for q in route):
                return p
        return fallback

    a = free(stop, approach, approach.copy())
    b = free(stop, a, np.array([stop[0], y1 - 0.2]))
    return [a, approach, lane[0], lane[1], stop, b], [0.0, 0.0, 0.0, 0.0, float(rng.uniform(2.0, 4.0)), 0.0]


class Walkers:
    """Each person loops through random waypoints in a free-floor rectangle at ~0.8 m/s. Given the robot base
    and the counter front, the loop also crosses the strip behind the base and pauses at the counter."""

    def __init__(self, n: int, floor: tuple[float, float, float, float], seed: int = 0, speed: float = 0.8,
                 base_xy=None, front_y: float | None = None, base_to=None):
        rng = np.random.default_rng(seed)
        x0, y0, x1, y1 = floor
        self.paths, self.pauses = [], []
        for _ in range(n):
            if base_xy is None:
                pts, pause = list(rng.uniform([x0, y0], [x1, y1], size=(4, 2))), [0.0] * 4
            else:
                pts, pause = _interaction_loop(rng, floor, base_xy, front_y, base_to)
            self.paths.append(pts)
            self.pauses.append(pause)
        self.pos = [p[0].copy() for p in self.paths]
        self.idx = [1] * n
        self.wait = [0.0] * n
        self.speed = speed

    def step(self, model, data, dt: float) -> list[list[float]]:
        out = []
        for i, path in enumerate(self.paths):
            if self.wait[i] > 0:
                self.wait[i] -= dt
                yaw = -np.pi / 2  # face the counter while stopped
            else:
                k = self.idx[i] % len(path)
                d = path[k] - self.pos[i]
                dist = float(np.linalg.norm(d))
                if dist < 0.05:
                    self.wait[i] = self.pauses[i][k]
                    self.idx[i] += 1
                else:
                    self.pos[i] += d / dist * min(dist, self.speed * dt)
                yaw = np.arctan2(d[1], d[0])
            mid = model.body(f"person_{i}").mocapid[0]
            data.mocap_pos[mid] = [self.pos[i][0], self.pos[i][1], 0.0]
            data.mocap_quat[mid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            out.append([float(self.pos[i][0]), float(self.pos[i][1])])
        return out
