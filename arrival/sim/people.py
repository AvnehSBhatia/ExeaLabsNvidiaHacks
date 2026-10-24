"""Kinematic walking people: proportioned capsule bodies on mocap roots, limbs swung by a gait phase,
on seeded waypoint loops through free floor that keep clear of fixture footprints. In some episodes one person
opens the fridge: the hinge is driven along a smooth profile while the hand holds the handle."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from arrival.sim.safety import yield_factor

CLEAR = 0.6  # min distance (m) from the base column for any path segment: 0.12 pedestal + 0.17 torso + margin
LANE = (0.5, 0.75)  # the crossing lane runs this far behind the base, closer than CLEAR on purpose


def frame(yaw: float) -> np.ndarray:
    """Rotation taking world xy into the frame where a fixture with this yaw faces +y (local = M @ world)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s], [-s, c]])


def _in_frame(f: dict, M: np.ndarray, yaw: float) -> dict:
    """A fixture dict moved into the frame `M` takes the world to (the one where a fixture with yaw `yaw` faces +y)."""
    x, y = M @ np.array(f["position"][:2], dtype=float)
    return {**f, "position": [float(x), float(y), *f["position"][2:]], "yaw": f.get("yaw", 0.0) - yaw}


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
BRAKE = (0.6, 0.15)  # into a stop the speed falls with the distance left over 0.6 m, to 15% at the end: the toe reaches
# 0.53 to 0.65 m ahead at full swing and 0.29 to 0.36 m at 15%; the arrival radius stops a person 4 to 5 cm short of
# the stop point, 0.35 m from the front, and that is what keeps the longest legs out
PATH_SAMPLE = 0.05
P_VISIT = 0.6  # share of episodes with people in which one person goes to the fridge
OPEN_ANGLE = (0.95, 1.30)  # radians the door is opened to, seeded; a wider swing puts the panel further from the
MAX_ANGLE = 1.45  # person in the doorway, so the angle is raised in 5 deg steps up to this when the stand point is close
SWING_T, REACH_T = 1.2, 0.3  # seconds to swing the door, and for the hand to settle on or leave the handle
LOOK_T = (1.5, 3.5)  # seconds spent in the open doorway
SLOW = 0.45  # the two short steps into and out of the doorway, as a share of walking speed; the stride scales with it
LET_GO = 0.30  # radians: the hand pulls the bar this far, then lets go and the door swings on; past about 20 deg the
# person is on the inner side of the panel while the bar is on the outer face, so an arm still on the bar would cross it
CATCH = 0.25  # radians: closing, the hand pushes the inner face at the free edge down to here, then the door shuts itself


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


def reach_arm(s: dict, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame elbow and wrist of the right arm reaching for a body-frame point: two-link IK with the elbow
    hanging below the shoulder-to-target line, the wrist stopping at 98% of the arm's length when out of reach."""
    sh = np.array([0.0, -s["shoulder_w"], s["shoulder_z"]])
    l1, l2 = s["upper_arm"], s["forearm"]
    d = target - sh
    dist = max(float(np.linalg.norm(d)), 1e-6)
    u = d / dist
    reach = min(dist, 0.98 * (l1 + l2))
    a = math.acos(float(np.clip((l1 * l1 + reach * reach - l2 * l2) / (2 * l1 * reach), -1.0, 1.0)))
    perp = np.array([0.0, 0.0, -1.0]) + u[2] * u  # the down direction, less its part along the arm
    n = float(np.linalg.norm(perp))
    perp = perp / n if n > 1e-6 else np.array([0.0, -1.0, 0.0])
    return sh + l1 * (math.cos(a) * u + math.sin(a) * perp), sh + reach * u


def toe_reach(s: dict, amp: float) -> float:
    """How far ahead of the body centre the toe gets at swing amplitude amp, including the foot's radius. Standing,
    the foot still points 0.13 h forward."""
    return max(limb_segments(s, ph, amp)["foot_r"][1][0] for ph in np.linspace(0.0, 2 * math.pi, 36)) + 0.8 * s["limb_r"]


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


def _rot(yaw: float, v) -> np.ndarray:
    c, si = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] - si * v[1], si * v[0] + c * v[1]])


def plan_fridge_visit(fixtures, floor, spec: dict, rng, base_xy=None) -> dict | None:
    """Where one person stands to open the fridge and how far the door goes. The person faces the fridge beside
    the handle, outside the door's swing (s1), reaches across to open it, takes a slow step into the doorway on the
    handle side (s2), and comes back to close it. None when the room has no fridge, a stand point is off the floor
    or in a footprint, a stand point or the swing comes within CLEAR of the robot base, or no opening angle keeps
    the panel clear of the person's body and toes (a wider swing moves the panel away from the doorway)."""
    from arrival.sim.scene import fridge_door_frame

    fr = next((f for f in fixtures or [] if f["type"].startswith("fridge")), None)
    if fr is None:
        return None
    g = fridge_door_frame(fr)
    r = spec["radius"]
    yaw, L, front = fr.get("yaw", 0.0), g["length"], g["front"]
    centre = np.array(fr["position"][:2], dtype=float)
    hinge = centre + _rot(yaw, g["hinge"])
    m = g["mirror"]  # +1 left-hinged; -1 right-hinged: every x across the fixture is mirrored
    s1 = centre + _rot(yaw, (m * (L / 2 + r), front + r + 0.20))  # 0.2 out: the shoulder stays on the bar's side longer
    s2 = centre + _rot(yaw, (m * (L / 2 - 0.15), front + r + 0.05))
    toe = _rot(yaw, (0.0, -toe_reach(spec, 0.0)))  # the standing toe tip, facing the fridge
    stride = toe_reach(spec, SLOW)
    x0, _, x1, y1 = floor  # the stand points are in the work strip in front of the floor rectangle, like the counter stop
    prints = Footprints(fixtures)
    for p in (s1, s2):
        if (not (x0 <= p[0] <= x1 and p[1] <= y1) or prints.blocked(p, r)
                or prints.blocked(p + toe, 0.8 * spec["limb_r"])):
            return None
    if base_xy is not None:
        base = np.asarray(base_xy, dtype=float)
        if (np.linalg.norm(base - hinge) < L + CLEAR or np.linalg.norm(s1 - base) < CLEAR
                or np.linalg.norm(s2 - base) < CLEAR):
            return None

    def panel_dist(p, q):  # distance from a stand point to the door panel at angle q
        return _seg_dist(hinge, hinge + L * _rot(yaw + m * q, (float(m), 0.0)), p)

    swing = np.linspace(0.0, OPEN_ANGLE[1], 40)
    if (min(panel_dist(s1, q) for q in swing) < r + 0.02
            or min(panel_dist(s1 + toe, q) for q in swing) < 0.8 * spec["limb_r"] + 0.02):
        return None
    q_open = float(rng.uniform(*OPEN_ANGLE))
    leg = [s1 + (s2 - s1) * t for t in np.linspace(0.0, 1.0, 12)]
    while q_open <= MAX_ANGLE:
        if (min(panel_dist(p, q_open) for p in leg) >= stride + 0.02
                and panel_dist(s2 + toe, q_open) >= 0.8 * spec["limb_r"] + 0.02):
            break
        q_open += math.radians(5)
    else:
        return None
    sector = {"position": [*(centre + _rot(yaw, (0.0, front + L / 2))), 0.0], "size": [L + 0.7, L + 0.7, 0.0],
              "yaw": yaw}  # the quarter disc the door sweeps, grown for toes: the other walkers route round it
    return {"fixture": fr["id"], "hinge": hinge, "yaw": yaw, "length": L, "handle": g["handle"], "centre": centre,
            "face": yaw - math.pi / 2, "mirror": m, "s1": s1, "s2": s2, "q_open": q_open, "look": float(rng.uniform(*LOOK_T)),
            "sector": sector}


def _interaction_loop(rng, floor, base_xy, front_y, prints: Footprints | None = None, r: float = 0.0,
                      base_to=None, keep=(), person_idx: int = 0, existing_stops: list | None = None,
                      other_starts: list | None = None):
    """Free point -> approach from behind -> walk a lane behind the base (crossing the robot's work strip)
    -> stop at the counter on the far side -> free point. Free points are resampled until no segment comes
    within CLEAR of the base; after 50 misses the point directly behind the stop or the lane is used.
    base_to (MAN-3) is where a mobile base drives to: the lane, the stop and the clearances then cover the whole
    straight drive from base_xy to base_to, so nobody stands on the route."""
    x0, y0, x1, y1 = floor
    base = np.asarray(base_xy, dtype=float)
    end = base if base_to is None else np.asarray(base_to, dtype=float)
    lo, hi = min(base[0], end[0]), max(base[0], end[0])
    drive = [base + k / 8 * (end - base) for k in range(9)]  # points the loop must keep CLEAR from
    room = {1.0: x1 - hi, -1.0: lo - x0}  # the stop needs 0.8 m beside the base; pick a side that has it
    side = rng.choice([-1.0, 1.0]) if min(room.values()) >= 1.3 else max(room, key=room.get)
    lane_y = min(max(base[1], end[1]) + rng.uniform(0.50, 0.56), y1 - 0.55)
    near, far = (hi, lo) if side > 0 else (lo, hi)  # the stop sits beyond the route's end on its side
    x_lane0 = np.clip(far - side, x0, x1)
    lane = [np.array([x_lane0, lane_y]), np.array([np.clip(near + side, x0, x1), lane_y])]
    app_y = y1 - 0.2
    approach = np.array([x_lane0, app_y])

    # Stagger stops so all stops stay >= 0.45m apart from each other and fridge
    has_fridge = existing_stops and any(abs(s[0] - 1.06) < 0.25 for s in existing_stops)
    max_u = min(1.65, room[side])
    if has_fridge:
        u = rng.uniform(1.05, 1.12)
    elif existing_stops and len(existing_stops) > 0:
        u0 = abs(existing_stops[0][0] - near)
        if u0 <= 1.25:
            u = min(u0 + rng.uniform(0.45, 0.52), max_u)
        else:
            u = max(u0 - rng.uniform(0.45, 0.52), 1.05)
    else:
        u = rng.uniform(1.05, 1.15)
    stop = np.array([near + side * u, front_y + 0.35])
    if any(np.linalg.norm(stop - c) < k for c, k in keep):  # the stop is in the door's swing: step on along the counter
        for _ in range(20):
            cand = np.array([near + side * rng.uniform(0.8, room[side]), front_y + 0.35])
            if not any(np.linalg.norm(cand - c) < k for c, k in keep):
                stop = cand
                break
    if existing_stops is not None:
        existing_stops.append(stop)
    def free(prev, nxt, fallback, min_other: bool = False):
        for _ in range(50):
            p = rng.uniform([x0, y0], [x1, y1])
            if (all(np.linalg.norm(p - q) >= CLEAR and _seg_dist(prev, p, q) >= CLEAR
                    and _seg_dist(p, nxt, q) >= CLEAR for q in drive) and not (prints and prints.blocked(p, r))
                    and not any(np.linalg.norm(p - c) < k for c, k in keep)):
                if min_other and other_starts:
                    if any(np.linalg.norm(p - s0) < 0.70 for s0 in other_starts):
                        continue
                return p
        return fallback

    a = free(stop, approach, approach.copy(), min_other=True)
    b_cand = np.array([np.clip(stop[0] + rng.uniform(0.05, 0.15), x0, x1), rng.uniform(y1 - 0.45, y1 - 0.25)])
    b = b_cand if not (prints and prints.blocked(b_cand, r)) else free(stop, a, np.array([stop[0], y1 - 0.2]))
    pts, pauses = [a, approach, lane[0], lane[1], stop, b], [0.0, 0.0, 0.0, 0.0, float(rng.uniform(2.0, 4.0)), 0.0]
    if prints is None:
        return pts, pauses
    routed, waits = [], []  # a leg that cuts through a fixture goes round it; the lane and stop stay where they are
    for k, p in enumerate(pts):
        via = route(pts[k - 1], p, prints, r, floor) or []
        routed += via + [p]
        waits += [0.0] * len(via) + [pauses[k]]
    return routed, waits


def _smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


class Walkers:
    """Each person loops through random waypoints in a free-floor rectangle at ~0.8 m/s, steering clear of
    the fixtures. Given the robot base and the counter front, the loop instead crosses the strip behind the base
    and pauses at the counter. Limbs follow a gait phase driven by distance walked; the swing fades out when standing.
    With a fridge in the room, P_VISIT of the seeds send one person to it first: they start beside the handle,
    hold it while the hinge swings open, stand in the doorway, come back, hold the handle while it swings shut,
    and then walk their loop. The others route round the door's swing. `fridge_visit` True or False forces it.
    `floor`, `base_xy` and `front_y` are in the frame where the fixture with yaw `yaw` faces +y (see `frame`; the
    hero kitchen's north wall has yaw 0, so there it is the world), `fixtures` are in the world. Everything is
    planned in that frame, so a run on any wall gets its strip, lane, counter stop and fridge trip, and the paths,
    the visit and the door geometry are turned back into the world before the first step. `front_y` None means no
    counter stop: free loops over the floor (the base is still kept clear of)."""

    def __init__(self, n: int, floor: tuple[float, float, float, float], seed: int = 0, speed: float = 0.8,
                 base_xy=None, front_y: float | None = None, fixtures=None, fridge_visit: bool | None = None,
                 yaw: float = 0.0, base_to=None, keep=None):
        rng = np.random.default_rng(seed)
        self.M, self.work_yaw = frame(yaw), yaw
        # `keep`: (world xy, radius) discs the robot works in, e.g. the swing of the door it opens. No counter stop,
        # free point or leg of the loop of a person other than the fridge visitor comes inside one.
        keep = [(self.M @ np.asarray(c, dtype=float), float(k)) for c, k in keep or []]
        fixtures = [_in_frame(f, self.M, yaw) for f in fixtures or []]
        self.specs = [person_spec(seed, i) for i in range(n)]
        self.visit = None
        if n and fixtures and fridge_visit is not False:  # its own stream, so seeds without a visit walk as before
            vrng = np.random.default_rng([seed, 7919])
            who = int(vrng.integers(n))
            if fridge_visit or vrng.random() < P_VISIT:
                plan = plan_fridge_visit(fixtures, floor, self.specs[who], vrng, base_xy)
                self.visit = {**plan, "person": who} if plan else None
        prints = Footprints(fixtures)
        avoid = Footprints(list(fixtures or []) + ([self.visit["sector"]] if self.visit else [])
                           + [{"position": [*c, 0.0], "size": [1.7 * k, 1.7 * k, 0.0], "yaw": 0.0} for c, k in keep])
        self.paths, self.pauses, self.faces, self.slow = [], [], [], []
        stops = []
        if self.visit:
            stops.append(self.visit["s1"])
        starts = []
        for i, s in enumerate(self.specs):
            visiting = self.visit is not None and i == self.visit["person"]
            pr = prints if visiting else avoid
            if front_y is None:
                pts = free_path(rng, floor, pr, s["radius"])
                pause = [0.0] * len(pts)
            else:
                pts, pause = _interaction_loop(rng, floor, base_xy, front_y, pr, s["radius"], base_to,
                                               keep=[] if visiting else keep,
                                               person_idx=i, existing_stops=stops, other_starts=starts)
            faces, slow = [None] * len(pts), [False] * len(pts)
            if visiting:
                pts, pause, faces, slow = self._splice_visit(pts, pause, s["radius"], floor, fixtures, base_xy)
            starts.append(pts[0])
            self.paths.append(pts)
            self.pauses.append(pause)
            self.faces.append(faces)
            self.slow.append(slow)
        self._to_world(yaw)
        self.pos = [p[0].copy() for p in self.paths]
        self.idx = [1] * n
        self.wait = [0.0] * n
        self.total = [0.0] * n  # length of the pause being waited out
        self.at = [0] * n  # waypoint index of that pause
        self.phase = [0.0] * n
        self.amp = [0.0] * n
        self.yaw = [0.0] * n
        self.reach = [None] * n  # (weight 0..1, world point) the right hand is going for
        self.speed = speed
        self.door_q, self.door_max = 0.0, 0.0
        self.driven_geoms: set[int] = set()  # the door's geoms while a person holds it open
        self._gids: list[dict[str, int]] | None = None
        self._door = None
        if self.visit:
            who = self.visit["person"]
            self.wait[who], self.total[who], self.at[who] = self.pauses[who][0], self.pauses[who][0], 0

    def _to_world(self, yaw: float) -> None:
        """Turn the paths and the planned visit from the work frame back into the world."""
        back = self.M.T
        self.paths = [[back @ p for p in path] for path in self.paths]
        if self.visit:
            v = self.visit
            for k in ("hinge", "centre", "s1", "s2"):
                v[k] = back @ v[k]
            v["yaw"] += yaw
            v["face"] += yaw

    def _splice_visit(self, pts, pause, r, floor, fixtures, base_xy):
        """Put the fridge trip at the front of the loop: s1 (open), s2 (look), s1 (close), then the loop's own
        points. The leg out of s1 and the leg back to it are routed round the fixtures and the robot base; the
        loop's own legs stay as they were (its lane runs closer to the base on purpose)."""
        v = self.visit
        hold = REACH_T + SWING_T + REACH_T
        block = list(fixtures or [])
        if base_xy is not None:
            block.append({"position": [base_xy[0], base_xy[1], 0.0], "size": [2 * CLEAR, 2 * CLEAR, 0.0], "yaw": 0.0})
        prints = Footprints(block)
        via_out = np.array([v["s1"][0], min(v["s1"][1] + 0.55, floor[3] - 0.4)])
        out = [via_out] + (route(via_out, pts[0], prints, r, floor) or [])
        back = route(pts[-1], v["s1"], prints, r, floor) or []
        path = [v["s1"], v["s2"], v["s1"]] + out + list(pts) + back
        waits = [hold, v["look"], hold] + [0.0] * len(out) + list(pause) + [0.0] * len(back)
        faces = ["open", "look", "close"] + [None] * (len(out) + len(pts) + len(back))
        slow = [False, True, True] + [False] * (len(out) + len(pts) + len(back))
        return path, waits, faces, slow

    def _bind(self, model) -> None:
        self._gids = [{k: model.geom(f"person_{i}_{k}").id for k in limb_segments(s, 0.0, 0.0)}
                      for i, s in enumerate(self.specs)]
        if self.visit:
            jn = f"{self.visit['fixture']}_door"
            try:
                j = model.joint(jn)
                body = model.body(jn).id
            except KeyError:
                self.visit = None  # a model built without the fridge: the person just walks
                return
            self._door = (int(j.qposadr[0]), int(j.dofadr[0]),
                          {g for g in range(model.ngeom) if model.geom_bodyid[g] == body})

    def door_point(self, q: float, lx: float, ly: float) -> np.ndarray:
        """A point of the door at angle q, from its frame: x along the panel from the hinge, y out of the front face
        (negative is the inner face), at the handle's height."""
        v = self.visit
        xy = v["hinge"] + _rot(v["yaw"] + v["mirror"] * q, (lx, ly))
        return np.array([xy[0], xy[1], v["handle"][2]])

    def _visit_step(self, i: int, tag: str) -> None:
        """During the open or close pause: the hand takes the bar, the door swings, the hand lets go (LET_GO).
        Closing, the hand pushes the inner face at the free edge and leaves it at CATCH; the door shuts itself."""
        v = self.visit
        hx, hy, _ = v["handle"]
        t = self.total[i] - self.wait[i]
        p = _smooth((t - REACH_T) / SWING_T)
        q = v["q_open"] * (p if tag == "open" else 1.0 - p)
        self.door_q = q
        self.door_max = max(self.door_max, q)
        off = 0.8 * self.specs[i]["limb_r"] + 0.02  # forearm radius plus a gap, so the hand sits on the surface
        if tag == "open":
            hold = 1.0 - _smooth((q - LET_GO) / 0.2)
            target = self.door_point(q, hx, hy + off)
        else:
            hold = _smooth((q - CATCH) / 0.2)
            target = self.door_point(q, v["mirror"] * (v["length"] - 0.05), -off - 0.065)  # 0.065 keeps the forearm capsule above the panel's top edge
        w = min(_smooth(t / REACH_T), _smooth((self.total[i] - t) / REACH_T)) * hold
        self.reach[i] = (w, target) if w > 0.0 else None

    def pose_limbs(self, model, i: int) -> None:
        """Write the limb capsules' body-frame pose for person i's current gait phase into the model."""
        if self._gids is None:
            self._bind(model)
        q = np.zeros(4)
        limbs = limb_segments(self.specs[i], self.phase[i], self.amp[i])
        if self.reach[i] is not None:
            w, target = self.reach[i]
            b = _rot(-self.yaw[i], target[:2] - self.pos[i])
            elbow, wrist = reach_arm(self.specs[i], np.array([b[0], b[1], target[2]]))
            sh, e0 = limbs["uarm_r"]
            _, w0 = limbs["farm_r"]
            e1 = (1 - w) * e0 + w * elbow
            limbs["uarm_r"], limbs["farm_r"] = (sh, e1), (e1, (1 - w) * w0 + w * wrist)
        for k, (a, b) in limbs.items():
            g = self._gids[i][k]
            model.geom_pos[g] = (a + b) / 2
            mujoco.mju_quatZ2Vec(q, b - a)
            model.geom_quat[g] = q

    def _pause_yaw(self, i: int) -> float:
        if self.faces[i][self.at[i]] in ("open", "look", "close") and self.visit:
            return self.visit["face"]  # square to the fridge; the toes stay out of the door's swing
        return -np.pi / 2 + self.work_yaw  # face the counter while stopped

    def step(self, model, data, dt: float, clearance=None) -> list[list[float]]:
        """`clearance(xy, radius)` is the walker's clearance to the closest of the robot base and arm-link
        bodies (safety.yield_clearance). A
        walker slows with it and stands still below safety.YIELD[0], then resumes; at a pause (the counter stop)
        the timer is held while the robot is that close. None: nobody yields, as before."""
        if self._gids is None:
            self._bind(model)
        out = []
        n = len(self.paths)

        u_dirs = []
        for i, path in enumerate(self.paths):
            k_i = self.idx[i] % len(path)
            d_target = path[k_i] - self.pos[i]
            dist_target = float(np.linalg.norm(d_target))
            u_dirs.append(d_target / dist_target if dist_target > 1e-9 else np.zeros(2))

        for i, path in enumerate(self.paths):
            moved = 0.0
            self.reach[i] = None
            give = yield_factor(clearance(self.pos[i], self.specs[i]["torso_r"])) if clearance else 1.0

            # Interpersonal collision avoidance: yield to other walkers in forward corridor
            tag_i = self.faces[i][self.at[i]] if self.at[i] < len(self.faces[i]) else None
            visiting_pause = self.wait[i] > 0 and tag_i in ("open", "look", "close")

            if n > 1 and not visiting_pause:
                u_i = u_dirs[i]
                for j in range(n):
                    if j == i:
                        continue
                    v_ij = self.pos[j] - self.pos[i]
                    d_ij = float(np.linalg.norm(v_ij))
                    if d_ij < 1e-6:
                        continue

                    s_i = float(np.dot(u_i, v_ij))
                    if s_i > 0:
                        lat = float(np.linalg.norm(v_ij - s_i * u_i))
                        r_sum = self.specs[i]["torso_r"] + self.specs[j]["torso_r"]

                        u_j = u_dirs[j]
                        s_j = float(np.dot(u_j, -v_ij))
                        mutual = s_j > 0

                        j_leaving = self.idx[j] == 5 and self.pos[j][1] < 1.45
                        i_leaving = self.idx[i] == 5 and self.pos[i][1] < 1.45

                        i_has_prio = False
                        if mutual:
                            if i_leaving and not j_leaving:
                                i_has_prio = True
                            elif j_leaving and not i_leaving:
                                i_has_prio = False
                            elif self.wait[j] > 0 and self.wait[i] <= 0:
                                i_has_prio = True
                            elif self.wait[i] > 0 and self.wait[j] <= 0:
                                i_has_prio = False
                            else:
                                i_has_prio = (i < j)

                        k_i = self.idx[i] % len(path)
                        target_p = path[k_i]
                        j_at_stop = float(np.linalg.norm(self.pos[j] - target_p)) < 0.45
                        j_paused = self.wait[j] > 0

                        if (j_paused or j_at_stop) and self.pos[i][1] > 1.40:
                            d_stop = max(r_sum + 0.16, 0.65)
                            d_slow = d_stop + 0.20
                        else:
                            d_stop = r_sum + 0.16
                            d_slow = d_stop + 0.22

                        lat_limit = 0.45
                        if lat < lat_limit:
                            if mutual and i_has_prio and d_ij > 0.42:
                                hg = 0.5
                            elif d_ij <= d_stop:
                                hg = 0.0
                            elif d_ij < d_slow:
                                hg = (d_ij - d_stop) / (d_slow - d_stop)
                            else:
                                hg = 1.0
                            give = min(give, hg)

            if self.wait[i] > 0:
                tag = self.faces[i][self.at[i]]
                if not (tag is None and give == 0.0):  # counter use pauses while the hand is at the walker's elbow
                    self.wait[i] -= dt
                if tag in ("open", "look", "close") and self.visit and tag != "look":
                    self._visit_step(i, tag)
                yaw = self._pause_yaw(i)
            else:
                k = self.idx[i] % len(path)
                d = path[k] - self.pos[i]
                dist = float(np.linalg.norm(d))
                if dist < 0.05:  # at a waypoint: pause there, or turn to the next one without breaking stride
                    self.wait[i] = self.total[i] = self.pauses[i][k]
                    self.at[i] = k
                    self.idx[i] += 1
                    d = path[self.idx[i] % len(path)] - self.pos[i]
                    dist = float(np.linalg.norm(d))
                if self.wait[i] <= 0 and dist > 1e-9:
                    nk = self.idx[i] % len(path)
                    v = self.speed * (SLOW if self.slow[i][nk] else 1.0)
                    if self.pauses[i][nk] > 0 and not self.slow[i][nk]:  # the slow fridge steps are planned for SLOW
                        v *= float(np.clip(dist / BRAKE[0], BRAKE[1], 1.0))
                    moved = min(dist, v * give * dt)
                    self.pos[i] += d / dist * moved
                yaw = self._pause_yaw(i) if self.wait[i] > 0 else np.arctan2(d[1], d[0])  # no one-frame spin on arrival
            self.yaw[i] = float(yaw)
            self.phase[i] += 2 * math.pi * moved / STRIDE
            target = min(1.0, moved / (self.speed * dt)) if moved > 0 else 0.0  # stride scaled to speed: a full stride at 45% speed puts the toe in the door swing
            self.amp[i] += float(np.clip(target - self.amp[i], -AMP_RATE * dt, AMP_RATE * dt))
            self.pose_limbs(model, i)
            mid = model.body(f"person_{i}").mocapid[0]
            bob = BOB * self.amp[i] * (1 - abs(math.cos(self.phase[i])))
            data.mocap_pos[mid] = [self.pos[i][0], self.pos[i][1], bob]
            data.mocap_quat[mid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            out.append([float(self.pos[i][0]), float(self.pos[i][1])])
        if self._door is not None:
            qadr, dadr, geoms = self._door
            data.qpos[qadr] = self.door_q
            data.qvel[dadr] = 0.0
            self.driven_geoms = geoms if self.door_q > 0.005 else set()
        return out
