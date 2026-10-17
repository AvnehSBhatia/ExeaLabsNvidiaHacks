"""Policies for the sim: act(obs) -> action (7 arm joint targets + 1 gripper opening target, meters).

ScriptedExpert  IK waypoint controller (the data generator).
VLMPlanner      instruction + scene objects -> plan JSON via Nemotron (Token Factory) or a rule-based fallback.
VLAClient       HTTP client for a GR00T N1.x policy server (GR00T_POLICY_URL); not bundled.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np

from arrival.sim import ik
from arrival.sim.robot import BASE_SPEED, BASE_TURN, FINGER_OPEN, HOME

MINJERK_PEAK = 1.875  # peak over mean speed of a min-jerk move: sizes a base drive so it never exceeds BASE_SPEED
STOP_CLEAR = 0.7  # the base waits while a person is this close ahead of it (0.12 pedestal + 0.17 torso + margin)


class Policy:
    def reset(self, env, task: dict) -> None:  # noqa: D401
        pass

    def act(self, obs: dict) -> np.ndarray:
        raise NotImplementedError


class ScriptedExpert(Policy):
    """Plans eef waypoints for the task, solves IK per waypoint (biased toward HOME), then plays joint targets
    with min-jerk timing; waypoints flagged `via` are passed through without stopping."""

    def __init__(self, seg_steps: int = 20, hold_steps: int = 8):
        self.seg_steps, self.hold_steps = seg_steps, hold_steps

    @staticmethod
    def _grip_yaw(env, yaw: float) -> float:
        """Parallel jaws are symmetric under a half turn: use yaw + pi when yaw is >120 deg from the hand's
        current heading, so the wrist never spins half a turn for an equivalent grasp."""
        x = env.data.site("eef").xmat.reshape(3, 3)[:, 0]
        return yaw + np.pi if np.cos(yaw - np.arctan2(x[1], x[0])) < -0.5 else yaw

    def _waypoints(self, env, task: dict) -> list[tuple]:
        """(pos, R, gripper, steps[, via[, base]]) list. base is a room-frame (x, y, yaw) the base stands at for
        that waypoint (None: where it already is); pos None keeps the arm's joints while the base drives."""
        fam, o, c = task["family"], FINGER_OPEN, 0.0
        if fam == "counter_to_counter" and getattr(env, "mobile", False):
            return self._drive_and_place(env, task, o, c)
        if fam in ("counter_to_sink", "counter_to_counter"):
            p = env.body_pos(task["object_id"])
            gz = p[2] + min(0.01, env.obj_half_h(task["object_id"]) * 0.2)
            tgt = env.place_point(task)
            rim = env.rim_grasp(task["object_id"])
            yaw = 0.0
            if rim is not None:  # bowl: pinch the rim nearest the robot, pads straddling the wall 2 cm below the lip
                off, yaw = rim
                p = p + np.array([off[0], off[1], 0.0])
                tgt = tgt + np.array([off[0], off[1], 0.0])
                gz = env.body_pos(task["object_id"])[2] + env.obj_half_h(task["object_id"]) - 0.02
            R = ik.down(self._grip_yaw(env, yaw))
            s = self.seg_steps
            return [((p[0], p[1], gz + 0.18), R, o, s), ((p[0], p[1], gz + 0.02), R, o, s),
                    ((p[0], p[1], gz), R, o, s // 2), ((p[0], p[1], gz), R, c, self.hold_steps),
                    ((p[0], p[1], gz + 0.22), R, c, s), ((tgt[0], tgt[1], gz + 0.22), R, c, 2 * s),
                    ((tgt[0], tgt[1], tgt[2]), R, c, s), ((tgt[0], tgt[1], tgt[2]), R, o, self.hold_steps),
                    ((tgt[0], tgt[1], tgt[2] + 0.2), R, o, s)]
        if fam == "open_drawer":
            h, axis = env.site_pos(task["handle"]), env.pull_axis(task)
            R = ik.down(self._grip_yaw(env, np.arctan2(axis[1], axis[0])))  # fingers close across the pull axis
            h = h + np.array([0, 0, 0.0])
            pre = h + np.array([0, 0, 0.12])
            s = self.seg_steps
            return [(pre, R, o, s), (h + 0.005 * axis, R, o, s), (h + 0.005 * axis, R, c, self.hold_steps)] + [
                (h + (0.005 + 0.32 * k / 6) * axis, R, c, s // 2, k < 6) for k in range(1, 7)] + [
                (h + 0.325 * axis, R, o, self.hold_steps), (h + 0.325 * axis + np.array([0, 0, 0.15]), R, o, s)]
        if fam == "put_in_drawer":
            # hand parallel to the counter front (yaw 90) so its long side stays clear of the overhang; the drop into
            # the drawer is a chain of short via-point steps, because one long joint-space move bows back over the
            # counter top and the fingers lose the mug on the slab
            p = env.body_pos(task["object_id"])
            gz = p[2] + min(0.01, env.obj_half_h(task["object_id"]) * 0.2)
            tgt = env.drawer_place_point(task)
            R = ik.down(self._grip_yaw(env, np.pi / 2))
            s = self.seg_steps
            lift = gz + 0.15
            n_steps = int(np.ceil((lift - tgt[2]) / 0.1))
            zs = np.linspace(lift, tgt[2], n_steps + 1)[1:]
            descent = [((tgt[0], tgt[1], z), R, c, s // 2, k < n_steps - 1) for k, z in enumerate(zs)]
            return [((p[0], p[1], gz + 0.18), R, o, s), ((p[0], p[1], gz + 0.02), R, o, s),
                    ((p[0], p[1], gz), R, o, s // 2), ((p[0], p[1], gz), R, c, self.hold_steps),
                    ((p[0], p[1], lift), R, c, s), ((tgt[0], tgt[1], lift), R, c, 2 * s)] + descent + [
                    ((tgt[0], tgt[1], tgt[2]), R, o, self.hold_steps), ((tgt[0], tgt[1], tgt[2] + 0.15), R, o, s)]
        if fam == "wipe_counter":
            # pinch the sponge across its short side, press it to the top and make two strokes along the counter
            p = env.body_pos(task["object_id"])
            top = env.place_point_top(task)
            hh = env.obj_half_h(task["object_id"])
            R = ik.down(self._grip_yaw(env, 0.0))
            s = self.seg_steps
            press = top + hh - 0.004
            strokes = [(p[0] + dx, p[1], press) for dx in (0.15, -0.15, 0.15, -0.15)]
            return [((p[0], p[1], p[2] + 0.15), R, o, s), ((p[0], p[1], p[2]), R, o, s),
                    ((p[0], p[1], p[2]), R, c, self.hold_steps), ((p[0], p[1], press), R, c, s // 2)] + [
                    (w, R, c, s, k < 3) for k, w in enumerate(strokes)] + [
                    ((p[0], p[1], press), R, c, s), ((p[0], p[1], press), R, o, self.hold_steps),
                    ((p[0], p[1], press + 0.15), R, o, s)]
        if fam == "close_drawer":
            # push the handle bar from outside with the closed fingers until the front meets the carcass
            h, axis = env.site_pos(task["handle"]), env.pull_axis(task)
            R = ik.down(self._grip_yaw(env, np.arctan2(axis[1], axis[0])))
            touch = h + 0.025 * axis  # closed finger block (6 mm half) against the bar (12 mm radius)
            s = self.seg_steps
            shut = touch - float(self.open_at) * axis
            return [(touch + 0.12 * axis + np.array([0, 0, 0.1]), R, c, s), (touch + 0.03 * axis, R, c, s)] + [
                (touch + (0.03 - (0.03 + float(self.open_at)) * k / 4) * axis, R, c, s // 2, k < 4) for k in range(1, 5)] + [
                (shut + 0.1 * axis, R, o, s)]
        if fam == "close_door":
            # push the handle bar back along its arc from the outside, hand nearly vertical as in open_door
            hinge, h, _ = env.door_geometry(task)
            r = np.linalg.norm((h - hinge)[:2])
            a_open = np.arctan2(*(h - hinge)[[1, 0]])
            s, up = self.seg_steps, np.array([0, 0, 1.0])
            grasp = np.array([0, 0, 0.04])

            def pt(a):
                return hinge + np.array([r * np.cos(a), r * np.sin(a), h[2] - hinge[2]]) + grasp

            def nrm(a):
                return -np.array([-np.sin(a), np.cos(a), 0.0])

            along = np.array([-nrm(a_open)[1], nrm(a_open)[0], 0.0])
            if np.dot(along, env.data.site("eef").xmat.reshape(3, 3)[:, 1]) < 0:
                along = -along

            def ori(a):
                rot = a - a_open
                ax = np.array([np.cos(rot) * along[0] - np.sin(rot) * along[1],
                               np.sin(rot) * along[0] + np.cos(rot) * along[1]])
                return ik.tilted(nrm(a)[:2], ax, np.radians(10))

            sweep = float(self.open_at) + np.radians(3)  # a little past closed so the latch gap closes
            out = [(pt(a_open) + 0.03 * nrm(a_open) + 0.15 * up, ori(a_open), c, s),
                   (pt(a_open) + 0.03 * nrm(a_open), ori(a_open), c, s)]
            for k in range(1, 9):
                a = a_open + sweep * k / 8
                out.append((pt(a) + 0.03 * nrm(a), ori(a), c, s // 2, k < 8))
            out.append((out[-1][0] + 0.15 * up, out[-1][1], o, s))
            return out
        if fam == "open_door":
            # The bar handle sits 2-3 cm below the counter-top overhang, so the hand comes down nearly vertical
            # (10 deg tilt keeps the hand body clear of the slab) with the fingers closing along the panel.
            # Segments interpolate in joint space, so the hand first goes out in front of the counter and high,
            # then straight down in front, then in to the bar; a direct move from HOME sweeps through the slab.
            hinge, h, _ = env.door_geometry(task)
            r = np.linalg.norm((h - hinge)[:2])
            a0 = np.arctan2(*(h - hinge)[[1, 0]])
            s, up = self.seg_steps, np.array([0, 0, 1.0])
            grasp = np.array([0, 0, 0.04])  # hold the bar near its top: a higher wrist reaches further along the arc

            def pt(a):
                return hinge + np.array([r * np.cos(a), r * np.sin(a), h[2] - hinge[2]]) + grasp

            def nrm(a):  # outward normal of the panel at door angle a (swings outward, clockwise from above)
                return -np.array([-np.sin(a), np.cos(a), 0.0])

            along = np.array([-nrm(a0)[1], nrm(a0)[0], 0.0])
            if np.dot(along, env.data.site("eef").xmat.reshape(3, 3)[:, 1]) < 0:
                along = -along  # equivalent parallel-jaw grasp that does not spin the wrist half a turn

            def ori(a):
                rot = a - a0
                ax = np.array([np.cos(rot) * along[0] - np.sin(rot) * along[1],
                               np.sin(rot) * along[0] + np.cos(rot) * along[1]])
                return ik.tilted(nrm(a)[:2], ax, np.radians(10))

            n0, g = nrm(a0), pt(a0)
            out = [(g + 0.2 * n0 + 0.35 * up, ori(a0), o, s), (g + 0.2 * n0, ori(a0), o, s),
                   (g + 0.005 * n0, ori(a0), o, s), (g + 0.005 * n0, ori(a0), c, self.hold_steps)]
            for k in range(1, 9):  # one smooth via-point pull: stop-and-go pulses let the bar slide off the pads
                a = a0 - np.radians(75) * k / 8
                out.append((pt(a) + 0.005 * nrm(a), ori(a), c, s // 2, k < 8))
            out.append((out[-1][0], out[-1][1], o, self.hold_steps))
            out.append((out[-1][0] + 0.15 * up, out[-1][1], o, s))
            return out
        raise ValueError(f"unknown family {fam}")

    def _drive_and_place(self, env, task: dict, o: float, c: float) -> list[tuple]:
        """counter_to_counter on a mobile base: grasp in front of the object, tuck the object in front of the base
        and above the counter top, drive along the counter front to the target, then place as counter_to_sink does."""
        from arrival.sim.collect import _front

        p = env.body_pos(task["object_id"])
        gz = p[2] + min(0.01, env.obj_half_h(task["object_id"]) * 0.2)
        tgt = env.place_point(task)
        rim = env.rim_grasp(task["object_id"])
        yaw = 0.0
        if rim is not None:
            off, yaw = rim
            p = p + np.array([off[0], off[1], 0.0])
            tgt = tgt + np.array([off[0], off[1], 0.0])
            gz = env.body_pos(task["object_id"])[2] + env.obj_half_h(task["object_id"]) - 0.02
        R = ik.down(self._grip_yaw(env, yaw))
        s = self.seg_steps
        here, there = env.base_pose(), env.drive_to
        n = _front(env.fx[task["source"]])[:2]
        carry = (here[0] - 0.25 * n[0], here[1] - 0.25 * n[1], 1.1)  # just off the counter front, above its top
        return [((p[0], p[1], gz + 0.18), R, o, s, False, here), ((p[0], p[1], gz + 0.02), R, o, s),
                ((p[0], p[1], gz), R, o, s // 2), ((p[0], p[1], gz), R, c, self.hold_steps),
                ((p[0], p[1], gz + 0.22), R, c, s), (carry, R, c, s),
                (None, None, c, s, False, there),
                ((tgt[0], tgt[1], gz + 0.22), R, c, 2 * s), ((tgt[0], tgt[1], tgt[2]), R, c, s),
                ((tgt[0], tgt[1], tgt[2]), R, o, self.hold_steps), ((tgt[0], tgt[1], tgt[2] + 0.2), R, o, s)]

    def reset(self, env, task: dict) -> None:
        from arrival.sim.collect import FPS

        self.env, self.last, self.held = env, None, 0
        if task["family"] in ("close_drawer", "close_door"):
            self.open_at = float(env.data.joint(task["joint"]).qpos[0])  # how far the part starts open
        q = np.array(HOME)
        self.plan: list[np.ndarray] = []
        base = env.base_pose()
        prev = np.concatenate([q, [FINGER_OPEN], base])
        self.ik_error = 0.0
        chain: list[tuple[np.ndarray, int]] = []
        for pos, R, g, steps, *rest in self._waypoints(env, task):
            via = bool(rest[0]) if rest else False
            nxt_base = np.asarray(rest[1], dtype=float) if len(rest) > 1 and rest[1] is not None else base
            if pos is not None:
                pin = (env.badr, nxt_base - env.base0) if getattr(env, "mobile", False) else None
                q, err = ik.solve(env.model, env.data, pos, R, q, env.qadr, env.dofadr, fixed=pin)
                self.ik_error = max(self.ik_error, err)
            d_xy = float(np.hypot(*(nxt_base[:2] - base[:2])))
            d_yaw = abs(float(nxt_base[2] - base[2]))
            if d_xy > 1e-6 or d_yaw > 1e-6:  # a drive: long enough that the min-jerk peak stays under the limits
                steps = max(steps, int(np.ceil(MINJERK_PEAK * d_xy / BASE_SPEED * FPS)) + 2,
                            int(np.ceil(MINJERK_PEAK * d_yaw / BASE_TURN * FPS)) + 2)
            base = nxt_base
            nxt = np.concatenate([q, [g], base])
            chain.append((nxt, steps))
            if via:
                continue
            total = sum(n for _, n in chain)
            knots = np.cumsum([0] + [n for _, n in chain]) / total
            pts = np.array([prev] + [p for p, _ in chain])
            for k in range(1, total + 1):
                t = min(1.0, k / max(1, total - 2))
                s = t ** 3 * (10 - 15 * t + 6 * t * t)
                self.plan.append(np.array([np.interp(s, knots, pts[:, j]) for j in range(pts.shape[1])]))
            prev, chain = nxt, []
        self.t = 0

    def done(self) -> bool:
        return self.t >= len(self.plan)

    def act(self, obs: dict) -> np.ndarray:
        a = self.plan[min(self.t, len(self.plan) - 1)]
        if self.last is not None and self.t < len(self.plan) and getattr(self.env, "mobile", False):
            cur = self.env.base_pose()[:2]
            d = a[8:10] - cur
            if np.linalg.norm(d) > 1e-5:  # about to drive: wait for anyone close in front, the plan resumes after
                for p in self.env.people_xy:
                    v = np.asarray(p, dtype=float) - cur
                    if np.linalg.norm(v) < STOP_CLEAR and v @ d > 0:
                        self.held += 1
                        return self.last
        self.last = a
        self.t += 1
        return a


# ---------- language planning ----------

def rule_plan(instruction: str, objects: list[str], fixtures: list[str]) -> dict:
    """Deterministic parser: family + object + target from keywords."""
    s = instruction.lower()
    obj = next((o for o in objects if o.split("_")[0] in s), None)
    if "drawer" in s:
        return {"family": "open_drawer", "object": None, "target": None}
    if "door" in s:
        return {"family": "open_door", "object": None, "target": None}
    if "sink" in s:
        return {"family": "counter_to_sink", "object": obj, "target": next((f for f in fixtures if "sink" in f), None)}
    return {"family": "counter_to_counter", "object": obj, "target": None}


class VLMPlanner:
    """Turns an instruction + scene inventory into a plan JSON with Nemotron; falls back to rule_plan.
    Maintains a response cache so tests and repeated runs execute offline without network."""

    def __init__(self, log=print, cache_path: Path | str | None = None):
        self.log = log
        self.source = None
        self.cache_path = Path(cache_path) if cache_path else Path(__file__).resolve().parents[1] / "models" / "nemotron_cache.json"
        self._cache = self._load_cache()

    def _cache_key(self, instruction: str, objects: list[str], fixtures: list[str]) -> str:
        return f"{instruction.strip().lower()}::{sorted(objects)}::{sorted(fixtures)}"

    def _load_cache(self) -> dict:
        if self.cache_path and self.cache_path.is_file():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception as e:
                self.log(f"[sim] VLMPlanner: cache load failed ({e!r}); starting with empty cache")
                return {}
        return {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=2) + "\n", encoding="utf-8")
        except Exception as e:
            self.log(f"[sim] VLMPlanner: warning, could not write cache to {self.cache_path}: {e}")

    def plan(self, instruction: str, objects: list[str], fixtures: list[str]) -> dict:
        from arrival.models.token_factory import TokenFactoryNotConfigured, complete

        key = self._cache_key(instruction, objects, fixtures)
        if key in self._cache:
            self.source = "cache"
            return dict(self._cache[key])

        try:
            txt = complete([{"role": "system", "content": "Return only JSON: {\"family\": one of counter_to_sink|"
                             "counter_to_counter|open_drawer|open_door, \"object\": id|null, \"target\": id|null}"},
                            {"role": "user", "content": json.dumps({"instruction": instruction, "objects": objects,
                                                                    "fixtures": fixtures})}], role="super")
            plan = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            self.source = "token_factory"
            self._cache[key] = plan
            self._save_cache()
        except TokenFactoryNotConfigured as e:
            self.log(f"[sim] VLMPlanner: Token Factory not configured ({e}); using rule-based parser")
            plan, self.source = rule_plan(instruction, objects, fixtures), "rule_based"
        except Exception as e:  # network / parse errors also fall back, loudly
            self.log(f"[sim] VLMPlanner: model call failed ({e!r}); using rule-based parser")
            plan, self.source = rule_plan(instruction, objects, fixtures), "rule_based"
        return plan


class VLANotConfigured(RuntimeError):
    pass


class VLAClient(Policy):
    """Client for a GR00T N1.x policy server (e.g. gr00t inference service behind an HTTP shim).

    POST {GR00T_POLICY_URL}/act  JSON:
      {"video.ego_view": <base64 PNG HxWx3 uint8>, "state.single_arm": [7], "state.gripper": [1],
       "state.base": [3], "annotation.human.task_description": "<instruction>"}
    Response: {"action.single_arm": [[7], ...], "action.gripper": [[1], ...], "action.base": [[3], ...]}
    (absolute joint targets, chunk; action.base is optional and the base holds without it)
    Matches meta/modality.json of datasets written by `arrival sim`.
    """

    def __init__(self, url: str | None = None, timeout: float = 10.0):
        self.url = url or os.environ.get("GR00T_POLICY_URL")
        if not self.url:
            raise VLANotConfigured("GR00T_POLICY_URL is not set; start a GR00T policy server and export its URL.")
        self.timeout = timeout
        self.queue: list[np.ndarray] = []

    def reset(self, env, task: dict) -> None:
        self.instruction = task["instruction"]
        self.queue = []

    def act(self, obs: dict) -> np.ndarray:
        if not self.queue:
            import io

            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(obs["ego_view"]).save(buf, format="PNG")
            body = json.dumps({"video.ego_view": base64.b64encode(buf.getvalue()).decode(),
                               "state.single_arm": obs["state"][:7].tolist(), "state.gripper": obs["state"][7:8].tolist(),
                               "state.base": obs["state"][8:11].tolist(),
                               "annotation.human.task_description": self.instruction}).encode()
            req = urllib.request.Request(self.url.rstrip("/") + "/act", data=body, method="POST",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.load(r)
            bases = out.get("action.base") or [[]] * len(out["action.single_arm"])  # absent: the base holds
            self.queue = [np.concatenate([a, g, b]) for a, g, b in zip(out["action.single_arm"], out["action.gripper"], bases)]
        return np.asarray(self.queue.pop(0), dtype=float)
