"""Policies for the sim: act(obs) -> action (7 arm joint targets + 1 gripper opening target, meters).

ScriptedExpert  IK waypoint controller (the data generator).
VLMPlanner      instruction + scene objects -> plan JSON via Nemotron (Token Factory) or a rule-based fallback.
VLAClient       HTTP client for a GR00T N1.x policy server (GR00T_POLICY_URL); not bundled.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

from arrival.sim import ik
from arrival.sim.scene import fridge_mirror
from arrival.sim.robot import BASE_SPEED, BASE_TURN, FINGER_OPEN, HAND, HOME

MINJERK_PEAK = 1.875  # peak over mean speed of a min-jerk move: sizes a base drive so it never exceeds BASE_SPEED
STOP_CLEAR = 0.7  # the base waits while a person is this close ahead of it (0.12 pedestal + 0.17 torso + margin)


class Policy:
    def reset(self, env, task: dict) -> None:  # noqa: D401
        pass

    def act(self, obs: dict) -> np.ndarray:
        raise NotImplementedError


FRIDGE_HOOK_ALONG = 0.015  # eef past the bar axis toward the free edge, so the hand clears the edge
FRIDGE_HOOK_OUT = 0.035  # eef out from the bar axis, so the open inner pad clears the panel on the way in
FRIDGE_HOOK = np.radians(40)  # door angle the hooked pull covers: the arm follows the bar exactly up to here
FRIDGE_TURN = 0.3  # share of the door's turn the hand follows while hooked; more folds the elbow onto its stop
DOOR_TURN = 0.0  # share of the door's turn the hand follows while pulling: the bar is vertical, so the pads can
# stay put and it turns between them; following it all the way spends joint 7's range on the swing and pins it at its stop
FRIDGE_FLIP = np.diag([-1.0, -1.0, 1.0])  # half a turn about the approach axis
FRIDGE_CLEAR = 0.04  # closed fingers slide this far out along the panel, off the bar's free side
FRIDGE_FAR = 0.90  # metres from the hinge where the hand rounds the free edge (the edge is at 0.745)
FRIDGE_BACK = 0.08  # how far inside the panel the hand rounds the edge and lines up before the push
FRIDGE_PUSH_AT = 0.62  # metres from the hinge where the closed fingertips push the inside face
FRIDGE_PUSH_GAP = 0.025  # eef behind the inside face: the closed fingertips reach 2.3 cm past the eef
FRIDGE_PUSH_END = np.radians(80)  # door angle the push is planned to
FRIDGE_PUSH_ROLL = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # fingers stacked vertically
# for the push, so the hand is narrow along the panel and its far finger stays off the edge
FRIDGE_PULL_STEPS = 10  # control steps per quarter of the hooked pull and per sixth of the push
FRIDGE_SETTLE_STEPS = 6  # held after sliding off the bar, so the door has stopped when its angle is read
FRIDGE_SETTLE_IN = 0.04  # m the hand eases back toward the hinge for that hold: held where it slid off, joint 4
# sat at its stop for 43 steps and the quality check (40) flagged every open_fridge episode
FRIDGE_REPLAN_IK = 0.02  # m: the push is re-planned only when the planned hook is reachable; past this (the
# U kitchen's side-wall fridge plans 1.70 m) re-planning made the flailing arm fling the door open and count a success
DOOR_SWITCH = np.radians(40)  # close_door: door angle where the push from the side hands over to the push from in front
DOOR_SIDE_YAWS = (0.0, -0.4, 0.4)  # rad the hand turns from the panel's direction for the side push, chosen per task
FLAP_OUT = 0.12  # m: how far out along the door normal the hand backs off after letting a flap door go
FAUCET_UNDER, FAUCET_PUSH = 0.045, 0.05  # m: closed fingers start this far below the lever tip, then travel up this far
FLAP_TILTS = (0.0, 0.35, 0.7, 1.35)  # rad below the door's horizontal for the hand on a flap-door bar; chosen per task
ROLL_TRIES = (0.0, 0.2, -0.2, 0.4, -0.4)  # rad of wrist roll about the tool axis tried when a plan parks a joint on its stop
MAX_REPLANS = 35  # (yaw, roll) combinations a re-plan may try
YAW_TRIES = (0.0, 0.15, -0.15, 0.3, -0.3, 0.5, -0.5)  # rad of grasp yaw about the vertical, likewise
LIMIT_PLAN_STEPS = 38  # a plan with a joint on its stop longer than this is re-planned off the stops (quality flags at 40)
REPLAN_ACCEPT_STEPS = 30  # a re-plan this clear of the stops is taken at once; a longer one only as the best of all
FLAP_RELEASE = 0.5  # rad; a bottom-hinged door is let go here and falls to its flat stop (its handle swings
# toward the base as it opens, and past this angle the wrist would have to fold back under the arm)


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


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
        fam, o, c = task["family"], getattr(env, "finger_open", FINGER_OPEN), 0.0
        if fam == "counter_to_counter" and env.drive_to is not None:
            return self._drive_and_place(env, task, o, c)
        if fam in ("counter_to_sink", "counter_to_counter", "move_on_table"):
            p = env.body_pos(task["object_id"])
            gz = p[2] + min(0.01, env.obj_half_h(task["object_id"]) * 0.2)
            tgt = env.place_point(task)
            g = env.grasp(task["object_id"])
            yaw = env.run_yaw()  # fingers close across the counter, whichever wall it stands on
            if g is not None:  # rim or handle grasp: same xy offset at the set-down, eef kept at its height over the body
                p = p + np.array([g["off"][0], g["off"][1], 0.0])
                tgt = tgt + np.array([g["off"][0], g["off"][1], g["z"] - env.body_pos(task["object_id"])[2]])
                gz, yaw = g["z"], g["yaw"]
            R = ik.down(self._grip_yaw(env, yaw))
            s = self.seg_steps
            lift = [((p[0], p[1], gz + 0.22), R, c, s)]
            if g is not None:  # a wide object tips in the pads as it leaves the counter: lift it in short steps
                lift = [((p[0], p[1], gz + dz), R, c, s, dz < 0.22) for dz in (0.03, 0.08, 0.15, 0.22)]
            return [((p[0], p[1], gz + 0.18), R, o, s), ((p[0], p[1], gz + 0.02), R, o, s),
                    ((p[0], p[1], gz), R, o, s // 2), ((p[0], p[1], gz), R, c, self.hold_steps)] + lift + [
                    ((tgt[0], tgt[1], gz + 0.22), R, c, 2 * s),
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
            R = ik.down(self._grip_yaw(env, env.run_yaw() + np.pi / 2))
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
            g = env.grasp(task["object_id"])
            R = ik.down(self._grip_yaw(env, env.run_yaw() if g is None else g["yaw"]))
            s = self.seg_steps
            gz, press = p[2], top + hh - 0.004
            if g is not None:  # the spatula: the pinch already has the pad tips over the top, so it is not pressed lower
                gz = press = g["z"]
            u = np.array([np.cos(env.run_yaw()), np.sin(env.run_yaw())])  # strokes run along the counter
            strokes = [(p[0] + d * u[0], p[1] + d * u[1], press) for d in (0.15, -0.15, 0.15, -0.15)]
            return [((p[0], p[1], gz + 0.15), R, o, s), ((p[0], p[1], gz), R, o, s),
                    ((p[0], p[1], gz), R, c, self.hold_steps), ((p[0], p[1], press), R, c, s // 2)] + [
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
            # push the handle bar back along its arc from the outside, hand nearly vertical as in open_door. The door
            # starts open past where its outer face turns toward the base, so the hand first reaches round the free
            # edge and pushes from the side; that reach folds the shoulder back over the panel and cannot follow the
            # bar to closed without joint 2 on its -1.76 rad stop. At DOOR_SWITCH the hand lifts off, goes back to
            # its start pose and pushes the rest of the way from in front
            hinge, h, _ = env.door_geometry(task)
            r = np.linalg.norm((h - hinge)[:2])
            a_open = np.arctan2(*(h - hinge)[[1, 0]])
            a_mid = a_open + float(self.open_at) - DOOR_SWITCH
            a_end = a_open + float(self.open_at) + np.radians(3)  # a little past closed so the latch gap closes
            s, up = self.seg_steps, np.array([0, 0, 1.0])
            hand = getattr(env, "hand", HAND)
            grasp = np.array([0, 0, hand["door_z"]])
            start = (env.site_pos("eef"), env.data.site("eef").xmat.reshape(3, 3).copy())

            def pt(a):
                return hinge + np.array([r * np.cos(a), r * np.sin(a), h[2] - hinge[2]]) + grasp

            def nrm(a):
                return -np.array([-np.sin(a), np.cos(a), 0.0])

            def ori(a, a_ref, flip=False, yaw=0.0):
                # fingers along the panel at a_ref in the heading that matches the start pose (or the other one),
                # turned with the door from there and `yaw` more
                along = np.array([-nrm(a_ref)[1], nrm(a_ref)[0], 0.0])
                if (np.dot(along, start[1][:, 1]) < 0) != flip:
                    along = -along
                rot = a - a_ref + yaw
                ax = np.array([np.cos(rot) * along[0] - np.sin(rot) * along[1],
                               np.sin(rot) * along[0] + np.cos(rot) * along[1]])
                return ik.tilted(nrm(a)[:2], ax, np.radians(10))

            def plan(yaw, flip):
                out = [(pt(a_open) + hand["door_push"] * nrm(a_open) + 0.15 * up, ori(a_open, a_open, yaw=yaw), c, s),
                       (pt(a_open) + hand["door_push"] * nrm(a_open), ori(a_open, a_open, yaw=yaw), c, s)]
                for k in range(1, 5):
                    a = a_open + (a_mid - a_open) * k / 4
                    out.append((pt(a) + hand["door_push"] * nrm(a), ori(a, a_open, yaw=yaw), c, s // 2, k < 4))
                out += [(out[-1][0] + 0.15 * up, out[-1][1], c, s), (*start, c, s),
                        (pt(a_mid) + 0.2 * nrm(a_mid) + 0.35 * up, ori(a_mid, a_mid, flip), c, s),
                        (pt(a_mid) + 0.2 * nrm(a_mid), ori(a_mid, a_mid, flip), c, s),
                        (pt(a_mid) + hand["door_push"] * nrm(a_mid), ori(a_mid, a_mid, flip), c, s)]
                for k in range(1, 5):
                    a = a_mid + (a_end - a_mid) * k / 4
                    out.append((pt(a) + hand["door_push"] * nrm(a), ori(a, a_mid, flip), c, s // 2, k < 4))
                out.append((out[-1][0] + 0.15 * up, out[-1][1], o, s))
                return out

            # the reach round the free edge can leave joint 7 on its +2.90 stop with the hand square to the panel, and
            # which jaw heading reaches the bar from in front depends on how far the arm turns: plan the variants
            # and keep the one reset() would, the smallest joint 7 peak among those within 5 mm of the best IK error
            plans = [(wps, *self._plan(env, wps)[1:])
                     for wps in (plan(yaw, flip) for yaw in DOOR_SIDE_YAWS for flip in (False, True))]
            best_err = min(err for _, err, _ in plans)
            return min((pl for pl in plans if pl[1] <= best_err + 0.005), key=lambda pl: pl[2])[0]
        if fam == "open_door":
            # The bar handle sits 2-3 cm below the counter-top overhang, so the hand comes down nearly vertical
            # (15 deg tilt keeps the hand body clear of the slab) with the fingers closing along the panel.
            # Segments interpolate in joint space, so the hand first goes out in front of the counter and high,
            # then straight down in front, then in to the bar; a direct move from HOME sweeps through the slab.
            hinge, h, _ = env.door_geometry(task)
            r = np.linalg.norm((h - hinge)[:2])
            a0 = np.arctan2(*(h - hinge)[[1, 0]])
            s, up = self.seg_steps, np.array([0, 0, 1.0])
            # Hold the bar near its top (the grasp fails from 0.115 up): a higher wrist reaches further along the arc,
            # and with the tilt it keeps the elbow (link4) above the swinging panel, which is solid to its full drawn
            # height (MAN-2).
            grasp = np.array([0, 0, 0.09])

            def pt(a):
                return hinge + np.array([r * np.cos(a), r * np.sin(a), h[2] - hinge[2]]) + grasp

            def nrm(a):  # outward normal of the panel at door angle a (swings outward, clockwise from above)
                return -np.array([-np.sin(a), np.cos(a), 0.0])

            along = np.array([-nrm(a0)[1], nrm(a0)[0], 0.0])
            if np.dot(along, env.data.site("eef").xmat.reshape(3, 3)[:, 1]) < 0:
                along = -along  # equivalent parallel-jaw grasp that does not spin the wrist half a turn

            def ori(a):
                rot = DOOR_TURN * (a - a0)
                ax = np.array([np.cos(rot) * along[0] - np.sin(rot) * along[1],
                               np.sin(rot) * along[0] + np.cos(rot) * along[1]])
                return ik.tilted(nrm(a)[:2], ax, np.radians(15))

            n0, g = nrm(a0), pt(a0)
            out = [(g + 0.2 * n0 + 0.35 * up, ori(a0), o, s), (g + 0.2 * n0, ori(a0), o, s),
                   (g + 0.005 * n0, ori(a0), o, s), (g + 0.005 * n0, ori(a0), c, self.hold_steps)]
            for k in range(1, 9):  # one smooth via-point pull: stop-and-go pulses let the bar slide off the pads
                a = a0 - np.radians(80) * k / 8
                out.append((pt(a) + 0.005 * nrm(a), ori(a), c, s // 2, k < 8))
            out.append((out[-1][0], out[-1][1], o, self.hold_steps))
            out.append((out[-1][0] + 0.15 * up, out[-1][1], o, s))
            return out
        if fam == "open_fridge":
            # The door is hinged at the fixture's left and swings out into the room. From this base the arm can
            # follow the bar's arc only to about 40 deg, so the plan has two parts: the fingers close round the
            # vertical bar from beyond the free edge and pull to 40 deg, then slide off the bar's free side, round
            # the free edge to the inside face and push the door on with the closed fingertips.
            hinge, h, _ = env.door_geometry(task)
            r = np.linalg.norm((h - hinge)[:2])
            a0 = np.arctan2(*(h - hinge)[[1, 0]])
            m = fridge_mirror(env.fx[task["fixture"]])  # -1: right-hinged, the whole plan is the mirror image
            yaw = env.fx[task["fixture"]].get("yaw", 0.0)  # a fridge on a side wall turns the whole plan with it
            qa = lambda q: yaw + (np.pi if m < 0 else 0.0) + m * q  # world heading of the panel at door angle q
            s, up = self.seg_steps, np.array([0, 0, 1.0])

            def along(a):  # from the hinge toward the free edge, panel heading a
                return np.array([np.cos(a), np.sin(a), 0.0])

            def nrm(a):  # outward normal of the panel (left: swings counter-clockwise from above; right: clockwise)
                return m * np.array([-np.sin(a), np.cos(a), 0.0])

            def pt(a):
                return hinge + (r + FRIDGE_HOOK_ALONG) * along(a) + FRIDGE_HOOK_OUT * nrm(a) + (h[2] - hinge[2]) * up

            def ori(a):  # the other parallel-jaw heading: the first one starts joint 7 on its +2.90 stop
                return ik.facing(-along(a0 + FRIDGE_TURN * (a - a0))[:2]) @ FRIDGE_FLIP

            def push(q):  # eef with the closed fingertips on the inside face at door angle q
                return hinge + FRIDGE_PUSH_AT * along(qa(q)) - FRIDGE_PUSH_GAP * nrm(qa(q)) + (h[2] - hinge[2]) * up

            def pori(q):
                return ik.facing(nrm(qa(q))[:2]) @ FRIDGE_PUSH_ROLL

            d0, g = along(a0), pt(a0)
            out = [(g + 0.25 * d0 + 0.15 * up, ori(a0), o, s), (g + 0.25 * d0, ori(a0), o, s),
                   (g + 0.08 * d0, ori(a0), o, s, True), (g, ori(a0), o, s), (g, ori(a0), c, self.hold_steps)]
            for k in range(1, 5):
                a = a0 + m * FRIDGE_HOOK * k / 4
                out.append((pt(a), ori(a), c, FRIDGE_PULL_STEPS, k < 4))
            a, q1 = a0 + m * FRIDGE_HOOK, FRIDGE_HOOK
            off = pt(a) + FRIDGE_CLEAR * along(a)
            back = hinge - off
            back[2] = 0.0
            out += [(off, ori(a), c, s), (off + FRIDGE_SETTLE_IN * back / np.linalg.norm(back), ori(a), c, FRIDGE_SETTLE_STEPS)]

            def rest(q1):
                # Round the free edge to the inside face and push on, planned for the door at q1. Sliding off the
                # bar knocks the door back by an amount that depends on the hinge side and the room (about 10 deg),
                # and the rounding point is only FRIDGE_BACK behind the panel, so act() calls this again with the
                # angle measured after FRIDGE_SETTLE_STEPS and replaces the nominal plan built here for FRIDGE_HOOK.
                far = hinge + FRIDGE_FAR * along(qa(q1)) + (h[2] - hinge[2]) * up  # clear of the free edge
                wps = [(far - FRIDGE_BACK * nrm(qa(q1)), pori(q1), c, s, True),
                       (push(q1) - FRIDGE_BACK * nrm(qa(q1)), pori(q1), c, s)]
                start = min(q1 - np.radians(3), FRIDGE_PUSH_END)
                for k in range(7):
                    q = start + (FRIDGE_PUSH_END - start) * k / 6
                    wps.append((push(q), pori(q), c, FRIDGE_PULL_STEPS, k < 6))
                q = FRIDGE_PUSH_END
                return wps + [(push(q) - 0.12 * nrm(qa(q)), pori(q), c, s),
                              (push(q) - 0.12 * nrm(qa(q)) + 0.1 * up, pori(q), c, s)]

            self.fridge_replan = (sum(w[3] for w in out), rest, f'{task["fixture"]}_door')
            return out + rest(q1)
        if fam in ("open_oven", "open_dishwasher"):
            # The bar sits on the outside of a bottom-hinged door, so a top-down grasp would put the hand body in
            # the panel as soon as the door tilts. The hand points into the panel with the fingers closing across
            # the bar vertically, and that frame turns with the door; at FLAP_RELEASE the gripper opens and gravity
            # takes the door to its flat stop.
            pivot, h, _ = env.flap_geometry(task)
            hand = getattr(env, "hand", HAND)  # pad and wrist geometry of the arm that is loaded
            n = env.pull_axis(task)  # outward normal of the closed panel
            s, up = self.seg_steps, np.array([0, 0, 1.0])
            a = np.cross(up, n)  # hinge axis direction
            rel = h - pivot

            def pt(q):
                """Handle position after the door turns q about the hinge (top edge forward and down)."""
                return pivot + rel * np.cos(q) + np.cross(a, rel) * np.sin(q) + a * np.dot(a, rel) * (1 - np.cos(q))

            def nrm(q):
                return n * np.cos(q) - up * np.sin(q)

            def ori(q, tilt):
                # pointing into the panel and `tilt` below the door's horizontal, fingers closing along the door's
                # up direction tilted the same way; a bigger tilt keeps the wrist higher and closer to the base
                nq, uq = nrm(q), up * np.cos(q) + n * np.sin(q)
                z = -nq * np.cos(tilt) - uq * np.sin(tilt)
                y = uq * np.cos(tilt) - nq * np.sin(tilt)
                if np.dot(y, env.data.site("eef").xmat.reshape(3, 3)[:, 1]) < 0:
                    y = -y
                return np.column_stack([np.cross(y, z), y, z])

            # sign check for pt: the handle must move outward (+n) as q grows
            if np.dot(pt(0.3) - pt(0.0), n) < 0:
                a = -a

            def chain(tilt):
                # the hand keeps the grasp orientation through the pull and the round bar turns between the pads:
                # turning with the door pitches the hand up as the bar swings in, which the wrist reaches only with
                # joint 5 or joint 2 on its stop
                R = ori(0, tilt)
                out = [(pt(0) + 0.18 * n + 0.05 * up, R, o, s), (pt(0) + hand["flap_bar"] * n, R, o, s),
                       (pt(0) + hand["flap_bar"] * n, R, c, self.hold_steps)]
                for k in range(1, 9):
                    q = FLAP_RELEASE * k / 8
                    out.append((pt(q) + hand["flap_bar"] * nrm(q), R, c, s // 2, k < 8))
                q = FLAP_RELEASE
                out.append((pt(q) + hand["flap_bar"] * nrm(q), R, o, self.hold_steps))
                out.append((pt(q) + FLAP_OUT * nrm(q) + hand["flap_up"] * up, R, o, s))
                return out

            # which tilt reaches depends on the bar's height and distance (oven bars sit higher than dishwasher
            # bars), so take the candidate whose whole chain has the smallest IK residual
            return min((chain(t) for t in FLAP_TILTS), key=lambda wps: self._chain_error(env, wps))
        if fam == "turn_on_faucet":
            # push the lever tip up from below with the closed fingers: hand horizontal, pointing at the wall
            tip = env.site_pos(task["handle"])
            n = env.pull_axis(task)  # the counter front direction; the lever sits at the back of the counter
            R = ik.facing(-n)
            if np.dot(R[:, 1], env.data.site("eef").xmat.reshape(3, 3)[:, 1]) < 0:
                R = R @ np.diag([-1, -1, 1])  # same grasp, half a turn less wrist roll
            s = self.seg_steps
            # the tip rises at most 3.8 cm over the lever's 1.2 rad range, so the push stops short of driving it
            # into the stop: 5 cm of hand travel from 3.5 cm below the tip lands the lever near 0.9 rad
            under = tip - np.array([0, 0, FAUCET_UNDER])
            lift = FAUCET_PUSH
            return [(under + 0.12 * n + np.array([0, 0, 0.06]), R, c, s), (under + 0.12 * n, R, c, s),
                    (under, R, c, s)] + [(under + np.array([0, 0, lift * k / 3]), R, c, s // 2, k < 3) for k in (1, 2, 3)] + [
                    (under + np.array([0, 0, lift]) + 0.12 * n, R, c, s)]
        if fam in ("load_dishwasher", "unload_dishwasher"):
            # pick by the body as for counter_to_sink; the set-down into the rack (or onto the counter) is a chain of
            # short via-point steps like put_in_drawer, so the joint-space path does not bow through the rack rails
            p = env.body_pos(task["object_id"])
            gz = p[2] + min(0.01, env.obj_half_h(task["object_id"]) * 0.2)
            tgt = env.rack_place_point(task) if fam == "load_dishwasher" else env.counter_place_point(task)
            R = ik.down(self._grip_yaw(env, env.run_yaw()))
            s = self.seg_steps
            lift = max(gz, tgt[2]) + 0.15
            n_steps = max(1, int(np.ceil((lift - tgt[2]) / 0.1)))
            zs = np.linspace(lift, tgt[2], n_steps + 1)[1:]
            descent = [((tgt[0], tgt[1], z), R, c, s // 2, k < n_steps - 1) for k, z in enumerate(zs)]
            # the approach to an object low in the rack is the same kind of chain: one long joint-space move from
            # HOME dips the hand onto the dishwasher's top edge on the way down
            n_app = max(1, int(np.ceil((lift - gz - 0.02) / 0.1)))
            zs = np.linspace(lift, gz + 0.02, n_app + 1)[1:]
            approach = [((p[0], p[1], z), R, o, s // 2, k < n_app - 1) for k, z in enumerate(zs)]
            rise = [((p[0], p[1], z), R, c, s // 2, k < n_app - 1) for k, z in enumerate(zs[::-1])]  # same pace up
            return [((p[0], p[1], lift), R, o, s)] + approach + [
                    ((p[0], p[1], gz), R, o, s // 2), ((p[0], p[1], gz), R, c, self.hold_steps)] + rise + [
                    ((p[0], p[1], lift), R, c, s // 2), ((tgt[0], tgt[1], lift), R, c, 2 * s)] + descent + [
                    ((tgt[0], tgt[1], tgt[2]), R, o, self.hold_steps), ((tgt[0], tgt[1], tgt[2] + 0.15), R, o, s)]
        raise ValueError(f"unknown family {fam}")

    @staticmethod
    def _chain_error(env, wps) -> float:
        q, worst = np.array(HOME), 0.0
        for pos, R, *_ in wps:
            q, err = ik.solve(env.model, env.data, pos, R, q, env.qadr, env.dofadr)
            worst = max(worst, err)
        return worst

    def _drive_and_place(self, env, task: dict, o: float, c: float) -> list[tuple]:
        """counter_to_counter on a mobile base: grasp in front of the object, tuck the object in front of the base
        and above the counter top, drive along the counter front to the target, then place as counter_to_sink does."""
        from arrival.sim.collect import _front

        p = env.body_pos(task["object_id"])
        gz = p[2] + min(0.01, env.obj_half_h(task["object_id"]) * 0.2)
        tgt = env.place_point(task)
        g = env.grasp(task["object_id"])
        yaw = env.run_yaw()
        if g is not None:  # rim or handle grasp: same xy offset at the set-down, eef kept at its height over the body
            p = p + np.array([g["off"][0], g["off"][1], 0.0])
            tgt = tgt + np.array([g["off"][0], g["off"][1], g["z"] - env.body_pos(task["object_id"])[2]])
            gz, yaw = g["z"], g["yaw"]
        R = ik.down(self._grip_yaw(env, yaw))
        s = self.seg_steps
        here, there = env.base_pose(), env.drive_to
        n = _front(env.fx[task["source"]])[:2]
        carry = (here[0] - 0.25 * n[0], here[1] - 0.25 * n[1], 1.1)  # just off the counter front, above its top
        lift = [((p[0], p[1], gz + 0.22), R, c, s)]
        if g is not None:  # a wide object tips in the pads as it leaves the counter: lift it in short steps
            lift = [((p[0], p[1], gz + dz), R, c, s, dz < 0.22) for dz in (0.03, 0.08, 0.15, 0.22)]
        return [((p[0], p[1], gz + 0.18), R, o, s, False, here), ((p[0], p[1], gz + 0.02), R, o, s),
                ((p[0], p[1], gz), R, o, s // 2), ((p[0], p[1], gz), R, c, self.hold_steps)] + lift + [
                (carry, R, c, s),
                (None, None, c, s, False, there),
                ((tgt[0], tgt[1], gz + 0.22), R, c, 2 * s), ((tgt[0], tgt[1], tgt[2]), R, c, s),
                ((tgt[0], tgt[1], tgt[2]), R, o, self.hold_steps), ((tgt[0], tgt[1], tgt[2] + 0.2), R, o, s)]

    def reset(self, env, task: dict) -> None:
        self.env, self.last, self.held = env, None, 0
        self.fridge_replan = None  # (plan step, phase-2 builder, door joint) set by the open_fridge plan
        if task["family"] in ("close_drawer", "close_door"):
            self.open_at = float(env.data.joint(task["joint"]).qpos[0])  # how far the part starts open
        wps = self._waypoints(env, task)
        # Parallel jaws make a half turn about the approach axis the same grasp. Plan both and keep the one whose
        # wrist stays furthest from the joint 7 stops: a fixed world yaw costs joint 7 the whole base swing, which
        # with one of the two headings pins it at the +2.90 rad stop for the second half of the move.
        flip = np.diag([-1.0, -1.0, 1.0])
        plans = [self._plan(env, wps)]
        if task["family"] not in ("open_door", "close_door"):
            # flipped, the cabinet-door grasp lands on the other wrist branch (joint 5 negative): link5 or the elbow
            # then meets the solid panel (#47) and the bar slips out of the pads at 26 to 36 deg; close_door picks
            # its heading per push in _waypoints
            plans.append(self._plan(env, [(p, None if R is None else R @ flip, *rest) for p, R, *rest in wps]))
        best_err = min(err for _, err, _ in plans)
        self.plan, self.ik_error, _ = min((pl for pl in plans if pl[1] <= best_err + 0.005), key=lambda pl: pl[2])
        if task["family"] != "close_door" and self._limit_run(env, self.plan) > LIMIT_PLAN_STEPS:
            # The plan parks a joint on its stop for a stretch the quality check would flag (and a real arm would
            # strain against). The grasp is free to yaw a little about the vertical (the bar is long and the hand
            # closes on it), to roll a little about the tool axis and to take either jaw heading. Re-plan with
            # those, smallest change first, and take the first that reaches as well and stays off the stops;
            # failing that, the one with the shortest stretch if that is still clear of the check.
            # close_door is left out: its push depends on the exact fingertip pose (a different yaw, roll or tilt
            # misses the handle on counter_b, and the other jaw heading puts a finger 2.5 cm into the door front).
            cands = sorted(itertools.product(YAW_TRIES, ROLL_TRIES), key=lambda t: abs(t[0]) + abs(t[1]))[:MAX_REPLANS]
            best = None
            for yaw, roll in cands:
                for f in (np.eye(3), flip):
                    pl = self._plan(env, [(p, None if R is None else _rz(yaw) @ R @ f @ _rz(roll), *rest)
                                          for p, R, *rest in wps])
                    run = self._limit_run(env, pl[0])
                    if pl[1] <= self.ik_error + 0.005 and run < self._limit_run(env, self.plan):
                        if best is None or run < best[0]:
                            best = (run, pl)
                        if run <= REPLAN_ACCEPT_STEPS:
                            break
                else:
                    continue
                break
            if best is not None and best[0] <= LIMIT_PLAN_STEPS:
                self.plan, self.ik_error, _ = best[1]
        self.t = 0

    @staticmethod
    def _limit_run(env, plan) -> int:
        """Longest run of plan steps with any arm joint inside the quality check's band at its range limit."""
        from arrival.export.quality import LIMIT_FRAC, longest_run

        lo, hi = env.model.jnt_range[[env.model.dof_jntid[i] for i in env.dofadr]].T
        q = np.asarray(plan)[:, :7]
        band = LIMIT_FRAC * (hi - lo)
        return max(longest_run(((q[:, j] <= lo[j] + band[j]) | (q[:, j] >= hi[j] - band[j]))) for j in range(7))

    @staticmethod
    def _plan(env, waypoints, start=None) -> tuple[list[np.ndarray], float, float]:
        """Joint-space plan for the waypoints: (targets per step, worst IK position error, peak |joint 7|). Every
        target is 11 wide: 7 arm joints, the gripper, then the base pose (the parked one unless a waypoint drives).
        `start` is the 11-wide target to continue from (default: HOME, gripper open, the parked base)."""
        from arrival.sim.collect import FPS

        q = np.array(HOME)
        plan: list[np.ndarray] = []
        base = env.base_pose()
        prev = np.concatenate([q, [getattr(env, "finger_open", FINGER_OPEN)], base])
        if start is not None:
            prev = np.asarray(start, dtype=float).copy()
            q, base = prev[:7].copy(), prev[8:].copy()
        ik_error, q7 = 0.0, abs(q[6])
        chain: list[tuple[np.ndarray, int]] = []
        for pos, R, g, steps, *rest in waypoints:
            via = bool(rest[0]) if rest else False
            nxt_base = np.asarray(rest[1], dtype=float) if len(rest) > 1 and rest[1] is not None else base
            if pos is not None:
                pin = (env.badr, nxt_base - env.base0) if env.mobile else None  # solve with the base where it will stand
                q, err = ik.solve(env.model, env.data, pos, R, q, env.qadr, env.dofadr, fixed=pin)
                ik_error, q7 = max(ik_error, err), max(q7, abs(float(q[6])))
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
                plan.append(np.array([np.interp(s, knots, pts[:, j]) for j in range(pts.shape[1])]))
            prev, chain = nxt, []
        return plan, ik_error, q7

    def done(self) -> bool:
        return self.t >= len(self.plan)

    def _replan_fridge(self) -> None:
        """Replace the rest of the open_fridge plan with one built for the door angle measured now."""
        _, rest, joint = self.fridge_replan
        self.fridge_replan = None
        wps = rest(float(self.env.data.joint(joint).qpos[0]))
        flip = np.diag([-1.0, -1.0, 1.0])
        plans = [self._plan(self.env, w, start=self.last) for w in
                 (wps, [(p, None if R is None else R @ flip, *r) for p, R, *r in wps])]
        best_err = min(err for _, err, _ in plans)
        plan, err, _ = min((pl for pl in plans if pl[1] <= best_err + 0.005), key=lambda pl: pl[2])
        self.plan = self.plan[:self.t] + plan
        self.ik_error = max(self.ik_error, err)

    def act(self, obs: dict) -> np.ndarray:
        if self.fridge_replan is not None and self.t == self.fridge_replan[0] and self.last is not None:
            if self.ik_error <= FRIDGE_REPLAN_IK:
                self._replan_fridge()
            self.fridge_replan = None  # an arm that cannot follow the hook is not helped by a new push plan
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
    closing = any(w in s for w in ("close", "shut", "push"))
    if "fridge" in s:
        return {"family": "open_fridge", "object": None, "target": None}
    if "wipe" in s or "scrub" in s or "clean" in s:
        return {"family": "wipe_counter", "object": obj, "target": None}
    if "drawer" in s:
        if closing:
            return {"family": "close_drawer", "object": None, "target": None}
        if obj and any(w in s for w in ("put", "place", "move")):
            return {"family": "put_in_drawer", "object": obj, "target": None}
        return {"family": "open_drawer", "object": None, "target": None}
    if "door" in s:
        return {"family": "close_door" if closing else "open_door", "object": None, "target": None}
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
        self._offline = False  # set by the first network failure so later tasks skip the 60 s timeout

    def _cache_key(self, instruction: str, objects: list[str], fixtures: list[str],
                   research_context: str | None = None) -> str:
        base = f"{instruction.strip().lower()}::{sorted(objects)}::{sorted(fixtures)}"
        if research_context:
            return f"{base}::tavily={hashlib.sha1(research_context.encode()).hexdigest()[:12]}"  # hash() is salted per process
        return base

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

    @staticmethod
    def _parse_plan(txt: str, objects: list[str], fixtures: list[str]) -> dict:
        """First JSON object in the reply, checked against the known families, objects and fixtures."""
        from arrival.sim.tasks import SIM_FAMILIES

        plan, _ = json.JSONDecoder().raw_decode(txt[txt.index("{"):])
        if not isinstance(plan, dict) or plan.get("family") not in SIM_FAMILIES:
            raise ValueError(f"plan has no known family: {plan!r}")
        if plan.get("object") is not None and plan["object"] not in objects:
            raise ValueError(f"plan object {plan['object']!r} is not in the scene")
        if plan.get("target") is not None and plan["target"] not in fixtures:
            raise ValueError(f"plan target {plan['target']!r} is not in the scene")
        return {"family": plan["family"], "object": plan.get("object"), "target": plan.get("target")}

    def plan(self, instruction: str, objects: list[str], fixtures: list[str],
             research_context: str | None = None) -> dict:
        from arrival.models.token_factory import TokenFactoryNotConfigured, complete

        key = self._cache_key(instruction, objects, fixtures, research_context)
        if key in self._cache:
            self.source = "cache"
            return dict(self._cache[key])
        if self._offline:
            self.source = "rule_based"
            return rule_plan(instruction, objects, fixtures)

        try:
            user_payload = {"instruction": instruction, "objects": objects, "fixtures": fixtures}
            if research_context:
                user_payload["research_context"] = research_context
            txt = complete([{"role": "system", "content": "Return only JSON: {\"family\": one of counter_to_sink|"
                             "counter_to_counter|open_drawer|open_door|close_drawer|close_door|put_in_drawer|wipe_counter|open_fridge, \"object\": id|null, \"target\": id|null}"},
                            {"role": "user", "content": json.dumps(user_payload)}], role="super")
            plan = self._parse_plan(txt, objects, fixtures)
            self.source = "token_factory"
            self._cache[key] = plan
            self._save_cache()
        except TokenFactoryNotConfigured as e:
            self.log(f"[sim] VLMPlanner: Token Factory not configured ({e}); using rule-based parser")
            plan, self.source = rule_plan(instruction, objects, fixtures), "rule_based"
        except Exception as e:  # network / parse errors also fall back, loudly
            if isinstance(e, OSError):  # URLError and timeouts: the endpoint is down, do not retry it per task
                self._offline = True
                self.log("[sim] VLMPlanner: endpoint unreachable; rule-based parser for the rest of this run")
            self.log(f"[sim] VLMPlanner: model call failed ({e!r}); using rule-based parser")
            plan, self.source = rule_plan(instruction, objects, fixtures), "rule_based"
        return plan


class VLANotConfigured(RuntimeError):
    pass


class VLAClient(Policy):
    """Client for a GR00T N1.x policy server (e.g. gr00t inference service behind an HTTP shim).

    POST {GR00T_POLICY_URL}/act  JSON:
      {"video.ego_view": <base64 PNG HxWx3 uint8>, "video.wrist": <base64 PNG>, "state.single_arm": [7],
       "state.gripper": [1], "annotation.human.task_description": "<instruction>"}
    Response: {"action.single_arm": [[7], ...], "action.gripper": [[1], ...]}  (absolute joint targets, chunk)
    With a mobile base the request also has "state.base" [3] and the response "action.base" [[3], ...] (11 floats a step).
    Matches meta/modality.json of datasets written by `arrival sim`; the wire helpers live in
    arrival.policy_server.protocol, which also checks the response (ProtocolError, a ValueError).
    """

    def __init__(self, url: str | None = None, timeout: float = 10.0, schema: str | None = None):
        from arrival import config as settings

        self.schema = schema  # "arm" (8 floats), "mobile" (11, adds the base pose) or None: "mobile" when the env has a mobile base

        self.url = url or settings.get("GR00T_POLICY_URL")
        if not self.url:
            raise VLANotConfigured("GR00T_POLICY_URL is not set; start a GR00T policy server and export its URL.")
        self.timeout = timeout
        self.queue: list[np.ndarray] = []

    def reset(self, env, task: dict) -> None:
        self.instruction = task["instruction"]
        self.queue = []
        self.active_schema = self.schema or ("mobile" if getattr(env, "mobile", False) else "arm")

    def act(self, obs: dict) -> np.ndarray:
        if not self.queue:
            from arrival.policy_server.protocol import decode_response, encode_request

            schema = getattr(self, "active_schema", self.schema or "arm")
            body = json.dumps(encode_request(obs, self.instruction, schema)).encode()
            req = urllib.request.Request(self.url.rstrip("/") + "/act", data=body, method="POST",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                self.queue = list(decode_response(json.load(r), schema))
        return np.asarray(self.queue.pop(0), dtype=float)
