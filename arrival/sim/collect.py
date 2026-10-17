"""Run sim episodes at 20 Hz, record real joint states / commanded targets / camera frames,
check success with physics, and write a LeRobot v2 dataset."""

from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from arrival.sim import robot
from arrival.sim.people import Walkers
from arrival.sim.safety import SafetyMonitor
from arrival.catalog.objects import HANDLE_HELD, RIM_HELD
from arrival.sim.scene import BOWL_WALL, PLATE_FLOOR, PLATE_RIM, build_mjcf, door_w, drawer_x

FPS = 20
SUBSTEPS = 25  # 0.002 s physics -> 20 Hz control
SETTLE_STEPS = 12
DRAWER_OPEN = 0.30  # start state (m) for close_drawer; open_drawer counts > 0.15 as open
HANDLE_ROOT = 0.017  # grasp point out from a pan's ring: pad half-width 0.01 plus 7 mm
PAD_BELOW_EEF = 0.023  # the finger pads reach this far below the eef site (robot.py hand geometry)
DRAWER_OPEN_PUT = 0.14  # start state for put_in_drawer: wide enough for a mug on the exposed tray, and the base
# can then park straight in front without the pulled-out front reaching its column
DOOR_OPEN = np.radians(70)  # start state for close_door; open_door counts > 60 deg as open
WIPE_MIN = 0.4  # metres the sponge must travel pressed to the counter for wipe_counter to count


def _front(f: dict) -> np.ndarray:
    y = f.get("yaw", 0.0)
    return np.array([-np.sin(y), np.cos(y), 0.0])


class SimEnv:
    def __init__(self, twin: dict, task: dict, res: int = 128, cameras=("ego_view", "wrist"), overview: bool = False):
        self.twin, self.task = twin, task
        self.fx = {f["id"]: f for f in twin["fixtures"]}
        self.objs = {o["id"]: o for o in twin["objects"]}
        obj_pos = {}
        if "object_id" in task:
            o = self.objs[task["object_id"]]
            src = self.fx[task["source"]]
            obj_pos[o["id"]] = [task["start"][0], task["start"][1], self._top(src) + o["size"][2] / 2 + 0.003]
        base, yaw = self._base_pose(task, obj_pos)
        xml = build_mjcf(twin, base, yaw, obj_pos, task.get("lighting", "neutral"), task.get("people", 0))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.qadr = [int(self.model.joint(j).qposadr[0]) for j in robot.ARM_JOINTS]
        self.dofadr = [int(self.model.joint(j).dofadr[0]) for j in robot.ARM_JOINTS]
        self.fadr = [int(self.model.joint(f"finger_{s}").qposadr[0]) for s in ("l", "r")]
        self.data.qpos[self.qadr] = robot.HOME
        self.data.qpos[self.fadr] = robot.FINGER_OPEN
        if task["family"] in ("close_drawer", "close_door", "put_in_drawer"):  # the part starts open
            jn = f'{task["fixture"]}_door' if task["family"] == "close_door" else f'{task["fixture"]}_drawer0'
            start = {"close_drawer": DRAWER_OPEN, "close_door": DOOR_OPEN, "put_in_drawer": DRAWER_OPEN_PUT}
            self.data.qpos[int(self.model.joint(jn).qposadr[0])] = start[task["family"]]
        self.data.ctrl[:] = list(robot.HOME) + [robot.FINGER_OPEN] * 2
        self.wiped = 0.0  # xy distance the sponge travelled while pressed to the counter (wipe_counter)
        free = self._free_floor()
        front = max(f["position"][1] + f["size"][1] / 2 for f in twin["fixtures"])
        self.walkers = Walkers(task.get("people", 0), free, seed=task.get("seed", 0), base_xy=base[:2],
                               front_y=front) if task.get("people") else None
        self.safety = SafetyMonitor(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.cameras, self.res = cameras, res
        self.renderer = mujoco.Renderer(self.model, res, res)
        self.big = mujoco.Renderer(self.model, 360, 480) if overview else None
        for _ in range(SETTLE_STEPS):
            self.step(np.array(list(robot.HOME) + [robot.FINGER_OPEN]))

    # ---- geometry helpers ----
    @staticmethod
    def _top(f: dict) -> float:
        return f["position"][2] + f["size"][2] / 2

    def _free_floor(self):
        xs = [p for w in self.twin["walls"] for p in (w["start"][0], w["end"][0])]
        ys = [p for w in self.twin["walls"] for p in (w["start"][1], w["end"][1])]
        deep = max(f["position"][1] + f["size"][1] / 2 for f in self.twin["fixtures"])
        return (min(xs) + 0.4, deep + 0.9, max(xs) - 0.4, max(ys) - 0.3)

    def _base_pose(self, task: dict, obj_pos: dict):
        fam = task["family"]
        if fam in ("counter_to_sink", "counter_to_counter"):
            src = self.fx[task["source"]]
            p = np.array(obj_pos[task["object_id"]])
            tgt = self.fx[task["target"]]
            mid = (p[0] + task["target_x"]) / 2
            n = _front(src)
            y = src["position"][1] + n[1] * (src["size"][1] / 2 + 0.28)
            base = (mid, y, 0.75)
            task.setdefault("_place", None)
        elif fam == "wipe_counter":
            src = self.fx[task["source"]]
            p = np.array(obj_pos[task["object_id"]])
            n = _front(src)
            base = (p[0], src["position"][1] + n[1] * (src["size"][1] / 2 + 0.28), 0.75)
        elif fam == "put_in_drawer":
            # straight in front, between the mug and the drawer, just beyond the pulled-out front and its handle;
            # parking beside the drawer instead puts the elbow into whatever appliance flanks the counter
            c = self.fx[task["fixture"]]
            n = _front(c)
            p = np.array(obj_pos[task["object_id"]])
            lx = drawer_x(c, self.twin["fixtures"])
            yaw = c.get("yaw", 0.0)
            mug_lx = np.cos(yaw) * (p[0] - c["position"][0]) + np.sin(yaw) * (p[1] - c["position"][1])
            along = (lx + mug_lx) / 2
            out = c["size"][1] / 2 + DRAWER_OPEN_PUT + 0.07 + 0.14  # front + handle bar + base column + margin
            base = (c["position"][0] + np.cos(yaw) * along, c["position"][1] + np.sin(yaw) * along + n[1] * out, 0.75)
        else:
            c = self.fx[task["fixture"]]
            n = _front(c)
            sx, sy = c["size"][0], c["size"][1]
            if fam in ("open_drawer", "close_drawer"):
                lx, lz = drawer_x(c, self.twin["fixtures"]), robot_drawer_z()
                along = 0.0
                h = 0.72
            else:
                # stand just left of the handle (away from the hinge), outside the door's swing, far enough out
                # that the far end of the handle arc stays within reach of the tilted-down grasp
                lx, lz = sx / 2 - 0.02 - door_w(c, self.twin["fixtures"]) + 0.06, 0.53
                along = -0.1
                h = 0.65
            yaw = c.get("yaw", 0.0)
            hx = c["position"][0] + np.cos(yaw) * (lx + along)
            hy = c["position"][1] + np.sin(yaw) * (lx + along) + n[1] * (sy / 2 + h)
            base = (hx, hy, 0.75)
        return base, float(np.degrees(np.arctan2(-n[1], -n[0])))

    def body_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def site_pos(self, name: str) -> np.ndarray:
        return self.data.site(name).xpos.copy()

    def obj_half_h(self, oid: str) -> float:
        return self.objs[oid]["size"][2] / 2

    def grasp(self, oid: str) -> dict | None:
        """How the expert holds an object it cannot grip by the body: xy offset from the body centre to the grasp
        point, the yaw for ik.down (fingers close across the wall or the handle) and the eef height. A bowl and a
        plate are pinched by the rim nearest the robot, the bowl 2 cm under its lip, the plate with the pads just
        above its bottom disc; a pan by its handle, 1.7 cm out from the ring. None for objects held by the body."""
        o = self.objs[oid]
        cid, p = o["catalog_id"], self.body_pos(oid)
        radius, sz = min(o["size"][0], o["size"][1]) / 2, o["size"][2]
        if cid in RIM_HELD:
            d = self.data.body("link1").xpos[:2] - p[:2]
            d = d / (np.linalg.norm(d) + 1e-9)
            wall = BOWL_WALL if cid == "bowl" else PLATE_RIM
            z = p[2] + sz / 2 - 0.02 if cid == "bowl" else p[2] - sz / 2 + PLATE_FLOOR + PAD_BELOW_EEF + 0.003
            return {"off": d * (radius - wall / 2), "yaw": float(np.arctan2(d[0], -d[1])), "z": z}
        if cid in HANDLE_HELD:  # the handle runs along +x from the ring (object_xml ignores the twin yaw); the pads
            # sit at its root, the thickest part; the pan tips and hangs from the handle during the carry
            return {"off": np.array([radius + HANDLE_ROOT, 0.0]), "yaw": 0.0, "z": p[2] + sz / 2 - 0.008}
        return None

    def place_point(self, task: dict) -> np.ndarray:
        tgt = self.fx[task["target"]]
        hh = self.obj_half_h(task["object_id"])
        top = self._top(tgt) if tgt["type"] != "sink" else max(
            self._top(f) for f in self.twin["fixtures"] if f["type"] == "counter")
        if tgt["type"] == "sink":
            return np.array([tgt["position"][0], tgt["position"][1], top + hh + 0.05])
        return np.array([task["target_x"], tgt["position"][1], top + hh + 0.012])

    def pull_axis(self, task: dict) -> np.ndarray:
        return _front(self.fx[task["fixture"]])

    def place_point_top(self, task: dict) -> float:
        return self._top(self.fx[task["source"]])

    def drawer_place_point(self, task: dict) -> np.ndarray:
        """Where to set an object down on the pulled-out part of the open drawer's tray."""
        c = self.fx[task["fixture"]]
        n = _front(c)
        tray = self.data.geom(f'{task["fixture"]}_drawer0_tray')
        half = self.model.geom(f'{task["fixture"]}_drawer0_tray').size
        front = np.array(c["position"][:2]) + n[:2] * (c["size"][1] / 2 - 0.02)  # a point on the carcass face
        along = float(np.dot(tray.xpos[:2] - front, n[:2]))  # tray centre, measured out from that face
        far = along + float(half[1])  # how far the tray reaches past the face
        mid = 0.02 + (far - 0.02) / 2  # middle of the exposed part of the tray
        xy = tray.xpos[:2] + n[:2] * (mid - along)  # keep the drawer's own x; the counter centre is not the drawer
        return np.array([xy[0], xy[1], tray.xpos[2] + half[2] + self.obj_half_h(task["object_id"]) + 0.01])

    def door_geometry(self, task: dict):
        b = self.data.body(f'{task["fixture"]}_door')
        return b.xpos.copy(), self.site_pos(f'{task["fixture"]}_door_handle'), b.xmat.reshape(3, 3).copy()

    # ---- stepping ----
    def state(self) -> np.ndarray:
        q = self.data.qpos
        return np.array(list(q[self.qadr]) + [float(np.mean(q[self.fadr]))], dtype=np.float32)

    def step(self, action: np.ndarray) -> None:
        self.data.ctrl[:7] = action[:7]
        self.data.ctrl[7:9] = action[7]
        before = self.body_pos(self.task["object_id"])[:2] if self.task["family"] == "wipe_counter" else None
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self.people_xy = self.walkers.step(self.model, self.data, 1.0 / FPS) if self.walkers else []
        if before is not None:
            p = self.body_pos(self.task["object_id"])
            if p[2] < self._top(self.fx[self.task["source"]]) + self.obj_half_h(self.task["object_id"]) + 0.01:
                self.wiped += float(np.linalg.norm(p[:2] - before))
        self.safety.update(self.model, self.data)

    def render(self, cam: str, big: bool = False) -> np.ndarray:
        r = self.big if big else self.renderer
        r.update_scene(self.data, cam)
        return r.render().copy()

    def obs(self) -> dict:
        o = {"state": self.state()}
        for c in self.cameras:
            o[c] = self.render(c)
        return o

    # ---- physics success ----
    def success(self, task: dict) -> tuple[bool, dict]:
        fam = task["family"]
        if fam in ("counter_to_sink", "counter_to_counter"):
            oid = task["object_id"]
            p = self.body_pos(oid)
            v = float(np.linalg.norm(self.data.joint(oid).qvel[:3]))
            tgt = self.fx[task["target"]]
            (cx, cy, _), (sx, sy, _) = tgt["position"], tgt["size"]
            inside = abs(p[0] - cx) < sx / 2 - 0.01 and abs(p[1] - cy) < sy / 2 - 0.01
            if tgt["type"] == "sink":
                top = max(self._top(f) for f in self.twin["fixtures"] if f["type"] == "counter")
                ok = inside and p[2] < top and v < 0.05
            else:
                ok = inside and abs(p[2] - (self._top(tgt) + self.obj_half_h(oid))) < 0.025 and v < 0.05
            return bool(ok), {"object_pos": p.round(4).tolist(), "object_speed": round(v, 4)}
        if fam == "put_in_drawer":
            oid = task["object_id"]
            p = self.body_pos(oid)
            v = float(np.linalg.norm(self.data.joint(oid).qvel[:3]))
            tray = self.data.geom(f'{task["fixture"]}_drawer0_tray')
            half = self.model.geom(f'{task["fixture"]}_drawer0_tray').size
            local = tray.xmat.reshape(3, 3).T @ (p - tray.xpos)
            inside = abs(local[0]) < half[0] - 0.02 and abs(local[1]) < half[1] - 0.02
            resting = abs(local[2] - (half[2] + self.obj_half_h(oid))) < 0.03
            return bool(inside and resting and v < 0.05), {
                "object_pos": p.round(4).tolist(), "object_speed": round(v, 4),
                "drawer_qpos": round(float(self.data.joint(task["joint"]).qpos[0]), 4)}
        if fam == "wipe_counter":
            oid = task["object_id"]
            p = self.body_pos(oid)
            v = float(np.linalg.norm(self.data.joint(oid).qvel[:3]))
            src = self.fx[task["source"]]
            (cx, cy, _), (sx, sy, _) = src["position"], src["size"]
            on_top = (abs(p[0] - cx) < sx / 2 and abs(p[1] - cy) < sy / 2
                      and abs(p[2] - (self._top(src) + self.obj_half_h(oid))) < 0.025)
            return bool(on_top and self.wiped >= WIPE_MIN and v < 0.05), {
                "object_pos": p.round(4).tolist(), "wiped_m": round(self.wiped, 3)}
        jn = task["joint"]
        q = float(self.data.joint(jn).qpos[0])
        ok = {"open_drawer": q > 0.15, "open_door": q > np.radians(60),
              "close_drawer": q < 0.03, "close_door": q < np.radians(5)}[fam]
        return bool(ok), {"joint": jn, "qpos": round(q, 4)}


def robot_drawer_z() -> float:
    from arrival.sim.scene import CARCASS_H

    return CARCASS_H - 0.01


def run_episode(env: SimEnv, policy, task: dict, max_steps: int = 400, preview: list | None = None) -> dict:
    policy.reset(env, task)
    env.safety.reset()
    states, actions, frames, people = [], [], {c: [] for c in env.cameras}, []
    tail = 0
    for t in range(max_steps):
        obs = env.obs()
        a = policy.act(obs)
        states.append(obs["state"])
        actions.append(np.asarray(a, dtype=np.float32))
        for c in env.cameras:
            frames[c].append(obs[c])
        if preview is not None:
            preview.append(env.render("overview", big=True))
        env.step(a)
        people.append(env.people_xy)
        if getattr(policy, "done", lambda: False)():
            tail += 1
            if tail > 10:
                break
    ok, detail = env.success(task)
    return {"states": np.array(states), "actions": np.array(actions), "videos": {c: np.array(v) for c, v in frames.items()},
            "success": ok, "detail": detail, "people_xy": people, "ik_error": getattr(policy, "ik_error", None),
            "safety": env.safety.summary()}


def collect(twin: dict, families: list[str], episodes: int, out_dir: str | Path = "out", people: int = 2,
            policy_name: str = "scripted", res: int = 128, seed: int = 0, save_preview: bool = False,
            planner_name: str = "rule", cache_path: str | Path | None = None,
            split: str = "train", use_generated_instructions: bool = False,
            log=print) -> dict:
    """Collect `episodes` attempts; failed attempts are dropped and counted."""
    from arrival.sim.export import SimDatasetWriter
    from arrival.sim.policy import ScriptedExpert, VLAClient, VLMPlanner
    from arrival.sim.tasks import make_tasks

    out_dir = Path(out_dir)
    room = twin["room_id"]
    tasks = make_tasks(twin, families, episodes, seed=seed, people=people,
                       split=split, use_generated_instructions=use_generated_instructions, log=log)
    for t in tasks:
        if t["family"] in ("open_drawer", "close_drawer", "put_in_drawer"):
            t["handle"], t["joint"] = f'{t["fixture"]}_drawer0_handle', f'{t["fixture"]}_drawer0'
        elif t["family"] in ("open_door", "close_door"):
            t["joint"] = f'{t["fixture"]}_door'
    policy = ScriptedExpert() if policy_name == "scripted" else VLAClient()
    planner = VLMPlanner(log=log, cache_path=cache_path) if planner_name == "vlm" else None
    writer = SimDatasetWriter(out_dir / "datasets" / f"{room}_sim", room, FPS, res, ("ego_view", "wrist"))
    t0 = time.perf_counter()
    per = {}
    preview = [] if save_preview else None
    for i, task in enumerate(tasks):
        if planner is not None:
            obj_ids = [o["id"] for o in twin.get("objects", [])]
            fx_ids = [f["id"] for f in twin.get("fixtures", [])]
            plan = planner.plan(task["instruction"], obj_ids, fx_ids)
            task["vlm_plan"] = plan
            task["planned_by"] = planner.source
            if plan.get("family") and plan["family"] in families:
                task["family"] = plan["family"]
            if plan.get("object") and plan["object"] in obj_ids:
                task["object_id"] = plan["object"]
            if plan.get("target") and plan["target"] in fx_ids:
                task["target"] = plan["target"]
            log(f"[sim] task {task['task_id']}: planned by {planner.source} -> {plan}")
        env = SimEnv(twin, task, res=res, overview=save_preview and i == 0)
        ep = run_episode(env, policy, task, preview=preview if i == 0 else None)
        fam = per.setdefault(task["family"], {"attempts": 0, "successes": 0, "human_contacts": 0})
        fam["attempts"] += 1
        contact = ep["safety"]["human_contact"]
        fam["human_contacts"] += contact
        tag = "DROP human_contact" if contact else ("OK " if ep["success"] else "MISS")
        log(f"[sim] {task['task_id']}: {tag} {ep['detail']} ik_err={ep['ik_error'] if ep['ik_error'] is None else round(ep['ik_error'], 4)} "
            f"min_human_d={ep['safety']['min_human_distance_m']}")
        if ep["success"] and not contact:
            fam["successes"] += 1
            writer.add_episode(ep, task)
        if save_preview and i == 0:
            _write_preview(out_dir / "previews", room, preview, ep)
    seconds = time.perf_counter() - t0
    info = writer.finish(len(tasks))
    succ = sum(f["successes"] for f in per.values())
    report = {"room_id": room, "generator": "mujoco_scripted_ik", "policy": policy_name,
              "planner": planner_name, "attempts": len(tasks),
              "successes": succ, "dropped": len(tasks) - succ, "seconds": round(seconds, 2),
              "demos_per_hour": round(succ / seconds * 3600, 1) if seconds else 0.0,
              "human_contacts": sum(f["human_contacts"] for f in per.values()),
              "families": {k: {**v, "success_rate": round(v["successes"] / v["attempts"], 3)} for k, v in per.items()},
              "people": people, "resolution": res, "physics_checked_success": True, "grasp": "friction"}
    rp = out_dir / "reports" / f"{room}_sim.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    return {"dataset": writer.root, "report": rp, "report_data": report, "info": info}


def _write_preview(d: Path, room: str, frames: list, ep: dict) -> None:
    from PIL import Image

    from arrival.export.lerobot import write_video

    d.mkdir(parents=True, exist_ok=True)
    write_video(d / f"{room}_sim_overview.mp4", np.array(frames), FPS)
    picks = [frames[int(k)] for k in np.linspace(0, len(frames) - 1, 6)]
    ego = ep["videos"]["ego_view"]
    egos = [np.array(Image.fromarray(ego[int(k)]).resize((480, 360))) for k in np.linspace(0, len(ego) - 1, 6)]
    sheet = np.concatenate([np.concatenate(picks[:3], 1), np.concatenate(picks[3:], 1),
                            np.concatenate(egos[:3], 1), np.concatenate(egos[3:], 1)], 0)
    Image.fromarray(sheet).save(d / f"{room}_sim_contact.png")
